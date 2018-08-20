#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include "../common/cuda_utils.h"
#include "gpu_swap.h"

namespace mxnet{

Swap* Swap::Get() {
  static Swap *s = _GetSharedRef().get();
  return s;
}

std::shared_ptr<Swap> Swap::_GetSharedRef() {
  static std::shared_ptr<Swap> inst(new Swap());
  return inst;
}

Swap::Swap() {
  std::cout << "Initialize Swap" << std::endl;
  memory_history_ = MemoryHistory::_GetSharedRef();
  memory_manager_ = GetMemoryManagerRef();
  swap_lock_ = PTHREAD_RWLOCK_INITIALIZER;
  swap_async_ = dmlc::GetEnv("MXNET_SWAP_ASYNC", true);
  for (int i = 0; i < NUMBER_OF_GPU; ++i) {
    locks_[i] = PTHREAD_RWLOCK_INITIALIZER;
    // TODO(fegin): This and other cuda related code should be protected
    //              by MXNET_USE_CUDA.
    cudaStreamCreate(&streams_[i]);
  }
  swap_locked_ = false;
  std::cout << "SWAP_ASYNC=" << swap_async_ << std::endl;
  std::cout << "Initialize Swap done" << std::endl;
}

Swap::~Swap() {
  std::cout << "Destroy Swap" << std::endl;
}

void Swap::SwapOutLocked(unsigned required_memory, int device_id, bool async) {
  pthread_rwlock_wrlock(&swap_lock_);
  SwapOut(required_memory, device_id, async);
  pthread_rwlock_unlock(&swap_lock_);
}

// Caller holds swap_lock_
void Swap::SwapOut(unsigned required_memory, int device_id, bool async) {
  while (!memory_manager_->TryAllocate(device_id, required_memory)) {
    SwapParams param = {0, required_memory, &divided_handles_[device_id]};
    handle_id_t victim = memory_history_->DecideVictim(
                            swappable_handles_[device_id], device_id, &param);
    if (swap_info_.find(victim) == swap_info_.end()) {
      std::cout << "Victim does not exist (deleted?)" << std::endl;
      CHECK(0);
    }
    SwapInfo *target = swap_info_[victim];
    //std::cout << "SwapOut " << victim << " " << target->size << " "
              //<< target->swap_count << std::endl;
    target->swap_count++;
    memory_history_->DevHistory(device_id).num_swap_out++;
    memory_history_->DevHistory(device_id).swap_out_total += target->size;
    if (target->cpu_address == nullptr) {
      target->cpu_address = new char[int(target->size)];
    }
    CHECK(target->cpu_address != nullptr);
    CHECK(target->swapped_in);
    CHECK(!target->is_swapping.test_and_set(std::memory_order_acquire));
    CHECK(target->dptr != nullptr);
    target->swapped_in = false;
    swappable_handles_[device_id].erase(victim);
    divided_handles_[device_id][target->size].erase(victim);
    pthread_rwlock_unlock(&swap_lock_);
    if (async) {
      memory_manager_->MemcpyAsync(device_id, target->cpu_address,
          target->dptr, target->size, cudaMemcpyDeviceToHost,
          streams_[device_id]);
      memory_manager_->StreamSynchronize(device_id, streams_[device_id]);
    } else {
      memory_manager_->Memcpy(device_id, target->cpu_address, target->dptr,
          target->size, cudaMemcpyDeviceToHost);
    }
    memory_manager_->Free(target->dptr, device_id);
    pthread_rwlock_wrlock(&swap_lock_);
    target->is_swapping.clear(std::memory_order_release);
  }
}

// Caller holds swap_lock_
void Swap::SwapIn(SwapInfo *info, bool async) {
  while (info->is_swapping.test_and_set(std::memory_order_acquire)) {
    // TODO(fegin): usleep may not be efficient and may cause unstable
    //              execution. This is not important for now but can be
    //              something to imporve in the future. However, this
    //              must be designed carefully. One mutex per handle is not
    //              acceptable unlike current atmoic variable design.
    pthread_rwlock_unlock(&swap_lock_);
    usleep(10);
    pthread_rwlock_wrlock(&swap_lock_);
  }
  if (!info->swapped_in) {
    CHECK(!info->swapped_in);
    CHECK(info->cpu_address != nullptr);
    //std::cout << "SwapIn "<< info->handle_id << " " << info->size << " "
    //          << info->swap_count << std::endl;
    SwapOut(info->size, info->device_id, async);
    pthread_rwlock_unlock(&swap_lock_);
    memory_history_->DevHistory(info->device_id).num_swap_in++;
    memory_history_->DevHistory(info->device_id).swap_in_total += info->size;
    memory_manager_->Malloc(info->dptr, info->size, info->device_id);
    if (async) {
      memory_manager_->MemcpyAsync(info->device_id, info->dptr,
          info->cpu_address, info->size, cudaMemcpyHostToDevice,
          streams_[info->device_id]);
      memory_manager_->StreamSynchronize(info->device_id,
          streams_[info->device_id]);
    } else {
      memory_manager_->Memcpy(info->device_id, info->dptr, info->cpu_address,
                             info->size, cudaMemcpyHostToDevice);
    }
    delete info->cpu_address;
    info->cpu_address = nullptr;
    pthread_rwlock_wrlock(&swap_lock_);
    info->swapped_in = true;
    swappable_handles_[info->device_id].insert(info->handle_id);
    divided_handles_[info->device_id][info->size].insert(info->handle_id);
  }
  info->is_swapping.clear(std::memory_order_release);
}

void Swap::SetAddr(handle_id_t handle_id, void* dptr, size_t size,
                  int device_id) {
  if (device_id != -1) {
    memory_history_->PutRecord(handle_id, device_id, MemoryHistory::SET_ADDR,
                               size);
  }
  if (dptr == nullptr) {
    return;
  }
  pthread_rwlock_wrlock(&swap_lock_);
  //std::cout << "SetAddr " << handle_id << std::endl;
  auto iter = swap_info_.find(handle_id);
  if (iter == swap_info_.end()) {
    SwapInfo* info = new SwapInfo{handle_id, true, device_id,
      dptr, nullptr, size, 0, ATOMIC_FLAG_INIT};
    swap_info_[handle_id] = info;
    // FIXME(Sotskin): Temporaty Fix
    if (device_id != -1 && size >= 20240) {
    //if (device_id != -1 && size >= 20240  && handle_id != 4341) {
      swappable_handles_[device_id].insert(handle_id);
      divided_handles_[device_id][size].insert(handle_id);
    }
  } else {
    std::cout << "SetAddr duplicated id " << handle_id << std::endl;
    std::cout << "SetAddr " << iter->second->size << " " << size << std::endl;
  }
  pthread_rwlock_unlock(&swap_lock_);
}

void Swap::FreeAddr(handle_id_t handle_id) {
  pthread_rwlock_wrlock(&swap_lock_);
  //std::cout << "FreeAddr " << handle_id << std::endl;
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1) {
    memory_history_->PutRecord(handle_id, info->device_id, MemoryHistory::DEL_ADDR,
                               info->size);
    if (swappable_handles_[info->device_id].find(handle_id)
        != swappable_handles_[info->device_id].end()) {
      swappable_handles_[info->device_id].erase(handle_id);
    }
    if (divided_handles_[info->device_id][info->size].find(handle_id)
        != divided_handles_[info->device_id][info->size].end()) {
      divided_handles_[info->device_id][info->size].erase(handle_id);
    }
  }
  size_t free, total;
  memory_manager_->MemGetInfo(info->device_id, &total, &free);
  if (info->swapped_in) {
    memory_manager_->Free(info->dptr, info->device_id);
  }
  if (info->cpu_address != nullptr) {
    delete info->cpu_address;
  }
  delete info;
  swap_info_.erase(handle_id);
  pthread_rwlock_unlock(&swap_lock_);
}

void Swap::DelAddr(handle_id_t handle_id) {
  pthread_rwlock_wrlock(&swap_lock_);
  //std::cout << "DelAddr " << handle_id << std::endl;
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1) {
    memory_history_->PutRecord(handle_id, info->device_id, MemoryHistory::DEL_ADDR,
                               info->size);
    if (swappable_handles_[info->device_id].find(handle_id)
        != swappable_handles_[info->device_id].end()) {
      swappable_handles_[info->device_id].erase(handle_id);
    }
    if (divided_handles_[info->device_id][info->size].find(handle_id)
        != divided_handles_[info->device_id][info->size].end()) {
      divided_handles_[info->device_id][info->size].erase(handle_id);
    }
  }
  if (info->cpu_address != nullptr) {
    delete info->cpu_address;
  }
  delete info;
  swap_info_.erase(handle_id);
  pthread_rwlock_unlock(&swap_lock_);
}

// TODO(sotskin) compatibility for MKLMEM
void* Swap::GetAddr(handle_id_t handle_id, bool prefetch) {
  pthread_rwlock_wrlock(&swap_lock_);
  //std::cout << "GetAddr: "<< handle_id << std::endl;
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1 && !prefetch) {
    memory_history_->DevHistory(info->device_id).num_get_addr++;
    memory_history_->PutRecord(handle_id, info->device_id, MemoryHistory::GET_ADDR,
                               info->size);
  }
  if (!info->swapped_in) {
    if (!prefetch) {
      ++(memory_history_->DevHistory(info->device_id).cache_miss);
    }
    SwapIn(info, swap_async_);
  }
  if (swap_locked_ &&
      swappable_handles_[info->device_id].find(handle_id) !=
      swappable_handles_[info->device_id].end()) {
    swappable_handles_[info->device_id].erase(handle_id);
    divided_handles_[info->device_id][info->size].erase(handle_id);
    locked_handles_[info->device_id].push(handle_id);
  }
  pthread_rwlock_unlock(&swap_lock_);
  return info->dptr;
}

void Swap::LockSwap() {
  pthread_rwlock_wrlock(&swap_lock_);
  swap_locked_ = true;
  pthread_rwlock_unlock(&swap_lock_);
}

void Swap::UnlockSwap() {
  if (swap_locked_ == false) return;
  pthread_rwlock_wrlock(&swap_lock_);
  swap_locked_ = false;
  for (int i = 0; i < NUMBER_OF_GPU; ++i) {
    while (!locked_handles_[i].empty()) {
      auto info = swap_info_.find(locked_handles_[i].top());
      if (info  == swap_info_.end()) {
        locked_handles_[i].pop();
        continue;
      } else {
        divided_handles_[i][info->second->size].insert(locked_handles_[i].top());
      }
      swappable_handles_[i].insert(locked_handles_[i].top());
      locked_handles_[i].pop();
    }
  }
  pthread_rwlock_unlock(&swap_lock_);
}

void Swap::PrintHandles() {
  std::cout << "Print Handles" << std::endl;
  //std::map<size_t, std::unordered_set<handle_id_t> > _divided_handles_;
  for (auto it : swap_info_) {
    //_divided_handles_[it.second->size].insert(it.first);
    std::cout << it.first << ": " << it.second->size << " "
              << it.second->swap_count << " " << it.second->device_id
              << std::endl;
  }
}

} // namespace mxnet
