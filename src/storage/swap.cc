#include <iostream>
#include <map>
#include <memory>
#include <mxnet/swap.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include "../common/cuda_utils.h"

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
  std::cout << "Initialize Swap" <<std::endl;
  memory_history_ = MemHistory::_GetSharedRef();
  memory_manager_ = MemoryManager::_GetSharedRef();
  swap_lock_ = PTHREAD_RWLOCK_INITIALIZER;
  for (int i = 0; i < NUMBER_OF_GPU; ++i){
    locks_[i] = PTHREAD_RWLOCK_INITIALIZER;
    free_memory_.push_back(0);
  }
  swap_locked_ = false;
  swap_algorithm_ = dmlc::GetEnv("SWAP_ALGORITHM", std::string("LRU"));
}

Swap::~Swap() {
  std::cout << "Destroy Swap" <<std::endl;
}

void Swap::SwapOutLocked(unsigned required_memory, int device_id) {
  pthread_rwlock_wrlock(&swap_lock_);
  SwapOut(required_memory, device_id);
  pthread_rwlock_unlock(&swap_lock_);
}

// Caller holds swap_lock_
void Swap::SwapOut(unsigned required_memory, int device_id) {
  if (memory_manager_->TryAllocate(device_id, required_memory)) {
    return;
  }
  while (!memory_manager_->TryAllocate(device_id, required_memory)) {
    handle_id_t victim;
    if (swap_algorithm_ == "SizeHistory" && 
        memory_history_->GetIterationIdx() > 2) {
      auto candidates = divided_handles_[device_id].lower_bound(required_memory);
      auto original_candidates = candidates;
      if (candidates == divided_handles_[device_id].end()) {
        candidates--;
      }
      bool reverse_flag = false;
      size_t no_swap_step = 80;
      //std::cout<<"Enter loop"<<std::endl;
      while (true) {
        //std::cout<<no_swap_step<<" "<<candidates->first<<" "<<
        // candidates->second.size()<<std::endl;
        if (candidates->second.size() != 0) {
          victim = memory_history_->DecideVictim(candidates->second, device_id, &no_swap_step);
          if(victim != 0) {
            //std::cout<<"Exit loop"<<std::endl;
            break;
          }
        }
        if (!reverse_flag) {
          candidates ++;
          if (candidates == divided_handles_[device_id].end()) {
            candidates = original_candidates;
            reverse_flag = true;
          }
        }
        if (reverse_flag) {
          if (candidates == divided_handles_[device_id].begin()) {
            candidates = original_candidates;
            reverse_flag = false;
            if( no_swap_step == 0) {
              std::cout << "Cannot find victim (algorithm error)" << std::endl;
              CHECK(0);
            }
            no_swap_step /= 2;
          } else {
            candidates --;
          }
        }
      }
    } else {
      victim = memory_history_->DecideVictim(swappable_handles_[device_id], device_id,
          nullptr);
    }
    if(swap_info_.find(victim) == swap_info_.end()) {
      std::cout<<"Victim does not exist (deleted?)"<<std::endl;
      CHECK(0);
    }
    SwapInfo *target = swap_info_[victim];
    target->swap_count++;
    memory_history_->num_swap_out++;
    memory_history_->swap_out_total += target->size;
    if(target->cpu_address == nullptr) {
      target->cpu_address = new char[int(target->size)];
    }

    CHECK(target->cpu_address != nullptr);
    CHECK(target->swapped_in);
    CHECK(target->dptr != nullptr);
    target->swapped_in = false;
    swappable_handles_[device_id].erase(victim);
    divided_handles_[device_id][target->size].erase(victim);
    pthread_rwlock_unlock(&swap_lock_);
    cudaError_t e = memory_manager_->Memcpy(device_id, target->cpu_address, target->dptr, target->size, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "Memcpy failed: " << cudaGetErrorString(e);
    }
    pthread_rwlock_wrlock(&swap_lock_);
    e = memory_manager_->Free(target->dptr, device_id);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "Free failed: " << cudaGetErrorString(e);
    }
  }
}

// Caller holds swap_lock_
void Swap::SwapIn(SwapInfo *info) {
  CHECK(!info->swapped_in);
  CHECK(info->cpu_address != nullptr);
  SwapOut(info->size, info->device_id);
  memory_history_->num_swap_in++;
  memory_history_->swap_in_total += info->size;
  cudaError_t e = memory_manager_->Malloc(info->dptr, info->size, info->device_id);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
  }
  info->swapped_in = true;
  swappable_handles_[info->device_id].insert(info->handle_id);
  divided_handles_[info->device_id][info->size].insert(info->handle_id);
  swap_info_[info->handle_id]->dptr = info->dptr;
  pthread_rwlock_unlock(&swap_lock_);
  e = memory_manager_->Memcpy(info->device_id, info->dptr, info->cpu_address, info->size,
      cudaMemcpyHostToDevice);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    LOG(FATAL) << "Memcpy failed: " << cudaGetErrorString(e);
  }
  pthread_rwlock_wrlock(&swap_lock_);
  delete info->cpu_address;
  info->cpu_address = nullptr;
}

void Swap::SetAddr(handle_id_t handle_id, void* dptr, size_t size, int device_id) {
  if (device_id != -1){
    memory_history_->PutRecord(handle_id, device_id, MemHistory::SET_ADDR, size);
  }
  if (dptr == nullptr) {
    return;
  }
  pthread_rwlock_wrlock(&swap_lock_);
  auto iter = swap_info_.find(handle_id);
  if (iter == swap_info_.end()){
    SwapInfo* info = new SwapInfo{handle_id, true, device_id, 
      dptr, nullptr, size, 0};
    swap_info_[handle_id] = info;
    // FIXME(Sotskin): Temporaty Fix
    if (device_id != -1 && size >= 20240){
      swappable_handles_[device_id].insert(handle_id);
      divided_handles_[device_id][size].insert(handle_id);
    }
  } else {
    std::cout << "SetAddr duplicated id " << handle_id << std::endl;
    std::cout << "SetAddr " << iter->second->size << " " << size << std::endl;
  }
  pthread_rwlock_unlock(&swap_lock_);
}

void Swap::DelAddr(handle_id_t handle_id) {
  pthread_rwlock_wrlock(&swap_lock_);
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1) {
    memory_history_->PutRecord(handle_id, info->device_id, MemHistory::DEL_ADDR, info->size);
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
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1 && !prefetch) {
    memory_history_->num_get_addr++;
    memory_history_->PutRecord(handle_id, info->device_id, MemHistory::GET_ADDR, info->size);
  }
  if (!info->swapped_in) {
    if(!prefetch)
      ++(memory_history_->cache_miss);
    SwapIn(info);
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
  if(swap_locked_ == false) return;
  pthread_rwlock_wrlock(&swap_lock_);
  swap_locked_ = false;
  for (int i = 0; i < NUMBER_OF_GPU; ++i) {
    while (!locked_handles_[i].empty()) {
      auto info = swap_info_.find(locked_handles_[i].top());
      if(info  == swap_info_.end()){
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

void Swap::PrintHandles(){
  /*
  std::cout << "Print Handles" << std::endl;
  std::map<size_t, std::unordered_set<handle_id_t> > divided_handles_;
  for(auto it : swap_info_) {
    divided_handles_[it.second->size].insert(it.first);
    std::cout<<it.first<<": "<<it.second->size<<" "
      <<it.second->swap_count<<" "<<it.second->device_id
      <<std::endl;
  }
  std::cout << "Print Handles II" << std::endl;
  for(auto it : divided_handles_) {
    std::cout<<it.first<<": "<<it.second.size()<<std::endl;
  }
  */
}

} // namespace mxnet
