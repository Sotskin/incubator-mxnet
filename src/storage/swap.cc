#include <iostream>
#include <memory>
#include <mxnet/swap.h>
#include <dmlc/logging.h>
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
}

Swap::~Swap() {
  std::cout << "Destroy Swap" <<std::endl;
}

void Swap::SwapOut(unsigned required_memory, int device_id) {
  std::cout << "Entering SwapOut devid = " << device_id << std::endl;
  if (memory_manager_->TryAllocate(device_id, required_memory)) {
    std::cout << "Swap Out does nothing " << device_id << " " << required_memory << std::endl;
    return;
  }
  std::cout << "SwapOut working " << device_id << " " << required_memory << std::endl;
  while (!memory_manager_->TryAllocate(device_id, required_memory)) {
    std::cout<<"0 decidevictim handle set size = " << 
      swappable_handles_[device_id].size()<<std::endl;
    handle_id_t victim = 
      memory_history_->DecideVictim(swappable_handles_[device_id], device_id);
    std::cout<<"1 victim = "<<victim<<std::endl;
    SwapInfo *target = swap_info_[victim];
    std::cout<<"2"<<std::endl;
    if(target->cpu_address == nullptr) {
      target->cpu_address = new char[int(target->size)];
      std::cout<<"2.5 Give CPU Address = "<<target->cpu_address<<std::endl;
    }
    CHECK(target->cpu_address != nullptr);
    CHECK(target->swapped_in);
    CHECK(target->dptr != nullptr);
    target->swapped_in = false;
    std::cout<<"3 "<<swap_info_[victim]->swapped_in<<" size="<<
      target->size<<" dst="<<(void*)(target->cpu_address)<<" src="
      <<target->dptr<<std::endl;
    swappable_handles_[device_id].erase(victim);
    cudaError_t e = memory_manager_->Memcpy(device_id, target->cpu_address, target->dptr, target->size, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "Memcpy failed: " << cudaGetErrorString(e);
    }
    std::cout<<"4"<<std::endl;
    e = memory_manager_->Free(target->dptr, device_id);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "Free failed: " << cudaGetErrorString(e);
    }
    
    std::cout<<"5"<<std::endl;
  }
  std::cout<<"Swapout End"<<std::endl;
}

void Swap::SwapIn(SwapInfo *info) {
  std::cout << "SwapIn working " << std::endl;
  CHECK(!info->swapped_in);
  CHECK(info->cpu_address != nullptr);
  SwapOut(info->size, info->device_id);
  cudaError_t e = memory_manager_->Malloc(info->dptr, info->size, info->device_id);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
  }
  std::cout<<"SwapIn Memcpy size="<<info->size<<" dst="<<info->dptr
    <<" src="<<(void*)(info->cpu_address)<<std::endl;
  e = memory_manager_->Memcpy(info->device_id, info->dptr, info->cpu_address, info->size,
      cudaMemcpyHostToDevice);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    LOG(FATAL) << "Memcpy failed: " << cudaGetErrorString(e);
  }
  info->swapped_in = true;
  delete info->cpu_address;
  info->cpu_address = nullptr;
  swappable_handles_[info->device_id].insert(info->handle_id);
  swap_info_[info->handle_id]->dptr = info->dptr;
}

void Swap::SetAddr(handle_id_t handle_id, void* dptr, size_t size, int device_id) {
  std::cout<<"SetAddr " << handle_id << " " << size <<" dev_id="<<device_id << std::endl;
  if (device_id != -1){
    memory_history_->PutRecord(handle_id, device_id, MemHistory::SET_ADDR, size);
  }
  if (dptr == nullptr) {
    std::cout<<"SetAddr Nullptr return"<<std::endl;
    return;
  }
  std::cout<<"SetAddr Grabbing Wlock"<<std::endl;
  pthread_rwlock_wrlock(&swap_lock_);
  std::cout<<"SetAddr Grabbed Wlock"<<std::endl;
  auto iter = swap_info_.find(handle_id);
  if (iter == swap_info_.end()){
    SwapInfo* info = new SwapInfo{handle_id, true, device_id, dptr, nullptr, size};
    swap_info_[handle_id] = info;
    if (device_id != -1){
      swappable_handles_[device_id].insert(handle_id);
    }
  } else {
    std::cout << "SetAddr duplicated id " << handle_id << std::endl;
    std::cout << "SetAddr " << iter->second->size << " " << size << std::endl;
  }
  pthread_rwlock_unlock(&swap_lock_);
  std::cout<<"SetAddr Releasing Wlock"<<std::endl;
}

void Swap::DelAddr(handle_id_t handle_id) {
  pthread_rwlock_wrlock(&swap_lock_);
  std::cout<<"DelAddr "<<handle_id<<std::endl;
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1) {
    memory_history_->PutRecord(handle_id, info->device_id, MemHistory::DEL_ADDR, info->size);
    if (swappable_handles_[info->device_id].find(handle_id) 
        != swappable_handles_[info->device_id].end()) {
      swappable_handles_[info->device_id].erase(handle_id);
    }
  }
  if (info->cpu_address != nullptr) {
    delete info->cpu_address;
  }
  delete info;
  swap_info_.erase(handle_id);
  pthread_rwlock_unlock(&swap_lock_);
  std::cout<<"DelAddr Over"<<std::endl;
}

// TODO(sotskin) compatibility for MKLMEM
void* Swap::GetAddr(handle_id_t handle_id) {
  std::cout<<"GetAddr Grabing Rlock"<<std::endl;
  pthread_rwlock_wrlock(&swap_lock_);
  std::cout<<"GetAddr " << handle_id << std::endl;
  auto info = swap_info_.at(handle_id);
  if(info->device_id != -1) {
    size_t a, b;
    memory_manager_->MemGetInfo(info->device_id, &a, &b);
  }
  if (info->device_id != -1) {
    memory_history_->PutRecord(handle_id, info->device_id, MemHistory::GET_ADDR, info->size);
  }
  std::cout<<"GetAddr size " << info->size << 
    " swapped in = " << info->swapped_in << std::endl;
  if (!info->swapped_in) {
    std::cout<<"GetAddr Swap in"<<std::endl;
    SwapIn(info);
  }
  pthread_rwlock_unlock(&swap_lock_);
  if(info->device_id != -1) {
    size_t a, b;
    memory_manager_->MemGetInfo(info->device_id, &a, &b);
  }
  std::cout<<"GetAddr Over"<<std::endl;
  return info->dptr;
}

int Swap::UpdateFree(int device) {
#if MXNET_USE_CUDA
  size_t free_mem, total;
  memory_manager_->MemGetInfo(device, &free_mem, &total);
  free_memory_[device] = free_mem;
#endif // MXNET_USE_CUDA
  return device;
}

} // namespace mxnet
