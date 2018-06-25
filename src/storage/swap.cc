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

Swap::Swap(){
  std::cout << "Initialize Swap" <<std::endl;
  memory_history_ = MemHistory::_GetSharedRef();
  memory_manager_ = MemoryManager::_GetSharedRef();
  swap_lock_ = PTHREAD_RWLOCK_INITIALIZER;
  for (int i = 0; i < NUMBER_OF_GPU; ++i){
    locks_[i] = PTHREAD_RWLOCK_INITIALIZER;
    free_memory_.push_back(0);
  }
}

Swap::~Swap(){
  std::cout << "Destroy Swap" <<std::endl;
}

void Swap::SwapOut(unsigned required_memory, int device){
  UpdateFree(device);
  if (free_memory_[device] > required_memory) {
    return;
  }
  while (free_memory_[device] < required_memory) {
    handle_id_t victim = 
      memory_history_->DecideVictim(swappable_handles_[device], device);
    SwapInfo *target = swap_info_[victim];
    if(target->cpu_address == nullptr) {
      target->cpu_address = new char[int(target->size)];
    }
    CHECK(target->swapped_in);
    CHECK(target->dptr != nullptr);
    target->swapped_in = false;
    swappable_handles_[device].erase(victim);
    memory_manager_->Memcpy(device, target->cpu_address, target->dptr,
        target->size, cudaMemcpyDeviceToHost);
    memory_manager_->Free(target->dptr, device);
    UpdateFree(device);
  }
}

void Swap::SwapIn(SwapInfo *info){
  CHECK(!info->swapped_in);
  CHECK(info->cpu_address != nullptr);
  SwapOut(info->size, info->device_id);
  e = memory_manager_->Malloc(&(info->dptr), info->size);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
  }
  memory_manager_->Memcpy(device, info->dptr, info->cpu_address, info->size,
      cudaMemcpyHostToDevice);
  info->swapped_in = true;
  swappable_handles_[old_device_].insert(info->handle_id);
}

void Swap::SetAddr(handle_id_t handle_id, void* dptr, size_t size, int dev_id){
  if (dev_id != -1){
    memory_history_->PutRecord(handle_id, dev_id, MemHistory::SET_ADDR, size);
  }
  if (dptr == nullptr) {
    return;
  }
  pthread_rwlock_wrlock(&swap_lock_);
  auto iter = swap_info_.find(handle_id);
  if (iter == swap_info_.end()){
    SwapInfo* info = new SwapInfo{handle_id, true, dev_id, dptr, nullptr, size};
    swap_info_[handle_id] = info;
    if (dev_id != -1){
      swappable_handles_[dev_id].insert(handle_id);
    }
  } else {
    std::cout << "SetAddr duplicated id " << handle_id << std::endl;
    std::cout << "SetAddr " << iter->second->size << " " << size << std::endl;
  }
  pthread_rwlock_unlock(&swap_lock_);
}

void Swap::DelAddr(handle_id_t handle_id){
  pthread_rwlock_wrlock(&swap_lock_);
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1) {
    memory_history_->PutRecord(handle_id, info->device_id, MemHistory::DEL_ADDR, info->size);
    if (swappable_handles_[device_id].find(handle_id) 
        != swappable_handles_.end()){
      swappable_handles_[device_id].erase(handle_id);
    }
  }
  if (iter->second->cpu_address != nullptr) {
    delete iter->second->cpu_address;
  }
  delete info;
  swap_info_.erase(handle_id);
  pthread_rwlock_unlock(&swap_lock_);
}

// TODO(sotskin) compatibility for MKLMEM
void* Swap::GetAddr(handle_id_t handle_id){
  pthread_rwlock_rdlock(&swap_lock_);
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1) {
    mhistory_->PutRecord(handle_id, info->device_id, MemHistory::GET_ADDR, info->size);
  }
#if MXNET_USE_CUDA
  if (!info->swapped_in) {
    SwapIn(info);
  }
#endif // MXNET_USE_CUDA
  pthread_rwlock_unlock(&swap_lock_);
  return info->dptr;
}

int Swap::UpdateFree(int device){
  // TODO(sotskin) all CUDA_CALL shall be replaced by custom mm call
#if MXNET_USE_CUDA
  size_t free_mem, total;
  memory_manager_->MemGetInfo(device, &free_mem, &total);
  free_memory_[device] = free_mem;
#endif // MXNET_USE_CUDA
  return device;
}

} // namespace mxnet
