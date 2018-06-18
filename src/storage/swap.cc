#include <iostream>
#include <memory>
#include <mxnet/swap.h>
#include <mxnet/gpu_swap_history.h>

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
  mhistory_ = MemHistory::_GetSharedRef();
}

Swap::~Swap(){
  std::cout << "Destroy Swap" <<std::endl;
}

void Swap::SwapOut(unsigned required_memory, int device){
  // Not implemented  
}

void Swap::SwapIn(SwapInfo *info){
  // Not implemented
}

// TODO(sotskin) compatibility for MKLMEM
void* Swap::GetAddr(handle_id_t handle_id, size_t size){
  pthread_rwlock_rdlock(&swap_lock_);
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1) {
    mhistory_->PutRecord(handle_id, info->device_id, MemHistory::DEL_ADDR, size);
  }
  pthread_rwlock_unlock(&swap_lock_);
  return info->dptr;
}

void Swap::SetAddr(handle_id_t handle_id, void* dptr, size_t size, int dev_id){
  if (dev_id != -1){
    mhistory_->PutRecord(handle_id, dev_id, MemHistory::SET_ADDR, size);
  }
  if (dptr == nullptr) {
    return;
  }
  pthread_rwlock_wrlock(&swap_lock_);
  auto iter = swap_info_.find(handle_id);
  if (iter == swap_info_.end()){
    SwapInfo* info = new SwapInfo{handle_id, true, dev_id, dptr, nullptr, size};
    swap_info_[handle_id] = info;
  } else {
    std::cout << "SetAddr duplicated id " << handle_id << std::endl;
    std::cout << "SetAddr " << iter->second->size << " " << size << std::endl;
  }
  pthread_rwlock_unlock(&swap_lock_);
}

void Swap::DelAddr(handle_id_t handle_id, size_t size){
  pthread_rwlock_wrlock(&swap_lock_);
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1) {
    mhistory_->PutRecord(handle_id, info->device_id, MemHistory::DEL_ADDR, size);
  }
  delete info;
  swap_info_.erase(handle_id);
  pthread_rwlock_unlock(&swap_lock_);
}

} // namespace mxnet
