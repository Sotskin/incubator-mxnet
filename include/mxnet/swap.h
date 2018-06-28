#ifndef MXNET_STORAGE_SWAP_H_
#define MXNET_STORAGE_SWAP_H_

#include <pthread.h>
#include <unordered_map>
#include <memory>
#include <mxnet/gpu_swap_history.h>
#include <mxnet/mem_mgr.h>
#if MXNET_USE_CUDA
#include "./cuda_runtime.h"
#endif // MXNET_USE_CUDA

namespace mxnet {

struct SwapInfo {
  handle_id_t handle_id;
  bool swapped_in;
  int device_id;
  void* dptr;
  char* cpu_address;
  size_t size;
};

class Swap {
public:
  ~Swap();
  static Swap* Get();
  static std::shared_ptr<Swap> _GetSharedRef();
  void SwapOut(unsigned required_memory, int device_id);
  void SwapIn(SwapInfo *info);
  void SetAddr(handle_id_t handle_id, void* dptr, size_t size, int device_id);
  void DelAddr(handle_id_t handle_id);
  void* GetAddr(handle_id_t handle_id);
  // Update size of free space for the device.
  int UpdateFree(int device); 

private:
  Swap();
  std::unordered_map<handle_id_t, SwapInfo*> swap_info_;
  std::unordered_set<handle_id_t> swappable_handles_[NUMBER_OF_GPU];
  std::vector<size_t> free_memory_;
  std::shared_ptr<MemHistory> memory_history_;
  std::shared_ptr<MemoryManager> memory_manager_;
  pthread_rwlock_t swap_lock_;
  pthread_rwlock_t locks_[NUMBER_OF_GPU];
}; // Class Swap

} // namespace mxnet

#endif // MXNET_STORAGE_SWAP_H_
