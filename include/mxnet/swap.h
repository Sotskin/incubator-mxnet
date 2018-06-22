#ifndef MXNET_STORAGE_SWAP_H_
#define MXNET_STORAGE_SWAP_H_

#include <pthread.h>
#include <unordered_map>
#include <memory>
#include <mxnet/gpu_swap_history.h>
#if MXNET_USE_CUDA
#include "./cuda_runtime.h"
#endif // MXNET_USE_CUDA

namespace mxnet {

using handle_id_t = unsigned long long;
using timestamp_t = unsigned long long;
using timestep_t = unsigned long long;

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
  void SwapOut(unsigned required_memory, int device);
  void SwapIn(SwapInfo *info);
  void SetAddr(handle_id_t handle_id, void* dptr, size_t size, int dev_id);
  void DelAddr(handle_id_t handle_id, size_t size);
  void* GetAddr(handle_id_t handle_id, size_t size);
  int UpdateFree(int device); 

private:
  Swap();
  std::unordered_map<handle_id_t, SwapInfo*> swap_info_;
  std::shared_ptr<MemHistory> mhistory_;
  std::vector<size_t> free_memory_;
  pthread_rwlock_t swap_lock_;
  pthread_rwlock_t locks_[NUMBER_OF_GPU];
}; // Class Swap

} // namespace mxnet

#endif // MXNET_STORAGE_SWAP_H_
