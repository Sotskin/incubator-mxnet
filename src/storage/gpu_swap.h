#ifndef MXNET_STORAGE_SWAP_H_
#define MXNET_STORAGE_SWAP_H_

#include <atomic>
#include <map>
#include <memory>
#include <pthread.h>
#include <stack>
#include <string>
#include <unistd.h>
#include <unordered_map>
#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif // MXNET_USE_CUDA
#include "./gpu_swap_history.h"
#include "./gpu_swap_memmgr.h"

namespace mxnet {

struct SwapInfo {
  handle_id_t handle_id;
  bool swapped_in;
  int device_id;
  void* dptr;
  char* cpu_address;
  size_t size;
  size_t swap_count;
  std::atomic_flag is_swapping;
};

class Swap {
public:
  ~Swap();
  static Swap* Get();
  static std::shared_ptr<Swap> _GetSharedRef();
  void SwapOut(unsigned required_memory, int device_id, bool async);
  void SwapOutLocked(unsigned required_memory, int device_id, bool async);
  void SwapIn(SwapInfo *info, bool async);
  void SetAddr(handle_id_t handle_id, void* dptr, size_t size, int device_id);
  void DelAddr(handle_id_t handle_id);
  void FreeAddr(handle_id_t handle_id);
  void* GetAddr(handle_id_t handle_id, bool prefetch = false);
  void LockSwap();
  void UnlockSwap();
  void PrintHandles();

private:
  Swap();
  std::unordered_map<handle_id_t, SwapInfo*> swap_info_;
  std::unordered_set<handle_id_t> swappable_handles_[NUMBER_OF_GPU];
  std::map<size_t, std::unordered_set<handle_id_t> > divided_handles_[NUMBER_OF_GPU];
  std::stack<handle_id_t> locked_handles_[NUMBER_OF_GPU];
  std::vector<size_t> free_memory_;
  std::shared_ptr<MemoryHistory> memory_history_;
  std::shared_ptr<MemoryManager> memory_manager_;
  pthread_rwlock_t swap_lock_;
  pthread_rwlock_t locks_[NUMBER_OF_GPU];
  bool swap_locked_;
  bool swap_async_;
  cudaStream_t streams_[NUMBER_OF_GPU];
}; // Class Swap

} // namespace mxnet

#endif // MXNET_STORAGE_SWAP_H_
