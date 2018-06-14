#ifndef MXNET_STORAGE_SWAP_H_
#define MXNET_STORAGE_SWAP_H_

#include <pthread.h>
#include <unordered_map>
#include <memory>

namespace mxnet {

using handle_id_t = unsigned long long;
using timestamp_t = unsigned long long;

struct SwapInfo {
  handle_id_t handle_id;
  bool swapped_in;
  int device;
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
  void* GetAddr(handle_id_t handle_id, size_t size);
  void SetAddr(handle_id_t handle_id, void* dptr, size_t size);
  void DelAddr(handle_id_t handle_id, size_t size);

private:
  Swap();
  std::unordered_map<handle_id_t, SwapInfo*> swap_info_;
  pthread_rwlock_t swap_lock_;
}; // Class Swap

} // namespace mxnet

#endif // MXNET_STORAGE_SWAP_H_
