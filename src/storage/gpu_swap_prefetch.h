#ifndef GPU_SWAP_PREFETCH_H
#define GPU_SWAP_PREFETCH_H

#include <mxnet/gpu_swap_history.h>

#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace mxnet {

class Prefetch {
public:

  ~Prefetch();
  static Prefetch* Get();
  static std::shared_ptr<Prefetch> _GetSharedRef();
  void StartPrefetching();
  void StopPrefetching();
  bool IsPrefetching() {return start_prefetching_;}

private:

  Prefetch();
  void Prefetching(int device);

  std::vector<int> lookahead_pos_;
  std::vector<std::thread> prefetcher_;
  pthread_rwlock_t swap_lock_;
  std::shared_ptr<MemHistory> history_;
  bool start_prefetching_;
  bool stop_prefetching_;
  std::string prefetch_algorithm_;
  int steps_ahead_;
  
  void (Prefetch::*DoPrefetch)(int);
  // Prefetch algorithm declarations
  void HistoryBasedPrefetch(int device);
  
}; // class prefetch

} // namespace mxnet


#endif
