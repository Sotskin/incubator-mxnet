#ifndef GPU_SWAP_PREFETCH_H
#define GPU_SWAP_PREFETCH_H

#include "./gpu_swap_history.h"

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
  void HistoryBasedPrefetch(int device);
  bool IsPrefetching() {return start_prefetching_;}

private:

  Prefetch();
  void Prefetching(int device);

  std::vector<int> lookahead_pos_ =
      std::vector<int>(NUMBER_OF_GPU);
  std::vector<std::thread> prefetcher_ =
      std::vector<std::thread>(NUMBER_OF_GPU);
  pthread_rwlock_t swap_lock_;
  std::shared_ptr<MemHistory> history_;
  bool start_prefetching_;
  bool stop_prefetching_;
  int algorithm_;
  int steps_ahead_;

}; // class prefetch

} // namespace mxnet


#endif
