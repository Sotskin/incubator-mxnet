#ifndef GPU_SWAP_PREFETCH_H
#define GPU_SWAP_PREFETCH_H

#include "./gpu_swap_history.h"

#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace mxnet {

class Prefetch {
public:

  std::vector<int> lookahead_pos = std::vector<int>(MemHistory::NUMBER_OF_GPU);

  ~Prefetch();
  static Prefetch* Get();
  static std::shared_ptr<Prefetch> _GetSharedRef();
  void StartPrefetching(int device);

private:
  Prefetch();
  bool begin_prefetching_;
  bool stop_prefetching_;
  int algorithm;
}; // class prefetch

} // namespace mxnet


#endif
