#ifndef GPU_SWAP_PREFETCH_H
#define GPU_SWAP_PREFETCH_H

#include <mxnet/gpu_swap_history.h>
#include <thread>

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
  void SignalStartComputing();
  void SignalStopComputing();
  bool IsPrefetching() {return start_prefetching_;}

private:
  Prefetch();
  void Prefetching(int device);
  void (Prefetch::*DoPrefetch)(int);
  // Prefetch algorithm declarations
  void HistoryBasedPrefetch(int device);
  void PrefetchWhileComputing(int device);

  bool computing_;
  std::vector<size_t> lookahead_pos_;
  std::vector<std::thread> prefetcher_;
  std::shared_ptr<MemoryHistory> history_;
  bool start_prefetching_;
  bool stop_prefetching_;
  std::string prefetch_algorithm_;
  size_t steps_ahead_;
}; // class prefetch

} // namespace mxnet


#endif
