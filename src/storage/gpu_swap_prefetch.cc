#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <dmlc/logging.h>
#include <mxnet/gpu_swap_history.h>
#include <mxnet/swap.h>
#include <mxnet/gpu_swap_prefetch.h>


namespace mxnet {

Prefetch::Prefetch() {
  begin_prefetching__ = false;
  stop_prefetching_ = false;
  algorithm = dmlc::GetEnv("SWAPPER_ALGORITHM", 0);
  for(int i = 0; i < MemHistory::NUMBER_OF_GPU; i++) {
    lookahead_pos[i] = -1;
  }
}

Prefetch::~Prefetch() {}

Prefetch* Prefetch::Get() {
  static Prefetch *s = _GetSharedRef().get();
  return s;
}

std::shared_ptr<Prefetch> Prefetch::_GetSharedRef() {
  static std::shared_ptr<Prefetch> inst(new Prefetch());
  return inst;
}


void Prefetch::StartPrefetching(int device) {
  pthread_t this_thread = pthread_self();
  while(stop_prefetching_) {

  }
}


} // namespace mxnet
