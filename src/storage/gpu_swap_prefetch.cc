#include <iostream>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <map>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include <mxnet/gpu_swap_history.h>
#include <mxnet/swap.h>
#include <mxnet/gpu_swap_prefetch.h>


namespace mxnet {

Prefetch::Prefetch() {
  start_prefetching_ = false;
  stop_prefetching_ = false;
  algorithm_ = dmlc::GetEnv("PREFETCH_ALGORITHM", 0);
  steps_ahead_ = dmlc::GetEnv("PREFETCH_STEP_AHEAD", 100);
  history_ = MemHistory::_GetSharedRef();
  for(int i = 0; i < MemHistory::NUMBER_OF_GPU; i++) {
    lookahead_pos_[i] = -1;
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


void Prefetch::StartPrefetching() {
  start_prefetching_ = false;
  stop_prefetching_ = false;
  for(int device = 0; device < MemHistory::NUMBER_OF_GPU; device++) {
    prefetcher_[device] = std::thread(&Prefetch::Prefetching, this, device);
  }
}


void Prefetch::StopPrefetching() {
  stop_prefetching_ = true;
  for(int device = 0; device < MemHistory::NUMBER_OF_GPU; device++) {
    prefetcher_[device].join();
  }
}


void Prefetch::Prefetching(int device) {
  while(!stop_prefetching_) {
    if(algorithm_ == 0) {
      HistoryBasedPrefetch(device);
    }
    start_prefetching_ = true;
    usleep(1);
  }
}


void Prefetch::HistoryBasedPrefetch(int device) {
  pthread_rwlock_rdlock(&swap_lock_);
  //bool has_begun = false;
  bool not_end =
      lookahead_pos_[device]+1 < history_->ordered_history[device].size();
  bool too_ahead =
      lookahead_pos_[device] - history_->record_idx[device] > steps_ahead_;
  while(not_end && !too_ahead) {
    MemHistory::MemRecord r =
        history_->ordered_history[device][++lookahead_pos_[device]];
    if(r.operation_id == MemHistory::GET_ADDR) {
      Swap::Get()->GetAddr(r.handle_id, r.size);
    } else {
      std::cout << "non-read operation found" << std::endl;
    }
  }
  pthread_rwlock_unlock(&swap_lock_);
}


} // namespace mxnet
