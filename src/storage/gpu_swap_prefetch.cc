#include <iostream>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <map>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include <mxnet/gpu_swap_history.h>
#include <mxnet/swap.h>
#include "./gpu_swap_prefetch.h"


namespace mxnet {

Prefetch::Prefetch() {
  start_prefetching_ = false;
  stop_prefetching_ = false;
  prefetch_algorithm_ = dmlc::GetEnv("PREFETCH_ALGORITHM", std::string("NaiveHistory"));
  steps_ahead_ = dmlc::GetEnv("PREFETCH_STEP_AHEAD", 100);
  history_ = MemHistory::_GetSharedRef();
  lookahead_pos_ = std::vector<int>(NUMBER_OF_GPU);
  prefetcher_ = std::vector<std::thread>(NUMBER_OF_GPU);
  for(int i = 0; i < NUMBER_OF_GPU; i++) {
    lookahead_pos_[i] = -1;
  }
  std::cout << "Prefetch Algorithm: " << prefetch_algorithm_ << std::endl;
  std::cout << "Prefetch Steps Ahead: " << steps_ahead_ << std::endl;
  if (prefetch_algorithm_ == "NaiveHistory") {
    DoPrefetch = &Prefetch::HistoryBasedPrefetch;
  } else { 
    std::cout << "Unknown Prefetch Algorithm: " << prefetch_algorithm_
      << std::endl;
    CHECK(0);
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
  history_->prefetch_count = 0;
  history_->cache_miss = 0;
  for(int device = 0; device < NUMBER_OF_GPU; device++) {
    prefetcher_[device] = std::thread(&Prefetch::Prefetching, this, device);
  }
}


void Prefetch::StopPrefetching() {
  stop_prefetching_ = true;
  for(int device = 0; device < NUMBER_OF_GPU; device++) {
    prefetcher_[device].join();
    lookahead_pos_[device] = -1;
  }
  std::cout << "=> total prefetch: " << history_->prefetch_count << std::endl;
  std::cout << "=> cache miss: " << history_->cache_miss << std::endl;
}


void Prefetch::Prefetching(int device) {
  while(!stop_prefetching_) {
    (this->*DoPrefetch)(device);
    start_prefetching_ = true;
    usleep(1);
  }
}

// TODO(sotskin): karll: Add algorithm discription
void Prefetch::HistoryBasedPrefetch(int device) {
  //pthread_rwlock_rdlock(&swap_lock_);
  //bool has_begun = false;
  if(lookahead_pos_[device] < history_->record_idx[device])
    lookahead_pos_[device] = history_->record_idx[device];
  while(lookahead_pos_[device]+1 < history_->ordered_history[device].size() &&
      lookahead_pos_[device]-(int)history_->record_idx[device]<=steps_ahead_) {
    MemHistory::MemRecord r =
        history_->ordered_history[device][++lookahead_pos_[device]];
    if(r.operation_id == MemHistory::GET_ADDR) {
      ++(history_->prefetch_count);
      Swap::Get()->GetAddr(r.handle_id, true);
    } else {
      std::cout << "non-read operation found" << std::endl;
    }
  }
  //pthread_rwlock_unlock(&swap_lock_);
}


} // namespace mxnet
