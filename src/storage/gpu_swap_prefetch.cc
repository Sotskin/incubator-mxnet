#include <iostream>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <map>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include <mxnet/gpu_swap.h>
#include <mxnet/gpu_swap_history.h>
#include "./gpu_swap_prefetch.h"


namespace mxnet {

Prefetch::Prefetch() {
  start_prefetching_ = false;
  stop_prefetching_ = false;
  computing_ = false;
  prefetch_algorithm_ = dmlc::GetEnv("MXNET_PREFETCH_ALGORITHM",
                                      std::string("NaiveHistory"));
  steps_ahead_ = dmlc::GetEnv("MXNET_PREFETCH_STEP_AHEAD", 100);
  history_ = MemoryHistory::_GetSharedRef();
  lookahead_pos_.resize(NUMBER_OF_GPU);
  prefetcher_.resize(NUMBER_OF_GPU);
  for (int i = 0; i < NUMBER_OF_GPU; i++) {
    lookahead_pos_[i] = 0; 
  }
  std::cout << "Prefetch Algorithm: " << prefetch_algorithm_ << std::endl;
  std::cout << "Prefetch Steps Ahead: " << steps_ahead_ << std::endl;
  if (prefetch_algorithm_ == "NaiveHistory") {
    DoPrefetch = &Prefetch::HistoryBasedPrefetch;
  } else if (prefetch_algorithm_ == "ComputePrefetch") {
    DoPrefetch = &Prefetch::PrefetchWhileComputing;
  } else if (prefetch_algorithm_ == "NoPrefetch") {
    DoPrefetch = nullptr;
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


void Prefetch::SignalStartComputing() {
  computing_ = true;
}


void Prefetch::SignalStopComputing() {
  computing_ = false;
}


void Prefetch::StartPrefetching() {
  start_prefetching_ = false;
  stop_prefetching_ = false;
  if (DoPrefetch == nullptr) {
    return;
  }
  for (int device = 0; device < NUMBER_OF_GPU; device++) {
    prefetcher_[device] = std::thread(&Prefetch::Prefetching, this, device);
  }
}


void Prefetch::StopPrefetching() {
  stop_prefetching_ = true;
  for (int device = 0; device < NUMBER_OF_GPU; device++) {
    prefetcher_[device].join();
    lookahead_pos_[device] = 0;
  }
  // Swap::Get()->PrintHandles();
}


void Prefetch::Prefetching(int device) {
  while (!stop_prefetching_) {
    (this->*DoPrefetch)(device);
    start_prefetching_ = true;
  }
}


void Prefetch::PrefetchWhileComputing(int device) {
  //TODO(karl): Use condition variable to replace the inefficient busy waiting

  /*
  std::unique_lock<std::mutex> lk(prefetch_lock_[device]);
  while (!computing_) {
    prefetch_cond_[device].wait(lk);
  }
  while (!computing_)
    usleep(1);
  */
  auto& history = history_->DevHistory(device);
  if (lookahead_pos_[device] < history.curr_idx) {
    lookahead_pos_[device] = history.curr_idx;
  }
  while (lookahead_pos_[device] + 1 < history.ordered_history.size() &&
         lookahead_pos_[device] >= history.curr_idx &&
         lookahead_pos_[device] - history.curr_idx < steps_ahead_ &&
         computing_) {
    MemoryHistory::MemRecord r =
        history.ordered_history[++lookahead_pos_[device]];
    if (r.operation_id == MemoryHistory::GET_ADDR) {
      ++history.prefetch_count;
      Swap::Get()->GetAddr(r.handle_id, true);
    } else {
      std::cout << "non-read operation found" << std::endl;
    }
  }
}

// TODO(sotskin): karll: Add algorithm discription
void Prefetch::HistoryBasedPrefetch(int device) {
  //pthread_rwlock_rdlock(&swap_lock_);
  //bool has_begun = false;
  auto& history = history_->DevHistory(device);
  if (lookahead_pos_[device] < history.curr_idx) {
    lookahead_pos_[device] = history.curr_idx;
  }
  while (lookahead_pos_[device] + 1 < history.ordered_history.size() &&
         lookahead_pos_[device] >= history.curr_idx &&
         lookahead_pos_[device] - history.curr_idx < steps_ahead_) {
    MemoryHistory::MemRecord r =
        history.ordered_history[++lookahead_pos_[device]];
    if (r.operation_id == MemoryHistory::GET_ADDR) {
      ++history.prefetch_count;
      Swap::Get()->GetAddr(r.handle_id, true);
    } else {
      std::cout << "non-read operation found" << std::endl;
    }
  }
  usleep(1);
  //pthread_rwlock_unlock(&swap_lock_);
}


} // namespace mxnet
