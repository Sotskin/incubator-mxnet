#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <list>
#include <unistd.h>
#include <unordered_set>
#include <vector>
#include <set>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include "./gpu_swap_history.h"
#include "./gpu_swap_prefetch.h"
#include "./gpu_swap_memmgr.h"
#include "./gpu_swap_util.h"

namespace mxnet {

MemoryHistory::MemoryHistory() {
  iteration_started_ = false;
  is_recording_ = false;
  pre_recording_ = false;
  iteration_idx_ = 0;
  swap_algorithm_ = dmlc::GetEnv("MXNET_SWAP_ALGORITHM", std::string("LRU"));
  adaptive_history_ = dmlc::GetEnv("MXNET_ADAPTIVE_HISTORY", false);
  dev_history_.resize(NUMBER_OF_GPU);
  std::cout << "Swap Algorithm: " << swap_algorithm_ << std::endl;
  if (swap_algorithm_ == "LRU") {
    DoDecide = &MemoryHistory::LRU;
  } else if (swap_algorithm_ == "NaiveHistory") {
    DoDecide = &MemoryHistory::NaiveHistory;
  } else if (swap_algorithm_ == "SizeHistory") {
    DoDecide = &MemoryHistory::SizeHistory;
  } else {
    std::cout << "Unknown Algorithm Name: " << swap_algorithm_ << std::endl;
    CHECK(0);
  }
}

MemoryHistory::~MemoryHistory() {}

std::shared_ptr<MemoryHistory> MemoryHistory::_GetSharedRef() {
  static std::shared_ptr<MemoryHistory> inst(new MemoryHistory());
  return inst;
}

MemoryHistory* MemoryHistory::Get() {
  static MemoryHistory *s = _GetSharedRef().get();
  return s;
}

void MemoryHistory::PreRecord(handle_id_t handle_id, record_t op,
                              DeviceHistory& history) {
  if (op == MemoryHistory::SET_ADDR) {
    history.lru_list.push_front(handle_id);
    history.lru_map[handle_id] = history.lru_list.begin();
  } else if (op == MemoryHistory::GET_ADDR) {
    if (history.lru_map[handle_id] == history.lru_list.end()) {
      history.lru_list.push_front(handle_id);
      history.lru_map[handle_id] = history.lru_list.begin();
    } else {
      std::list<handle_id_t>::iterator hid = history.lru_map[handle_id];
      history.lru_list.erase(hid);
      history.lru_list.push_front(handle_id);
      history.lru_map[handle_id] = history.lru_list.begin();
    }
  } else {
    std::list<handle_id_t>::iterator hid = history.lru_map[handle_id];
    history.lru_list.erase(hid);
    history.lru_map.erase(handle_id);
  }
}

void MemoryHistory::PutRecord(handle_id_t handle_id, int device,
                              record_t op, size_t size) {
  if (!IterationStarted()) {
    return;
  }
  auto& history = dev_history_[device];
  if (IsPreRecording()) {
    std::lock_guard<std::mutex> lock(mutex_[device]);
    MemoryHistory::PreRecord(handle_id, op, history);
  }
  if (IsRecording()) {
    std::lock_guard<std::mutex> lock(mutex_[device]);
    timestamp_t t = (duration_cast<microseconds>
        (high_resolution_clock::now() - begin_time_)).count();
    size_t record_step = history.curr_idx;
    MemRecord record = {handle_id, op, t, record_step, size};
    history.all_handle_history.back()[handle_id].push_back(record);
    history.all_ordered_history.back().push_back(record);
  }
  history.curr_idx++;
}

// LRU: Swapout the least recently used handle
handle_id_t MemoryHistory::LRU(std::unordered_set<handle_id_t> handles,
                               int device, void* arg) {
  auto& history = dev_history_[device];
  handle_id_t victim = -1;
  while (history.lru_list.size() != 0 &&
    handles.find(history.lru_list.back()) == handles.end()) {
    handle_id_t temp_id = history.lru_list.back();
    history.lru_map[temp_id] = history.lru_list.end();
    history.lru_list.pop_back();
  }
  if (history.lru_list.size() == 0) {
    std::cout << "LRU: No Swappable Handle Found" << std::endl;
    CHECK(0);
  } else {
    victim = history.lru_list.back();
    history.lru_map[victim] = history.lru_list.end();
    history.lru_list.pop_back();
  }
  return victim;
}

// NaiveHistory: assume iterations remain the same; choose the handle
// whose next reference is furthest in the future as victim.
handle_id_t MemoryHistory::NaiveHistory(
  std::unordered_set<handle_id_t> handles, int device, void* arg) {
  auto& history = dev_history_[device];
  SwapParams* params = (SwapParams*)arg;
  size_t latest_step = 0;
  handle_id_t latest_id = 0;
  for (auto &id : handles) {
    MemoryHistory::MemRecord r = {0, MemoryHistory::GET_ADDR, 0,
                               history.curr_idx, 0};
    auto it = std::upper_bound(history.handle_history->at(id).begin(),
                               history.handle_history->at(id).end(), r,
                               CompareByStep);
    if (it == history.handle_history->at(id).end()) {
      /*
      if (it != history.handle_history->at(id).begin() &&
          history.curr_idx - history.handle_history->at(id).back().record_step < 10) {
        // Victim just used, skip
        continue;
      }
      */
      return id;
    } else if (it->record_step - history.curr_idx < params->no_swap_steps) {
      continue;
    } else if (it->record_step > latest_step) {
      latest_step = it->record_step;
      latest_id = id;
    }
  }
  return latest_id;
}

handle_id_t MemoryHistory::SizeHistory(
    std::unordered_set<handle_id_t> handles, int device, void* arg) {
  auto divided_handles  = ((SwapParams*)arg)->divided_handles;
  auto candidates = divided_handles->lower_bound(((SwapParams*)arg)->required_memory);
  auto original_candidates = candidates;
  bool reverse_flag = false;
  //FIXME: Empirical result may need a better way to know how to choose this.
  size_t no_swap_step = 80;
  if (candidates == divided_handles->end()) {
    candidates--;
  }
  while (true) {
    if (candidates->second.size() != 0) {
      SwapParams new_params = {no_swap_step, 0, nullptr};
      handle_id_t victim = NaiveHistory(candidates->second, device, &new_params);
      if (victim != 0) {
        return victim;
      }
    }
    if (!reverse_flag) {
      candidates++;
      if (candidates == divided_handles->end()) {
        candidates = original_candidates;
        reverse_flag = true;
      }
    }
    if (reverse_flag) {
      if (candidates == divided_handles->begin()) {
        candidates = original_candidates;
        reverse_flag = false;
        if (no_swap_step == 0) {
          std::cout << "Cannot find victim (algorithm error)" << std::endl;
          CHECK(0);
        }
        no_swap_step /= 2;
      } else {
        candidates --;
      }
    }
  }
  return 0;
}

handle_id_t MemoryHistory::DecideVictim(std::unordered_set<handle_id_t> handles,
                                        int device, void* arg) {
  std::lock_guard<std::mutex> lock(mutex_[device]);
  if (iteration_idx_ <= kBeginRecordAt) {
    return MemoryHistory::LRU(handles, device, nullptr);
  } else {
    return (this->*DoDecide)(handles, device, arg);
  }
}

void MemoryHistory::PrintRecord(int device) {
  std::lock_guard<std::mutex> lock(mutex_[device]);
  auto& history = dev_history_[device];
  std::ofstream fp;
  fp.open("history_log.txt");
  std::vector<MemRecord> records;
  std::map<handle_id_t, std::vector<MemRecord> >::iterator it;
  for (it = history.handle_history->begin();
       it != history.handle_history->end(); ++it) {
    for (size_t i = 0; i < (it->second).size(); i++) {
      records.push_back(it->second[i]);
    }
  }
  std::sort(records.begin(), records.end(), MemoryHistory::CompareByStep);
  for (size_t i = 0; i < records.size(); i++) {
    MemRecord r = records[i];
    fp << "No." << i << std::endl;
    fp << "Step: " << r.record_step << std::endl;
    fp << "Handle ID: " << r.handle_id << std::endl;
    fp << "Operation: ";
    if (r.operation_id == GET_ADDR) {
      fp << "get";
    } else if (r.operation_id == SET_ADDR) {
      fp << "set";
    } else {
      fp << "del";
    }
    fp << std::endl;
    fp << "Time: " << r.time << std::endl;
    fp << "Size: " << r.size << std::endl;
    fp << std::endl;
  }
  fp.close();
}

void MemoryHistory::StartIteration() {
  iteration_started_ = true;
  for (int i = 0; i < NUMBER_OF_GPU; i++) {
    dev_history_[i].curr_idx = 0;
  }
  // LRU needs to record every iteration. As a result, it is mandatory to do LRU
  // recording even at kBeginRecordAt iteration because the desired swapping
  // algorithm will kick in from (kBeginRecordAt + 1) iteration.
  if (iteration_idx_ <= kBeginRecordAt || swap_algorithm_ == "LRU") {
    pre_recording_ = true;
  }
  if ((adaptive_history_ && iteration_idx_ >= kBeginRecordAt) ||
      iteration_idx_ == kBeginRecordAt) {
    // Each history is stored in a cyclic list (mimiced by std::list).
    // The LAST history in a cyclic list is used to be recorded in current
    // iteration. The (LAST-1) history is used by DoDecide(). However, if
    // adaptive_history_ is false, The LAST history is used by DoDecide().
    for (int i = 0; i < NUMBER_OF_GPU; i++) {
      auto& history = dev_history_[i];
      if (history.all_ordered_history.size() ==
          DeviceHistory::kMaxPreservedIteration) {
        history.all_ordered_history.splice(history.all_ordered_history.end(),
                                           history.all_ordered_history,
                                           history.all_ordered_history.begin());
        history.all_handle_history.splice(history.all_handle_history.end(),
                                          history.all_handle_history,
                                          history.all_handle_history.begin());
      } else {
        size_t size = history.all_ordered_history.size();
        history.all_ordered_history.resize(size + 1);
        history.all_handle_history.resize(size + 1);
      }
      auto ordered_it = history.all_ordered_history.rbegin();
      auto handle_it  = history.all_handle_history.rbegin();
      if (history.all_ordered_history.size() > 1) {
        ordered_it = std::next(ordered_it, 1);
        handle_it = std::next(handle_it, 1);
      }
      history.ordered_history = &(*ordered_it);
      history.handle_history = &(*handle_it);
      history.all_ordered_history.rbegin()->clear();
      history.all_handle_history.rbegin()->clear();
    }
    is_recording_ = true;
  }
  begin_time_ = high_resolution_clock::now();
  for (int device = 0; device < NUMBER_OF_GPU; device++) {
    dev_history_[device].prefetch_count = 0;
    dev_history_[device].cache_miss = 0;
    dev_history_[device].num_swap_in = 0;
    dev_history_[device].num_swap_out = 0;
    dev_history_[device].swap_in_total = 0;
    dev_history_[device].swap_out_total = 0;
    dev_history_[device].num_get_addr = 0;
  }

  // We can't start the prefetching too early, otherwise, the prefetch_count
  // may be incorrect.
  if (iteration_idx_ > kBeginRecordAt) {
    Prefetch::Get()->StartPrefetching();
  }
}

void MemoryHistory::StopIteration() {
  pre_recording_ = false;
  is_recording_ = false;
  iteration_started_ = false;
  if (Prefetch::Get()->IsPrefetching()) {
    Prefetch::Get()->StopPrefetching();
  }
  ++iteration_idx_;
  //for (int device = 0; device < NUMBER_OF_GPU; device++) {
    //auto& history = dev_history_[device];
    //if (adaptive_history_ && iteration_started_ >= kBeginRecordAt) {
      //history.all_ordered_history.push_back(history.ordered_history);
      //history.all_handle_history.push_back(history.handle_history);
    //}
  //}
}

void MemoryHistory::Statistics() {
  if (adaptive_history_) {
    PrintSimilarity();
  }
  for (int device = 0; device < NUMBER_OF_GPU; device++) {
    auto& history = dev_history_[device];
    std::cout << "GPU" << device << " statistics:" << std::endl
              << "=> Number of prefetch: "
              << history.prefetch_count << std::endl
              << "=> Number of cache miss: "
              << history.cache_miss << std::endl
              << "=> Number of getaddr: "
              << history.num_get_addr << std::endl
              << "=> Number of swap in: "
              << history.num_swap_in << std::endl
              << "=> Total swap in size: "
              << GBString(history.swap_in_total) << std::endl
              << "=> Number of swap out: "
              << history.num_swap_out << std::endl
              << "=> Total swap out size: "
              << GBString(history.swap_out_total)<< std::endl;
  }
  GetMemoryManager()->Statistics();
}

double MemoryHistory::LCS_Similarity(std::vector<MemRecord>& base,
                                     std::vector<MemRecord>& target) {
  std::vector<size_t> *prev_table, *curr_table, *temp;
  size_t size1 = base.size();
  size_t size2 = target.size();
  prev_table = new std::vector<size_t>(size2 + 1, 0);
  curr_table = new std::vector<size_t>(size2 + 1, 0);
  for (size_t i = 1; i <= size1; i++) {
    for (size_t j = 1; j <= size2; j++) {
      if (base[i - 1].handle_id == target[j - 1].handle_id) {
        (*curr_table)[j] = (*prev_table)[j - 1] + 1;
      } else {
        (*curr_table)[j] = std::max((*curr_table)[j - 1], (*prev_table)[j]);
      }
    }
    temp = curr_table;
    curr_table = prev_table;
    prev_table = temp;
  }
  return (*prev_table)[size2] / (double)std::max(size1, size2);
}

void MemoryHistory::PrintSimilarity() {
  double min = 1.0, max = 0.0, mean = 0.0, delta = 0.0, delta2 = 0.0, M2 = 0.0;
  int count = 0;
  if (dev_history_[0].all_ordered_history.size() < kBeginRecordAt) {
    return;
  }
  for (int device = 0; device < NUMBER_OF_GPU; device++) {
    std::cout << "GPU" << device << " history similarity:" << std::endl;
    auto& history = dev_history_[device];
    for (size_t i = 0; i < history.all_ordered_history.size() - 1; i++) {
      auto base_it = std::next(history.all_ordered_history.begin(), i);
      auto target_it = std::next(base_it, 1);
      while (target_it != history.all_ordered_history.end()) {
        double sim = LCS_Similarity(*base_it, *target_it);
        min = (sim < min) ? sim : min;
        max = (sim > max) ? sim : max;
        count += 1;
        delta = sim - mean;
        mean = mean + delta / count;
        delta2 = sim - mean;
        M2 += delta * delta2;
        target_it++;
      }
    }
  }
  std::cout << "min: " << min << ", max: " << max << ", mean: " << mean << " " << M2
            << " stddev: " << sqrt(M2 / count - 1) << std::endl;
}

} // namespace mxnet
