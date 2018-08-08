#include <iostream>
#include <fstream>
#include <unistd.h>
#include <algorithm>
#include <unordered_set>
#include <list>
#include <vector>
#include <set>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include <mxnet/gpu_swap_history.h>
#include "./gpu_swap_prefetch.h"

namespace mxnet {

MemHistory::MemHistory() {
  iteration_started_ = false;
  is_recording_ = false;
  pre_recording_ = false;
  iteration_idx_ = 0;
  swap_algorithm_ = dmlc::GetEnv("MXNET_SWAP_ALGORITHM", std::string("LRU"));
  dev_history_.resize(NUMBER_OF_GPU);
  std::cout << "Swap Algorithm: " << swap_algorithm_ << std::endl;
  if (swap_algorithm_ == "LRU") {
    DoDecide = &MemHistory::LRU;
  } else if (swap_algorithm_ == "NaiveHistory") {
    DoDecide = &MemHistory::NaiveHistory;
  } else if (swap_algorithm_ == "SizeHistory") {
    DoDecide = &MemHistory::SizeHistory;
  } else {
    std::cout << "Unknown Algorithm Name: " << swap_algorithm_ << std::endl;
    CHECK(0);
  }
}

MemHistory::~MemHistory() {}

std::shared_ptr<MemHistory> MemHistory::_GetSharedRef() {
  static std::shared_ptr<MemHistory> inst(new MemHistory());
  return inst;
}

MemHistory* MemHistory::Get() {
  static MemHistory *s = _GetSharedRef().get();
  return s;
}

void MemHistory::PreRecord(handle_id_t id, record_t op,
                           DeviceHistory& history) {
  if (op == MemHistory::SET_ADDR) {
    history.lru_list.push_front(id);
    history.lru_map[id] = history.lru_list.begin();
  } else if (op == MemHistory::GET_ADDR) {
    if (history.lru_map[id] == history.lru_list.end()) {
      history.lru_list.push_front(id);
      history.lru_map[id] = history.lru_list.begin();
    } else {
      std::list<handle_id_t>::iterator hid = history.lru_map[id];
      history.lru_list.erase(hid);
      history.lru_list.push_front(id);
      history.lru_map[id] = history.lru_list.begin();
    }
  } else {
    std::list<handle_id_t>::iterator hid = history.lru_map[id];
    history.lru_list.erase(hid);
    history.lru_map.erase(id);
  }
}

void MemHistory::PutRecord(handle_id_t handle_id, int device,
                           record_t operation_id, size_t size) {
  if (!IterationStarted()) {
    return;
  }
  auto& history = dev_history_[device];
  if (IsPreRecording()) {
    std::lock_guard<std::mutex> lock(mutex_[device]);
    MemHistory::PreRecord(handle_id, operation_id, history);
  }
  if (IsRecording()) {
    std::lock_guard<std::mutex> lock(mutex_[device]);
    timestamp_t t = (duration_cast<microseconds>
        (high_resolution_clock::now() - begin_time_)).count();
    size_t record_step = history.curr_idx;
    MemRecord record = {handle_id, operation_id, t, record_step, size};
    history.handle_history[handle_id].push_back(record);
    history.ordered_history.push_back(record);
  }
  history.curr_idx++;
}

// LRU: Swapout the least recently used handle
handle_id_t MemHistory::LRU(std::unordered_set<handle_id_t> handles, int device, void* arg) {
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
handle_id_t MemHistory::NaiveHistory(
  std::unordered_set<handle_id_t> handles, int device, void* arg) {
  auto& history = dev_history_[device];
  SwapParams* params = (SwapParams*)arg;
  size_t latest_step = 0;
  handle_id_t latest_id = 0;
  size_t loop_count = 0;
  for (auto &id : handles) {
    loop_count += 1;
    MemHistory::MemRecord r = {0, MemHistory::GET_ADDR, 0,
                               history.curr_idx, 0};
    auto it = std::upper_bound(history.handle_history[id].begin(),
                               history.handle_history[id].end(), r,
                               CompareByStep);
    if (it == history.handle_history[id].end()) {
      /*
      if (it != history.handle_history[id].begin() &&
          history.curr_idx - history.handle_history[id].back().record_step < 10) {
        // Victim just used, skip
        continue;
      }
      */
      std::cout << "loop_count : " << loop_count << std::endl;
      return id;
    } else if (it->record_step - history.curr_idx < params->no_swap_steps) {
      continue;
    } else if (it->record_step > latest_step) {
      latest_step = it->record_step;
      latest_id = id;
    }
  }
  std::cout << "loop_count : " << loop_count << std::endl;
  return latest_id;
}

handle_id_t MemHistory::SizeHistory(
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

handle_id_t MemHistory::DecideVictim(std::unordered_set<handle_id_t> handles, int device,
                                     void* arg) {
  std::lock_guard<std::mutex> lock(mutex_[device]);
  if (iteration_idx_ <= 2) {
    return MemHistory::LRU(handles, device, nullptr);
  } else {
    return (this->*DoDecide)(handles, device, arg);
  }
}

void MemHistory::PrintRecord(int device) {
  std::lock_guard<std::mutex> lock(mutex_[device]);
  auto& history = dev_history_[device];
  std::ofstream fp;
  fp.open("history_log.txt");
  std::vector<MemRecord> records;
  std::map<handle_id_t, std::vector<MemRecord> >::iterator it;
  for (it = history.handle_history.begin();
       it != history.handle_history.end(); ++it) {
    for (size_t i = 0; i < (it->second).size(); i++) {
      records.push_back(it->second[i]);
    }
  }
  std::sort(records.begin(), records.end(), MemHistory::CompareByStep);
  for (size_t i = 0; i < records.size(); i++) {
    MemRecord r = records[i];
    fp << "No." << i << std::endl;
    fp << "Step: " << r.record_step << std::endl;
    fp << "Handle ID: " << r.handle_id << std::endl;
    fp << "Operation: ";
    if (r.operation_id == GET_ADDR)
      fp << "get";
    else if (r.operation_id == SET_ADDR)
      fp << "set";
    else
      fp << "del";
    fp << std::endl;
    fp << "Time: " << r.time << std::endl;
    fp << "Size: " << r.size << std::endl;
    fp << std::endl;
  }
  fp.close();
}

void MemHistory::StartIteration() {
  iteration_started_ = true;
  for (int i = 0; i < NUMBER_OF_GPU; i++) {
    dev_history_[i].curr_idx = 0;
  }
  if (iteration_idx_ <= 2 || swap_algorithm_ == "LRU") {
    pre_recording_ = true;
  }
  if (iteration_idx_ == 2) {
    is_recording_ = true;
  } else if (iteration_idx_ > 2) {
    Prefetch::Get()->StartPrefetching();
    //while (!Prefetch::Get()->IsPrefetching())
    //  usleep(1);
  }
  begin_time_ = high_resolution_clock::now();
  // Log variables
  num_swap_in = 0;
  num_swap_out = 0;
  swap_in_total = 0;
  swap_out_total = 0;
  num_get_addr = 0;
}

void MemHistory::StopIteration() {
  pre_recording_ = false;
  is_recording_ = false;
  iteration_started_ = false;
  if (Prefetch::Get()->IsPrefetching()) {
    Prefetch::Get()->StopPrefetching();
  }
  ++iteration_idx_;
  std::cout << "num_get_addr " << num_get_addr << std::endl
    << "num_swap_in: " << num_swap_in << " "
    << "total: " << swap_in_total / 1e9 << "GB " << std::endl
    << "num_swap_out " << num_swap_out << " "
    << "total: " << swap_out_total / 1e9 << "GB " << std::endl;
}

} // namespace mxnet



