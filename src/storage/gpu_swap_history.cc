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
  swap_algorithm_ = dmlc::GetEnv("SWAP_ALGORITHM", std::string("LRU"));
  history.resize(NUMBER_OF_GPU);
  ordered_history.resize(NUMBER_OF_GPU);
  lru_list.resize(NUMBER_OF_GPU);
  lru_map.resize(NUMBER_OF_GPU);
  record_idx.resize(NUMBER_OF_GPU);
  if(swap_algorithm_ == "LRU"){
    DoDecide = &MemHistory::LRU;
  } else if(swap_algorithm_ == "NaiveHistory") {
    DoDecide = &MemHistory::NaiveHistoryBased;
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

void MemHistory::PreRecord(handle_id_t id, record_t op, int device) {
  if(op == MemHistory::SET_ADDR) {
    lru_list[device].push_front(id);
    lru_map[device][id] = lru_list[device].begin();
  } else if(op == MemHistory::GET_ADDR) {
    if(lru_map[device][id] == lru_list[device].end()) {
      lru_list[device].push_front(id);
      lru_map[device][id] = lru_list[device].begin();
    } else {
      std::list<handle_id_t>::iterator hid = lru_map[device][id];
      lru_list[device].erase(hid);
      lru_list[device].push_front(id);
      lru_map[device][id] = lru_list[device].begin();
    }
  } else {
    std::list<handle_id_t>::iterator hid = lru_map[device][id];
    lru_list[device].erase(hid);
    lru_map[device].erase(id);
  }
}

void MemHistory::PutRecord(handle_id_t handle_id, int device,
                          record_t operation_id, size_t size) {
  if(!IterationStarted()) { 
    return;
  }
  if(IsPreRecording()) {
    std::lock_guard<std::mutex> lock(mutex_[device]);
    MemHistory::PreRecord(handle_id, operation_id, device);
  } 
  if(IsRecording()) {
    std::lock_guard<std::mutex> lock(mutex_[device]);
    timestamp_t t = (duration_cast<microseconds>
        (high_resolution_clock::now() - begin_time_)).count();
    size_t record_step = record_idx[device];
    MemRecord record = {handle_id, operation_id, t, record_step, size};
    history[device][handle_id].push_back(record);
    ordered_history[device].push_back(record);
  }
  record_idx[device]++;
}

// LRU: Swapout the least recently used handle
handle_id_t MemHistory::LRU(std::unordered_set<handle_id_t> handles, int device) {
  handle_id_t victim = -1;
  while(lru_list[device].size() != 0 &&
    handles.find(lru_list[device].back()) == handles.end()) {
    handle_id_t temp_id = lru_list[device].back();
    lru_map[device][temp_id] = lru_list[device].end();
    lru_list[device].pop_back();
  }
  if(lru_list[device].size() == 0) {
    std::cout << "LRU: No Swappable Handle Found" << std::endl;
    CHECK(0);
  } else {
    victim = lru_list[device].back();
    lru_map[device][victim] = lru_list[device].end();
    lru_list[device].pop_back();
  }
  return victim;
}

// NaiveHistory: assume iterations remain the same; choose the handle
// whose next reference is furthest in the future as victim.
handle_id_t MemHistory::NaiveHistoryBased(
  std::unordered_set<handle_id_t> handles, int device) {
  size_t latest_step = 0;
  handle_id_t latest_id = 0;
  for(auto &id : handles) {
    MemHistory::MemRecord r = {0,MemHistory::GET_ADDR,0,record_idx[device],0};
    auto it = std::upper_bound(history[device][id].begin(), 
        history[device][id].end(), r, CompareByStep);
    if(it == history[device][id].end()){
      /*
      if(it != history[device][id].begin() && 
          record_idx[device] - history[device][id].back().record_step < 10) {
        // Victim just used, skip
        continue;
      }
      */
      return id;
    } 
    /*
    else if(it != history[device][id].begin() &&
        std::prev(it) != history[device][id].begin() &&
        record_idx[device] - std::prev(it)->record_step < 10){
      continue;
    }
    */
    if(it->record_step > latest_step) {
      latest_step = it->record_step;
      latest_id = id;
    }
  }
  return latest_id;

}

handle_id_t MemHistory::DecideVictim(std::unordered_set<handle_id_t> handles, int device) {
  std::lock_guard<std::mutex> lock(mutex_[device]);
  if (iteration_idx_ <= 2) {
    return MemHistory::LRU(handles, device);
  } else {
    return (this->*DoDecide)(handles, device);
  }
}

void MemHistory::PrintRecord(int device) {
  std::lock_guard<std::mutex> lock(mutex_[device]);
  std::ofstream fp;
  fp.open("history_log.txt");
  std::vector<MemRecord> records;
  std::map<handle_id_t, std::vector<MemRecord> >::iterator it;
  for(it = history[device].begin(); it != history[device].end(); ++it) {
    for(size_t i = 0; i < (it->second).size(); i++) {
      records.push_back(it->second[i]);
    }
  }
  std::sort(records.begin(), records.end(), MemHistory::CompareByStep);
  for(size_t i = 0; i < records.size(); i++) {
    MemRecord r = records[i];
    fp << "No." << i << std::endl;
    fp << "Step: " << r.record_step << std::endl;
    fp << "Handle ID: " << r.handle_id << std::endl;
    fp << "Operation: ";
    if(r.operation_id == GET_ADDR)
      fp << "get";
    else if(r.operation_id == SET_ADDR)
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
  for(int i = 0; i < NUMBER_OF_GPU; i++) {
    record_idx[i] = 0;
  }
  if(iteration_idx_ <= 2 || swap_algorithm_ == "LRU") {
    pre_recording_ = true;
  }
  if(iteration_idx_ == 2) {
    is_recording_ = true;
  } else if(iteration_idx_ > 2) {
    /*
    Prefetch::Get()->StartPrefetching();
    while(!Prefetch::Get()->IsPrefetching())
      usleep(5);
    */
  }
  begin_time_ = high_resolution_clock::now();
}

void MemHistory::StopIteration() {
  pre_recording_ = false;
  is_recording_ = false;
  iteration_started_ = false;
  /*
  if(Prefetch::Get()->IsPrefetching()) {
    Prefetch::Get()->StopPrefetching();
  }
  */
  ++iteration_idx_;
}

} // namespace mxnet



