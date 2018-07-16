#include <iostream>
#include <fstream>
#include <unistd.h>
#include <algorithm>
#include <unordered_set>
#include <list>
#include <vector>
#include <set>
#include <dmlc/logging.h>
#include <mxnet/gpu_swap_history.h>
#include <mxnet/gpu_swap_prefetch.h>

namespace mxnet {

MemHistory::MemHistory() {
  iteration_started_ = false;
  is_recording_ = false;
  pre_recording_ = false;
  iteration_idx_ = 0;
  history.resize(NUMBER_OF_GPU);
  ordered_history.resize(NUMBER_OF_GPU);
  lru_list.resize(NUMBER_OF_GPU);
  lru_map.resize(NUMBER_OF_GPU);
  record_idx.resize(NUMBER_OF_GPU);
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
    std::cout<<"Prerecord setaddr"<<std::endl;
    lru_list[device].push_front(id);
    lru_map[device][id] = lru_list[device].begin();
  } else if(op == MemHistory::GET_ADDR) {
    std::cout<<"Prerecord getaddr"<<std::endl;
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
    std::cout<<"Prerecord deladdr"<<std::endl;
    std::list<handle_id_t>::iterator hid = lru_map[device][id];
    lru_list[device].erase(hid);
    lru_map[device].erase(id);
  }
}

void MemHistory::PutRecord(handle_id_t handle_id, int device,
                          record_t operation_id, size_t size) {
  if(!IterationStarted()) { 
    std::cout << "iteration not started" << std::endl;
    return;
  }
  if(IsPreRecording()) {
    std::lock_guard<std::mutex> lock(mutex_[device]);
    std::cout<<"Prerecord "<<handle_id<<std::endl;
    MemHistory::PreRecord(handle_id, operation_id, device);
  } 
  if(IsRecording()) {
    std::lock_guard<std::mutex> lock(mutex_[device]);
    std::cout << "Record "<< handle_id << " " << size << std::endl;
    timestamp_t t = (duration_cast<microseconds>
        (high_resolution_clock::now() - begin_time_)).count();
    size_t record_step = record_idx[device];
    MemRecord record = {handle_id, operation_id, t, record_step, size};
    history[device][handle_id].push_back(record);
    ordered_history[device].push_back(record);
  } else {
    std::cout << "Not recording" << std::endl;
  }
  record_idx[device]++;
}

handle_id_t MemHistory::LRU(std::unordered_set<handle_id_t> handles, int device) {
  handle_id_t victim = -1;
  while(lru_list[device].size() != 0 &&
    handles.find(lru_list[device].back()) == handles.end()) {
    std::cout<<"list size:"<<lru_list[device].size()<<std::endl;
    handle_id_t temp_id = lru_list[device].back();
    lru_map[device][temp_id] = lru_list[device].end();
    lru_list[device].pop_back();
  }
  if(lru_list[device].size() == 0) {
    std::cout << "No swappable handle found" << std::endl;
  } else {
    victim = lru_list[device].back();
    lru_map[device][victim] = lru_list[device].end();
    lru_list[device].pop_back();
  }
  return victim;
}

// optimal algorithm: assume iterations remain the same; choose the handle
// whose next reference is furthest in the future as victim.
handle_id_t MemHistory::DecideVictim(std::unordered_set<handle_id_t> handles, int device) {
  std::lock_guard<std::mutex> lock(mutex_[device]);
  if (iteration_idx_ <= 2) {
    std::cout << "No real history, lru used" << std::endl;
    return MemHistory::LRU(handles, device);
  }
  std::cout << "History based victim decision" << std::endl;
  size_t latest_step = 0;
  handle_id_t latest_id = 0;
  for(auto &id : handles) {
    std::cout<<"Handle:"<<id<<";History size:"<<history[device][id].size()<<std::endl;
    MemHistory::MemRecord r = {0,MemHistory::GET_ADDR,0,record_idx[device],0};
    std::vector<MemHistory::MemRecord>::iterator it =
      std::upper_bound(history[device][id].begin(), history[device][id].end(),
      r, CompareByStep);
    // if record not found, will return -1
    if(it == history[device][id].end())
      return id;
    if(it->record_step > latest_step) {
      latest_step = it->record_step;
      latest_id = id;
    }
  }
  return latest_id;
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
  std::cout<<"Start Iteration: " << iteration_idx_<<std::endl;
  iteration_started_ = true;
  for(int i = 0; i < NUMBER_OF_GPU; i++) {
    record_idx[i] = 0;
  }
  if(iteration_idx_ <= 2) {
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
  std::cout<<"StartIteration end"<<std::endl;
}

void MemHistory::StopIteration() {
  std::cout<<"Iteration Stopped"<<std::endl;
  pre_recording_ = false;
  is_recording_ = false;
  iteration_started_ = false;
  /*
  if(Prefetch::Get()->IsPrefetching()) {
    Prefetch::Get()->StopPrefetching();
  }
  */
  ++iteration_idx_;
  /*
  if(Prefetch::IsPrefetching()) {
    for(int device = 0; device < NUMBER_OF_GPU; device++) {
      prefetcher_[device].join();
    }
  }
  */
}

MemHistory::MemRecord* MemHistory::find(std::vector<MemHistory::MemRecord> 
    records, size_t step) {
  return nullptr;
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // NOTE: The following code is obsolete and may be buggy
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  /*
  size_t start = 0;
  size_t end = (records.size() == 0) ? 0 : records.size() - 1;
  std::cout << "start: "<<start<<";end: "<<end<<";step: " << step<<std::endl;
  while(start < end) {
    if(start == records.size()-1)
      return &records[start];
    if(end == 0)
      return &records[end];
    size_t mid = (start + end) / 2;
    size_t c_step = records[mid].record_step;
    bool right_before = c_step < step && records[mid+1].record_step > step;
    if(c_step == step || right_before) {
      return &records[mid+1];
    } else if(c_step < step) {
      start = mid + 1;
    } else{
      end = mid - 1;
    }
  }
  // suppress warning of reaching end of non-void function
  // but actually should not reach here
  return nullptr;
  */
}


} // namespace mxnet



