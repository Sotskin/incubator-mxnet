#include <iostream>
#include <fstream>
#include <unistd.h>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <set>
#include <dmlc/logging.h>
#include <mxnet/gpu_swap_history.h>
#include <mxnet/gpu_swap_prefetch.h>

namespace mxnet {

MemHistory::MemHistory() {
  iteration_started_ = false;
  is_recording_ = false;
  iteration_idx_ = 0;
  fifo_index_ = 0;
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

void MemHistory::PutRecord(handle_id_t handle_id, int device,
                          record_t operation_id, size_t size) {
  if(!IterationStarted())
    return;
  if(!IsRecording()) {
  } else {
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

// optimal algorithm: assume iterations remain the same; choose the handle
// whose next reference is furthest in the future as victim.
handle_id_t MemHistory::DecideVictim(std::unordered_set<handle_id_t> handles, int device) {
  std::lock_guard<std::mutex> lock(mutex_[device]);
  if (iteration_idx_ == 0) {
    while (handles.find(ordered_history[device][fifo_index_].handle_id)
        == handles.end()) {
      fifo_index_ ++;
      if (fifo_index_ == ordered_history[device].size()) {
        // (sotskin) none of the handles is in the history
        // should never happen
        return -1;
      }
    }
    return ordered_history[device][fifo_index_++].handle_id;
  }
  size_t latest_step = 0;
  handle_id_t latest_id = 0;
  for(auto &id : handles) {
    MemHistory::MemRecord r = 
        MemHistory::find(history[device][id], record_idx[device]);
    if(r.record_step > latest_step) {
      latest_step = r.record_step;
      latest_id = r.handle_id;
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
  iteration_started_ = true;
  for(int i = 0; i < NUMBER_OF_GPU; i++) {
    record_idx[i] = 0;
  }
  if(iteration_idx_ <= 1) {
    is_recording_ = true;
  }
  if(iteration_idx_ > 1) {
    /*
    for(int device = 0; device < NUMBER_OF_GPU; device++) {
      prefetcher_[device] = std::thread(&Prefetch::StartPrefetching, this, device);
    }
    */
    Prefetch::Get()->StartPrefetching();
    while(!Prefetch::Get()->IsPrefetching())
      usleep(5);
  }
  begin_time_ = high_resolution_clock::now();
}

void MemHistory::StopIteration() {
  if (iteration_idx_ == 0) {
    for (auto& _ordered_history : ordered_history) {
      _ordered_history.clear();
    }
    for (auto& _history : history) {
      _history.clear(); 
    }
    fifo_index_ = 0;
  }
  is_recording_ = false;
  iteration_started_ = false;
  Prefetch::Get()->StopPrefetching();
  ++iteration_idx_;
  /*
  if(Prefetch::IsPrefetching()) {
    for(int device = 0; device < NUMBER_OF_GPU; device++) {
      prefetcher_[device].join();
    }
  }
  */
}

MemHistory::MemRecord MemHistory::find(std::vector<MemHistory::MemRecord> 
    records, size_t step) {
  size_t start = 0;
  size_t end = records.size() - 1;
  while(start < end) {
    if(start == records.size()-1)
      return records[start];
    if(end == 0)
      return records[end];
    size_t mid = (start + end) / 2;
    size_t c_step = records[mid].record_step;
    bool right_before = c_step < step && records[mid+1].record_step > step;
    if(c_step == step || right_before) {
      return records[mid+1];
    } else if(c_step < step) {
      start = mid + 1;
    } else{
      end = mid - 1;
    }
  }
  // suppress warning of reaching end of non-void function
  // but actually should not reach here
  return records[0];
}


} // namespace mxnet



