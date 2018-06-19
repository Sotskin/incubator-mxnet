#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <set>
#include <mxnet/gpu_swap_history.h>

namespace mxnet {

MemHistory::MemHistory() {
  iteration_started_ = false;
  is_recording_ = false;
  iteration_idx_ = 0;
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
    size_t record_step = record_idx;
    MemRecord record = {handle_id, operation_id, t, record_step, size};
    history[device][handle_id].push_back(record);
  }
  record_idx++;
}

handle_id_t MemHistory::DecideVictim(std::vector<handle_id_t> handles, int device) {
  std::lock_guard<std::mutex> lock(mutex_[device]);

  size_t latest_step = 0;
  handle_id_t latest_id = 0;
  std::vector<handle_id_t>::iterator it;
  for(it = handles.begin(); it != handles.end(); ++it) {
    handle_id_t id = *it;
    MemHistory::MemRecord r = 
        MemHistory::find(history[device][id], record_idx);
    if(r.record_step > latest_step) {
      latest_step = r.record_step;
      latest_id = r.handle_id;
    }
  }

  return latest_id;
}

void MemHistory::PrintRecord(int device) {
  std::lock_guard<std::mutex> lock(mutex_[device]);
  std::ofstream f;
  f.open("history_log.txt");
  std::vector<MemRecord> v;
  std::map<handle_id_t, std::vector<MemRecord> >::iterator it;
  for(it = history[device].begin(); it != history[device].end(); ++it) {
    for(size_t i = 0; i < (it->second).size(); i++) {
      v.push_back(it->second[i]);
    }
  }
  std::sort(v.begin(), v.end(), MemHistory::CompareByStep);
  for(size_t i = 0; i < v.size(); i++) {
    MemRecord r = v[i];
    f << "No." << i << std::endl;
    f << "Step: " << r.record_step << std::endl;
    f << "Handle ID: " << r.handle_id << std::endl;
    f << "Operation: ";
    if(r.operation_id == GET_ADDR)
      f << "get";
    else if(r.operation_id == SET_ADDR)
      f << "set";
    else
      f << "del";
    f << std::endl;
    f << "Time: " << r.time << std::endl;
    f << "Size: " << r.size << std::endl;
    f << std::endl;
  }
}

void MemHistory::StartIteration() {
  iteration_started_ = true;
  record_idx = 0;
  if(iteration_idx_ == 0)
    is_recording_ = true;
  begin_time_ = high_resolution_clock::now();
}

void MemHistory::StopIteration() {
  is_recording_ = false;
  iteration_started_ = false;
  ++iteration_idx_;
}

MemHistory::MemRecord MemHistory::find(std::vector<MemHistory::MemRecord> v,
    size_t step) {
  size_t i = 0;
  size_t j = v.size() - 1;
  while(i < j) {
    if(i == v.size()-1)
      return v[i];
    if(j == 0)
      return v[j];
    size_t mid = (i + j) / 2;
    size_t c_step = v[mid].record_step;
    bool right_before = c_step < step && v[mid+1].record_step > step;
    if(c_step == step || right_before) {
      return v[mid+1];
    } else if(c_step < step) {
      i = mid + 1;
    } else{
      j = mid - 1;
    }
  }
  // suppress warning of reaching end of non-void function
  // but actually should not reach here
  return v[0];
}


} // namespace mxnet



