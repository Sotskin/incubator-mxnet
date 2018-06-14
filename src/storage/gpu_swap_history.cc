#include <iostream>
#include <vector>
#include <set>
#include <mxnet/gpu_swap_history.h>
using namespace mxnet;

MemHistory::MemHistory() {
  iteration_started_ = false;
  do_record_ = false;
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
  if(!DoRecord()) {
  } else {
    std::lock_guard<std::mutex> lock(mutex_[device]);
    timestamp_t t = (duration_cast<microseconds>
        (high_resolution_clock::now() - begin_time_)).count();
    MemRecord record = {handle_id, operation_id, t, size};
    history[device].push_back(record);
  }
  record_idx++;
}

handle_id_t MemHistory::GetFurthest(std::vector<handle_id_t> handles, int device) {
  // TODO(karl): not implemented
  //std::lock_guard<std::mutex> lock(mutex_[device]);
  return 0;
}

void MemHistory::PrintRecord(int device) {
  std::lock_guard<std::mutex> lock(mutex_[device]);
  //std::vector<MemRecord>::iterator it;
  //for(it = history[device].begin(); it != history[device].end(); ++it) {
  //}
  MemHistory::MemRecord r = history[device][history[device].size()-1];
  std::cout << "Device: " << device << std::endl;
  std::cout << "Size of history: " << history[device].size() << std::endl;

  std::cout << "Latest record: ";
  std::cout << "Handle id " << r.handle_id;
  std::cout << ", Operation ";
  if(r.operation_id == MemHistory::GET_ADDR)
    std::cout << "get_addr";
  else if(r.operation_id == MemHistory::SET_ADDR)
    std::cout << "set_addr";
  else
    std::cout << "del_addr";
  std::cout << std::endl;
  std::cout << ", Size: " << r.size;
  std::cout << std::endl;
}

void MemHistory::StartIteration() {
  iteration_started_ = true;
  record_idx = 0;
  if(iteration_idx_ == 1)
    do_record_ = true;
  begin_time_ = high_resolution_clock::now();
}

void MemHistory::EndIteration() {
  do_record_ = false;
  iteration_started_ = false;
  ++iteration_idx_;
}



