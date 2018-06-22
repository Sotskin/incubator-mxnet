#ifndef GPU_SWAP_HISTORY_H_
#define GPU_SWAP_HISTORY_H_

#include <chrono>
#include <map>
#include <vector>
#include <thread>
#include <mutex>

#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std::chrono;

namespace mxnet {

using handle_id_t = unsigned long long;
using timestamp_t = unsigned long long;

const int NUMBER_OF_GPU = 8;

class MemHistory {
public:

  enum record_t {GET_ADDR, SET_ADDR, DEL_ADDR};
  struct MemRecord {
    handle_id_t handle_id;
    record_t operation_id;
    timestamp_t time;
    size_t record_step;
    size_t size;
  };
  static bool CompareByStep(const MemRecord &r1, const MemRecord &r2) {
    return r1.record_step < r2.record_step;
  }

  //std::vector<std::vector<MemRecord> > history 
  //    = std::vector<std::vector<MemRecord> >(NUMBER_OF_GPU);
  std::vector<std::map<handle_id_t, std::vector<MemRecord> > > history
      = std::vector<std::map<handle_id_t, std::vector<MemRecord> > >
      (NUMBER_OF_GPU);
  size_t record_idx;

  ~MemHistory();
  static MemHistory* Get();
  static std::shared_ptr<MemHistory> _GetSharedRef();
  bool IterationStarted() {return iteration_started_;}
  bool IsRecording() {return is_recording_;}
  void PutRecord(handle_id_t handle_id, int device, record_t type, size_t size);
  handle_id_t DecideVictim(std::vector<handle_id_t> handles, int device);
  void PrintRecord(int device);
  void StartIteration();
  void StopIteration();
  MemRecord find(std::vector<MemRecord> records, size_t target_step);

private:
  MemHistory();
  bool iteration_started_;
  bool is_recording_;
  size_t iteration_idx_;
  high_resolution_clock::time_point begin_time_;
  std::vector<std::mutex> mutex_ = std::vector<std::mutex>(NUMBER_OF_GPU);
};  // class MemHistory

} // namespace mxnet

#endif
