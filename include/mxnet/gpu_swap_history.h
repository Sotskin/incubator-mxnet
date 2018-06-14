#ifndef GPU_SWAP_HISTORY_H_
#define GPU_SWAP_HISTORY_H_

#include <chrono>
#include <vector>
#include "storage.h"
#include <thread>
#include <mutex>

#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif


using namespace std::chrono;

namespace mxnet {

class MemHistory {
public:

  enum record_t {GET_ADDR, SET_ADDR, DEL_ADDR};
  struct MemRecord {
    handle_id_t handle_id;
    record_t operation_id;
    timestamp_t time;
    size_t size;
  };

  std::vector<std::vector<MemRecord> > 
      history = std::vector<std::vector<MemRecord> >(8);
  size_t record_idx;

  ~MemHistory();
  static MemHistory* Get();
  static std::shared_ptr<MemHistory> _GetSharedRef();
  bool IterationStarted() {return iteration_started_;}
  bool DoRecord() {return do_record_;}
  void PutRecord(handle_id_t handle_id, int device, record_t type, size_t size);
  handle_id_t GetFurthest(std::vector<handle_id_t> handles, int device);
  void PrintRecord(int device);
  void StartIteration();
  void EndIteration();

private:
  MemHistory();
  bool iteration_started_;
  bool do_record_;
  size_t iteration_idx_;
  high_resolution_clock::time_point begin_time_;
  std::vector<std::mutex> mutex_ = std::vector<std::mutex>(8);
};

}

#endif
