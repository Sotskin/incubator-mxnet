#ifndef GPU_SWAP_HISTORY_H_
#define GPU_SWAP_HISTORY_H_

#include <chrono>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <list>
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
using timestep_t = unsigned long long;

const int NUMBER_OF_GPU = 1;

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

  unsigned prefetch_count;
  unsigned cache_miss;

  std::vector<std::map<handle_id_t, std::vector<MemRecord> > > history;
  std::vector<std::vector<MemRecord> > ordered_history;
  std::vector<std::list<handle_id_t> > lru_list;
  std::vector<std::unordered_map<handle_id_t, std::list<handle_id_t>::iterator> >
      lru_map;
  std::vector<size_t> record_idx; 

  ~MemHistory();
  static MemHistory* Get();
  static std::shared_ptr<MemHistory> _GetSharedRef();
  bool IterationStarted() {return iteration_started_;}
  bool IsPreRecording() {return pre_recording_;}
  bool IsRecording() {return is_recording_;}
  size_t GetIterationIdx() {return iteration_idx_;}
  void PreRecord(handle_id_t handle_id, record_t operation_id, int device);
  void PutRecord(handle_id_t handle_id, int device, record_t type, size_t size);
  void PrintRecord(int device);
  void StartIteration();
  void StopIteration();
  handle_id_t DecideVictim(std::unordered_set<handle_id_t> handles, int device, void* arg);
  // Logs
  size_t num_swap_in;
  size_t num_swap_out;
  size_t swap_in_total;
  size_t swap_out_total;
  size_t num_get_addr;

private:
  MemHistory();
  bool iteration_started_;
  bool is_recording_;
  bool pre_recording_;
  size_t iteration_idx_;
  std::string swap_algorithm_;
  high_resolution_clock::time_point begin_time_;
  std::vector<std::mutex> mutex_ = std::vector<std::mutex>(NUMBER_OF_GPU);
  handle_id_t (MemHistory::*DoDecide)(std::unordered_set<handle_id_t>, int, void*);
  // Swap algorithm declaration
  handle_id_t LRU(std::unordered_set<handle_id_t> handles, int device, void* arg);
  handle_id_t NaiveHistoryBased(std::unordered_set<handle_id_t> handles,
    int device, void* arg);

};  // class MemHistory

} // namespace mxnet

#endif
