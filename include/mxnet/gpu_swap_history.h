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

struct SwapParams {
  size_t no_swap_steps;
  size_t required_memory;
  std::map<size_t, std::unordered_set<handle_id_t> >* divided_handles;
};

class MemoryHistory {
public:
  enum record_t {GET_ADDR, SET_ADDR, DEL_ADDR};
  struct MemRecord {
    handle_id_t handle_id;
    record_t operation_id;
    timestamp_t time;
    size_t record_step;
    size_t size;
  };
  struct DeviceHistory {
    static const size_t kMaxPreservedIteration = 10;
    std::list<std::map<handle_id_t, std::vector<MemRecord>>> all_handle_history;
    std::map<handle_id_t, std::vector<MemRecord>> *handle_history;
    std::list<std::vector<MemRecord>> all_ordered_history;
    std::vector<MemRecord> *ordered_history;
    std::list<handle_id_t> lru_list;
    std::unordered_map<handle_id_t, std::list<handle_id_t>::iterator> lru_map;
    size_t curr_idx;
    // Statistics
    size_t prefetch_count;
    size_t cache_miss;
    size_t num_swap_in;
    size_t num_swap_out;
    size_t swap_in_total;
    size_t swap_out_total;
    size_t num_get_addr;
  };
  static const size_t kBeginRecordAt = 3;

  ~MemoryHistory();
  static bool CompareByStep(const MemRecord &r1, const MemRecord &r2) {
    return r1.record_step < r2.record_step;
  }
  static MemoryHistory* Get();
  static std::shared_ptr<MemoryHistory> _GetSharedRef();
  bool IterationStarted() {return iteration_started_;}
  bool IsPreRecording() {return pre_recording_;}
  bool IsRecording() {return is_recording_;}
  DeviceHistory& DevHistory(int device) {return dev_history_[device];}
  size_t GetIterationIdx() {return iteration_idx_;}
  void PreRecord(handle_id_t handle_id, record_t op, DeviceHistory& history);
  void PutRecord(handle_id_t handle_id, int device, record_t op, size_t size);
  void PrintRecord(int device);
  void StartIteration();
  void StopIteration();
  void Statistics();
  handle_id_t DecideVictim(std::unordered_set<handle_id_t> handles, int device,
                           void* arg);

private:
  MemoryHistory();
  std::vector<std::mutex> mutex_ = std::vector<std::mutex>(NUMBER_OF_GPU);
  handle_id_t (MemoryHistory::*DoDecide)(std::unordered_set<handle_id_t>, int, void*);
  void PrintSimilarity();
  double LCS_Similarity(std::vector<MemRecord>& base,
                        std::vector<MemRecord>& target);
  // Swap algorithm declaration
  handle_id_t LRU(std::unordered_set<handle_id_t> handles, int device, void* arg);
  handle_id_t NaiveHistory(std::unordered_set<handle_id_t> handles, int device,
      void* arg);
  handle_id_t SizeHistory(std::unordered_set<handle_id_t> handles, int device,
      void* arg);

  std::vector<DeviceHistory> dev_history_;
  bool iteration_started_;
  bool is_recording_;
  bool pre_recording_;
  size_t iteration_idx_;
  bool adaptive_history_;
  std::string swap_algorithm_;
  high_resolution_clock::time_point begin_time_;
};  // class MemoryHistory

} // namespace mxnet

#endif
