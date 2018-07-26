#ifndef MXNET_MEM_MGR_H_
#define MXNET_MEM_MGR_H_

#include <mxnet/buddy.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <math.h>
#include <stdio.h>
#include <string>

namespace mxnet {

const double GPU_UTIL_RATIO = 0.96; //save some memory for each device(subject to change)

typedef enum {
  memStatus_Sucess,
  memStatus_InvalidValue,
  memStatus_OutOfMemory,
  memStatus_CUDAError
} memStatus_t;

inline std::string MemGetStatusString(memStatus_t status) {
  switch (status) {
    case memStatus_Sucess: return "Sucess";
    case memStatus_InvalidValue: return "Invalid value";
    case memStatus_OutOfMemory: return "Out of memory";
    case memStatus_CUDAError: return "CUDA error";
  }
  return "Unknown error";
}
 
class MemoryManager {
  private:  
    BuddySystem** buddy_;
    std::mutex mutex_;
    int deviceCount_;
    MemoryManager();    

  public:
    ~MemoryManager();
    static MemoryManager* Get();
    static std::shared_ptr<MemoryManager> _GetSharedRef();
    cudaError_t Malloc(void*& devptr, size_t size, int deviceIdx);
    cudaError_t Free(void* devptr, int deviceIdx);
    cudaError_t Memcpy(int deviceIdx, void* dst, 
                       const void* src, size_t count, enum cudaMemcpyKind kind);
    cudaError_t MemGetInfo(int deviceId, size_t* total, size_t* free);   
    bool TryAllocate(int deviceId, size_t size);
};  // Class MemoryManager
} //namespace mxnet
#endif // MXNET_MEM_MGR_H_
