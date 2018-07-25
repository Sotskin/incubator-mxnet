#ifndef MXNET_MEM_MGR_H_
#define MXNET_MEM_MGR_H_

#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <string>

namespace mxnet {

class MemoryManager {
  std::mutex mutex_;
  public:
    ~MemoryManager();
    static MemoryManager* Get();
    static std::shared_ptr<MemoryManager> _GetSharedRef();
    cudaError_t Malloc(void*& devptr, size_t size, int deviceIdx);
    cudaError_t Free(void* devptr, int deviceIdx);
    cudaError_t Memcpy(int deviceId, void* dst, 
                       const void* src, size_t count, enum cudaMemcpyKind kind);
    cudaError_t MemGetInfo(int deviceId, size_t* total, size_t* free);   
    bool TryAllocate(int deviceId, size_t size);

  private:
    MemoryManager();
};  // Class MemoryManager

} //namespace mxnet

#endif // MXNET_MEM_MGR_H_
