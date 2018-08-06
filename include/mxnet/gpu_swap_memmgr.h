#ifndef MXNET_MEM_MGR_H_
#define MXNET_MEM_MGR_H_

#include <cuda_runtime_api.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <math.h>
#include <mxnet/buddy.h>
#include <stdio.h>
#include <string>

namespace mxnet {

// Save some memory for each device(subject to change).
const double GPU_UTIL_RATIO = 0.96;

class MemoryManager {
  public:
    cudaError_t Memcpy(int device_id, void* dst, const void* src, size_t count,
                       cudaMemcpyKind kind);
    cudaError_t MemcpyAsync(int device_id, void* dst, const void* src,
                            size_t count, cudaMemcpyKind kind,
                            cudaStream_t stream);
    cudaError_t StreamSynchronize(int device_id, cudaStream_t stream);
    virtual cudaError_t MemGetInfo(int device_id, size_t* total,
                                   size_t* free) = 0;
    virtual bool TryAllocate(int device_id, size_t size) = 0;
    virtual cudaError_t Malloc(void*& devptr, size_t size, int device_id) = 0;
    virtual cudaError_t Free(void* devptr, int device_id) = 0;
}; // Class MemoryManager

class CudaMemoryManager : public MemoryManager {
  public:
    cudaError_t MemGetInfo(int device_id, size_t* total, size_t* free);
    bool TryAllocate(int device_id, size_t size);
    cudaError_t Malloc(void*& devptr, size_t size, int device_id);
    cudaError_t Free(void* devptr, int device_id);

    friend std::shared_ptr<MemoryManager> GetMemoryManagerRef();

  private:
    CudaMemoryManager();
    ~CudaMemoryManager();
}; // Class CudaMemoryManager

class BuddyMemoryManager : public MemoryManager {
  public:
    cudaError_t MemGetInfo(int device_id, size_t* total, size_t* free);
    bool TryAllocate(int device_id, size_t size);
    cudaError_t Malloc(void*& devptr, size_t size, int device_id);
    cudaError_t Free(void* devptr, int device_id);

    friend std::shared_ptr<MemoryManager> GetMemoryManagerRef();

  private:
    BuddySystem** buddy_;
    std::mutex mutex_;
    int deviceCount_;

    BuddyMemoryManager();
    ~BuddyMemoryManager();
}; // Class BuddyMemoryManager

std::shared_ptr<MemoryManager> GetMemoryManagerRef();
MemoryManager* GetMemoryManager();
} //namespace mxnet

#endif // MXNET_MEM_MGR_H_
