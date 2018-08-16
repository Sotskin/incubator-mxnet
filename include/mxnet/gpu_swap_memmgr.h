#ifndef MXNET_GPU_SWAP_MEMMGR_H_
#define MXNET_GPU_SWAP_MEMMGR_H_

#include <cuda_runtime_api.h>
#include <iostream>
#include <map>
#include <math.h>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <string>

#include <mxnet/gpu_swap_buddy.h>
#include <mxnet/gpu_swap_history.h>

namespace mxnet {

// Save some memory for each device(subject to change).

class MemoryManager {
  public:
    static constexpr double kGPUUtilRatio = 0.95;
    static const size_t kMB = 1L << 20;
    static const size_t kGB = 1L << 30;

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
    BuddyMemoryManager();
    ~BuddyMemoryManager();

    std::vector<BuddySystem*> buddy_;
    // Note that this line means we assume there will no more than 16 GPUs.
    std::array<std::mutex, 16> mutex_;
}; // Class BuddyMemoryManager

//class SlabMemoryManager : public MemoryManager {
  //public:
    //cudaError_t MemGetInfo(int device_id, size_t* total, size_t* free);
    //bool TryAllocate(int device_id, size_t size);
    //cudaError_t Malloc(void*& devptr, size_t size, int device_id);
    //cudaError_t Free(void* devptr, int device_id);

    //friend std::shared_ptr<MemoryManager> GetMemoryManagerRef();

  //private:
    //SlabMemoryManager();
    //~SlabMemoryManager();

    //std::vector<SlabSystem*> buddy_;
    //// Note that this line means we assume there will no more than 16 GPUs.
    //std::array<std::mutex, 16> mutex_;
//}; // Class SlabMemoryManager
std::shared_ptr<MemoryManager> GetMemoryManagerRef();
MemoryManager* GetMemoryManager();

} //namespace mxnet

#endif // MXNET_MEM_MGR_H_
