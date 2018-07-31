#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include <math.h>
#include <mxnet/mem_mgr.h>
#include "../common/cuda_utils.h"

namespace mxnet {

CudaMemoryManager::CudaMemoryManager() {
  std::cout << "Initialize CUDA Memory Allocator" << std::endl;
}

CudaMemoryManager::~CudaMemoryManager() {
  std::cout << "Destroy Cuda Memory Allocator" << std::endl;
}

cudaError_t CudaMemoryManager::Malloc(void*& devptr, size_t size,
                                      int device_id) {
  cudaSetDevice(device_id);
  cudaError_t e = cudaMalloc(&devptr, size);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << "Malloc failed: " << cudaGetErrorString(e) << std::endl;
  }
  return e;
}

cudaError_t CudaMemoryManager::Free(void* devptr, int device_id) {
  cudaSetDevice(device_id);
  cudaError_t e = cudaFree(devptr);
  if(e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << "Free failed: " << cudaGetErrorString(e) << std::endl;
  }
  return e;
}

cudaError_t CudaMemoryManager::Memcpy(int device_id, void* dst, const void* src,
                                      size_t count, enum cudaMemcpyKind kind) {
  cudaSetDevice(device_id);
  cudaSetDevice(device_id);
  cudaError_t e = cudaMemcpy(dst, src, count, kind);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << "Memcpy failed: " << cudaGetErrorString(e) << std::endl;
  }
  return e;
}

cudaError_t CudaMemoryManager::MemGetInfo(int device_id, size_t *total,
                                          size_t* free) {
  std::cout<<"MemGetInfo: Check"<<std::endl;
  cudaError_t e = cudaSetDevice(device_id);
  if (e != cudaSuccess) {
    std::cout << e << " Check setdevice failed: " << cudaGetErrorString(e)
              << std::endl;
  }
  size_t free_, total_;
  e = cudaMemGetInfo(&free_, &total_);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << e << " Check GetInfo failed: " << cudaGetErrorString(e)
              << std::endl;
  } else {
    std::cout << free_ << " " << total_ << std::endl;
  }
  std::cout << "MemGetInfo: Check Over" << std::endl;
  return cudaSuccess;
}

bool CudaMemoryManager::TryAllocate(int device_id, size_t size) {
  cudaError_t e = cudaSetDevice(device_id);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << e << " TryAlloc SetDevice failed: " << cudaGetErrorString(e)
              << std::endl;
  }
  size_t free, total;
  e = cudaMemGetInfo(&free, &total);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << e << " TryAlloc GetInfo failed: " << cudaGetErrorString(e)
              << std::endl;
  }
  return free > size + 1000000000;
}

// FEGIN
static std::shared_ptr<MemoryManager> GetMemoryManagerRef() {
  static std::shared_ptr<MemoryManager> inst;
  static bool set = false;
  if (set) {
    return inst;
  } else {
    std::string mem_mgr_type = dmlc::GetEnv("MXNET_MEM_MGR_TYPE",
                                            std::string("CUDA"));
    if (mem_mgr_type == "CUDA") {
      inst.reset(new CudaMemoryManager());
    } else if (mem_mgr_type == "Buddy") {
      // FIXME:
    }
    set = true
  }
}

static MemoryManager* GetMemoryManager() {
  static MemoryManager* mm = GetMemoryManagerRef().get()
  return mm;
}


} // namespace mxnet


