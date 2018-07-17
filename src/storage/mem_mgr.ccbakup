#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <mxnet/mem_mgr.h>
#include "../common/cuda_utils.h"

namespace mxnet {

const size_t hack_size = 0;

MemoryManager* MemoryManager::Get() {
  static MemoryManager* mm = _GetSharedRef().get();
  return mm;
}

std::shared_ptr<MemoryManager> MemoryManager::_GetSharedRef() {
  static std::shared_ptr<MemoryManager> inst(new MemoryManager());
  return inst;
}

MemoryManager::MemoryManager() {
  std::cout << "Initialize Memory Allocator" << std::endl;
}

MemoryManager::~MemoryManager() {
  std::cout << "Destroy Memory Allocator" << std::endl;
}

cudaError_t MemoryManager::Malloc(void*& devptr, size_t size, int device_id){
  cudaSetDevice(device_id);
  cudaError_t e = cudaMalloc(&devptr, size);
  if(e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << "Malloc failed: " << cudaGetErrorString(e) << std::endl;
  }
  return e;
}

cudaError_t MemoryManager::Free(void* devptr, int device_id){
  cudaSetDevice(device_id);
  cudaError_t e = cudaFree(devptr);
  if(e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << "Free failed: " << cudaGetErrorString(e) << std::endl;
  }
  return e;
}

cudaError_t MemoryManager::Memcpy(int device_id, void* dst, const void* src,
    size_t count, enum cudaMemcpyKind kind) {
  cudaSetDevice(device_id);
  cudaError_t e =cudaMemcpy(dst, src, count, kind);
  if(e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << "Memcpy failed: " << cudaGetErrorString(e) << std::endl;
  }
  return e;
}

cudaError_t MemoryManager::MemGetInfo(int device_id, size_t *total, 
    size_t* free) {
  std::cout<<"MemGetInfo: Check"<<std::endl;
  cudaError_t e = cudaSetDevice(device_id);
  if(e != cudaSuccess) {
    std::cout << e << " Check setdevice failed: " <<
      cudaGetErrorString(e) << std::endl;
  }
  size_t free_, total_;
  e = cudaMemGetInfo(&free_, &total_);
  if(e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << e << " Check GetInfo failed: " << 
      cudaGetErrorString(e) << std::endl;
  } else {
    std::cout<<free_<<" "<<total_<<std::endl;
  }
  std::cout<<"MemGetInfo: Check Over"<<std::endl;
  return cudaSuccess;
}


bool MemoryManager::TryAllocate(int device_id, size_t size) {
  cudaError_t e = cudaSetDevice(device_id);
  if(e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << e << " TryAlloc SetDevice failed: " << cudaGetErrorString(e) << std::endl;
  }
  size_t free, total;
  e = cudaMemGetInfo(&free, &total);
  if(e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << e << " TryAlloc GetInfo failed: " << cudaGetErrorString(e) << std::endl;
  }
  return free > size + 1000000000;
}

} // namespace mxnet


