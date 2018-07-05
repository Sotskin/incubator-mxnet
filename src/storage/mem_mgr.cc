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
  return cudaMalloc(&devptr, size);
}

cudaError_t MemoryManager::Free(void* devptr, int device_id){
  cudaSetDevice(device_id);
  return cudaFree(devptr);
}

cudaError_t MemoryManager::Memcpy(int device_id, void* dst, const void* src,
    size_t count, enum cudaMemcpyKind kind) {
  cudaSetDevice(device_id);
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t MemoryManager::MemGetInfo(int device_id, size_t *total, 
    size_t* free) {
  return cudaSuccess;
}


bool MemoryManager::TryAllocate(int device_id, size_t size) {
  cudaSetDevice(device_id);
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  return free  > size;
}

} // namespace mxnet


