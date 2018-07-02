#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <mxnet/mem_mgr.h>
#include <../common/cuda_utils.h>

namespace mxnet {

MemoryManager* MemoryManager::Get() {
  static MemoryManager* mm = _GetSharedRef().get();
  return mm;
}

std::shared_ptr<MemoryManager> MemoryManager::_GetSharedRef() {
  static std::shared_ptr<MemoryManager> inst(new MemoryManager());
  return inst;
}

MemoryManager::MemoryManager() {
}

MemoryManger::~MemoryManger() {
}

cudaError_t MemoryManger::Malloc(void*& devptr, size_t size, int device_id){

  return cudaSuccess;
}

cudaError_t MemroyManger::Free(void* devptr, int device_id){

  return cudaSuccess;
}

cudaError_t MemoryManger::Memcpy(int device_id, void* dst, const void* src,
    size_t, count, enum cudaMemcpyKind kind) {
  return cudaSuccess;
}

cudaError_t MemoryManager::MemGetInfo(int device_id, size_t *total, 
    size_t* free) {
  return cudaSuccess;
}


bool MemoryManager::TryAllocate(int device_id, size_t size) {
  return true;
}

} // namespace mxnet


