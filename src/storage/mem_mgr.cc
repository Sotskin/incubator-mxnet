#include <cuda_runtime_api.h>
#include <mxnet/mem_mgr.h>

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
  std::cout << "Initialize Memory Allocator" << std::endl;
}

MemoryManager::~MemoryManager() {
  std::cout << "Destory Memory Allocator" << std::endl;
}

cudaError_t MemoryManager::Malloc(void** devptr, size_t size) {
  return cudaMalloc(devptr, size);
}

cudaError_t MemoryManager::Free(void* devptr) {
  return cudaFree(devptr);
}

} //namespace mxnet
