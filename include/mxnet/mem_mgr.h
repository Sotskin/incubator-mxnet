#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdio.h>

namespace mxnet {

class MemoryManager {
  public:
    static MemoryManager* Get();
    static std::shared_ptr<MemoryManager> _GetSharedRef();
    ~MemoryManager();
    cudaError_t Malloc(void** devptr, size_t size);
    cudaError_t Free(void* devptr);

  private:
    MemoryManager();
};

} //namespace mxnet

