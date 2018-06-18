#include <cuda_runtime_api.h>
#include <iostream>
#include <stdio.h>

namespace mxnet {

class MemoryManager {
  public:
    MemoryManager();
    ~MemoryManager();
    cudaError_t Malloc(void** devptr, size_t size);
    cudaError_t Free(void* devptr);
};

}

