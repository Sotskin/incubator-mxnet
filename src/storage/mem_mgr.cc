#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <mxnet/mem_mgr.h>
#include "../common/cuda_utils.h"

namespace mxnet {

#define CUDA_CALL(func) 					 \
  {								 \
    cudaError_t e = (func);     				 \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)     \
        << "CUDA: " << cudaGetErrorString(e);                    \
  }

static inline void CHECK_CUDA_ERROR() {									   \
  cudaError_t e = cudaGetLastError();					   
  CHECK_EQ(e, cudaSuccess) << "CUDA: " << cudaGetErrorString(e);         
}

void* BuddySystem::Alloc(size_t size) {

}

cudaError_t BuddySystem::Free(void* ptr) {

}

MemoryManager* MemoryManager::Get() {
  static MemoryManager* mm = _GetSharedRef().get();
  return mm;
}

std::shared_ptr<MemoryManager> MemoryManager::_GetSharedRef() {
  static std::shared_ptr<MemoryManager> inst(new MemoryManager());
  return inst;
} 

MemoryManager::MemoryManager() {
  int deviceNum;
  CUDA_CALL(cudaGetDeviceCount(&deviceNum));
  buddy_ = new BuddySystem*[deviceNum];
  
  for (int deviceIdx = 0; deviceIdx < deviceNum; deviceIdx++) {
    buddy_[deviceIdx] = NULL;
    CUDA_CALL(cudaSetDevice(deviceIdx));
    
    size_t avail, total;
    size_t mb = 1 << 20;
    CUDA_CALL(cudaMemGetInfo(&avail, &total));
  
    avail = static_cast<size_t>(avail * GPUUTILRATIO);  
    char* wholeMemory = NULL;
    while (cudaMalloc((void**)&wholeMemory, avail) == cudaErrorMemoryAllocation) {
        avail -= mb;
        if (avail <= 0) break;
    }
 
    if (avail > 0) buddy_[deviceIdx] = new BuddySystem(new Block(wholeMemory, avail), avail, deviceIdx);
  } 
}

//MemoryManager::~MemoryManager() {
  //TODO(qingsen): need implementation 
//  cout << "Memory manager destructed";
//}

cudaError_t MemoryManager::Malloc(void*& devptr, size_t size, int deviceIdx) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(deviceIdx));
  devptr = buddy_[deviceIdx]->Alloc(size);
  if (!devptr) return cudaErrorMemoryAllocation;
  return cudaSuccess;
}

cudaError_t MemoryManager::Free(void* devptr, int deviceIdx) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(deviceIdx));
  buddy_[deviceIdx]->Free(devptr);
  return cudaSuccess;
}
        
cudaError_t MemoryManager::Memcpy(int deviceId, void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
  //TODO(qingsen): need implementation
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t MemoryManager::MemGetInfo(int deviceIdx, size_t* total, size_t* free) {
  //TODO(qingsen): need implementation
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(deviceIdx));
  return cudaMemGetInfo(free, total);
}

bool MemoryManager::TryAllocate(int deviceIdx, size_t size) {
  //CUDA_CALL(cudaSetDevice(deviceIdx));
  //if (size > buddy_->maxBlock_) {
  //  return false;
  //} else {
  //  return true;
  //}
}
} //namespace mxnet
