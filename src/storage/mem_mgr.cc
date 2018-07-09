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

static inline void CHECK_CUDA_ERROR() {									   
  cudaError_t e = cudaGetLastError();					   
  CHECK_EQ(e, cudaSuccess) << "CUDA: " << cudaGetErrorString(e);         
}

BuddySystem::BuddySystem(Block* start, size_t total, int gpuIdx) 
  : start_(start),
    total_(total),
    allocated(0),
    free_(total),
    gpuIdx_(gpuIdx) {
  freeListSize_ = getListSize(total);
  freeList_ = new Block*[freeListSize_];
  for (int i = 0; i < freeListSize_; i++) {
    freeList_[i] = NULL;
  }
  if (freeListSize_ > 0) freeList_[freeListSize_ - 1] = start;
}

void* BuddySystem::Alloc(size_t size) {
  int listIdx = getListIdx(size);
  int currIdx = listIdx;
  bool found = false;
  Block* blockToBeAllocated;

  while(!found) {
    if (freeList_[listIdx] != NULL) {
      blockToBeAllocated = freeList[listIdx];
      freeList[listIdx] = blockToBeAllocated->getNext();
      blockToBeAllocated->setNext(NULL);
      found = true; 
    } else if (currIdx < freeListSize_) {
      currIdx++:
      if (freeList[currIdx] != NULL) {
        Block* blockToBeRemoved = freeList[currIdx];
        int blockSize = getListBlockSize(currIdx - 1);
        InsertBlock(new Block(blockToBeRemoved->getData(), (size_t)blockSize));
        InsertBlock(new Block(blockToBeRemoved->getData() + blockSize, blockToBeRemoved->getSize() - blockSize));
        freeList[currIdx] = blockToBeRemoved->getNext();
        blockToBeRemoved->setNext(NULL);
        currIdx = listIdx;
      }
    } else {
      break;
    }
  }
    
  if (found) {
    allocated += blockToBeAllocated->getSize();
    free -= blockToBeAllocated->getSize();
    return (void*)(blockToBeAllocated->getData());
  } else {
    return NULL;
  }    
}

cudaError_t BuddySystem::Free(void* ptr) {

}

void BuddySystem::InsertBlock(Block* block) {
  int idx = getListIdx(block->getSize());
  
  if (freeList_[idx] == NULL) {
    freeList_[idx] = block;
    return;
  }

  Block* curr, prev;
  curr = freeList_[idx];
 
  while (curr != NULL) {
    prev = curr;
  
  }  
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
  CUDA_CALL(cudaSetDevice(deviceIdx));
  BuddySystem* buddy = buddy_[deviceIdx];
  Block** freeList = buddy->getFreeList();
  int freeListSize = buddy->getFreeListSize();
  int idx = getListIdx(size);
  if (idx == 0) idx = 1;

  for (int i = idx; i < freeListSize; i++) {
    if (freeList[i] != NULL) return true;
  }

  return false;
}

} //namespace mxnet
