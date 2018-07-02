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

//allocate one single biggest available block in the last freelist entry
MemoryManager::MemoryManager() {
  for (int i = 0; i < FREELISTSIZE_; i++) {
    char* data;
    size_t size = (size_t)exp2((double)(i + 7);
    cudaError_t err = cudaMalloc((void**)&data, size);
    if (err != cudaSuccess) {    
      CHECK_CUDA_ERROR(); 
    }
    else {
      Block b = new block(data, size);
      freeList_[i] = &b; 
    }   
  }
  usedList_ = NULL;
}

MemoryManager::~MemoryManager() {
  //TODO(qingsen): need implementation 
}

cudaError_t MemoryManager::Malloc(void*& devptr, size_t size, int deviceIdx) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(deviceIdx));
  int idx = getFreeListIdx(size);
  Block* prev = NULL;
  Block* b = findFirstFit(idx, prev, size);
  if (b == NULL) b = allocateBlock(size);  
  if (b == NULL) {
    devptr = NULL;
    return cudaErrorMemoryAllocation; 
  }
  splitAndPlace(b, prev, idx, size);
  b->setAllocated();
  b->setNext(allocatedList_);
  allocatedList_ = b;
  devptr = b->getData();
  return cudaSuccess;
}

//TODO(qingsen): Try to coalesce when freeing allocated blocks?
cudaError_t MemoryManager::Free(void* devptr, int deviceIdx) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(deviceIdx));
  Block* curr = allocatedList_;
  Block* prev = NULL;
 
  while (curr) {
    if (curr->getData() == devptr) break;
    prev = curr;
    curr = curr.getNext();
  }

  if (!curr) return cudaErrorInvalidValue;
  
  curr->setFree();
  if (prev) {
    prev->setNext(curr->getNext());
  }
  else {
    allocatedList_ = curr->getNext();
  } 

  int idx = getFreeListIdx(curr->getSize());
  curr->setNext(freeList_[idx]);
  freeList_[idx] = curr;
  return cudaSuccess;
}
        
cudaError_t MemoryManager::Memcpy(int deviceId, void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
  //TODO(qingsen): need implementation
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t MemoryManager::MemGetInfo(int deviceId, size_t* total, size_t* free) {
  //TODO(qingsen): need implementation
}

Block* MemoryManager::findFirstFit(int idx, Block* prev, size_t size) {
  Block* b = freeList_[idx];
  prev = NULL;
  if (b == NULL) return NULL;
  while (b != NULL) {
    if (b->getSize() >= size) return b;
    prev = b;
    b = b->getNext();
  }
  return NULL;
}

Block* MemoryManager::allocateBlock(size_t size) {
  char* data;
  cudaError_t err = cudaMalloc((void**)&data, size);
  if (err != cudaSuccess) {
    CHECK_CUDA_ERROR();
    return NULL;
  }
  Block b = new Block(data, size);
  return &b;
}

void MemoryManager::splitAndPlace(Block* b, Block* prev, int idx, size_t size) {
  if ((b->getSize() - size) < 128) {
    if (prev) {
      prev->setNext(b->getNext());
    }
    else {
      freeList_[idx] = b->getNext();
    }
  }
  else {
    size_t split = b->getSize() - size;
    Block splitBlock = new Block(b->getData() + size, b->getSize() - size);
    int splitIdx = getFreeListIdx(splitBlock->getSize());
    splitBlock.setNext(freeList_[splitIdx]);
    freeList_[spliltIdx] = &splitBlock; 
  }
}

} //namespace mxnet
