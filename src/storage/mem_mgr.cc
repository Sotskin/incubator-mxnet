#include <assert.h>
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
    allocated_(0),
    free_(total),
    gpuIdx_(gpuIdx) {
  freeListSize_ = GetListSize(total);
  freeList_ = new Block*[freeListSize_];
  for (int i = 0; i < freeListSize_; i++) {
    freeList_[i] = NULL;
  }
  if (freeListSize_ > 0) freeList_[freeListSize_ - 1] = start;
}

void* BuddySystem::Alloc(size_t size) {
  int listIdx = GetListIdx(size);
  int currIdx = listIdx;
  bool found = false;
  Block* blockToBeAllocated;

  while(!found) {
    if (freeList_[listIdx] != NULL) {
      blockToBeAllocated = freeList_[listIdx];
      freeList_[listIdx] = blockToBeAllocated->GetNext();
      blockToBeAllocated->SetNext(NULL);
      found = true; 
    } else if (currIdx < freeListSize_) {
      currIdx++;
      if (freeList_[currIdx] != NULL) {
        Block* blockToBeRemoved = freeList_[currIdx];
        int blockSize = GetListBlockSize(currIdx - 1);
        InsertBlock(new Block(blockToBeRemoved->GetData(), (size_t)blockSize));
        InsertBlock(new Block(blockToBeRemoved->GetData() + blockSize, blockToBeRemoved->GetSize() - blockSize));
        freeList_[currIdx] = blockToBeRemoved->GetNext();
        blockToBeRemoved->SetNext(NULL);
        currIdx = listIdx;
      }
    } else {
      break;
    }
  }
    
  if (found) {
    size_t size = blockToBeAllocated->GetSize();
    allocated_ += size;
    free_ -= size;
    memPool_[blockToBeAllocated->GetData()] = blockToBeAllocated;
    return (void*)(blockToBeAllocated->GetData());
  } else {
    return NULL;
  }    
}

cudaError_t BuddySystem::Free(void* ptr) {
  std::map<char*, Block*>::iterator itr = memPool_.find((char*)ptr);
  if (itr == memPool_.end()) return cudaErrorInvalidValue;
  Block* blockToBeInserted = itr->second;
  memPool_.erase(itr);
  allocated_ -= blockToBeInserted->GetSize();
  free_ += blockToBeInserted->GetSize();
  InsertBlock(blockToBeInserted);
  Merge(blockToBeInserted);
  return cudaSuccess;
}

void BuddySystem::InsertBlock(Block* block) {
  int idx = GetListIdx(block->GetSize()); 
  if (freeList_[idx] == NULL) {
    freeList_[idx] = block;
    return;
  }

  Block* curr;
  Block* prev;
  prev = NULL;
  curr = freeList_[idx];
 
  while (curr != NULL) {
    if (curr->GetData() > block->GetData()) break;
    prev = curr;
    curr = curr->GetNext();
  }  

  if (prev != NULL) {
    prev->SetNext(block);
    block->SetNext(curr);
  } else {
    block->SetNext(freeList_[idx]);
    freeList_[idx] = block;
  } 
}

void BuddySystem::Merge(Block* block) {
  int idx = GetListIdx(block->GetSize());
  size_t listBlockSize = GetListBlockSize((size_t)idx);
  Block* curr = freeList_[idx];
  Block* prev = NULL;
  
  while (curr != block && curr != NULL) {
    prev = curr;
    curr = curr->GetNext();
  } 

  if (curr == NULL) return;
  if (curr->GetNext() != NULL) {
    Block* next = curr->GetNext();
    if ((curr->GetData() + listBlockSize) == next->GetData()) {
      curr->SetSize(curr->GetSize() + next->GetSize());
      curr->SetNext(next->GetNext());
      next->SetNext(NULL);
    }
  }
  if (prev != NULL) {
    if ((prev->GetData() + listBlockSize) == curr->GetData()) {
      prev->SetSize(prev->GetSize() + curr->GetSize());
      prev->SetNext(curr->GetNext());
      curr->SetNext(NULL);
      InsertBlock(prev);
      return;
    }
  }
  InsertBlock(curr);
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
  deviceCount_ = deviceNum;
  buddy_ = new BuddySystem*[deviceNum];
  
  for (int deviceIdx = 0; deviceIdx < deviceNum; deviceIdx++) {
    buddy_[deviceIdx] = NULL;
    CUDA_CALL(cudaSetDevice(deviceIdx));
    
    size_t avail, total;
    size_t mb = 1 << 20;
    CUDA_CALL(cudaMemGetInfo(&avail, &total));
  
    avail = static_cast<size_t>(avail * GPU_UTIL_RATIO);  
    char* wholeMemory = NULL;
    while (cudaMalloc((void**)&wholeMemory, avail) == cudaErrorMemoryAllocation) {
        avail -= mb;
        if (avail <= 0) break;
    }

    if (avail > 0) {
      buddy_[deviceIdx] = new BuddySystem(new Block(wholeMemory, avail), avail, deviceIdx);
    } else {
      std::cout << "Warning: There's no memory left on device: " << deviceIdx << std::endl;
    }
  } 
}

MemoryManager::~MemoryManager() {
  typedef std::map<char*, Block*> MemoryPool;
  for (int deviceIdx = 0; deviceIdx < deviceCount_; deviceIdx++) {
    CUDA_CALL(cudaSetDevice(deviceIdx));
    BuddySystem* buddy = buddy_[deviceIdx];
    MemoryPool mp = buddy->GetMemPool();
    while (!mp.empty()) {
      buddy->Free((void*)(mp.begin()->first));
    }
    cudaFree((void*)buddy->GetStart());    
  }
}

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
        
cudaError_t MemoryManager::Memcpy(int deviceIdx, void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
  //TODO(qingsen): need implementation
  CUDA_CALL(cudaSetDevice(deviceIdx));
  return cudaMemcpy(dst, src, count, kind);
}

//returns total memory and total free memory(not necessarily consequtive) in mmu
cudaError_t MemoryManager::MemGetInfo(int deviceIdx, size_t* total, size_t* free) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(deviceIdx));
  if (buddy_[deviceIdx] == NULL) return cudaErrorInvalidValue;
  *total = buddy_[deviceIdx]->GetTotal();
  *free = buddy_[deviceIdx]->GetFree();
  return cudaSuccess;
}

bool MemoryManager::TryAllocate(int deviceIdx, size_t size) {
  CUDA_CALL(cudaSetDevice(deviceIdx));
  BuddySystem* buddy = buddy_[deviceIdx];
  Block** freeList = buddy->GetFreeList();
  int freeListSize = buddy->GetFreeListSize();
  int idx = GetListIdx(size);
  if (idx == 0) idx = 1;

  for (int i = idx; i < freeListSize; i++) {
    if (freeList[i] != NULL) return true;
  }

  return false;
}
} //namespace mxnet
