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
  std::cout << "Initializing Buddy System No." << gpuIdx << std::endl;
  freeListSize_ = GetListSize(total);
  freeList_ = new Block*[freeListSize_];
  for (int i = 0; i < freeListSize_; i++) {
    freeList_[i] = NULL;
  }
  if (freeListSize_ > 0) freeList_[freeListSize_ - 1] = start;
  std::cout << "Buddy System No." << gpuIdx << " initialization finished." <<std::endl;
}

void* BuddySystem::Alloc(size_t size) {
  std::cout << "Buddy System No." << gpuIdx_ << ": Allocating size = " << size << " bytes" << std::endl;
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
      std::cout << "Found block of size = " << size << " bytes" << std::endl; 
    } else if (currIdx < freeListSize_) {
      currIdx++;
      if (freeList_[currIdx] != NULL) {
        //std::cout << "Spliting in progress: listIdx = " << listIdx << " and currIdx = " << currIdx << std::endl;
	Block* blockToBeRemoved = freeList_[currIdx];
        //std::cout << "Block to split has size: " << blockToBeRemoved->GetSize() << std::endl;
	unsigned long blockSize = GetListBlockSize(currIdx - 1);
        //IMPORTANT: size_t blockSize = size;
	//std::cout << "The list to be inserted in has block size: " << blockSize << std::endl;
        std::cout << "Blocks supposed to be inserted at list index = " << currIdx - 1 << std::endl;
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
    std::cout << "Generating requested block" << std::endl;
    size_t size = blockToBeAllocated->GetSize();
    allocated_ += size;
    free_ -= size;
    assert(memPool_.find(blockToBeAllocated->GetData()) == memPool_.end());
    memPool_[blockToBeAllocated->GetData()] = blockToBeAllocated;
    std::cout << "SUCCESS: Buddy System No." << gpuIdx_ << " list index = " << listIdx << " block size = " << size << 
                 " at address = " << (void*)blockToBeAllocated->GetData() << std::endl;
    return (void*)(blockToBeAllocated->GetData());
  } else {
    std::cout << "FAILURE: Buddy System No." << gpuIdx_ << " cannot allocate size = " << size << " bytes" << std::endl;
    return NULL;
  }    
}

cudaError_t BuddySystem::Free(void* ptr) {
  std::cout << "Buddy System No." << gpuIdx_ << " trying to free pointer: " << ptr <<std::endl;
  std::map<char*, Block*>::iterator itr = memPool_.find((char*)ptr);
  if (itr == memPool_.end()) {
    std::cout << "FAILURE: Buddy System No." << gpuIdx_ << ": Can't free pointer at " << ptr << std::endl;
    return cudaErrorInvalidValue;
  }
  Block* blockToBeInserted = itr->second;
  memPool_.erase(itr);
  allocated_ -= blockToBeInserted->GetSize();
  free_ += blockToBeInserted->GetSize();
  int idx = GetListIdx(blockToBeInserted->GetSize());
  std::cout << "Block suppposed to be inserted at index = " << idx << std::endl;
  InsertBlock(blockToBeInserted);
  Merge(blockToBeInserted);
  std::cout << "SUCCESS: Free completed: " << ptr << std::endl;
  std::cout << "Total free memory after Free: size = " << free_ << " bytes" << std::endl;
  std::cout << "Total allocated memory after Free: size = " << allocated_ << " bytes" << std::endl;
  return cudaSuccess;
}

void BuddySystem::InsertBlock(Block* block) {
  //std::cout << "Block to insert has size: " << block->GetSize() << std::endl;
  int idx = GetListIdx(block->GetSize()); 
  std::cout << "Block actually inserted at list index = " << idx << std::endl;
  //std::cout << "Block should be inserted at list index: " << idx << std::endl;
  if (freeList_[idx] == NULL) {
    //std::cout << "Block inserted at head of list index: " << idx << std::endl;
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
  //std::cout << "Block inserted in the middle of list index: " << idx << std::endl;
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
      std::cout << "Merged with the next block" << std::endl;
    }
  }
  if (prev != NULL) {
    if ((prev->GetData() + listBlockSize) == curr->GetData()) {
      prev->SetSize(prev->GetSize() + curr->GetSize());
      prev->SetNext(curr->GetNext());
      curr->SetNext(NULL);
      InsertBlock(prev);
      std::cout << "Merged with the previous block" << std::endl;
      return;
    }
  }
  InsertBlock(curr);
}

void RemoveDuplicateBlockPtr() {
  return;
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
  std::cout << "Initializing Memory Manager" << std::endl;
  int deviceNum;
  CUDA_CALL(cudaGetDeviceCount(&deviceNum));
  std::cout << "device num = " << deviceNum << std::endl;
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
      std::cout << "Buddy System No." << deviceIdx << " initialized with size = " << avail << " bytes"  << std::endl;
    } else {
      std::cout << "Warning: There's no memory left on device: " << deviceIdx << std::endl;
    }
  }
  std::cout << "Memory Manager initialization completed" << std::endl; 
}

MemoryManager::~MemoryManager() {
  std::cout << "Destructing Memory Manager" << std::endl;
  typedef std::map<char*, Block*> MemoryPool;
  for (int deviceIdx = 0; deviceIdx < deviceCount_; deviceIdx++) {
    CUDA_CALL(cudaSetDevice(deviceIdx));
    BuddySystem* buddy = buddy_[deviceIdx];
    MemoryPool mp = buddy->GetMemPool();
    while (!mp.empty()) {
      buddy->Free((void*)(mp.begin()->first));
    }
    cudaFree((void*)buddy->GetStart());    
    std::cout << "Buddy System No." << buddy->GetGPUIdx() << " destructed" << std::endl;
  }
  std::cout << "Memory Manager destruction completed" << std::endl;
}

cudaError_t MemoryManager::Malloc(void*& devptr, size_t size, int deviceIdx) {
  std::cout << "Malloc size = " << size << " bytes on Buddy System No. " << deviceIdx << std::endl;
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
  std::cout << "Buddy System No." << deviceIdx << " has free = " << buddy_[deviceIdx]->GetFree() <<
	       " and allocate = " << buddy_[deviceIdx]->GetAllocated() << std::endl;
  std::cout << "Buddy System No." << deviceIdx << ": Trying to allocate size = " << size << std::endl;
  CUDA_CALL(cudaSetDevice(deviceIdx));
  BuddySystem* buddy = buddy_[deviceIdx];
  Block** freeList = buddy->GetFreeList();
  int freeListSize = buddy->GetFreeListSize();
  int idx = GetListIdx(size);
  if (idx == 0) idx = 1;

  for (int i = idx; i < freeListSize; i++) {
    if (freeList[i] != NULL) {
      std::cout << "SUCCESS: There is enough space" << std::endl;
      return true;
    }
  }
  
  std::cout << "FAILURE: There isn't enough space" << std::endl;
  return false;
}
} //namespace mxnet
