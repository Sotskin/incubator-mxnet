#ifndef MXNET_BUDDY_H_
#define MXNET_BUDDY_H_

#include <cuda_runtime_api.h>
#include <iostream>
#include <map>
#include <math.h>
#include <set>
#include <stdio.h>
#include <string>

namespace mxnet {

typedef enum {
  blockStatus_Uninitialized,
  blockStatus_Free,
  blockStatus_Allocated
} blockStatus_t;
  
const int MIN_ALLOC_SIZE = 128;
// TODO(fegin): This fixed value is not acceptable.
const size_t CLEAN_UP_BOUNDRY = 500000000;

// TODO(fegin): Since everything is integer, we can use << and >> to replace pow and log2.
inline int GetListIdx(size_t size) {
  if (size <= 128) return 0;
  return ceil(log2(static_cast<double>(size)) - log2(static_cast<double>(MIN_ALLOC_SIZE)));
}

inline unsigned long GetListSize(size_t size) {
  return GetListIdx(size) + 1;
}

inline size_t GetListBlockSize(int idx) {
  return pow(2, idx + log2(static_cast<double>(MIN_ALLOC_SIZE)));
}

class Block {
  private:
    char* data_;
    std::size_t size_;
    Block* nextBlock_;
    blockStatus_t status_;

  public:
    Block(char* data, size_t size)
      : data_(data),
        size_(size),
        nextBlock_(NULL),
        status_(blockStatus_Uninitialized) {
    }

    char* GetData() { return data_; }
    size_t GetSize() { return size_; }
    Block* GetNext() { return nextBlock_; }

    void SetSize(size_t size) { size_ = size; }
    void SetNext(Block* b) { nextBlock_ = b; }
    void SetAllocated() { status_ = blockStatus_Allocated; }
    void SetFree() { status_ = blockStatus_Free; }
}; // Class Block

class BuddySystem {
  private:
    Block* start_;
    Block** freeList_;
    size_t total_;
    size_t allocated_;
    size_t free_;
    int freeListSize_;
    int gpuIdx_;
    typedef std::map<char*, Block*> MemoryPool;
    MemoryPool memPool_;
    void InsertBlock(Block* block);
    Block* Merge(Block* block, int idx);
    void MergeFreeList();
    void PrintFreeList();
    void CheckDuplicate();
    void PrintMemPool();

  public:
    BuddySystem(Block* start, size_t total, int gpuIdx);
    ~BuddySystem();
    Block* GetStart() { return start_; }
    size_t GetTotal() { return total_; }
    size_t GetFree() { return free_; }
    size_t GetAllocated() { return allocated_; }
    int GetFreeListSize() { return freeListSize_; }
    Block** GetFreeList() { return freeList_; }
    int GetGPUIdx() { return gpuIdx_; }
    MemoryPool GetMemPool() { return memPool_; }
    void* Alloc(size_t size);
    cudaError_t Free(void* ptr);
    void CleanUp();
}; //Class BuddySystem

} //namespace mxnet

#endif // MXNET_BUDDY_H_

