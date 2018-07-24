#ifndef MXNET_MEM_MGR_H_
#define MXNET_MEM_MGR_H_

#include <cuda_runtime_api.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <math.h>
#include <stdio.h>
#include <string>

namespace mxnet {

const int MIN_ALLOC_SIZE = 128;
const double GPU_UTIL_RATIO = 0.95; //save some memory for each device(subject to change)

typedef enum {
  memStatus_Sucess,
  memStatus_InvalidValue,
  memStatus_OutOfMemory,
  memStatus_CUDAError
} memStatus_t;

typedef enum {
  blockStatus_Uninitialized,
  blockStatus_Free,
  blockStatus_Allocated
} blockStatus_t;

inline std::string MemGetStatusString(memStatus_t status) {
  switch (status) {
    case memStatus_Sucess: return "Sucess";
    case memStatus_InvalidValue: return "Invalid value";
    case memStatus_OutOfMemory: return "Out of memory";
    case memStatus_CUDAError: return "CUDA error";
  }
  return "Unknown error";
}

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
    void Merge(Block* block);
    void PrintFreeList();
    void CheckDuplicate();
 
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
}; //Class BuddySystem

class MemoryManager {
  private:  
    BuddySystem** buddy_;
    std::mutex mutex_;
    int deviceCount_;
    MemoryManager();    

  public:
    ~MemoryManager();
    static MemoryManager* Get();
    static std::shared_ptr<MemoryManager> _GetSharedRef();
    cudaError_t Malloc(void*& devptr, size_t size, int deviceIdx);
    cudaError_t Free(void* devptr, int deviceIdx);
    cudaError_t Memcpy(int deviceIdx, void* dst, 
                       const void* src, size_t count, enum cudaMemcpyKind kind);
    cudaError_t MemGetInfo(int deviceId, size_t* total, size_t* free);   
    bool TryAllocate(int deviceId, size_t size);
};  // Class MemoryManager
} //namespace mxnet
#endif // MXNET_MEM_MGR_H_
