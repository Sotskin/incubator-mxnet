#ifndef MXNET_MEM_MGR_H_
#define MXNET_MEM_MGR_H_

#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <math.h>
#include <stdio.h>
#include <string>

namespace mxnet {

const int MINALLOCSIZE_ = 128;
const double GPUUTILRATIO = 0.95; //save some memory for each device(subject to change)

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

inline std::string memGetStatusString(memStatus_t status) {
  switch (status) {
    case memStatus_Sucess: return "Sucess";
    case memStatus_InvalidValue: return "Invalid value";
    case memStatus_OutOfMemory: return "Out of memory";
    case memStatus_CUDAError: return "CUDA error";
  }
  return "Unknown error";
}

inline int nextPowerOfTwo(int size) {
  int result = pow(2, ceil(log(size) / log(2)));
  return result;
}

class Block {
  private: 
    char* data_;
    std::size_t size_;
    //Block* nextBlock_;
    blockStatus_t status_;

  public:
    Block(char* data, size_t size)
      : data_(data),
        size_(size),
        //nextBlock_(NULL),
        status_(blockStatus_Uninitialized) {
    }

    char* getData() { return data_; }
    size_t getSize() { return size_; }
    //Block* getNext() {return nextBlock_; }

    //void setNext(Block* b) { nextBlock_ = b; }
    void setAllocated() { status_ = blockStatus_Allocated; }
    void setFree() { status_ = blockStatus_Free; }
}; // Class Block

class BuddySystem {
  private:
    Block* start_;
    size_t total_;
    size_t allocated_;
    size_t free_;
    size_t maxBlock_;
    int gpuIdx_;
  
  public:
    BuddySystem(Block* start, size_t total, int gpuIdx)
      : start_(start), 
        total_(total),
        gpuIdx_(gpuIdx) {
    }
    ~BuddySystem();  

  public:
    void* Alloc(size_t size);
    cudaError_t Free(void* ptr); 
    
}; //Class BuddySystem

class MemoryManager {
  BuddySystem** buddy_;
  std::mutex mutex_;
  public:
    static MemoryManager* Get();
    static std::shared_ptr<MemoryManager> _GetSharedRef();
    cudaError_t Malloc(void*& devptr, size_t size, int deviceIdx);
    cudaError_t Free(void* devptr, int deviceIdx);
    cudaError_t Memcpy(int deviceId, void* dst, 
                       const void* src, size_t count, enum cudaMemcpyKind kind);
    cudaError_t MemGetInfo(int deviceId, size_t* total, size_t* free);   
    bool TryAllocate(int deviceId, size_t size);

  private:
    MemoryManager();
};  // Class MemoryManager
} //namespace mxnet

#endif // MXNET_MEM_MGR_H_
