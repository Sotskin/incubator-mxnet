#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdio.h>

namespace mxnet {

const int FREELISTSIZE = 32;

typedef enum {
  memStatus_Sucess,
  memStatus_InvalidValue,
  memStatus_OutOfMemory,
  memStatus_CUDAError,
} memStatus_t;

inline char* memGetStatusString(memStatus_t status) {
  switch (status) {
    case memStatus_Sucess: return "Sucess";
    case memStatus_InvalidValue: return "Invalid value";
    case memStatus_OutOfMemory: return "Out of memory";
    case memStatus_CUDAError: return "CUDA error";
  }
}

inline int getFreeListIdx(size_t size) {
  int idx = 0;
  while ((idx < FREELISTSIZE - 1) && (size > 1)) {
    size >>= 1;
    idx++;
  }  
  return idx;
}

class Block {
  private: 
    char* data_;
    std::size_t size_;
    Block* nextBlock_;
    bool isFree_;
    bool isHead_;

  public:
    Block(char* data, size_t size)
      : data_(data)
      , size_(size)
      , nextBlock_(NULL)
      , isFree_(true)
      , isHead_(false) {
    }

    char* getData() { return data_; }
    size_t getSize() { return size_; }
    Block* getNext() {return nextBlock_; }
    bool isHead() { return isHead_; }

    void setHead() { isHead_ = true; }
    void setNext(Block* b) { nextBlock_ = b; }
};

class MemoryManager {
  Block* freeList_[FREELISTSIZE];
  std::mutex mutex_;
  public:
    static MemoryManager* Get();
    static std::shared_ptr<MemoryManager> _GetSharedRef();
    ~MemoryManager();
    cudaError_t Malloc(void** devptr, size_t size, int deviceIdx);
    cudaError_t Free(void* devptr, int deviceIdx);
    cudaError_t Memcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);

  private:
    MemoryManager();
};

} //namespace mxnet

