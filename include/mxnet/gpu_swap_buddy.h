#ifndef MXNET_BUDDY_H_
#define MXNET_BUDDY_H_

#include <cuda_runtime_api.h>
#include <map>

namespace mxnet {

class Block {
  public:
    Block(char* data, size_t size)
      : data_(data), size_(size), next_block_(NULL) {};
    char* Data() { return data_; }
    size_t Size() { return size_; }
    Block* Next() { return next_block_; }
    void SetSize(size_t size) { size_ = size; }
    void SetNext(Block* b) { next_block_ = b; }

  private:
    char* data_;
    std::size_t size_;
    Block* next_block_;
}; // Class Block

typedef std::map<char*, Block*> MemoryPool;

static inline size_t Log2(size_t x) {
  size_t y;
  asm ( "\tbsr %1, %0\n"
      : "=r"(y)
      : "r" (x)
  );
  return y;
}

class BuddySystem {
  public:
    static const size_t MIN_ALLOC_SIZE = 128;
    // TODO(fegin): This fixed value is not acceptable.
    static const size_t CLEAN_UP_BOUNDRY = 500000000;

    static inline size_t GetListIdx(size_t size) {
      size_t size_log = Log2(size);
      size_log += (size_log ^ size) ? 1 : 0;
      return size_log - Log2(MIN_ALLOC_SIZE);
    }
    static inline size_t GetListSize(size_t size) {
      return GetListIdx(size) + 1;
    }
    static inline size_t GetListBlockSize(int idx) {
      return 2 << (idx + Log2(MIN_ALLOC_SIZE));
    }

    BuddySystem(Block* start, size_t total, size_t device_id);
    ~BuddySystem();
    Block* GetStart() { return start_; }
    size_t GetTotal() { return total_; }
    size_t GetFree() { return free_; }
    size_t GetAllocated() { return allocated_; }
    int GetFreeListSize() { return free_list_size_; }
    Block** GetFreeList() { return free_list_; }
    MemoryPool GetMemPool() { return mem_pool_; }
    void* Alloc(size_t size);
    cudaError_t Free(void* ptr);
    void CleanUp();

  private:
    void InsertBlock(Block* block);
    Block* Merge(Block* block, int idx);
    void MergeFreeList();
    void PrintFreeList();
    void CheckDuplicate();
    void PrintMemPool();

    size_t device_id_;
    Block* start_;
    Block** free_list_;
    size_t total_;
    size_t allocated_;
    size_t free_;
    int free_list_size_;
    MemoryPool mem_pool_;
}; //Class BuddySystem

} //namespace mxnet

#endif // MXNET_BUDDY_H_

