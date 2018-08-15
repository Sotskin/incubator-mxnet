#ifndef MXNET_BUDDY_H_
#define MXNET_BUDDY_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <map>
#include <set>

namespace mxnet {

class Block {
  public:
    Block() : data_(nullptr), size_(0) {};
    Block(void* data, size_t size) : data_(data), size_(size) {};
    void* Data() const { return data_; }
    size_t Size() const { return size_; }
    void SetSize(size_t size) { size_ = size; }

    friend bool operator< (const Block& lhs, const Block& rhs);

  private:
    void* data_;
    std::size_t size_;
}; // Class Block

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
    static const size_t kMinAllocateSize = 128;
    // TODO(fegin): This fixed value is not acceptable.
    static const size_t kCleanUpBoundary = 500000000;

    static inline size_t ListIdx(size_t size, bool carry=false) {
      if (size <= kMinAllocateSize) {
        return 0;
      } else {
        size_t size_log = Log2(size);
        size_log += (size_log ^ size && carry) ? 1 : 0;
        return size_log - Log2(kMinAllocateSize);
      }
    }
    static inline size_t ListSize(size_t size) {
      return ListIdx(size) + 1;
    }
    static inline size_t ListBlockSize(int idx) {
      return 1L << (idx + Log2(kMinAllocateSize));
    }

    BuddySystem(void* memory, size_t size, size_t device_id);
    ~BuddySystem();
    void* Memory() { return memory_; }
    void MemoryUsage(size_t* total, size_t* free) {
      *total = total_size_;
      *free = free_size_;
    }
    int FreeListSize() { return free_list_size_; }
    bool TryAllocate(size_t size);
    void* Malloc(size_t size);
    cudaError_t Free(void* ptr);

  private:
    void InsertBlock(const Block& block);
    void MergeBlock(std::set<Block>* free_list, size_t idx, bool reinsert);
    void MergeFreeList();
    void CleanUp();
    void PrintFreeList();
    void CheckDuplicate();
    void PrintMemPool();

    size_t device_id_;
    void* memory_;
    std::vector<std::set<Block>> free_list_;
    size_t total_size_;
    size_t allocated_size_;
    size_t free_size_;
    int free_list_size_;
    std::map<void*, Block> mem_pool_;
}; //Class BuddySystem

} //namespace mxnet

#endif // MXNET_BUDDY_H_

