#ifndef MXNET_BUDDY_H_
#define MXNET_BUDDY_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>

#ifdef __GNUC__
#define mem_likely(x)       __builtin_expect(!!(x), 1)
#define mem_unlikely(x)     __builtin_expect(!!(x), 0)
#else
#define mem_likely(x)       (x)
#define mem_unlikely(x)     (x)
#endif

namespace mxnet {

class Block {
  public:
    Block() : data_(nullptr), size_(0) {};
    Block(void* data, size_t size) : data_(data), size_(size) {};
    void* Data() const { return data_; }
    size_t Size() const { return size_; }
    void SetSize(size_t size) { size_ = size; }
    inline bool IsLeftBlock(const void* base, size_t block_size) const {
      return (((size_t)data_ - (size_t)base) & block_size) == 0;
    }

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

constexpr size_t Log2Const(size_t x) {
  return (x <= 1) ? 0 : 1 + Log2Const(x >> 1);
}

class BuddySystem {
  public:
    BuddySystem(void* memory, size_t size, size_t device_id);
    ~BuddySystem();
    void* Memory() { return memory_; }
    void MemoryUsage(size_t* total, size_t* free) {
      *total = total_size_;
      *free = available_size_;
    }
    bool TryAllocate(size_t size);
    void* Malloc(size_t size);
    cudaError_t Free(void* ptr);
    void Statistics();

  private:
    static const size_t kMinAllocateSize = 1;
    static constexpr size_t kLogBase = Log2Const(kMinAllocateSize);
    static inline size_t AllocListIdx(size_t size) {
      if (mem_unlikely(size <= kMinAllocateSize)) {
        return 0;
      } else {
        size_t size_log = Log2(size);
        size_log += (size_log ^ size) ? 1 : 0;
        return size_log - kLogBase;
      }
    }
    static inline size_t BlockListIdx(size_t size) {
      if (mem_unlikely(size <= kMinAllocateSize)) {
        return 0;
      } else {
        return Log2(size) - kLogBase;
      }
    }
    static inline size_t ListSize(size_t size) {
      return BlockListIdx(size) + 1;
    }
    static inline size_t ListBlockSize(int idx) {
      return 1L << (idx + kLogBase);
    }

    std::set<Block>::iterator InsertBlock(const Block& block);
    void MergeBlock(std::set<Block>::iterator iter, int idx);
    void PrintFreeList();
    void PrintMemPool();
    void CheckDuplicate();
    void CheckSize();

    size_t device_id_;
    void* memory_;
    std::unordered_map<void*, Block> mem_pool_;
    std::vector<std::set<Block>> free_list_;
    int free_list_size_;
    size_t total_size_;
    size_t allocated_size_;
    size_t available_size_;
    size_t merge_count_;
}; //Class BuddySystem

} //namespace mxnet

#endif // MXNET_BUDDY_H_

