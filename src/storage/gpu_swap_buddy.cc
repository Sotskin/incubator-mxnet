#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <set>

#include <dmlc/logging.h>
#include <mxnet/gpu_swap_buddy.h>

#define BUDDY_DEBUG 0

namespace mxnet {

bool operator< (const Block& lhs, const Block& rhs) {
  return (char*)lhs.data_ < (char*)rhs.data_;
}

BuddySystem::BuddySystem(void* memory, size_t size, size_t device_id)
  : device_id_(device_id), total_size_(size), allocated_size_(0),
    free_size_(size) {
  free_list_size_ = ListSize(size);
  free_list_.resize(free_list_size_);
  memory_ = memory;
  free_list_[free_list_size_ - 1].insert(Block(memory, size));
}

BuddySystem::~BuddySystem() {}

bool BuddySystem::TryAllocate(size_t size) {
  for (int i = AllocListIdx(size); i < free_list_size_; i++) {
    if (free_list_[i].size() != 0) {
      return true;
    }
  }
  return false;
}

void BuddySystem::InsertBlock(const Block& block) {
  int idx = BlockListIdx(block.Size());
  free_list_[idx].insert(block);
}

void* BuddySystem::Malloc(size_t size) {
  int list_idx = AllocListIdx(size);
  int curr_idx = list_idx;

  while (curr_idx < free_list_size_ && free_list_[curr_idx].size() == 0) {
    curr_idx++;
  }
  if (curr_idx < free_list_size_) {
    while (curr_idx > list_idx) {
      auto victim_it = free_list_[curr_idx].begin();
      size_t block_size = ListBlockSize(curr_idx - 1);
      InsertBlock(Block(victim_it->Data(), block_size));
      InsertBlock(Block((char*)victim_it->Data() + block_size,
                        victim_it->Size() - block_size));
      free_list_[curr_idx].erase(victim_it);
      curr_idx--;
    }
    size_t block_size = ListBlockSize(list_idx);
    Block allocated_block = *(free_list_[list_idx].begin());
    free_list_[list_idx].erase(free_list_[list_idx].begin());
    if (allocated_block.Size() > block_size) {
      InsertBlock(Block((char*)allocated_block.Data() + block_size,
                        allocated_block.Size() - block_size));
      allocated_block.SetSize(block_size);
    }
    allocated_size_ += block_size;
    free_size_ -= block_size;
    CHECK(mem_pool_.find(allocated_block.Data()) == mem_pool_.end());
    mem_pool_[allocated_block.Data()] = allocated_block;
    return allocated_block.Data();
  } else {
    return nullptr;
  }
}

cudaError_t BuddySystem::Free(void* ptr) {
  static int count = 0;
  auto iter = mem_pool_.find(ptr);
  if (iter == mem_pool_.end()) {
    CHECK(iter != mem_pool_.end());
    return cudaErrorInvalidValue;
  }
  count += 1;
  allocated_size_ -= iter->second.Size();
  free_size_ += iter->second.Size();
  InsertBlock(iter->second);
  MergeFreeList(BlockListIdx(iter->second.Size()));
  mem_pool_.erase(iter);
  CheckSize();
  //PrintFreeList();
  return cudaSuccess;
}

bool BuddySystem::MergeBlock(std::set<Block>* free_list, size_t idx) {
  if (free_list->size() <= 1) {
    return false;
  }
  std::set<Block>::iterator iter = free_list->begin();
  size_t block_size = ListBlockSize(idx);
  for (auto iter = free_list->begin(); iter != free_list->end(); iter++) {
    if (iter->IsLeftBlock(memory_, block_size)) {
      std::set<Block>::iterator next_iter = std::next(iter, 1);
      if (next_iter != free_list->end() &&
          (char*)iter->Data() + iter->Size() == (char*)next_iter->Data()) {
        // A trick to workaround constness of std::set elements.
        const Block &block = *iter;
        (const_cast<Block&>(block)).SetSize(iter->Size() + next_iter->Size());
        InsertBlock(block);
        free_list->erase(iter);
        free_list->erase(next_iter);
        return true;
      }
    }
  }
  return false;
}

void BuddySystem::MergeFreeList(size_t idx) {
  // We can't merge blocks, if any, inthe last free_list.
  for (size_t i = idx; i < (size_t)free_list_size_ - 1; i++) {
    if (!MergeBlock(&(free_list_[i]), (size_t)i)) {
      break;
    }
  }
}

void BuddySystem::CheckSize() {
#if BUDDY_DEBUG
  size_t size = 0;
  for (auto& free_list : free_list_) {
    for (auto& block : free_list) {
      size += block.Size();
    }
  }
  CHECK(size == free_size_) << "Size = " << size << " free_size_ = "
                            << free_size_;
#endif
}

void BuddySystem::CheckDuplicate() {
#if BUDDY_DEBUG
  std::set<void*> addr;
  bool abort = false;
  for (int i = 0; i < free_list_size_; i++) {
    for (const auto& block : free_list_[i]) {
      if (addr.find(block.Data()) != addr.end()) {
        std::cout << "This block with address = " << block.Data()
                  << " appeared more than once." << std::endl;
        abort = true;
      } else {
        addr.insert(block.Data());
      }
    }
  }
  CHECK(!abort);
#endif
}

void BuddySystem::PrintFreeList() {
  std::cout << "=================================================" << std::endl
            << "Free List Info:" << std::endl
            << "=================================================" << std::endl;
  std::cout << "Allocated size = " << allocated_size_ << std::endl;
  for (int i = 0; i < free_list_size_; i++ ) {
    std::cout << "Free List Index = " << i
              << ", size = " << free_list_[i].size() << std::endl;
  }
}

void BuddySystem::PrintMemPool() {
  if (mem_pool_.empty()) {
    std::cout << "Memory pool is empty" << std::endl;
    return;
  }
  std::cout << "=================================================" << std::endl
            << "Memory Pool Info:" << std::endl
            << "=================================================" << std::endl;
  for (const auto& block : mem_pool_) {
    std::cout << "Block addr = " << block.first
              << " size = " << block.second.Size() << std::endl;
  }
}

} //namespace mxnet
