#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <set>

#include <dmlc/logging.h>
#include "./gpu_swap_buddy.h"
#include "./gpu_swap_util.h"

#define BUDDY_DEBUG 0

namespace mxnet {

bool operator< (const Block& lhs, const Block& rhs) {
  return (char*)lhs.data_ < (char*)rhs.data_;
}

BuddySystem::BuddySystem(void* memory, size_t size, size_t device_id)
  : device_id_(device_id), total_size_(size), allocated_size_(0),
    available_size_(size) {
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
std::set<Block>::iterator BuddySystem::InsertBlock(const Block& block) {
  int idx = BlockListIdx(block.Size());
  auto ret = free_list_[idx].insert(block);
  return ret.first;
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
      size_t split_block_size = ListBlockSize(curr_idx - 1);
      InsertBlock(Block(victim_it->Data(), split_block_size));
      InsertBlock(Block((char*)victim_it->Data() + split_block_size,
                        victim_it->Size() - split_block_size));
      free_list_[curr_idx].erase(victim_it);
      curr_idx--;
    }
    size_t block_size = ListBlockSize(list_idx);
    auto allocated_it = free_list_[list_idx].begin();
    if (allocated_it->Size() > block_size) {
      InsertBlock(Block((char*)allocated_it->Data() + block_size,
                        allocated_it->Size() - block_size));
    }
    allocated_size_ += block_size;
    total_allocated_size_ += block_size;
    available_size_ -= block_size;
    CHECK(mem_pool_.find(allocated_it->Data()) == mem_pool_.end());
    mem_pool_[allocated_it->Data()] = Block(allocated_it->Data(), block_size);
    void* ret = allocated_it->Data();
    free_list_[list_idx].erase(allocated_it);
    return ret;
  } else {
    return nullptr;
  }
}

cudaError_t BuddySystem::Free(void* ptr) {
  auto iter = mem_pool_.find(ptr);
  if (iter == mem_pool_.end()) {
    CHECK(iter != mem_pool_.end());
    return cudaErrorInvalidValue;
  }
  allocated_size_ -= iter->second.Size();
  available_size_ += iter->second.Size();
  MergeBlock(InsertBlock(iter->second), BlockListIdx(iter->second.Size()));
  mem_pool_.erase(iter);
#if BUDDY_DEBUG
  CheckSize();
#endif
  return cudaSuccess;
}

void BuddySystem::Statistics() {
  CheckSize();
  std::cout << "=> BuddySystem:" << std::endl
            << "=> Total allocated size: " << GBString(total_allocated_size_)
            << std::endl
            << "=> Merge count: " << merge_count_ << std::endl;
  PrintFreeList();
  merge_count_ = 0;
  total_allocated_size_ = 0;
}

void BuddySystem::MergeBlock(std::set<Block>::iterator iter, int idx) {
  size_t block_size = ListBlockSize(idx);
  while (idx < free_list_size_ - 1) {
    if (iter->IsLeftBlock(memory_, block_size)) {
      auto next_iter = std::next(iter, 1);
      if (next_iter != free_list_[idx].end() &&
          (char*)iter->Data() + iter->Size() == (char*)next_iter->Data()) {
        // A trick to workaround constness of std::set elements.
        const Block &block = *iter;
        (const_cast<Block&>(block)).SetSize(iter->Size() + next_iter->Size());
        auto new_iter = InsertBlock(block);
        free_list_[idx].erase(iter);
        free_list_[idx].erase(next_iter);
        iter = new_iter;
      } else {
        break;
      }
    } else {
      auto prev_iter = std::prev(iter, 1);
      if (iter != free_list_[idx].begin() &&
          (char*)prev_iter->Data() + prev_iter->Size() == (char*)iter->Data()) {
        // A trick to workaround constness of std::set elements.
        const Block &block = *prev_iter;
        (const_cast<Block&>(block)).SetSize(prev_iter->Size() + iter->Size());
        auto new_iter = InsertBlock(block);
        free_list_[idx].erase(prev_iter);
        free_list_[idx].erase(iter);
        iter = new_iter;
      } else {
        break;
      }
    }
    merge_count_++;
    idx += 1;
    block_size <<= 1;
  }
}

void BuddySystem::CheckSize() {
  size_t size = 0;
  for (auto& free_list : free_list_) {
    for (auto& block : free_list) {
      size += block.Size();
    }
  }
  CHECK(size == available_size_) << "Size = " << size << " available_size = "
                                 << available_size_;
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
  std::cout << "=> Available size = " << GBString(available_size_) << std::endl;
  for (int i = 0; i < free_list_size_; i++ ) {
    if (free_list_[i].size() > 0) {
      std::cout << "=> Free List Index = " << i
                << ", size = " << free_list_[i].size() << std::endl;
    }
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
