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
  //MergeFreeList();
  //size_t idx = BlockListIdx(iter->second.Size());
  //MergeBlock(&(free_list_[idx]), idx, false);
  MergeFreeList(BlockListIdx(iter->second.Size()));
  mem_pool_.erase(iter);
  CheckSize();
  //PrintFreeList();
  return cudaSuccess;
}

#if 0
void BuddySystem::MergeBlock(std::set<Block>* free_list, size_t idx,
                             bool reinsert=true) {
  if (free_list->size() <= 1) {
    return;
  }
  std::set<Block>::iterator iter = free_list->begin();
#if 1
  size_t block_size = ListBlockSize(idx);
  while (iter != free_list->end()) {
    const Block &block = *iter;
    if (iter->IsLeftBlock(memory_, block_size)) {
      std::set<Block>::iterator next_iter = std::next(iter, 1);
      if (next_iter != free_list->end() &&
          (char*)iter->Data() + iter->Size() == (char*)next_iter->Data()) {
        // A trick to workaround constness of std::set elements.
        (const_cast<Block&>(block)).SetSize(iter->Size() + next_iter->Size());
        InsertBlock(block);
        free_list->erase(iter);
        iter = free_list->erase(next_iter);
        continue;
      }
    }
    iter++;
  }
#else
  size_t block_size = ListBlockSize(idx + 1);
  while (iter != free_list->end()) {
    std::set<Block>::iterator next_iter = std::next(iter, 1);
    if (next_iter != free_list->end() &&
        (char*)iter->Data() + iter->Size() == (char*)next_iter->Data()) {
      size_t size = iter->Size() + next_iter->Size();
      // A trick to workaround constness of std::set elements.
      const Block &block = *iter;
      (const_cast<Block&>(block)).SetSize(size);
      free_list->erase(next_iter);
    } else {
      if (reinsert && iter->Size() >= block_size) {
        InsertBlock(*iter);
        free_list->erase(iter);
      }
      iter = next_iter;
    }
  }
#endif
}
#endif
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

#if 0
void BuddySystem::CleanUp() {
  //insert all nodes in the free list into a temp list
  std::set<Block> temp_list;
  for (int i = 0; i < free_list_size_ - 1; i++) {
    temp_list.insert(free_list_[i].begin(), free_list_[i].end());
    free_list_[i].clear();
  }
  //merge the nodes in the temp list
  MergeBlock(&temp_list, 0, false);
  //insert the nodes in the temp list back into the free list
  for (auto& block : temp_list) {
    InsertBlock(block);
  }
  CheckSize();
}
#endif

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
