#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <set>

#include <dmlc/logging.h>
#include <mxnet/gpu_swap_buddy.h>

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
  PrintFreeList();
}

BuddySystem::~BuddySystem() {}

bool BuddySystem::TryAllocate(size_t size) {
  size_t idx = ListIdx(size, true);
  // FIXME(fegin): This is in the original implementation. Is this necessary?
  //if (idx == 0) {
    //idx = 1;
  //}
  if (allocated_size_ < kCleanUpBoundary) {
    CleanUp();
  }
  for (int i = idx; i < free_list_size_; i++) {
    if (free_list_[i].size() != 0) {
      return true;
    }
  }
  return false;
}

void BuddySystem::InsertBlock(const Block& block) {
  int idx = ListIdx(block.Size());
  free_list_[idx].insert(block);
}

void* BuddySystem::Malloc(size_t size) {
  int list_idx = ListIdx(size, true);
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
    Block allocated_block = *(free_list_[list_idx].begin());
    free_list_[list_idx].erase(free_list_[list_idx].begin());
    allocated_size_ += allocated_block.Size();
    free_size_ -= allocated_block.Size();
    CHECK(mem_pool_.find(allocated_block.Data()) == mem_pool_.end());
    mem_pool_[allocated_block.Data()] = allocated_block;
    return allocated_block.Data();
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
  free_size_ += iter->second.Size();
  InsertBlock(iter->second);
  mem_pool_.erase(iter);
  MergeFreeList();
  return cudaSuccess;
}

void BuddySystem::MergeBlock(std::set<Block>* free_list, size_t idx,
                             bool reinsert=true) {
  if (free_list->size() == 0) {
    return;
  }
  std::set<Block>::iterator iter = free_list->begin();
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
      if (reinsert && iter->Size() > ListBlockSize(idx)) {
        InsertBlock(*iter);
        free_list->erase(iter);
        iter = next_iter;
      }
      iter = next_iter;
    }
  }
}

void BuddySystem::MergeFreeList() {
  for (int i = 0; i < free_list_size_; i++) {
    MergeBlock(&(free_list_[i]), (size_t)i);
  }
}

void BuddySystem::CleanUp() {
  //std::cout << "Before the clean up: " << std::endl;
  PrintFreeList();
  //insert all nodes in the free list into a temp list
  std::set<Block> temp_list;
  for (int i = 0; i < free_list_size_; i++) {
    temp_list.insert(free_list_[i].begin(), free_list_[i].end());
    free_list_[i].clear();
  }
  //merge the nodes in the temp list
  MergeBlock(&temp_list, 0, false);
  //insert the nodes in the temp list back into the free list
  for (auto& block : temp_list) {
    InsertBlock(block);
  }
  //std::cout << "After the clean up: " << std::endl;
  PrintFreeList();
}

//currently disabled for better log
void BuddySystem::PrintFreeList() {
  return;
  std::cout << "=================================================" << std::endl
            << "Free List Info:" << std::endl
            << "=================================================" << std::endl;
  for (int i = 0; i < free_list_size_; i++ ) {
    std::cout << "Free List Index = " << i << std::endl;
    for (const auto& block : free_list_[i]) {
      std::cout << "Block addr = " << block.Data()
                << " size = " << block.Size() << std::endl;
    }
    std::cout << "-----------------------------------------------" << std::endl;
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

void BuddySystem::CheckDuplicate() {
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
}

} //namespace mxnet
