#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <set>

#include <dmlc/logging.h>
#include <mxnet/gpu_swap_buddy.h>

namespace mxnet {

BuddySystem::BuddySystem(void* memory, size_t size, size_t device_id)
  : device_id_(device_id), total_size_(size), allocated_size_(0),
    free_size_(size) {
  free_list_size_ = ListSize(size);
  free_list_.resize(free_list_size_);
  head_block_ = new Block(memory, size);
  for (int i = 0; i < free_list_size_; i++) {
    free_list_[i] = NULL;
  }
  if (free_list_size_ > 0) {
    free_list_[free_list_size_ - 1] = head_block_;
  }
  PrintFreeList();
}

BuddySystem::~BuddySystem() {
  while (!mem_pool_.empty()) {
    Free((void*)mem_pool_.begin()->first);
  }
}

bool BuddySystem::TryAllocate(size_t size) {
  size_t free_list_size = FreeListSize();
  size_t idx = ListIdx(size);
  // FIXME(fegin): ????
  if (idx == 0) {
    idx = 1;
  }

  // FIXME(fegin): Can we combine this to the next for loop?
  for (size_t i = idx; i < free_list_size; i++) {
    if (free_list_[i] != nullptr) {
      return true;
    }
  }

  // FIXME(fegin): ???
  if (allocated_ < kCleanUpBoundary) {
    std::cout << "Starting clean up process" << std::endl;
    CleanUp();
  }

  for (size_t i = idx; i < free_list_size; i++) {
    if (free_list_[i] != nullptr) {
      return true;
    }
  }
  return false;
}

void* BuddySystem::Malloc(size_t size) {
  int list_idx = ListIdx(size);
  int curr_idx = list_idx;
  bool found = false;
  Block* allocated_block;

  while(!found) {
    if (free_list_[list_idx] != NULL) {
      allocated_block = free_list_[list_idx];
      free_list_[list_idx] = allocated_block->Next();
      allocated_block->SetNext(NULL);
      found = true;
      std::cout << "Found block of size = " << size << " bytes" << std::endl;
    } else if (curr_idx < free_list_size_) {
      curr_idx++;
      if (free_list_[curr_idx] != NULL) {
        Block* victim_block = free_list_[curr_idx];
        unsigned long block_size = ListBlockSize(curr_idx - 1);
        std::cout << "Blocks supposed to be inserted at list index = "
                  << curr_idx - 1 << std::endl;
        InsertBlock(new Block(victim_block->Data(), (size_t)block_size));
        InsertBlock(new Block(victim_block->Data() + block_size,
                              victim_block->Size() - block_size));
        free_list_[curr_idx] = victim_block->Next();
        victim_block->SetNext(NULL);
        curr_idx = list_idx;
      }
    } else {
      break;
    }
  }

  if (found) {
    size_t size = allocated_block->Size();
    allocated_size_ += size;
    free_size_ -= size;
    assert(mem_pool_.find(allocated_block->Data()) == mem_pool_.end());
    mem_pool_[allocated_block->Data()] = allocated_block;
    PrintFreeList();
    return (void*)(allocated_block->Data());
  } else {
    PrintFreeList();
    return NULL;
  }
}

cudaError_t BuddySystem::Free(void* ptr) {
  std::map<void*, Block*>::iterator iter = mem_pool_.find(ptr);
  if (iter == mem_pool_.end()) {
    return cudaErrorInvalidValue;
  }
  Block* free_block = iter->second;
  mem_pool_.erase(iter);
  allocated_size_ -= free_block->Size();
  free_size_ += free_block->Size();
  InsertBlock(free_block);
  MergeFreeList();
  PrintMemPool();
  PrintFreeList();
  return cudaSuccess;
}

void BuddySystem::InsertBlock(Block* block) {
  int idx = ListIdx(block->Size());
  if (free_list_[idx] == NULL) {
    free_list_[idx] = block;
    return;
  }

  Block* curr;
  Block* prev;
  prev = NULL;
  curr = free_list_[idx];

  while (curr != NULL) {
    if (curr->Data() > block->Data()) {
      break;
    }
    prev = curr;
    curr = curr->Next();
  }

  if (prev != NULL) {
    prev->SetNext(block);
    block->SetNext(curr);
  } else {
    block->SetNext(free_list_[idx]);
    free_list_[idx] = block;
  }
}

Block* BuddySystem::Merge(Block* block, int idx) {
  Block* curr = free_list_[idx];
  Block* prev = NULL;
  Block* prev_prev = NULL;
  bool merged_with_prev  = false;
  bool merged_with_next = false;

  while (curr != block && curr != NULL) {
    prev_prev = prev;
    prev = curr;
    curr = curr->Next();
  }

  if (curr == NULL) {
    return NULL;
  }

  if (prev != NULL) {
    // If can merge with previous block, merge and remove curr block.
    if ((prev->Data() + ListBlockSize((size_t)idx)) == curr->Data()) {
      prev->SetSize(prev->Size() + curr->Size());
      prev->SetNext(curr->Next());
      curr->SetNext(NULL);
      merged_with_prev  = true;
    }
  }

  // If merge with previous block, check if it can be merged with next block.
  if (merged_with_prev ) {
    if (prev->Next() != NULL) {
      Block* next = prev->Next();
      if ((prev->Data() + prev->Size()) == next->Data()) {
        prev->SetSize(prev->Size() + next->Size());
        prev->SetNext(next->Next());
        next->SetNext(NULL);
        merged_with_next = true;
      }
    }
  } else {
    // If did not merge with previous block(i.e curr still exists),
    // check if it can be merged with next block.
    if (curr->Next() != NULL) {
      Block* next = curr->Next();
      if ((curr->Data() + ListBlockSize((size_t)idx)) == next->Data()) {
        curr->SetSize(curr->Size() + next->Size());
        curr->SetNext(next->Next());
        next->SetNext(NULL);
        merged_with_next = true;
      }
    }
  }

  if (merged_with_prev ) {
    if (prev_prev != NULL) {
      prev_prev->SetNext(prev->Next());
      prev->SetNext(NULL);
      InsertBlock(prev);
      return prev_prev->Next();
    } else {
      free_list_[idx] = prev->Next();
      prev->SetNext(NULL);
      InsertBlock(prev);
      return free_list_[idx];
    }
  } else if (merged_with_next) {
    if (prev != NULL) {
      prev->SetNext(curr->Next());
      curr->SetNext(NULL);
      InsertBlock(curr);
      return curr->Next();
    } else {
      free_list_[idx] = curr->Next();
      curr->SetNext(NULL);
      InsertBlock(curr);
      return free_list_[idx];
    }
  }
  return block->Next();
}

void BuddySystem::MergeFreeList() {
  for (int i = 0; i < free_list_size_; i++) {
    Block* curr = free_list_[i];
    while (curr != NULL) {
      curr = Merge(curr, i);
    }
  }
}

//currently disabled for better log
void BuddySystem::PrintFreeList() {
  std::cout << "=================================================" << std::endl;
  std::cout << "Free List Info" << std::endl;
  std::cout << "=================================================" << std::endl;
  for (int i = 0; i < free_list_size_; i++ ) {
    std::cout << "Free List Index = " << i << std::endl;
    Block* curr = free_list_[i];
    while (curr != NULL) {
      std::cout << "Block addr = " << (void*)curr->Data() << " size = "
                << curr->Size() << " ===>" << std::endl;
      curr = curr->Next();
    }
    std::cout << "-----------------------------------------------" << std::endl;
  }
}

void BuddySystem::CheckDuplicate() {
  std::set<void*> addr;
  bool abort = false;
  for (int i = 0; i < free_list_size_; i++) {
    Block * curr = free_list_[i];
    while (curr != NULL) {
      const bool exists = addr.find(curr->Data()) != addr.end();
      if (exists) {
        std::cout << "This block with address = " << (void*)curr->Data()
                  << " appeared more than once." << std::endl;
        abort = true;
      }
      curr = curr->Next();
    }
  }
  CHECK(!abort);
}

void BuddySystem::CleanUp() {
  std::cout << "Starting clean up" << std::endl;
  Block* temp_list = NULL;

  //insert all nodes in the free list into a temp list
  for (int i = 0; i < free_list_size_; i++) {
    Block* curr = free_list_[i];

    while (curr != NULL) {
      free_list_[i] = curr->Next();
      curr->SetNext(NULL);

      Block* temp = temp_list;
      Block* prev = NULL;

      while (temp != NULL) {
        if (temp->Data() > curr->Data()) {
          break;
        }
        prev = temp;
        temp = temp->Next();
      }

      if (prev != NULL) {
        prev->SetNext(curr);
        curr->SetNext(temp);
      } else {
        curr->SetNext(temp_list);
        temp_list = curr;
      }
      curr = free_list_[i];
    }
  }

  std::cout << "First remove all nodes in the existing free list" << std:: endl;
  PrintFreeList();
  std::cout << "Display all nodes" << std::endl;
  Block* print_ptr = temp_list;
  while (print_ptr != NULL) {
    std::cout << "Block addr = " << (void*)print_ptr->Data() << " size = "
              << print_ptr->Size() << " ==========>" << std::endl;
    print_ptr = print_ptr->Next();
  }

  //merge the nodes in the temp list
  Block* curr = temp_list;
  while (curr != NULL) {
    if (curr->Next() != NULL) {
      Block* next = curr->Next();
      if ((curr->Data() + curr->Size()) == next->Data()) {
        curr->SetSize(curr->Size() + next->Size());
        curr->SetNext(next->Next());
        next->SetNext(NULL);
      } else {
        curr = next;
      }
    } else {
      curr = curr->Next();
    }
  }

  std::cout << "After the merge we have nodes:" << std::endl;
  print_ptr = temp_list;
  while (print_ptr != NULL) {
    std::cout << "Block addr = " << (void*)print_ptr->Data() << " size = "
              << print_ptr->Size() << " ==========>" << std::endl;
    print_ptr = print_ptr->Next();
  }

  //insert the nodes in the temp list back into the free list
  curr = temp_list;
  while (curr != NULL) {
    Block* next = curr->Next();
    curr->SetNext(NULL);
    InsertBlock(curr);
    curr = next;
  }
  std::cout << "After the clean up the free list is: " << std::endl;
}

void BuddySystem::PrintMemPool() {
  if (mem_pool_.empty()) {
    std::cout << "Memory pool is empty" << std::endl;
    return;
  }

  std::cout << "=================================================" << std::endl;
  std::cout << "Printing Memory Pool:" << std::endl;
  std::map<void*, Block*>::const_iterator iter = mem_pool_.begin();
  while (iter != mem_pool_.end()) {
    std::cout << "Block addr = " << (void*)iter->first << " size = "
              << iter->second->Size() << std::endl;
    iter++;
  }
}

} //namespace mxnet
