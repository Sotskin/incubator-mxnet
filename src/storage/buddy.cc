#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <mxnet/buddy.h>

namespace mxnet {

BuddySystem::BuddySystem(Block* start, size_t total, int gpuIdx)
  : start_(start),
    total_(total),
    allocated_(0),
    free_(total),
    gpuIdx_(gpuIdx) {
  std::cout << "Initializing Buddy System No." << gpuIdx << std::endl;
  freeListSize_ = GetListSize(total);
  freeList_ = new Block*[freeListSize_];
  for (int i = 0; i < freeListSize_; i++) {
    freeList_[i] = NULL;
  }
  if (freeListSize_ > 0) freeList_[freeListSize_ - 1] = start;
  std::cout << "Buddy System No." << gpuIdx << " initialization finished." <<std::endl;
  PrintFreeList();
}

BuddySystem::~BuddySystem() {
  while (!memPool_.empty()) {
    std::cout << "Destructing block at addr: " << (void*)memPool_.begin()->first << std::endl;
    Free((void*)memPool_.begin()->first);
  }
}

void* BuddySystem::Alloc(size_t size) {
  std::cout << "Buddy System No." << gpuIdx_ << ": Allocating size = " << size << " bytes" << std::endl;
  int listIdx = GetListIdx(size);
  int currIdx = listIdx;
  bool found = false;
  Block* blockToBeAllocated;
  std::cout << "Block of required size are supposed to be allocated from list index = " << listIdx << std::endl;

  while(!found) {
    if (freeList_[listIdx] != NULL) {
      blockToBeAllocated = freeList_[listIdx];
      freeList_[listIdx] = blockToBeAllocated->GetNext();
      blockToBeAllocated->SetNext(NULL);
      found = true;
      std::cout << "Found block of size = " << size << " bytes" << std::endl;
    } else if (currIdx < freeListSize_) {
      currIdx++;
      if (freeList_[currIdx] != NULL) {
        //std::cout << "Spliting in progress: listIdx = " << listIdx << " and currIdx = " << currIdx << std::endl;
        Block* blockToBeRemoved = freeList_[currIdx];
        //std::cout << "Block to split has size: " << blockToBeRemoved->GetSize() << std::endl;
        unsigned long blockSize = GetListBlockSize(currIdx - 1);
        //IMPORTANT: size_t blockSize = size;
        //std::cout << "The list to be inserted in has block size: " << blockSize << std::endl;
        std::cout << "Blocks supposed to be inserted at list index = " << currIdx - 1 << std::endl;
        InsertBlock(new Block(blockToBeRemoved->GetData(), (size_t)blockSize));
        InsertBlock(new Block(blockToBeRemoved->GetData() + blockSize, blockToBeRemoved->GetSize() - blockSize));
        freeList_[currIdx] = blockToBeRemoved->GetNext();
        blockToBeRemoved->SetNext(NULL);
        currIdx = listIdx;
      }
    } else {
      break;
    }
  }

  if (found) {
    std::cout << "Generating requested block" << std::endl;
    size_t size = blockToBeAllocated->GetSize();
    allocated_ += size;
    free_ -= size;
    assert(memPool_.find(blockToBeAllocated->GetData()) == memPool_.end());
    memPool_[blockToBeAllocated->GetData()] = blockToBeAllocated;
    std::cout << "SUCCESS: Buddy System No." << gpuIdx_ << " list index = " << listIdx << " block size = " << size <<
                 " at address = " << (void*)blockToBeAllocated->GetData() << std::endl;
    PrintFreeList();
    return (void*)(blockToBeAllocated->GetData());
  } else {
    std::cout << "FAILURE: Buddy System No." << gpuIdx_ << " cannot allocate size = " << size << " bytes" << std::endl;
    PrintFreeList();
    return NULL;
  }
}

cudaError_t BuddySystem::Free(void* ptr) {
  std::cout << "Buddy System No." << gpuIdx_ << " trying to free pointer: " << ptr <<std::endl;
  std::map<char*, Block*>::iterator itr = memPool_.find((char*)ptr);
  if (itr == memPool_.end()) {
    std::cout << "FAILURE: Buddy System No." << gpuIdx_ << ": Can't free pointer at " << ptr << std::endl;
    return cudaErrorInvalidValue;
  }
  Block* blockToBeInserted = itr->second;
  memPool_.erase(itr);
  allocated_ -= blockToBeInserted->GetSize();
  free_ += blockToBeInserted->GetSize();
  int idx = GetListIdx(blockToBeInserted->GetSize());
  std::cout << "Block suppposed to be inserted at index = " << idx << std::endl;
  std::cout << "Initially: ";
  InsertBlock(blockToBeInserted);
  MergeFreeList();
  std::cout << "SUCCESS: Free completed: " << ptr << std::endl;
  std::cout << "Total free memory after Free: size = " << free_ << " bytes" << std::endl;
  std::cout << "Total allocated memory after Free: size = " << allocated_ << " bytes" << std::endl;
  PrintMemPool();
  //if (allocated_ < CLEAN_UP_BOUNDRY) CleanUp();
  PrintFreeList();
  return cudaSuccess;
}

void BuddySystem::InsertBlock(Block* block) {
  //std::cout << "Block to insert has size: " << block->GetSize() << std::endl;
  int idx = GetListIdx(block->GetSize());
  std::cout << "Block inserted at list index = " << idx << std::endl;
  //std::cout << "Block should be inserted at list index: " << idx << std::endl;
  if (freeList_[idx] == NULL) {
    //std::cout << "Block inserted at head of list index: " << idx << std::endl;
    freeList_[idx] = block;
    return;
  }

  Block* curr;
  Block* prev;
  prev = NULL;
  curr = freeList_[idx];

  while (curr != NULL) {
    if (curr->GetData() > block->GetData()) break;
    prev = curr;
    curr = curr->GetNext();
  }

  if (prev != NULL) {
    prev->SetNext(block);
    block->SetNext(curr);
  } else {
    block->SetNext(freeList_[idx]);
    freeList_[idx] = block;
  }
  //std::cout << "Block inserted in the middle of list index: " << idx << std::endl;
}

Block* BuddySystem::Merge(Block* block, int idx) {
  //std::cout << "Trying to merge desired block" << std::endl;
  //idx = GetListIdx(block->GetSize());
  size_t listBlockSize = GetListBlockSize((size_t)idx);
  Block* curr = freeList_[idx];
  Block* prev = NULL;
  Block* prevPrev = NULL;
  bool mergedWithPrev = false;
  bool mergedWithNext = false;
  //bool merged = false;

  while (curr != block && curr != NULL) {
    prevPrev = prev;
    prev = curr;
    curr = curr->GetNext();
  }

  if (curr == NULL) return NULL;

  if (prev != NULL) {
    //if can merge with previous block, merge and remove curr block
    if ((prev->GetData() + listBlockSize) == curr->GetData()) {
      prev->SetSize(prev->GetSize() + curr->GetSize());
      prev->SetNext(curr->GetNext());
      curr->SetNext(NULL);
      mergedWithPrev = true;
      std::cout << "Merged with the previous block" << std::endl;
    }
  }

  //if merge with previous block, check if it can be merged with next block
  if (mergedWithPrev) {
    if (prev->GetNext() != NULL) {
      Block* next = prev->GetNext();
      if ((prev->GetData() + prev->GetSize()) == next->GetData()) {
        prev->SetSize(prev->GetSize() + next->GetSize());
        prev->SetNext(next->GetNext());
        next->SetNext(NULL);
        mergedWithNext = true;
        std::cout << "Merged with the next block" << std::endl;
      }
    }
  } else { //if did not merge with previous block(i.e curr still exists), check if it can be merged with next block
    if (curr->GetNext() != NULL) {
      Block* next = curr->GetNext();
      if ((curr->GetData() + listBlockSize) == next->GetData()) {
        curr->SetSize(curr->GetSize() + next->GetSize());
        curr->SetNext(next->GetNext());
        next->SetNext(NULL);
        mergedWithNext = true;
        std::cout << "Merged with the next block" << std::endl;
      }
    }
  }
 
  if (mergedWithPrev) {
    if (prevPrev != NULL) {
      prevPrev->SetNext(prev->GetNext());
      prev->SetNext(NULL);
      std::cout << "Inserting merged block: ";
      InsertBlock(prev);
      return prevPrev->GetNext();
    } else {
      freeList_[idx] = prev->GetNext();
      prev->SetNext(NULL);
      std::cout << "Inserting merged block: ";
      InsertBlock(prev);
      return freeList_[idx];
    }
  } else if (mergedWithNext) {
      if (prev != NULL) {
        prev->SetNext(curr->GetNext());
        curr->SetNext(NULL);
        std::cout << "Inserting merged block: ";
        InsertBlock(curr);
        return curr->GetNext();
      } else {
        freeList_[idx] = curr->GetNext();
        curr->SetNext(NULL);
        std::cout << "Inserting merged block: ";
        InsertBlock(curr);
        return freeList_[idx];
      }
  }

  return block->GetNext();
}

void BuddySystem::MergeFreeList() {
  for (int i = 0; i < freeListSize_; i++) {
    Block* curr = freeList_[i];
    while (curr != NULL) {
      curr = Merge(curr, i);
    }
  }
}

//currently disabled for better log
void BuddySystem::PrintFreeList() {
  std::cout << "====================================================" << std::endl;
  std::cout << "Free List Info" << std::endl;
  std::cout << "====================================================" << std::endl;
  for (int i = 0; i < freeListSize_; i++ ) {
    std::cout << "Free List Index = " << i << std::endl;
    Block* curr = freeList_[i];
    while (curr != NULL) {
      std::cout << "Block addr = " << (void*)curr->GetData() << " size = " << curr->GetSize() << " ===>" << std::endl;
      curr = curr->GetNext();
    }
    std::cout << "---------------------------------------------------" << std::endl;
  }
}

void BuddySystem::CheckDuplicate() {
  std::set<char*> addr;
  for (int i = 0; i < freeListSize_; i++) {
    Block * curr = freeList_[i];
    while (curr != NULL) {
      const bool inSet = addr.find(curr->GetData()) != addr.end();
      assert(inSet == false);
      if (inSet == true) std::cout << "This block with address = " << (void*)curr->GetData() <<
                                      " appeared more than once." << std::endl; 
      curr = curr->GetNext();
    }
  }
}

void BuddySystem::CleanUp() {
  std::cout << "Starting clean up" << std::endl;
  Block* tempList = NULL;

  //insert all nodes in the free list into a temp list
  for (int i = 0; i < freeListSize_; i++) {
    Block* curr = freeList_[i];

    while (curr != NULL) {
      freeList_[i] = curr->GetNext();
      curr->SetNext(NULL);

      Block* temp = tempList;
      Block* prev = NULL;

      while (temp != NULL) {
        if (temp->GetData() > curr->GetData()) break;
        prev = temp;
        temp = temp->GetNext();
      }

      if (prev != NULL) {
        prev->SetNext(curr);
        curr->SetNext(temp);
      } else {
        curr->SetNext(tempList);
        tempList = curr;
      }
      curr = freeList_[i];
    }
  }

  std::cout << "First remove all nodes in the existing free list" << std:: endl;
  PrintFreeList();
  std::cout << "Display all nodes" << std::endl;
  Block* printPtr = tempList;
  while (printPtr != NULL) {
    std::cout << "Block addr = " << (void*)printPtr->GetData() << " size = " << printPtr->GetSize() <<
                 " ==========>" << std::endl;
    printPtr = printPtr->GetNext();
  }

  //merge the nodes in the temp list
  Block* curr = tempList;
  while (curr != NULL) {
    if (curr->GetNext() != NULL) {
      Block* next = curr->GetNext();
      if ((curr->GetData() + curr->GetSize()) == next->GetData()) {
        curr->SetSize(curr->GetSize() + next->GetSize());
        curr->SetNext(next->GetNext());
        next->SetNext(NULL);
      } else {
        curr = next;
      }
    } else {
      curr = curr->GetNext();
    }
  }

  std::cout << "After the merge we have nodes:" << std::endl;
  printPtr = tempList;
  while (printPtr != NULL) {
    std::cout << "Block addr = " << (void*)printPtr->GetData() << " size = " << printPtr->GetSize() <<
                 " ==========>" << std::endl;
    printPtr = printPtr->GetNext();
  }

  //insert the nodes in the temp list back into the free list
  curr = tempList;
  while (curr != NULL) {
    Block* next = curr->GetNext();
    curr->SetNext(NULL);
    InsertBlock(curr);
    curr = next;
  }

  std::cout << "After the clean up the free list is: " << std::endl;
}

void BuddySystem::PrintMemPool() {
  if (memPool_.empty()) {
    std::cout << "Memory pool is empty" << std::endl;
    return;
  }

  std::cout << "===================================================" << std::endl;
  std::cout << "Printing Memory Pool:" << std::endl;

  std::map<char*, Block*>::const_iterator itr = memPool_.begin();
  while (itr != memPool_.end()) {
    std::cout << "Block addr = " << (void*)itr->first << " size = " << itr->second->GetSize() << std::endl;
    itr++;
  }
}

} //namespace mxnet

