#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include <math.h>
#include <mxnet/buddy.h>
#include <mxnet/mem_mgr.h>
#include "../common/cuda_utils.h"

namespace mxnet {

#define CUDA_CALL(func) 					 \
  {								 \
    cudaError_t e = (func);     				 \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)     \
        << "CUDA: " << cudaGetErrorString(e);                    \
  }

static inline void CHECK_CUDA_ERROR() {									
  cudaError_t e = cudaGetLastError();					
  CHECK_EQ(e, cudaSuccess) << "CUDA: " << cudaGetErrorString(e);
}

CudaMemoryManager::CudaMemoryManager() {
  std::cout << "Initialize CUDA Memory Allocator" << std::endl;
}

CudaMemoryManager::~CudaMemoryManager() {
  std::cout << "Destroy Cuda Memory Allocator" << std::endl;
}

cudaError_t CudaMemoryManager::Malloc(void*& devptr, size_t size,
                                      int device_id) {
  cudaSetDevice(device_id);
  cudaError_t e = cudaMalloc(&devptr, size);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << "Malloc failed: " << cudaGetErrorString(e) << std::endl;
  }
  return e;
}

cudaError_t CudaMemoryManager::Free(void* devptr, int device_id) {
  cudaSetDevice(device_id);
  cudaError_t e = cudaFree(devptr);
  if(e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << "Free failed: " << cudaGetErrorString(e) << std::endl;
  }
  return e;
}

cudaError_t CudaMemoryManager::Memcpy(int device_id, void* dst, const void* src,
                                      size_t count, enum cudaMemcpyKind kind) {
  cudaSetDevice(device_id);
  cudaSetDevice(device_id);
  cudaError_t e = cudaMemcpy(dst, src, count, kind);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << "Memcpy failed: " << cudaGetErrorString(e) << std::endl;
  }
  return e;
}

cudaError_t CudaMemoryManager::MemGetInfo(int device_id, size_t* total,
                                          size_t* free) {
  std::cout << "MemGetInfo: Check" <<std::endl;
  cudaError_t e = cudaSetDevice(device_id);
  if (e != cudaSuccess) {
    std::cout << "Check setdevice failed: " << cudaGetErrorString(e)
              << std::endl;
  }
  size_t free_, total_;
  e = cudaMemGetInfo(&free_, &total_);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << "Check GetInfo failed: " << cudaGetErrorString(e)
              << std::endl;
  } else {
    std::cout << free_ << " " << total_ << std::endl;
  }
  *total = total_;
  *free = free_;
  std::cout << "MemGetInfo: Check Over" << std::endl;
  return cudaSuccess;
}

bool CudaMemoryManager::TryAllocate(int device_id, size_t size) {
  cudaError_t e = cudaSetDevice(device_id);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << e << " TryAlloc SetDevice failed: " << cudaGetErrorString(e)
              << std::endl;
  }
  size_t free, total;
  e = cudaMemGetInfo(&free, &total);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    std::cout << e << " TryAlloc GetInfo failed: " << cudaGetErrorString(e)
              << std::endl;
  }
  return free > size + 1000000000;
}


BuddyMemoryManager::BuddyMemoryManager() {
  std::cout << "Initializing Memory Manager" << std::endl;
  int deviceNum;
  CUDA_CALL(cudaGetDeviceCount(&deviceNum));
  std::cout << "device num = " << deviceNum << std::endl;
  deviceCount_ = deviceNum;
  buddy_ = new BuddySystem*[deviceNum];

  for (int deviceIdx = 0; deviceIdx < deviceNum; deviceIdx++) {
    buddy_[deviceIdx] = NULL;
    CUDA_CALL(cudaSetDevice(deviceIdx));

    size_t avail, total;
    size_t mb = 1 << 20;
    CUDA_CALL(cudaMemGetInfo(&avail, &total));

    avail = static_cast<size_t>(avail * GPU_UTIL_RATIO);
    char* wholeMemory = NULL;
    while (cudaMalloc((void**)&wholeMemory, avail) == cudaErrorMemoryAllocation) {
        avail -= mb;
        if (avail <= 0) break;
    }

    if (avail > 0) {
      buddy_[deviceIdx] = new BuddySystem(new Block(wholeMemory, avail), avail, deviceIdx);
      std::cout << "Buddy System No." << deviceIdx << " initialized with size = " << avail << " bytes"  << std::endl;
    } else {
      std::cout << "Warning: There's no memory left on device: " << deviceIdx << std::endl;
    }
  }
  std::cout << "Memory Manager initialization completed" << std::endl;
}

BuddyMemoryManager::~BuddyMemoryManager() {
  std::cout << "DestructingBuddy Memory Manager" << std::endl;
  typedef std::map<char*, Block*> MemoryPool;
  for (int deviceIdx = 0; deviceIdx < deviceCount_; deviceIdx++) {
    CUDA_CALL(cudaSetDevice(deviceIdx));
    BuddySystem* buddy = buddy_[deviceIdx];
    buddy->~BuddySystem();
    //MemoryPool mp = buddy->GetMemPool();
    //while (!mp.empty()) {
    //  std::cout << "Destructing block at addr: " << (void*)mp.bigin()->first() << std::endl;
    //  buddy->Free((void*)(mp.begin()->first));
    //}
    cudaFree((void*)buddy->GetStart());
    std::cout << "Buddy System No." << buddy->GetGPUIdx() << " destructed" << std::endl;
  }
  std::cout << "Memory Manager destruction completed" << std::endl;
}

cudaError_t BuddyMemoryManager::Malloc(void*& devptr, size_t size, int deviceIdx) {
  std::cout << "Malloc size = " << size << " bytes on Buddy System No. " << deviceIdx << std::endl;
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(deviceIdx));
  devptr = buddy_[deviceIdx]->Alloc(size);
  if (!devptr) return cudaErrorMemoryAllocation;
  return cudaSuccess;
}

cudaError_t BuddyMemoryManager::Free(void* devptr, int deviceIdx) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(deviceIdx));
  buddy_[deviceIdx]->Free(devptr);
  return cudaSuccess;
}

cudaError_t BuddyMemoryManager::Memcpy(int deviceIdx, void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
  //TODO(qingsen): need implementation
  std::cout<<"Memcpy"<<std::endl;
  CUDA_CALL(cudaSetDevice(deviceIdx));
  return cudaMemcpy(dst, src, count, kind);
}

//returns total memory and total free memory(not necessarily consequtive) in mmu
cudaError_t BuddyMemoryManager::MemGetInfo(int deviceIdx, size_t* total, size_t* free) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(deviceIdx));
  if (buddy_[deviceIdx] == NULL) return cudaErrorInvalidValue;
  *total = buddy_[deviceIdx]->GetTotal();
  *free = buddy_[deviceIdx]->GetFree();
  return cudaSuccess;
}

bool BuddyMemoryManager::TryAllocate(int deviceIdx, size_t size) {
  std::cout << "Buddy System No." << deviceIdx << " has free = " << buddy_[deviceIdx]->GetFree() <<
	       " and allocate = " << buddy_[deviceIdx]->GetAllocated() << std::endl;
  std::cout << "Buddy System No." << deviceIdx << ": Trying to allocate size = " << size << std::endl;
  CUDA_CALL(cudaSetDevice(deviceIdx));
  BuddySystem* buddy = buddy_[deviceIdx];
  Block** freeList = buddy->GetFreeList();
  int freeListSize = buddy->GetFreeListSize();
  int idx = GetListIdx(size);
  if (idx == 0) idx = 1;

  for (int i = idx; i < freeListSize; i++) {
    if (freeList[i] != NULL) {
      std::cout << "SUCCESS: There is enough space" << std::endl;
      return true;
    }
  }

  if (buddy->GetAllocated() < CLEAN_UP_BOUNDRY) {
    std::cout << "Starting clean up process" << std::endl;
    buddy->CleanUp();
  }

  for (int i = idx; i < freeListSize; i++) {
    if (freeList[i] != NULL) {
      std::cout << "SUCCESS: There is enough space" << std::endl;
      return true;
    }
  }

  std::cout << "FAILURE: There isn't enough space" << std::endl;
  return false;
}

// Factory functions.
std::shared_ptr<MemoryManager> GetMemoryManagerRef() {
  static std::shared_ptr<MemoryManager> inst;
  static bool set = false;
  if (!set) {
    std::string mem_mgr_type = dmlc::GetEnv("MXNET_MEM_MGR_TYPE",
                                            std::string("CUDA"));
    std::cout << "MXNET_MEM_MGR_TYPE: " << mem_mgr_type << std::endl;
    if (mem_mgr_type == "CUDA") {
      inst.reset(dynamic_cast<MemoryManager*>(new CudaMemoryManager()));
    } else if (mem_mgr_type == "Buddy") {
      inst.reset(dynamic_cast<MemoryManager*>(new BuddyMemoryManager()));
    }
    set = true;
  }
  return inst;
}

MemoryManager* GetMemoryManager() {
  static MemoryManager* mm = GetMemoryManagerRef().get();
  return mm;
}

} // namespace mxnet
