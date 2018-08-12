#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include <math.h>

#include <mxnet/gpu_swap_memmgr.h>
#include "../common/cuda_utils.h"

namespace mxnet {

#define CUDA_CALL(func) {                                   \
  cudaError_t e = (func);                                   \
  CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)  \
        << __FUNCTION__ << ":" << __LINE__                  \
        << "has a CUDA error: " << cudaGetErrorString(e);   \
}

cudaError_t MemoryManager::Memcpy(int device_id, void* dst, const void* src,
                                  size_t count, enum cudaMemcpyKind kind) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaMemcpy(dst, src, count, kind));
  return cudaSuccess;
}

cudaError_t MemoryManager::MemcpyAsync(
        int device_id, void* dst, const void* src, size_t count,
        cudaMemcpyKind kind, cudaStream_t stream) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaMemcpyAsync(dst, src, count, kind, stream));
  return cudaSuccess;
}

cudaError_t MemoryManager::StreamSynchronize(int device_id,
                                             cudaStream_t stream) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaStreamSynchronize(stream));
  return cudaSuccess;
}

CudaMemoryManager::CudaMemoryManager() {
  std::cout << "Initialize CUDA Memory Allocator" << std::endl;
}

CudaMemoryManager::~CudaMemoryManager() {
  std::cout << "Destroy Cuda Memory Allocator" << std::endl;
}

cudaError_t CudaMemoryManager::Malloc(void*& devptr, size_t size,
                                      int device_id) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaMalloc(&devptr, size));
  return cudaSuccess;
}

cudaError_t CudaMemoryManager::Free(void* devptr, int device_id) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaFree(devptr));
  return cudaSuccess;
}

cudaError_t CudaMemoryManager::MemGetInfo(int device_id, size_t* total,
                                          size_t* free) {
  //std::cout<<"MemGetInfo: Check"<<std::endl;
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaMemGetInfo(free, total));
  //std::cout << *free << " " << *total << std::endl;
  //std::cout << "MemGetInfo: Check Over" << std::endl;
  return cudaSuccess;
}

bool CudaMemoryManager::TryAllocate(int device_id, size_t size) {
  CUDA_CALL(cudaSetDevice(device_id));
  size_t free, total;
  CUDA_CALL(cudaMemGetInfo(&free, &total));
  // TODO(fegin): This fixed threshold is not acceptable.
  // FIXME(fegin): The maximum threshould I used in the old MXNet is 512 MB.
  //               We should figure out why such a large threshold is needed
  //               for current implementation.
  return free > size + 1500000000;
}

BuddyMemoryManager::BuddyMemoryManager() {
  std::cout << "Initializing Memory Manager" << std::endl;
  buddy_ = new BuddySystem*[NUMBER_OF_GPU];
  for (int device  = 0; device < NUMBER_OF_GPU; device++) {
    buddy_[device] = NULL;
    CUDA_CALL(cudaSetDevice(device));
    size_t avail, total;
    size_t mb = 1 << 20;
    CUDA_CALL(cudaMemGetInfo(&avail, &total));
    avail = static_cast<size_t>(avail * GPU_UTIL_RATIO);
    char* wholeMemory = NULL;
    while (cudaMalloc((void**)&wholeMemory, avail) == cudaErrorMemoryAllocation) {
      avail -= mb;
      if (avail <= 0) {
        break;
      }
    }
    if (avail > 0) {
      buddy_[device] = new BuddySystem(new Block(wholeMemory, avail), avail,
                                       device);
      std::cout << "Buddy System No." << device
                << " initialized with size = " << avail << " bytes"
                << std::endl;
    } else {
      std::cout << "Warning: There's no memory left on device: " << device
                << std::endl;
    }
  }
  std::cout << "Memory Manager initialization completed" << std::endl;
}

BuddyMemoryManager::~BuddyMemoryManager() {
  for (int device = 0; device < NUMBER_OF_GPU; device++) {
    CUDA_CALL(cudaSetDevice(device));
    BuddySystem* buddy = buddy_[device];
    buddy->~BuddySystem();
    CUDA_CALL(cudaFree((void*)buddy->GetStart()));
    std::cout << "Buddy System No." << buddy->GetGPUIdx() << " destructed"
              << std::endl;
  }
}

cudaError_t BuddyMemoryManager::Malloc(void*& devptr, size_t size,
                                       int device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(device_id));
  devptr = buddy_[device_id]->Alloc(size);
  if (!devptr) {
    return cudaErrorMemoryAllocation;
  }
  return cudaSuccess;
}

cudaError_t BuddyMemoryManager::Free(void* devptr, int device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(device_id));
  buddy_[device_id]->Free(devptr);
  return cudaSuccess;
}

//returns total memory and total free memory(not necessarily consequtive) in mmu
cudaError_t BuddyMemoryManager::MemGetInfo(int device_id, size_t* total,
                                           size_t* free) {
  std::lock_guard<std::mutex> lock(mutex_);
  CUDA_CALL(cudaSetDevice(device_id));
  if (buddy_[device_id] == NULL) {
    return cudaErrorInvalidValue;
  }
  *total = buddy_[device_id]->GetTotal();
  *free = buddy_[device_id]->GetFree();
  return cudaSuccess;
}

bool BuddyMemoryManager::TryAllocate(int device_id, size_t size) {
  CUDA_CALL(cudaSetDevice(device_id));
  BuddySystem* buddy = buddy_[device_id];
  Block** freeList = buddy->GetFreeList();
  int freeListSize = buddy->GetFreeListSize();
  int idx = GetListIdx(size);
  if (idx == 0) {
    idx = 1;
  }

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
