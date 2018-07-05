/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#if MXNET_USE_CUDA
  #include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA
#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <mxnet/mem_mgr.h>
#include <new>
#include "./storage_manager.h"
#include "../common/cuda_utils.h"


namespace mxnet {
namespace storage {

#if MXNET_USE_CUDA
/*!
 * \brief Storage manager with a memory pool on gpu.
 */
class GPUPooledStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  GPUPooledStorageManager(int device_id) {
    reserve_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_RESERVE", 5);
    memory_manager_ = MemoryManager::_GetSharedRef();
    swap_ = Swap::_GetSharedRef();
    do_reuse_ = true;
    device_id_ = device_id;
  }
  /*!
   * \brief Default destructor.
   */
  ~GPUPooledStorageManager() {
    ReleaseAll();
  }

  void Alloc(Storage::Handle* handle) override;
  void Free(Storage::Handle handle) override;

  void DirectFree(Storage::Handle handle) override {
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
    DirectFreeNoLock(handle);
  }

 private:
  void DirectFreeNoLock(Storage::Handle handle) {
    cudaError_t err = memory_manager_->Free(handle.GetDptr(), device_id_);
    size_t size = handle.size + NDEV;
    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }
    used_memory_ -= size;
  }
 
 private:
  void ReleaseAll();
  // used memory
  size_t used_memory_ = 0;
  // percentage of reserved memory
  int reserve_;
  // shared pointer for swap
  std::shared_ptr<Swap> swap_;
  // number of devices
  const int NDEV = 32;
  //shared pointer for memory manager
  std::shared_ptr<MemoryManager> memory_manager_;
  // memory pool
  std::unordered_map<size_t, std::vector<void*>> memory_pool_;
  // whether to reuse freed memory
  bool do_reuse_;
  int device_id_;
  DISALLOW_COPY_AND_ASSIGN(GPUPooledStorageManager);
};  // class GPUPooledStorageManager

void GPUPooledStorageManager::Alloc(Storage::Handle* handle) {
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  size_t size = handle->size + NDEV;
  auto&& reuse_it = memory_pool_.find(size);
  if (reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
    size_t free, total;
    if (do_reuse_ && 
        ( !memory_manager_->TryAllocate(device_id_, size + total * reserve_ / 100) 
        || !memory_manager_->TryAllocate(device_id_, total * reserve_ / 100))) {
      std::cout<<"!!!!!!!!!!!!!!!Disable reuse"<<std::endl;
      do_reuse_ = false;
      ReleaseAll();
    }
    std::cout <<"swap out " << size << " " << device_id_ << std::endl;
    swap_->SwapOut(size, device_id_);
    void* ret = nullptr;
    cudaError_t e = memory_manager_->Malloc(ret, size, device_id_);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
    }
    used_memory_ += size;
    handle->SetDptr(ret, device_id_);
  } else {
    auto&& reuse_pool = reuse_it->second;
    auto ret = reuse_pool.back();
    reuse_pool.pop_back();
    handle->SetDptr(ret, device_id_);
  }
}

void GPUPooledStorageManager::Free(Storage::Handle handle) {
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  if (do_reuse_) {
    size_t size = handle.size + NDEV;
    auto&& reuse_pool = memory_pool_[size];
    reuse_pool.push_back(handle.GetDptr());
  } else {
    DirectFreeNoLock(handle);
  }
}

// FIXME(sotskin):DirectFreeNoLock does not know the device id of memory to be
// freed
void GPUPooledStorageManager::ReleaseAll() {
  for (auto&& i : memory_pool_) {
    for (auto&& j : i.second) {
      Storage::Handle handle;
      // (Sotskin)No need to swap memories that are about to be released
      handle.SetDptr(j, -1); 
      handle.size = i.first - NDEV;
      DirectFreeNoLock(handle);
    }
  }
  memory_pool_.clear();
}

#endif  // MXNET_USE_CUDA

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
