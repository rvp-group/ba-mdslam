// Copyright 2022 Luca Di Giammarino
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

// force align to 16B for eigen-cuda-cpp classes
#ifdef __CUDACC__
#define ALIGN(x) __align__(x)
#else
#define ALIGN(x) alignas(x)
#endif

namespace md_slam {

  static void HandleError(cudaError_t err, const char* file, int line) {
    // CUDA error handeling from the "CUDA by example" book
    if (err != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
      exit(EXIT_FAILURE);
    }
  }

#define CUDA_CHECK(err) (HandleError(err, __FILE__, __LINE__))

  inline int getDeviceInfo() {
    int n_devices = 0;
    // int available_shmem = 0;
    cudaGetDeviceCount(&n_devices);
    // check for devices
    for (int i = 0; i < n_devices; ++i) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("device number: %d\n", i);
      printf("  device name: %s\n", prop.name);
      printf("  memory clock rate [KHz]: %d\n", prop.memoryClockRate);
      printf("  memory bus width [bits]: %d\n", prop.memoryBusWidth);
      printf("  peak memory bandwidth [GB/s]: %f\n",
             2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
      printf("  shared mem size per block [KB]: %i\n", prop.sharedMemPerBlock);
      std::cerr << "_______________________________________________" << std::endl;
      // available_shmem = prop.sharedMemPerBlock;
    }
    if (n_devices > 1) {
      std::cerr << "multiple devices found, using devices number 0" << std::endl;
      std::cerr << "_______________________________________________" << std::endl;
    }
    std::cerr << std::endl;
    return n_devices;
  }

} // namespace md_slam