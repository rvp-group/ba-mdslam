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

#include "cuda_utils.cuh"

// blocksize for shared mem optimized for LinearSystemEntry struct
// TODO use dynamic shared mem
// #define BLOCKSIZE 128 // TODO grosso come na casa
#define BLOCKSIZE 64 // TODO grosso come na casa

namespace md_slam {

  /*
     This version is completely unrolled, unless warp shuffle is available (commented at the
     moment), then shuffle is used within a loop.  It uses a template parameter to achieve optimal
     code for any (power of 2) number of threads. This requires a switch statement in the host code
     to handle all the different thread block sizes at compile time. When shuffle is available, it
     is used to reduce warp synchronization.

      Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
      In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
      If blockSize > 32, allocate blockSize*sizeof(T) bytes.
  */
  template <class T, unsigned int blockSize>
  __global__ void sum_reduce(T* g_odata, const T* g_idata, const size_t N) {
    __shared__ T sdata[BLOCKSIZE];
    unsigned int tid = threadIdx.x; // local thread index
    unsigned int i   = blockIdx.x * (blockDim.x * 2) +
                     threadIdx.x; // global thread index - fictitiously double the block dimension

    // --- performs the first level of reduction in registers when reading from global memory
    T mySum = (i < N) ? g_idata[i] : T();
    if (i + blockDim.x < N)
      mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;

    // --- before going further, we have to make sure that all the shared memory loads have been
    // completed
    __syncthreads();

    // --- reduction in shared memory, fully unrolled loop
    if ((blockSize >= 512) && (tid < 256)) {
      sdata[tid] = mySum = mySum + sdata[tid + 256];
    }
    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
      sdata[tid] = mySum = mySum + sdata[tid + 128];
    }
    __syncthreads();

    if ((blockSize >= 128) && (tid < 64)) {
      sdata[tid] = mySum = mySum + sdata[tid + 64];
    }
    __syncthreads();
    // #if (__CUDA_ARCH__ >= 300)
    //   // --- Single warp reduction by shuffle operations
    //   if (tid < 32) {
    //     // --- Last iteration removed from the for loop, but needed for shuffle reduction
    //     mySum += sdata[tid + 32];
    //     // --- Reduce final warp using shuffle
    //     for (int offset = warpSize / 2; offset > 0; offset /= 2)
    //       // mySum += __shfl_down(mySum, offset);
    //       mySum += __shfl_down_sync(0xffffffff, mySum, offset);
    //     // for (int offset=1; offset < warpSize; offset *= 2) mySum += __shfl_xor(mySum, i);
    //   }
    // #else
    // --- Reduction within a single warp. Fully unrolled loop.
    if ((blockSize >= 64) && (tid < 32)) {
      sdata[tid] = mySum = mySum + sdata[tid + 32];
    }
    __syncthreads();

    if ((blockSize >= 32) && (tid < 16)) {
      sdata[tid] = mySum = mySum + sdata[tid + 16];
    }
    __syncthreads();

    if ((blockSize >= 16) && (tid < 8)) {
      sdata[tid] = mySum = mySum + sdata[tid + 8];
    }
    __syncthreads();

    if ((blockSize >= 8) && (tid < 4)) {
      sdata[tid] = mySum = mySum + sdata[tid + 4];
    }
    __syncthreads();

    if ((blockSize >= 4) && (tid < 2)) {
      sdata[tid] = mySum = mySum + sdata[tid + 2];
    }
    __syncthreads();

    if ((blockSize >= 2) && (tid < 1)) {
      sdata[tid] = mySum = mySum + sdata[tid + 1];
    }
    __syncthreads();
    // #endif

    // --- write result for this block to global memory, at the end of the kernel, global memory
    // will contain the results for the summations of individual blocks
    if (tid == 0)
      g_odata[blockIdx.x] = mySum;
  }

  template <class T>
  void sum_reduce_wrapper(T* g_odata,
                          const T* g_idata,
                          const size_t N,
                          const int num_blocks_,
                          const int num_threads_) {
    switch (num_threads_) {
      case 512:
        sum_reduce<T, 512><<<num_blocks_, num_threads_>>>(g_odata, g_idata, N);
        break;
      case 256:
        sum_reduce<T, 256><<<num_blocks_, num_threads_>>>(g_odata, g_idata, N);
        break;
      case 128:
        sum_reduce<T, 128><<<num_blocks_, num_threads_>>>(g_odata, g_idata, N);
        break;
      case 64:
        sum_reduce<T, 64><<<num_blocks_, num_threads_>>>(g_odata, g_idata, N);
        break;
      case 32:
        sum_reduce<T, 32><<<num_blocks_, num_threads_>>>(g_odata, g_idata, N);
        break;
      case 16:
        sum_reduce<T, 16><<<num_blocks_, num_threads_>>>(g_odata, g_idata, N);
        break;
      case 8:
        sum_reduce<T, 8><<<num_blocks_, num_threads_>>>(g_odata, g_idata, N);
        break;
      case 4:
        sum_reduce<T, 4><<<num_blocks_, num_threads_>>>(g_odata, g_idata, N);
        break;
      case 2:
        sum_reduce<T, 2><<<num_blocks_, num_threads_>>>(g_odata, g_idata, N);
        break;
      case 1:
        sum_reduce<T, 1><<<num_blocks_, num_threads_>>>(g_odata, g_idata, N);
        break;
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
} // namespace md_slam
