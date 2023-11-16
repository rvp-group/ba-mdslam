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

#include <gtest/gtest.h>
#include <iostream>
#include <md_slam/factor.cu>
#include <md_slam/linear_system_entry.cuh>
#include <random>
#include <srrg_test/test_helper.hpp>

// clock stuff
#include <chrono>
#include <ctime>
#include <ratio>

using namespace std::chrono;

#ifndef MD_TEST_DATA_FOLDER
#error "NO TEST DATA FOLDER"
#endif

using namespace md_slam;

using Entry = LinearSystemEntry;

TEST(DUMMY, HostVsDeviceStaticSharedMemory) {
  int nDevices;
  int available_shmem = 0;
  cudaGetDeviceCount(&nDevices);
  // TODO remove this shit from here
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate [KHz]: %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width [bits]: %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth [GB/s]: %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Shared mem size per block [KB]: %i\n", prop.sharedMemPerBlock);
    available_shmem = prop.sharedMemPerBlock;
  }

  const int N = 200000; // 100000;

  // create vec in host
  std::vector<Entry> h_vec(N, Entry::Ones());
  // random 0/1 generator to simulate test bool inside kernel
  // auto gen = std::bind(std::uniform_int_distribution<>(0, 1), std::default_random_engine());
  // for (int i = 0; i < h_vec.size(); ++i) {
  //   h_vec[i].is_good = gen();
  // }

  // copy to gpu
  Entry* d_vec;
  CUDA_CHECK(cudaMalloc((void**) &d_vec, sizeof(Entry) * N));
  CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(Entry) * N, cudaMemcpyHostToDevice));

  // host summation - simple life
  high_resolution_clock::time_point hstart = high_resolution_clock::now();
  Entry sum;
  for (int i = 0; i < h_vec.size(); ++i) {
    sum += h_vec[i];
  }
  high_resolution_clock::time_point hstop = high_resolution_clock::now();
  duration<double> htime                  = duration_cast<duration<double>>(hstop - hstart);
  printf("sum reduce (host) - Elapsed time:  %3.3f ms \n", htime * 1e3);

  std::cerr << "chi: " << sum.chi << "\nH:\n"
            << sum.upperH.transpose() << "\nb: " << sum.b.transpose() << std::endl;

  const int num_threads    = BLOCKSIZE; // macro optimized for Linear System Entries
  const int num_blocks     = (N + num_threads - 1) / num_threads;
  const int required_shmem = num_threads * sizeof(Entry);

  printf("total num of data points: %i | num threads: %i | num blocks: %i\n",
         N,
         num_threads,
         num_blocks);
  printf("available shared mem : %i | required shared mem: %i\n", available_shmem, required_shmem);
  ASSERT_LT(required_shmem, available_shmem);
  //   TODO check if shmem exceed, if yes make smaller pow2 blocksize

  // --- creating events for timing
  float dtime;
  cudaEvent_t dstart, dstop;
  cudaEventCreate(&dstart);
  cudaEventCreate(&dstop);

  Entry* d_vec_block;
  CUDA_CHECK(cudaMalloc((void**) &d_vec_block, sizeof(Entry) * num_blocks));

  cudaEventRecord(dstart, 0);
  sum_reduce_wrapper(d_vec_block, d_vec, N, num_blocks, num_threads);
  cudaEventRecord(dstop, 0);
  cudaEventSynchronize(dstop);
  cudaEventElapsedTime(&dtime, dstart, dstop);
  printf("sum reduce (device) - Elapsed time:  %3.3f ms \n", dtime);

  // --- the last part of the reduction, which would be expensive to perform on the device, is
  // executed on the host

  Entry* h_vec_block = new Entry[num_blocks];
  CUDA_CHECK(
    cudaMemcpy(h_vec_block, d_vec_block, num_blocks * sizeof(Entry), cudaMemcpyDeviceToHost));

  Entry d_sum;
  for (int i = 0; i < num_blocks; i++) {
    d_sum += h_vec_block[i];
  }

  ASSERT_LT(d_sum.chi - sum.chi, 1e-9);
  ASSERT_LT(d_sum.b(0) - sum.b(0), 1e-9);
  ASSERT_LT(d_sum.b(1) - sum.b(1), 1e-9);
  ASSERT_LT(d_sum.b(2) - sum.b(2), 1e-9);
  ASSERT_LT(d_sum.b(3) - sum.b(3), 1e-9);
  ASSERT_LT(d_sum.b(4) - sum.b(4), 1e-9);
  ASSERT_LT(d_sum.b(5) - sum.b(5), 1e-9);

  const double errH = d_sum.upperH.squaredNorm() - sum.upperH.squaredNorm();
  ASSERT_LT(errH, 1e-9);

  // std::cerr << errH << std::endl;
  std::cerr << "chi: " << d_sum.chi << "\nH:\n"
            << d_sum.upperH.transpose() << "\nb: " << d_sum.b.transpose() << std::endl;

  CUDA_CHECK(cudaFree(d_vec));
  CUDA_CHECK(cudaFree(d_vec_block));
  delete h_vec_block;
}

TEST(LSEntry, ReproducibleDeviceStaticSharedMemory) {
  const int num_experiments = 10;
  std::vector<Entry> entries(num_experiments, Entry());

  const int N = 2000000; // 100000;

  // create vec in host
  std::vector<Entry> h_vec(N);
  for (int i = 0; i < N; ++i) {
    h_vec[i] = Entry::Random();
  }

  for (int i = 0; i < num_experiments; ++i) {
    // copy to gpu
    Entry* d_vec;
    CUDA_CHECK(cudaMalloc((void**) &d_vec, sizeof(Entry) * N));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(Entry) * N, cudaMemcpyHostToDevice));

    const int num_threads    = BLOCKSIZE; // macro optimized for Linear System Entries
    const int num_blocks     = (N + num_threads - 1) / num_threads;
    const int required_shmem = num_threads * sizeof(Entry);

    //   TODO check if shmem exceed, if yes make smaller pow2 blocksize

    // --- creating events for timing
    float dtime;
    cudaEvent_t dstart, dstop;
    cudaEventCreate(&dstart);
    cudaEventCreate(&dstop);

    Entry* d_vec_block;
    CUDA_CHECK(cudaMalloc((void**) &d_vec_block, sizeof(Entry) * num_blocks));

    cudaEventRecord(dstart, 0);
    sum_reduce_wrapper(d_vec_block, d_vec, N, num_blocks, num_threads);
    cudaEventRecord(dstop, 0);
    cudaEventSynchronize(dstop);
    cudaEventElapsedTime(&dtime, dstart, dstop);
    printf("sum reduce (device) - Elapsed time:  %3.3f ms \n", dtime);

    // --- the last part of the reduction, which would be expensive to perform on the device, is
    // executed on the host

    Entry* h_vec_block = new Entry[num_blocks];
    CUDA_CHECK(
      cudaMemcpy(h_vec_block, d_vec_block, num_blocks * sizeof(Entry), cudaMemcpyDeviceToHost));

    Entry d_sum;
    for (int i = 0; i < num_blocks; i++) {
      d_sum += h_vec_block[i];
    }

    entries[i] = d_sum;

    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_vec_block));
    delete h_vec_block;
  }

  for (int i = 1; i < num_experiments; ++i) {
    // std::cerr << entries[0].b.transpose() << std::endl;
    // std::cerr << entries[i].b.transpose() << std::endl;
    // std::cerr << "======================" << std::endl;
    ASSERT_EQ_EIGEN(entries[0].b, entries[i].b);
    ASSERT_EQ_EIGEN(entries[0].upperH, entries[i].upperH);
    // std::cerr << entries[0].upperH.transpose() << std::endl;
    // std::cerr << entries[i].upperH.transpose() << std::endl;
    // std::cerr << "======================" << std::endl;
    ASSERT_EQ(entries[0].chi, entries[i].chi);
    // std::cerr << entries[0].chi << std::endl;
    // std::cerr << entries[i].chi << std::endl;
    // std::cerr << "======================" << std::endl;
    // std::cerr << std::endl;
  }
}

TEST(Double, ReproducibleDeviceStaticSharedMemory) {
  const int num_experiments = 10;
  std::vector<double> entries(num_experiments, 0.0);

  const int N              = 1000000;
  const double lower_bound = 0;
  const double upper_bound = 10000;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  // create vec in host and populate
  std::vector<double> h_vec(N);
  for (int i = 0; i < h_vec.size(); ++i) {
    h_vec[i] = unif(re);
  }

  for (int i = 0; i < num_experiments; ++i) {
    // copy to gpu
    double* d_vec;
    CUDA_CHECK(cudaMalloc((void**) &d_vec, sizeof(double) * N));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * N, cudaMemcpyHostToDevice));

    const int num_threads    = BLOCKSIZE; // macro optimized for Linear System Entries
    const int num_blocks     = (N + num_threads - 1) / num_threads;
    const int required_shmem = num_threads * sizeof(double);

    //   TODO check if shmem exceed, if yes make smaller pow2 blocksize

    // --- creating events for timing
    float dtime;
    cudaEvent_t dstart, dstop;
    cudaEventCreate(&dstart);
    cudaEventCreate(&dstop);

    double* d_vec_block;
    CUDA_CHECK(cudaMalloc((void**) &d_vec_block, sizeof(double) * num_blocks));

    cudaEventRecord(dstart, 0);
    sum_reduce_wrapper(d_vec_block, d_vec, N, num_blocks, num_threads);
    cudaEventRecord(dstop, 0);
    cudaEventSynchronize(dstop);
    cudaEventElapsedTime(&dtime, dstart, dstop);
    printf("sum reduce (device) - Elapsed time:  %3.3f ms \n", dtime);

    // --- the last part of the reduction, which would be expensive to perform on the device, is
    // executed on the host

    double* h_vec_block = new double[num_blocks];
    CUDA_CHECK(
      cudaMemcpy(h_vec_block, d_vec_block, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));

    double d_sum = 0.0;
    for (int i = 0; i < num_blocks; i++) {
      d_sum += h_vec_block[i];
    }

    entries[i] = d_sum;

    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_vec_block));
    delete h_vec_block;
  }

  for (int i = 0; i < num_experiments; ++i) {
    std::cerr << entries[i] << std::endl;
  }

  for (int i = 1; i < num_experiments; ++i) {
    std::cerr << entries[0] << " " << entries[i] << std::endl;
    ASSERT_EQ(entries[0], entries[i]);
  }
}

TEST(Float, ReproducibleDeviceStaticSharedMemory) {
  const int num_experiments = 10;
  std::vector<float> entries(num_experiments, 0.f);

  const int N             = 1000000;
  const float lower_bound = 0.f;
  const float upper_bound = 10000.f;
  std::uniform_real_distribution<float> unif(lower_bound, upper_bound);
  std::default_random_engine re;
  re.seed(1);

  // create vec in host and populate
  std::vector<float> h_vec(N);
  for (int i = 0; i < h_vec.size(); ++i) {
    h_vec[i] = unif(re);
  }

  for (int i = 0; i < num_experiments; ++i) {
    // copy to gpu
    float* d_vec;
    CUDA_CHECK(cudaMalloc((void**) &d_vec, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

    const int num_threads    = BLOCKSIZE; // macro optimized for Linear System Entries
    const int num_blocks     = (N + num_threads - 1) / num_threads;
    const int required_shmem = num_threads * sizeof(float);

    //   TODO check if shmem exceed, if yes make smaller pow2 blocksize

    // --- creating events for timing
    float dtime;
    cudaEvent_t dstart, dstop;
    cudaEventCreate(&dstart);
    cudaEventCreate(&dstop);

    float* d_vec_block;
    CUDA_CHECK(cudaMalloc((void**) &d_vec_block, sizeof(float) * num_blocks));

    cudaEventRecord(dstart, 0);
    sum_reduce_wrapper(d_vec_block, d_vec, N, num_blocks, num_threads);
    cudaEventRecord(dstop, 0);
    cudaEventSynchronize(dstop);
    cudaEventElapsedTime(&dtime, dstart, dstop);
    printf("sum reduce (device) - Elapsed time:  %3.3f ms \n", dtime);

    // --- the last part of the reduction, which would be expensive to perform on the device, is
    // executed on the host

    float* h_vec_block = new float[num_blocks];
    CUDA_CHECK(
      cudaMemcpy(h_vec_block, d_vec_block, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float d_sum = 0.f;
    for (int i = 0; i < num_blocks; i++) {
      d_sum += h_vec_block[i];
    }

    entries[i] = d_sum;

    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_vec_block));
    delete h_vec_block;
  }

  for (int i = 0; i < num_experiments; ++i) {
    std::cerr << entries[i] << std::endl;
  }

  for (int i = 1; i < num_experiments; ++i) {
    std::cerr << entries[0] << " " << entries[i] << std::endl;
    ASSERT_EQ(entries[0], entries[i]);
  }
}

int main(int argc, char** argv) {
  return srrg2_test::runTests(argc, argv, true /*use test folder*/);
}
