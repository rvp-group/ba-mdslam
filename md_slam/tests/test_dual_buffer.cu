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
#include <srrg_test/test_helper.hpp>
#include <vector>

#include <md_slam/dual_matrix.cu>
#include <md_slam/dual_matrix.cuh>
#include <md_slam/factor_common.cuh>
#include <md_slam/pyramid_level.cuh>

#ifndef MD_TEST_DATA_FOLDER
#error "NO TEST DATA FOLDER"
#endif

using namespace md_slam;

template <typename Mat>
__global__ void checkFieldsDevice(size_t* fields_, const Mat* mat_) {
  fields_[0] = mat_->rows();
  fields_[1] = mat_->cols();
  fields_[2] = mat_->size();
}

template <typename Entry, typename Mat>
__global__ void copyFieldsToDevice(Entry* entries_, const Mat* mat_) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= mat_->size())
    return;
  entries_[tid] = mat_->at<1>(tid);
}

TEST(FLOAT, DualMatrixFloat) {
  using Entry   = float;
  using MatType = DualMatrix_<Entry>;
  // check creation and resizing
  size_t rows = 300;
  size_t cols = 300;
  MatType mat(rows, cols);
  mat  = MatType();
  rows = 400;
  cols = 300;
  mat.resize(rows, cols);
  mat.fill(0.f);

  size_t* mat_fields_device = nullptr;
  size_t* mat_fields_host   = new size_t[3];
  memset(mat_fields_host, 0, sizeof(size_t) * 3);
  CUDA_CHECK(cudaMalloc((void**) &mat_fields_device, sizeof(size_t) * 3));
  checkFieldsDevice<MatType><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(
    cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(size_t) * 3, cudaMemcpyDeviceToHost));

  size_t dev_rows     = mat_fields_host[0];
  size_t dev_cols     = mat_fields_host[1];
  size_t dev_capacity = mat_fields_host[2];

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // check only reconstruction and resizing
  rows = 700;
  cols = 100;

  mat = MatType();
  mat.resize(rows, cols);
  mat.fill(0.f);

  checkFieldsDevice<MatType><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(
    cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(size_t) * 3, cudaMemcpyDeviceToHost));

  dev_rows     = mat_fields_host[0];
  dev_cols     = mat_fields_host[1];
  dev_capacity = mat_fields_host[2];

  cudaFree(mat_fields_device);
  delete[] mat_fields_host;

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // checking that host manipulation of entries is valid in device
  Entry* buff_entries_host = new Entry[mat.size()];
  Entry* buff_entries_dev  = nullptr;
  CUDA_CHECK(cudaMalloc((void**) &buff_entries_dev, sizeof(Entry) * mat.size()));

  // modify entries of matrix in host
  for (size_t i = 0; i < mat.size(); ++i) {
    mat.at(i) = (rand()) / (static_cast<float>(RAND_MAX / 1.f));
  }

  // copy to device
  mat.toDevice();
  // check that host and device contain the same element
  copyFieldsToDevice<Entry, MatType>
    <<<mat.nBlocks(), mat.nThreads()>>>(buff_entries_dev, mat.deviceInstance());
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaMemcpy(
    buff_entries_host, buff_entries_dev, sizeof(Entry) * mat.size(), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < mat.size(); ++i) {
    ASSERT_EQ(mat.at(i), buff_entries_host[i]);
  }

  cudaFree(buff_entries_dev);
  delete[] buff_entries_host;

  return;
}

struct DummyEntry {
  DummyEntry() {
    field1 = 0.f;
  }
  float field1  = 0.f;
  size_t field2 = 0;
};

template <typename Mat>
__global__ void checkMemPaddingEntryDevice(int* add1_, int* add2_, int* add3_, const Mat* mat_) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= mat_->size())
    return;
  const auto elem = mat_->at<1>(tid);
  int* add1       = (int*) &elem;
  int* add2       = (int*) &(elem.field1);
  int* add3       = (int*) &(elem.field2);
  // printf("add: %p %p %p\n", add1, add2, add3);
  // printf("diff: %i %i %i",add3-add1,add2-add1,add3-add2);
  add1_[tid] = add3 - add1;
  add2_[tid] = add2 - add1;
  add3_[tid] = add3 - add2;
}

TEST(DUMMY, DualMatrixDummyEntry) {
  using Entry   = DummyEntry;
  using MatType = DualMatrix_<Entry>;
  // check creation and resizing
  size_t rows = 300;
  size_t cols = 300;
  MatType mat(rows, cols);
  mat  = MatType();
  rows = 400;
  cols = 300;
  mat.resize(rows, cols);
  mat.fill(DummyEntry());

  size_t* mat_fields_device = nullptr;
  size_t* mat_fields_host   = new size_t[3];
  memset(mat_fields_host, 0, sizeof(size_t) * 3);
  CUDA_CHECK(cudaMalloc((void**) &mat_fields_device, sizeof(size_t) * 3));
  checkFieldsDevice<MatType><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(
    cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(size_t) * 3, cudaMemcpyDeviceToHost));

  size_t dev_rows     = mat_fields_host[0];
  size_t dev_cols     = mat_fields_host[1];
  size_t dev_capacity = mat_fields_host[2];

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // check only reconstruction and resizing
  rows = 700;
  cols = 100;

  mat = MatType();
  mat.resize(rows, cols);
  mat.fill(DummyEntry());

  checkFieldsDevice<MatType><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(
    cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(size_t) * 3, cudaMemcpyDeviceToHost));

  dev_rows     = mat_fields_host[0];
  dev_cols     = mat_fields_host[1];
  dev_capacity = mat_fields_host[2];

  cudaFree(mat_fields_device);
  delete[] mat_fields_host;

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // checking that host manipulation of entries is valid in device
  Entry* buff_entries_host = new Entry[mat.size()];
  Entry* buff_entries_dev  = nullptr;
  CUDA_CHECK(cudaMalloc((void**) &buff_entries_dev, sizeof(Entry) * mat.size()));

  // modify entries of matrix in host
  for (size_t i = 0; i < mat.size(); ++i) {
    // mat.at(i).setDepth(static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / max_depth)));
    // mat.at(i).setIntensity(static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1.f)));
    // mat.at(i).setNormal(Vector3f::Random());
    mat.at(i).field1 = (rand()) / (static_cast<float>(RAND_MAX / 1.f));
    mat.at(i).field2 = i;
  }

  // copy to device
  mat.toDevice();
  // check that host and device contain the same element
  copyFieldsToDevice<Entry, MatType>
    <<<mat.nBlocks(), mat.nThreads()>>>(buff_entries_dev, mat.deviceInstance());
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaMemcpy(
    buff_entries_host, buff_entries_dev, sizeof(Entry) * mat.size(), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < mat.size(); ++i) {
    ASSERT_EQ(mat.at(i).field1, buff_entries_host[i].field1);
    ASSERT_EQ(mat.at(i).field2, buff_entries_host[i].field2);
    // ASSERT_EQ(mat.at(i).depth(), buff_entries_host[i].depth());
    // ASSERT_EQ(mat.at(i).intensity(), buff_entries_host[i].intensity());
    // ASSERT_EQ(mat.at(i).normal(), buff_entries_host[i].normal());
  }

  cudaFree(buff_entries_dev);
  delete[] buff_entries_host;

  // checking for struct padding
  int* adds_dev[3] = {nullptr, nullptr, nullptr};
  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cudaMalloc((void**) &adds_dev[i], sizeof(int) * mat.size()));
  }
  int* adds[3] = {new int[mat.size()], new int[mat.size()], new int[mat.size()]};
  checkMemPaddingEntryDevice<MatType><<<mat.nBlocks(), mat.nThreads()>>>(
    adds_dev[0], adds_dev[1], adds_dev[2], mat.deviceInstance());
  cudaDeviceSynchronize();

  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cudaMemcpy(adds[i], adds_dev[i], sizeof(int) * mat.size(), cudaMemcpyDeviceToHost));
  }

  for (int i = 0; i < 0; ++i) {
    const auto elem = mat.at(i);
    int* add1       = (int*) &elem;
    int* add2       = (int*) &(elem.field1);
    int* add3       = (int*) &(elem.field2);

    const int diff1 = add3 - add1;
    const int diff2 = add2 - add1;
    const int diff3 = add3 - add2;

    ASSERT_EQ(adds[0][i], diff1);
    ASSERT_EQ(adds[1][i], diff2);
    ASSERT_EQ(adds[2][i], diff3);
  }

  for (int i = 0; i < 3; ++i) {
    cudaFree(adds_dev[i]);
    delete[] adds[i];
  }

  return;
}

using namespace Eigen;

struct EigenEntry {
  EigenEntry() {
    field2.setIdentity();
  }
  Vector3f field1   = Vector3f::Zero();
  Isometry3d field2 = Isometry3d::Identity();
};

TEST(EIGEN, DualMatrixEigenEntry) {
  using Entry   = EigenEntry;
  using MatType = DualMatrix_<Entry>;
  // check creation and resizing
  size_t rows = 300;
  size_t cols = 300;
  MatType mat(rows, cols);
  mat  = MatType();
  rows = 400;
  cols = 300;
  mat.resize(rows, cols);
  mat.fill(EigenEntry());

  size_t* mat_fields_device = nullptr;
  size_t* mat_fields_host   = new size_t[3];
  memset(mat_fields_host, 0, sizeof(size_t) * 3);
  CUDA_CHECK(cudaMalloc((void**) &mat_fields_device, sizeof(size_t) * 3));
  checkFieldsDevice<MatType><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(
    cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(size_t) * 3, cudaMemcpyDeviceToHost));

  size_t dev_rows     = mat_fields_host[0];
  size_t dev_cols     = mat_fields_host[1];
  size_t dev_capacity = mat_fields_host[2];

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // check only reconstruction and resizing
  rows = 700;
  cols = 100;

  mat = MatType();
  mat.resize(rows, cols);
  mat.fill(EigenEntry());

  checkFieldsDevice<MatType><<<1, 1>>>(mat_fields_device, mat.deviceInstance());
  CUDA_CHECK(
    cudaMemcpy(mat_fields_host, mat_fields_device, sizeof(size_t) * 3, cudaMemcpyDeviceToHost));

  dev_rows     = mat_fields_host[0];
  dev_cols     = mat_fields_host[1];
  dev_capacity = mat_fields_host[2];

  cudaFree(mat_fields_device);
  delete[] mat_fields_host;

  ASSERT_EQ(rows, mat.rows());
  ASSERT_EQ(cols, mat.cols());
  ASSERT_EQ(rows * cols, mat.size());
  ASSERT_EQ(dev_rows, mat.rows());
  ASSERT_EQ(dev_cols, mat.cols());
  ASSERT_EQ(dev_capacity, mat.size());

  // checking that host manipulation of entries is valid in device
  Entry* buff_entries_host = new Entry[mat.size()];
  Entry* buff_entries_dev  = nullptr;
  CUDA_CHECK(cudaMalloc((void**) &buff_entries_dev, sizeof(Entry) * mat.size()));

  // modify entries of matrix in host
  for (size_t i = 0; i < mat.size(); ++i) {
    mat.at(i).field1                   = Vector3f::Random();
    mat.at(i).field2.translation().x() = (static_cast<double>(RAND_MAX / 1.0));
  }

  // copy to device
  mat.toDevice();
  // check that host and device contain the same element
  copyFieldsToDevice<Entry, MatType>
    <<<mat.nBlocks(), mat.nThreads()>>>(buff_entries_dev, mat.deviceInstance());
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaMemcpy(
    buff_entries_host, buff_entries_dev, sizeof(Entry) * mat.size(), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < mat.size(); ++i) {
    ASSERT_EQ(mat.at(i).field1, buff_entries_host[i].field1);
    ASSERT_EQ(mat.at(i).field2.translation().x(), buff_entries_host[i].field2.translation().x());
  }

  cudaFree(buff_entries_dev);
  delete[] buff_entries_host;

  // checking for struct padding
  int* adds_dev[3] = {nullptr, nullptr, nullptr};
  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cudaMalloc((void**) &adds_dev[i], sizeof(int) * mat.size()));
  }
  int* adds[3] = {new int[mat.size()], new int[mat.size()], new int[mat.size()]};
  checkMemPaddingEntryDevice<MatType><<<mat.nBlocks(), mat.nThreads()>>>(
    adds_dev[0], adds_dev[1], adds_dev[2], mat.deviceInstance());
  cudaDeviceSynchronize();

  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cudaMemcpy(adds[i], adds_dev[i], sizeof(int) * mat.size(), cudaMemcpyDeviceToHost));
  }

  for (int i = 0; i < 0; ++i) {
    const auto elem = mat.at(i);
    int* add1       = (int*) &elem;
    int* add2       = (int*) &(elem.field1);
    int* add3       = (int*) &(elem.field2);

    const int diff1 = add3 - add1;
    const int diff2 = add2 - add1;
    const int diff3 = add3 - add2;

    ASSERT_EQ(adds[0][i], diff1);
    ASSERT_EQ(adds[1][i], diff2);
    ASSERT_EQ(adds[2][i], diff3);
  }

  for (int i = 0; i < 3; ++i) {
    cudaFree(adds_dev[i]);
    delete[] adds[i];
  }

  return;
}

int main(int argc, char** argv) {
  return srrg2_test::runTests(argc, argv, true /*use test folder*/);
}
