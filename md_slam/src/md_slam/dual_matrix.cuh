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
#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include <iterator>
#include <srrg_data_structures/matrix.h>

#define N_THREADS 256


namespace md_slam {

  enum MemType { Host = 0, Device = 1 };

  // TODO DualMat mat; mat.resize(100, 100); mat = DualMat(1, 1); valgrind not passing do copy
  // constructor

  template <typename CellType_>
  class DualMatrix_ {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using ThisType = DualMatrix_<CellType_>;
    using CellType = CellType_;

    inline void _copyHeader() {
      _n_threads = N_THREADS;
      _n_blocks  = (_capacity + _n_threads - 1) / _n_threads;
      // once class fields are populated copy ptr on device
      CUDA_CHECK(cudaMemcpy(_device_instance, this, sizeof(ThisType), cudaMemcpyHostToDevice));
    }

    DualMatrix_(const size_t rows_, const size_t cols_) {
      CUDA_CHECK(cudaMalloc((void**) &_device_instance, sizeof(ThisType)));
      resize(rows_, cols_);
    }

    DualMatrix_() :
      _buffers{nullptr, nullptr},
      _device_instance(nullptr),
      _rows(0),
      _cols(0),
      _capacity(0) {
      CUDA_CHECK(cudaMalloc((void**) &_device_instance, sizeof(ThisType)));
      CUDA_CHECK(cudaMemcpy(_device_instance, this, sizeof(ThisType), cudaMemcpyHostToDevice));
    }

    DualMatrix_(const DualMatrix_& src_) : DualMatrix_(src_._rows, src_._cols) {
      memcpy(_buffers[Host], src_._buffers[Host], sizeof(CellType) * _capacity);
      CUDA_CHECK(cudaMemcpy(_buffers[Device],
                            src_._buffers[Device],
                            sizeof(CellType) * _capacity,
                            cudaMemcpyDeviceToDevice));
    }

    DualMatrix_& operator=(const DualMatrix_& src_) {
      resize(src_._rows, src_._cols);
      memcpy(_buffers[Host], src_._buffers[Host], sizeof(CellType) * _capacity);
      CUDA_CHECK(cudaMemcpy(_buffers[Device],
                            src_._buffers[Device],
                            sizeof(CellType) * _capacity,
                            cudaMemcpyDeviceToDevice));
      return *this;
    }

    inline void _sync() {
      if (_capacity == _rows * _cols) {
        _copyHeader();
        return;
      }

      if (_buffers[Device]) {
        cudaFree(_buffers[Device]);
        _buffers[Device] = nullptr;
      }

      if (_buffers[Host]) {
        delete[] _buffers[Host];
        _buffers[Host] = nullptr;
      }
      _capacity = _rows * _cols;
      if (_capacity) {
        _buffers[Host] = new CellType[_capacity];
        CUDA_CHECK(cudaMalloc((void**) &_buffers[Device], sizeof(CellType) * _capacity));
        CUDA_CHECK(cudaMemcpy(
          _buffers[Device], _buffers[Host], sizeof(CellType) * _capacity, cudaMemcpyHostToDevice));
      }

      _copyHeader();
    }

    ~DualMatrix_() {
      if (_device_instance)
        cudaFree(_device_instance);
      if (_buffers[Host])
        delete[] _buffers[Host];
      if (_buffers[Device])
        cudaFree(_buffers[Device]);
    }

    __host__ inline void resize(const size_t rows_, const size_t cols_) {
      // if size is ok, do nothing
      if (rows_ == _rows && cols_ == _cols)
        return;
      _rows = rows_;
      _cols = cols_;
      _sync();
    }

    // clang-format off
    __host__ __device__ inline ThisType* deviceInstance() { return _device_instance; }
    __host__ inline const ThisType* deviceInstance() const { return _device_instance; } 
    __host__ void fill(const CellType& value_, const bool device_only_ = false);
    __host__ inline const size_t nThreads() const { return _n_threads; }
    __host__ inline const size_t nBlocks() const { return _n_blocks; }
    __host__ __device__ inline const size_t rows() const { return _rows; }
    __host__ __device__ inline const size_t cols() const { return _cols; }
    __host__ __device__ inline const size_t size() const { return _capacity; }
    __host__ __device__ inline const bool empty() const { return _capacity == 0; };
    // clang-format on

    __host__ __device__ inline bool inside(const size_t row_, const size_t col_) const {
      return row_ >= 0 && col_ >= 0 && row_ < _rows && col_ < _cols;
    }

    __host__ __device__ inline bool onBorder(const size_t row_, const size_t col_) const {
      return row_ == 0 || col_ == 0 || row_ == _rows - 1 || col_ == _cols - 1;
    }

    template <int MemType = 0>
    __host__ __device__ inline const CellType& at(const size_t index_) const {
      return _buffers[MemType][index_];
    }

    template <int MemType = 0>
    __host__ __device__ inline CellType& at(const size_t index_) {
      return _buffers[MemType][index_];
    }

    template <int MemType = 0>
    __host__ __device__ inline const CellType& at(const size_t row_, const size_t col_) const {
      return _buffers[MemType][row_ * _cols + col_];
    }

    template <int MemType = 0>
    __host__ __device__ inline CellType& at(const size_t row_, const size_t col_) {
      return _buffers[MemType][row_ * _cols + col_];
    }

    template <int MemType = 0>
    __host__ __device__ inline CellType& operator()(const size_t row_, const size_t col_) {
      return _buffers[MemType][row_ * _cols + col_];
    }

    template <int MemType = 0>
    __host__ __device__ inline CellType& operator[](const size_t index_) {
      return _buffers[MemType][index_];
    }

    template <int MemType = 0>
    __host__ __device__ inline const CellType& operator[](const size_t index_) const {
      return _buffers[MemType][index_];
    }

    template <int MemType = 0>
    __host__ __device__ inline const CellType* data() const {
      return _buffers[MemType];
    }

    template <int MemType = 0>
    __host__ __device__ inline CellType* data() {
      return _buffers[MemType];
    }

    // copy whole device buffer to host, for debugging at the moment
    __host__ inline void toHost() {
      CUDA_CHECK(cudaMemcpy(
        _buffers[Host], _buffers[Device], sizeof(CellType) * _capacity, cudaMemcpyDeviceToHost));
    }

    __host__ inline void toDevice() {
      CUDA_CHECK(cudaMemcpy(
        _buffers[Device], _buffers[Host], sizeof(CellType) * _capacity, cudaMemcpyHostToDevice));
    }

    __host__ inline void clearDeviceBuffer() {
      if (_buffers[Device])
        cudaFree(_buffers[Device]);
    }

    CellType* _buffers[2]      = {nullptr, nullptr};
    ThisType* _device_instance = nullptr;
    size_t _rows               = 0;
    size_t _cols               = 0;
    size_t _capacity           = 0;
    size_t _n_threads          = 0;
    size_t _n_blocks           = 0;
  };
} // namespace md_slam
