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
#include <srrg_geometry/geometry_defs.h>

namespace md_slam {
  using namespace srrg2_core;

  // linear system vector reflecting upper
  // triangular part of hessian
  using Vector21d = Eigen::Matrix<double, 21, 1>;

  struct LinearSystemEntry {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vector21d upperH;
    Vector6d b;
    size_t is_good;
    size_t is_inlier;
    float chi;

    __host__ __device__ LinearSystemEntry() {
      upperH.setZero(); // we store only the upper triangular of H
      b.setZero();
      chi       = 0.f;
      is_good   = 0;
      is_inlier = 0;
    }

    __host__ __device__ LinearSystemEntry(const LinearSystemEntry& entry_) {
      upperH    = entry_.upperH;
      b         = entry_.b;
      chi       = entry_.chi;
      is_good   = entry_.is_good;
      is_inlier = entry_.is_inlier;
    }

    __host__ __device__ LinearSystemEntry& operator+=(const LinearSystemEntry& other_) {
      this->upperH += other_.upperH;
      this->b += other_.b;
      this->chi += other_.chi;
      this->is_good += other_.is_good;
      this->is_inlier += other_.is_inlier;
      return *this;
    }

    __host__ __device__ LinearSystemEntry& operator=(const LinearSystemEntry& other_) {
      this->upperH    = other_.upperH;
      this->b         = other_.b;
      this->chi       = other_.chi;
      this->is_good   = other_.is_good;
      this->is_inlier = other_.is_inlier;
      return *this;
    }

    __host__ __device__ LinearSystemEntry& operator+(const LinearSystemEntry& other_) {
      this->upperH += other_.upperH;
      this->b += other_.b;
      this->chi += other_.chi;
      this->is_good += other_.is_good;
      this->is_inlier += other_.is_inlier;
      return *this;
    }

    __host__ static LinearSystemEntry Ones() {
      LinearSystemEntry e;
      e.upperH.setOnes();
      e.b.setOnes();
      e.chi = 1.f;
      return e;
    }

    __host__ static LinearSystemEntry Random() {
      LinearSystemEntry e;
      e.upperH.setRandom();
      e.b.setRandom();
      e.chi = float(rand()) / float((RAND_MAX));
      return e;
    }
  };

  struct LinearSystemEntryBi : public LinearSystemEntry {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Matrix6d Hij;
    Vector21d upperHjj;
    Vector6d bj;

    __host__ __device__ LinearSystemEntryBi() : LinearSystemEntry() {
      upperHjj.setZero();
      Hij.setZero();
      bj.setZero();
    }

    __host__ __device__ LinearSystemEntryBi(const LinearSystemEntryBi& entry_) :
      LinearSystemEntry(entry_) {
      Hij      = entry_.Hij;
      upperHjj = entry_.upperHjj;
      bj       = entry_.bj;
    }

    __host__ __device__ LinearSystemEntryBi& operator+=(const LinearSystemEntryBi& other_) {
      LinearSystemEntry::operator+=(other_);
      this->Hij += other_.Hij;
      this->upperHjj += other_.upperHjj;
      this->bj += other_.bj;
      return *this;
    }

    __host__ __device__ LinearSystemEntryBi& operator=(const LinearSystemEntryBi& other_) {
      LinearSystemEntry::operator=(other_);
      this->Hij                  = other_.Hij;
      this->upperHjj             = other_.upperHjj;
      this->bj                   = other_.bj;
      return *this;
    }

    __host__ __device__ LinearSystemEntryBi& operator+(const LinearSystemEntryBi& other_) {
      LinearSystemEntry::operator+(other_);
      this->Hij += other_.Hij;
      this->upperHjj += other_.upperHjj;
      this->bj += other_.bj;
      return *this;
    }

    __host__ static LinearSystemEntryBi Ones() {
      LinearSystemEntryBi e;
      e.upperH.setOnes();
      e.Hij.setOnes();
      e.upperHjj.setOnes();
      e.b.setOnes();
      e.bj.setOnes();
      e.chi = 1.f;
      return e;
    }

    __host__ static LinearSystemEntryBi Random() {
      LinearSystemEntryBi e;
      e.upperH.setRandom();
      e.Hij.setRandom();
      e.upperHjj.setRandom();
      e.b.setRandom();
      e.bj.setRandom();
      e.chi = float(rand()) / float((RAND_MAX));
      return e;
    }
  };

} // namespace md_slam