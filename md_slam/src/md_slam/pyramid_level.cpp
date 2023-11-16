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

#include "pyramid_level.cuh"
#include <srrg_system_utils/char_array.h>

namespace md_slam {
  using namespace srrg2_core;

  MDPyramidLevel::~MDPyramidLevel() {
  }

  //! generates a scaled pyramid from this one. Scale should be an int
  void MDPyramidLevel::writeToCharArray(char*& dest, size_t& size) const {
    srrg2_core::writeToCharArray(dest, size, rows());
    srrg2_core::writeToCharArray(dest, size, cols());
    srrg2_core::writeToCharArray(dest, size, min_depth);
    srrg2_core::writeToCharArray(dest, size, max_depth);
    srrg2_core::writeToCharArray(dest, size, camera_matrix);
    srrg2_core::writeToCharArray(dest, size, sensor_offset);
    srrg2_core::writeToCharArray(dest, size, camera_type);
    for (int i = 0; i < 3; ++i)
      srrg2_core::writeToCharArray(dest, size, thresholds[i]);
    for (int i = 0; i < 3; ++i)
      srrg2_core::writeToCharArray(dest, size, policies[i]);
    srrg2_core::writeToCharArray(dest, size, mask_grow_radius);
    for (size_t r = 0; r < rows(); ++r) {
      for (size_t c = 0; c < cols(); ++c) {
        const MDPyramidMatrixEntry& entry = matrix.at(r, c);
        float v                           = 0;
        if (entry.masked()) {
          srrg2_core::writeToCharArray(dest, size, v);
          continue;
        }
        srrg2_core::writeToCharArray(dest, size, entry.depth());
        srrg2_core::writeToCharArray(dest, size, entry.intensity());
        for (int i = 0; i < 3; ++i) {
          srrg2_core::writeToCharArray(dest, size, entry.value[2 + i]);
        }
      }
    }
  }

  //! reads a pyramid from a byte array
  void MDPyramidLevel::readFromCharArray(const char*& src, size_t& size) {
    size_t rows_, cols_;
    srrg2_core::readFromCharArray(rows_, src, size);
    srrg2_core::readFromCharArray(cols_, src, size);
    resize(rows_, cols_);
    srrg2_core::readFromCharArray(min_depth, src, size);
    srrg2_core::readFromCharArray(max_depth, src, size);
    srrg2_core::readFromCharArray(camera_matrix, src, size);
    srrg2_core::readFromCharArray(sensor_offset, src, size);
    srrg2_core::readFromCharArray(camera_type, src, size);
    for (int i = 0; i < 3; ++i)
      srrg2_core::readFromCharArray(thresholds[i], src, size);
    for (int i = 0; i < 3; ++i)
      srrg2_core::readFromCharArray(policies[i], src, size);
    srrg2_core::readFromCharArray(mask_grow_radius, src, size);
    for (size_t r = 0; r < rows(); ++r) {
      for (size_t c = 0; c < cols(); ++c) {
        MDPyramidMatrixEntry& entry = matrix.at(r, c);
        float v;
        srrg2_core::readFromCharArray(v, src, size);
        if (v == 0) {
          entry.setMasked(true);
          continue;
        }
        entry.setMasked(false);
        entry.setDepth(v);
        srrg2_core::readFromCharArray(v, src, size);
        entry.setIntensity(v);
        for (int i = 0; i < 3; ++i) {
          srrg2_core::readFromCharArray(entry.value[2 + i], src, size);
        }
      }
    }
    updateDerivatives();
  }

  void MDPyramidLevel::write(std::ostream& os) const {
    static constexpr size_t b_size = 1024 * 1024 * 50; // 50 MB of buffer
    char* buffer                   = new char[b_size];
    size_t size                    = b_size;
    char* buffer_end               = buffer;
    writeToCharArray(buffer_end, size);
    size_t real_size = buffer_end - buffer;
    os.write(buffer, real_size);
    delete[] buffer;
  }

  bool MDPyramidLevel::read(std::istream& is) {
    static constexpr size_t b_size = 1024 * 1024 * 50; // 50 MB of buffer
    char* buffer                   = new char[b_size];
    is.read(buffer, b_size);
    size_t real_size       = is.gcount();
    const char* buffer_end = buffer;
    this->readFromCharArray(buffer_end, real_size);
    delete[] buffer;
    return true;
  }

  MDPyramidLevel::MDPyramidLevel(size_t rows_, size_t cols_) {
    resize(rows_, cols_);
    // sensor_offset = Isometry3f::Identity();
  }

  void MDPyramidLevel::resize(const size_t& rows_, const size_t& cols_) {
    matrix.resize(rows_, cols_);
  }

  template <typename Matrix_>
  void applyPolicy(MDPyramidMatrixEntry& entry,
                   Matrix_&& m,
                   FilterPolicy policy,
                   float squared_threshold) {
    if (entry.masked())
      return;

    float n = m.squaredNorm();
    if (n < squared_threshold)
      return;

    switch (policy) {
      case Suppress:
        entry.setMasked(1);
        break;
      case Clamp:
        m *= sqrt(squared_threshold / n);
        break;
      default:;
    }
  }

  void MDPyramidLevel::updateDerivatives() {
    // these are for the normalization
    const float i2 = pow(thresholds[Intensity], 2);
    const float d2 = pow(thresholds[Depth], 2);
    const float n2 = pow(thresholds[Normal], 2);

    const size_t& rows = matrix.rows();
    const size_t& cols = matrix.cols();
    // we start from 1st row
    for (size_t r = 1; r < rows - 1; ++r) {
      // fetch the row vectors
      // in the iteration below we start from the 1st column
      // so we increment the pointers by 1

      for (size_t c = 1; c < cols - 1; ++c) {
        MDPyramidMatrixEntry& entry          = matrix.at(r, c);
        const MDPyramidMatrixEntry& entry_r0 = matrix.at(r - 1, c);
        const MDPyramidMatrixEntry& entry_r1 = matrix.at(r + 1, c);
        const MDPyramidMatrixEntry& entry_c0 = matrix.at(r, c - 1);
        const MDPyramidMatrixEntry& entry_c1 = matrix.at(r, c + 1);

        // retrieve value
        const Vector5f& v_r0 = entry_r0.value;
        const Vector5f& v_r1 = entry_r1.value;
        const Vector5f& v_c0 = entry_c0.value;
        const Vector5f& v_c1 = entry_c1.value;

        // compute derivatives
        Matrix5_2f& derivatives = entry.derivatives;
        derivatives.col(1)      = .5 * v_r1 - .5 * v_r0;
        derivatives.col(0)      = .5 * v_c1 - .5 * v_c0;

        // here we ignore, clamp or suppress
        // the derivatives according to the selected policy
        applyPolicy(entry, derivatives.row(0), policies[Intensity], i2);
        applyPolicy(entry, derivatives.row(1), policies[Depth], d2);
        applyPolicy(entry, derivatives.block<3, 2>(2, 0), policies[Normal], n2);
      }
    }
  }

  void MDPyramidLevel::getIntensity(ImageFloat& intensity) const {
    intensity.resize(rows(), cols());
    intensity.fill(0);
    for (size_t k = 0; k < matrix.size(); ++k)
      intensity.at(k) = matrix.at(k).intensity();
  }

  void MDPyramidLevel::getDepth(ImageFloat& depth) const {
    depth.resize(rows(), cols());
    depth.fill(0);
    for (size_t k = 0; k < matrix.size(); ++k)
      depth.at(k) = matrix.at(k).depth();
  }

  void MDPyramidLevel::getNormals(ImageVector3f& normals) const {
    normals.resize(rows(), cols());
    for (size_t k = 0; k < matrix.size(); ++k)
      normals.at(k) = matrix.at(k).normal();
  }

  void MDPyramidLevel::growMask() {
    const int& radius = mask_grow_radius;
    std::vector<int> ball_offsets;
    int r2 = pow(radius, 2);
    for (int r = -radius; r < radius + 1; ++r) {
      for (int c = -radius; c < radius + 1; ++c) {
        int idx = r * cols() + c;
        if ((r * r + c * c) <= r2) {
          ball_offsets.push_back(idx);
        }
      }
    }

    ImageUInt8 mask(rows(), cols());
    for (size_t i = 0; i < mask.size(); ++i)
      mask[i] = matrix.at(i).masked();
    for (size_t i = 0; i < mask.size(); ++i) {
      if (!mask[i])
        continue;
      for (auto offset : ball_offsets) {
        int target = offset + i;
        if (target < 0 || target >= (int) matrix.size())
          continue;
        matrix.at(target).setMasked(true);
      }
    }
  }

} // namespace md_slam
