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
#include "dual_matrix.cuh"
#include "utils.cuh"
#include <iostream>
#include <srrg_boss/blob.h>
#include <srrg_geometry/geometry_defs.h>
#include <srrg_image/image.h>
#include <srrg_pcl/point_types.h>
#include <vector>

namespace md_slam {

  using srrg2_core::ImageFloat;
  using srrg2_core::ImageUInt8;
  using srrg2_core::ImageVector3f;
  using srrg2_core::Isometry3f;
  using srrg2_core::Matrix3f;
  using srrg2_core::Matrix5_2f;
  using srrg2_core::PointNormalIntensity3f;
  using srrg2_core::Vector2f;
  using srrg2_core::Vector5f;
  // TODO this needs to be modified, pcl needs to be moved entirely in GPU
  using MDMatrixVectorCloud =
    srrg2_core::PointCloud_<srrg2_core::Matrix_<srrg2_core::PointNormalIntensity3f,
                                                Eigen::aligned_allocator<PointNormalIntensity3f>>>;
  using MDMatrixCloud = DualMatrix_<srrg2_core::PointNormalIntensity3f>;
  using MDVectorCloud = srrg2_core::PointNormalIntensity3fVectorCloud;

  enum ChannelType { Intensity = 0x0, Depth = 0x1, Normal = 0x2 };
  enum FilterPolicy { Ignore = 0, Suppress = 1, Clamp = 2 };
  enum PointStatusFlag {
    Good            = 0x00,
    Outside         = 0x1,
    DepthOutOfRange = 0x2,
    Masked          = 0x3,
    Occluded        = 0x4,
    DepthError      = 0x5,
    Invalid         = 0x7
  };

  // MDPyramidMatrixEntry: represents an entry of a MD Image,
  // in terms of both point and derivatives
  struct MDPyramidMatrixEntry {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Matrix5_2f derivatives; // col and row derivatives
    Vector5f value;         // [i, d, nx, ny, nz]: intensity, depth, normal
#ifdef _MD_ENABLE_SUPERRES_
    float r = 0, c = 0; // subpixel coordinates
#endif
    bool _masked = true;
    MDPyramidMatrixEntry() {
      value.setZero();
      derivatives.setZero();
    }
    // clang-format off
    __host__ __device__ inline float intensity() const { return value(0); }
    __host__ __device__ inline void setIntensity(float intensity_) { value(0) = intensity_; }
    __host__ __device__ inline float depth() const { return value(1); }
    __host__ __device__ inline void setDepth(float depth_) { value(1) = depth_; }
    __host__ __device__ inline Eigen::Vector3f normal() const { return value.block<3, 1>(2, 0); }
    __host__ __device__ inline void setNormal(const Eigen::Vector3f& n) { value.block<3, 1>(2, 0) = n; }
    __host__ __device__ inline bool masked() const { return _masked; }
    __host__ __device__ inline void setMasked(bool masked_) { _masked = masked_; }
    // clang-format on
  };

  using MDPyramidMatrix = DualMatrix_<MDPyramidMatrixEntry>;

  __device__ __forceinline__ bool getSubPixel(Vector5f& value_,
                                              Matrix5_2f& derivative_,
                                              const MDPyramidMatrix* mat_,
                                              const Vector2f& image_point_) {
    float c = image_point_.x();
    float r = image_point_.y();
    int r0  = (int) r;
    int c0  = (int) c;
    if (!mat_->inside(r0, c0))
      return false;

    int r1 = r0 + 1;
    int c1 = c0 + 1;
    if (!mat_->inside(r1, c1))
      return false;

    const MDPyramidMatrixEntry p00 = mat_->at<1>(r0, c0);
    const MDPyramidMatrixEntry p01 = mat_->at<1>(r0, c1);
    const MDPyramidMatrixEntry p10 = mat_->at<1>(r1, c0);
    const MDPyramidMatrixEntry p11 = mat_->at<1>(r1, c1);
    if (p00.masked() || p01.masked() || p10.masked() || p11.masked())
      return false;

    const float dr  = r - (float) r0;
    const float dc  = c - (float) c0;
    const float dr1 = 1.f - dr;
    const float dc1 = 1.f - dc;

    value_ = (p00.value * dc1 + p01.value * dc) * dr1 + (p10.value * dc1 + p11.value * dc) * dr;

    derivative_ = (p00.derivatives * dc1 + p01.derivatives * dc) * dr1 +
                  (p10.derivatives * dc1 + p11.derivatives * dc) * dr;

    return true;
  }

  // MDPyramidLevel: contains all the data of a specific pyramid level, i.e.
  //  - the image as a MDPyramidMatrix
  //  - a mask of valid points
  //  - camera matrix at this level of pyramid
  //  - image size
  //  - the corresponding cloud
  // the MDPyramidGenerator takes care of initialize Levels
  class MDPyramidLevel : public srrg2_core::BLOB {
    friend class MDImagePyramid;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ~MDPyramidLevel();

    //! @brief PyramidLevel c'tor
    MDPyramidLevel(size_t rows_ = 0, size_t cols_ = 0);

    //! @brief resizes image and mask
    void resize(const size_t& rows_, const size_t& cols_);

    //! @brief get the MDPyramidMatrixEntry obtained with bilinear interpolation. False if outside
    // __device__ inline bool
    // getSubPixel(Vector5f& v, Matrix5_2f& d, const Vector2f& image_point) const;

    //! @brief rows of a level
    const size_t rows() const {
      return matrix.rows();
    }
    //! @brief cols of a level
    const size_t cols() const {
      return matrix.cols();
    }

    //! @brief generates a cloud based on the pyramid
    void toCloud(MDMatrixVectorCloud& target) const; // puts a cloud rendered from self in target
    __host__ void toCloudDevice(MDMatrixCloud* target) const; // does the same but in device

    //! @brief generates the level based on the cloud
    //! the matrix should be resized before calling the method
    //! uses the embedded parameters
    void fromCloud(MDMatrixVectorCloud& src_cloud); // fills own values from src_cloud

    //! @brief produces a 3x3  tiled image of the pyramid for debug
    void toTiledImage(ImageVector3f& canvas);

    //! @brief: these are to get a cue stripped from the pyramid level
    void getIntensity(ImageFloat& intensity) const;
    void getDepth(ImageFloat& depth) const;
    void getNormals(ImageVector3f& normals) const;

    //! generates a scaled pyramid from this one. Scale should be an int
    void scaleTo(MDPyramidLevel& dest, int scale) const; // constructs a scaled version of self

    void write(std::ostream& os) const override;
    bool read(std::istream& is) override;

    // init first bigger variable to avoid padding
    MDPyramidMatrix matrix;                            // TODO change name - image with all stuff;
    Isometry3f sensor_offset = Isometry3f::Identity(); // sensor offset

    Matrix3f camera_matrix = Matrix3f::Identity(); // camera matrix for this level

    // parameters used to compute the derivatives and the mask
    float thresholds[3]      = {10.f, 0.5f, 0.5f};
    FilterPolicy policies[3] = {Ignore, Suppress, Clamp};

    CameraType camera_type = Pinhole; // pinhole or speherical
    float min_depth        = 0.3f;    // minimum depth of the cloud
    float max_depth        = 50.f;    // max depth of the cloud
    int mask_grow_radius   = 3;

    //! generates a scaled pyramid from this one. Scale should be an int
    void writeToCharArray(char*& dest, size_t& size) const;

    //! reads a pyramid from a byte array2
    void readFromCharArray(const char*& src, size_t& size);

  protected:
    void growMask();
    void updateDerivatives();
  };

  using MDPyramidLevelPtr = std::shared_ptr<MDPyramidLevel>;
} // namespace md_slam
