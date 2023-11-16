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
#include <cstdint>
#include <srrg_geometry/geometry_defs.h>
#include <srrg_image/image.h>
#include <srrg_pcl/point_types.h>

namespace md_slam {
  using namespace srrg2_core;

  enum CameraType { Pinhole = 0, Spherical = 1, Unknown = -1 };

  // loads an image, depending on the type of img
  bool loadImage(srrg2_core::BaseImage& img, const std::string filename);
  void showImage(const srrg2_core::BaseImage& img, const std::string& title, int wait_time = 0);

  // handles damn images with odd rows/cols
  void prepareImage(ImageFloat& dest,
                    const ImageFloat& src,
                    int max_scale,
                    int row_scale      = 1,
                    int col_scale      = 1,
                    bool suppress_zero = false);

  // !
  // ! projections, sparse and dense
  void sparseProjection(Point2fVectorCloud& dest_,
                        const Point3fVectorCloud& source_,
                        const Matrix3f& cam_matrix_,
                        const float& min_depth_,
                        const float& max_depth_,
                        const int& n_rows_,
                        const int& n_cols_,
                        const CameraType& camera_type_);

  __host__ __device__ inline bool project(Vector2f& image_point_,
                                          Vector3f& camera_point_,
                                          float& depth_,
                                          const Vector3f& point_,
                                          const CameraType& camera_type_,
                                          const Matrix3f& camera_mat_,
                                          const float& min_depth_,
                                          const float& max_depth_) {
    switch (camera_type_) {
      case CameraType::Pinhole:
        depth_ = point_.z();
        if (depth_ < min_depth_ || depth_ > max_depth_) {
          return false;
        }
        camera_point_ = camera_mat_ * point_;
        image_point_  = camera_point_.head<2>() * 1.f / depth_;
        break;
      case CameraType::Spherical:
        depth_ = point_.norm();
        if (depth_ < min_depth_ || depth_ > max_depth_) {
          return false;
        }
        camera_point_.x() = atan2(point_.y(), point_.x());
        camera_point_.y() = atan2(point_.z(), point_.head<2>().norm());
        camera_point_.z() = depth_;

        image_point_.x() = camera_mat_(0, 0) * camera_point_.x() + camera_mat_(0, 2);
        image_point_.y() = camera_mat_(1, 1) * camera_point_.y() + camera_mat_(1, 2);
        break;
        // default:
        //   throw std::runtime_error("utils::project | unknown camera type");
    }
    return true;
  }

  // clang-format off
  // ! redifining some useful numeric limits for device code
  template<class T> struct numeric_limits{
      typedef T type;
      __host__ __device__ static type min()  { return type(); }
      __host__ __device__ static type max() { return type(); }
  };

  template<> struct numeric_limits<unsigned long>{
      typedef unsigned long type;
      __host__ __device__ static type min() { return 0; }
      __host__ __device__ static type max() { return ULONG_MAX; }
  };

  template<> struct numeric_limits<unsigned long long>{
      typedef unsigned long long type;
      __host__ __device__ static type min() { return 0; }
      __host__ __device__ static type max() { return UINT64_MAX; }
  };

  template<> struct numeric_limits<float>{
      typedef float type;
      __host__ __device__ static type min() { return 1.175494351e-38f; }
      __host__ __device__ static type max() { return 3.402823466e+38f; }
  };

  template<> struct numeric_limits<double>{
      typedef double type;
      __host__ __device__ static type min() { return 2.2250738585072014e-308; }
      __host__ __device__ static type max() { return 1.7976931348623158e+308; }
  };
  // clang-format on

} // namespace md_slam
