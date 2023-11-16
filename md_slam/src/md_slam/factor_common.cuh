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
#include "image_pyramid.h"
#include "linear_system_entry.cuh"
#include <srrg_geometry/geometry3d.h>
#include <srrg_system_utils/chrono.h>

namespace md_slam {
  using namespace srrg2_core;

  __device__ __forceinline__ void
  computeAtxA(Matrix6d& dest, const Matrix5_6f& src, const float& lambda) {
    for (int c = 0; c < 6; ++c) {
      for (int r = 0; r <= c; ++r) {
        dest(r, c) += static_cast<double>(src.col(r).dot(src.col(c)) * lambda);
      }
    }
  }

  __host__ inline void copyLowerTriangleUp(Matrix6d& A) {
    for (int c = 0; c < 6; ++c) {
      for (int r = 0; r < c; ++r) {
        A(c, r) = A(r, c);
      }
    }
  }

  __host__ inline Matrix6d VectorToMat6dUpperTriangular(const Vector21d& vec_) {
    Matrix6d mat = Matrix6d::Zero();
    int k        = 0;
    for (int i = 0; i < Matrix6d::RowsAtCompileTime; ++i) {
      for (int j = i; j < Matrix6d::ColsAtCompileTime; ++j) {
        mat(i, j) = vec_(k++);
      }
    }
    return mat;
  }

  __device__ __forceinline__ Vector21d Mat6dUpperTriangularToVector(const Matrix6d& mat_) {
    Vector21d vec;
    int k = 0;
    for (int i = 0; i < Matrix6d::RowsAtCompileTime; ++i) {
      for (int j = i; j < Matrix6d::ColsAtCompileTime; ++j) {
        vec(k++) = mat_(i, j);
      }
    }
    return vec;
  }

  // ! pack int and float into a single uint64
  __device__ __forceinline__ unsigned long long pack(int a, float b) {
    return (((unsigned long long) (*(reinterpret_cast<unsigned*>(&b)))) << 32) +
           *(reinterpret_cast<unsigned*>(&a));
  }

  // ! unpack an uint64 into an int and a float
  __device__ __forceinline__ void unpack(int& a, float& b, unsigned long long val) {
    unsigned ma = (unsigned) (val & 0x0FFFFFFFFULL);
    a           = *(reinterpret_cast<int*>(&ma));
    unsigned mb = (unsigned) (val >> 32);
    b           = *(reinterpret_cast<float*>(&mb));
  }

  //! a solver entry
  __host__ __device__ struct ALIGN(16) WorkspaceEntry {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // clang-format off
    __host__ __device__ WorkspaceEntry() { _prediction[1] = numeric_limits<float>::max(); }
    Eigen::Matrix<float, 5, 1> _prediction = Eigen::Matrix<float, 5, 1>::Zero();
    Eigen::Matrix<float, 5, 1> _error = Eigen::Matrix<float, 5, 1>::Zero();
    Vector3f _point = Vector3f::Zero(); // point in original frame
    Vector3f _normal = Vector3f::Zero(); // normal in the moving frame, before applying transform
    Vector3f _transformed_point = Vector3f::Zero(); // point after applying transform
    Vector3f _camera_point = Vector3f::Zero(); // camera point in camera coords
    Vector2f _image_point   = Vector2f::Zero(); // point in image coords
    int _index              = -1; // index of projected point
    float _chi              = 0.f; // chi error 
    PointStatusFlag _status = Good; // point status
    unsigned long long depth_idx = numeric_limits<unsigned long long>::max(); // encode float depth and tid     
    __host__ __device__ const float& intensity() { return _prediction[0]; }
    __host__ __device__ const float& depth() { return _prediction[1]; }
    // clang-format on
  };

  // common stuff for tracking and adjustment factors
  // defines workspace and functions for projecting pyramid
  using Workspace = DualMatrix_<WorkspaceEntry>;

  class ALIGN(16) MDFactorCommon {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using MDMatrixCloudPtr = std::shared_ptr<MDMatrixCloud>;

    MDFactorCommon();                                     // c'tor for initializing factor vars
    void setFixed(const MDPyramidLevel* pyramid_level_);  // pyramid of the current factor
    void setMoving(const MDPyramidLevel& pyramid_level_); // cloud of the current factor
    void toTiledImage(ImageVector3f& canvas) const;       // viewer debug

    Chrono::ChronoMap timings;

    // clang-format off
    inline float omegaDepth() const { return _omega_depth; }
    inline float omegaIntensity() const { return _omega_intensity; }
    inline float omegaNormals() const { return _omega_normals; }
    inline float depthRejectionThreshold() const { return _depth_error_rejection_threshold; }
    inline float kernelChiThreshold() const { return _kernel_chi_threshold; }
    inline void setOmegaDepth(float v) { _omega_depth = v; }
    inline void setOmegaIntensity(float v) { _omega_intensity = v; }
    inline void setOmegaNormals(float v) { _omega_normals = v; }
    inline void setDepthRejectionThreshold(float v) { _depth_error_rejection_threshold = v; }
    inline void setKernelChiThreshold(float v) { _kernel_chi_threshold = v; }
    // clang-format on

  protected:
    //! performs the occlusion check
    void computeProjections();
    void setMovingInFixedEstimate(const Isometry3f&);

    // moving cloud in robot frame
    MDMatrixCloud* _cloud = nullptr;
    char pad_cloud[56];

    // Matrix3_6f _J_icp; // this needs to be here, padding issues

    Isometry3f _sensor_offset_inverse;
    Isometry3f _SX; // product between _sensor_in_robot_inverse and X

    Matrix3f _neg2rotSX; // two times the rotation of SX, used in jacobians
    Matrix3f _sensor_offset_rotation_inverse;
    Matrix3f _camera_matrix;

    const MDPyramidLevel* _level_ptr = nullptr;
    Workspace* _workspace            = nullptr;
    // MDMatrixCloud* _cloud            = nullptr;

    CameraType _camera_type;

    // default parameters
    float _omega_intensity = 1.f;
    float _omega_depth     = 5.f;
    float _omega_normals   = 1.f;
    float _omega_intensity_sqrt;
    float _omega_depth_sqrt;
    float _omega_normals_sqrt;
    float _depth_error_rejection_threshold = .25f; // if depth error>that eliminate
    float _kernel_chi_threshold            = 1.f;  // if chi> this suppress/kernelize
    float _min_depth;
    float _max_depth;

    int _rows;
    int _cols;
    int _max_level;

    // Matrix3_6f _J_icp; // this needs to be here, padding issues
  };
} // namespace md_slam
