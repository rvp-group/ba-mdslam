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
#include "dual_matrix.cu"
#include "factor.cuh"
#include "sum_reduce.cu"
#include <srrg_system_utils/chrono.h>

#include <fstream>
#include <iostream>
#include <string>

namespace md_slam {

  using namespace srrg2_core;

  __device__ PointStatusFlag errorAndJacobian(Vector5f& e_,
                                              Matrix5_6f& J_,
                                              WorkspaceEntry& entry_,
                                              const MDPyramidMatrix* level_matrix_,
                                              const Isometry3f& SX_,
                                              const Matrix3f& neg2rotSX_,
                                              const Matrix3f& cam_matrix_,
                                              const CameraType& cam_type_,
                                              const float& omega_intensity_sqrt_,
                                              const float& omega_depth_sqrt_,
                                              const float& omega_normals_sqrt_,
                                              const float& depth_error_rejection_threshold_,
                                              bool chi_only_) {
    PointStatusFlag status           = Good;
    const float z                    = entry_.depth();
    const float iz                   = 1.f / z;
    const Vector3f point             = entry_._point;
    const Vector3f normal            = entry_._normal;
    const Vector3f transformed_point = entry_._transformed_point;
    const Vector3f camera_point      = entry_._camera_point;
    const Vector2f image_point       = entry_._image_point;

    Vector5f measurement;
    Matrix5_2f image_derivatives;
    // todo level ptr
    const bool ok = getSubPixel(measurement, image_derivatives, level_matrix_, image_point);
    if (!ok) {
      return Masked;
    }

    // in error put the difference between prediction and measurement
    e_            = entry_._prediction - measurement;
    entry_._error = e_;
    e_(0) *= omega_intensity_sqrt_;
    e_(1) *= omega_depth_sqrt_;
    e_.tail(3) *= omega_normals_sqrt_;

    // if depth error is to big we drop
    const float depth_error = e_(1) * e_(1);
    if (depth_error > depth_error_rejection_threshold_)
      return DepthError;
    if (chi_only_)
      return status;

    J_.setZero();

    // compute the pose jacobian, including sensor offset
    Matrix3_6f J_icp        = Matrix3_6f::Zero();
    J_icp.block<3, 3>(0, 0) = SX_.linear();
    J_icp.block<3, 3>(0, 3) = neg2rotSX_ * geometry3d::skew((const Vector3f)(point));

    // extract values from hom for readability
    const float iz2 = iz * iz;

    // extract the values from camera matrix
    const float fx = cam_matrix_(0, 0);
    const float fy = cam_matrix_(1, 1);
    const float cx = cam_matrix_(0, 2);
    const float cy = cam_matrix_(1, 2);

    // computes J_hom*K explicitly to avoid matrix multiplication and stores it in J_proj
    Matrix2_3f J_proj = Matrix2_3f::Zero();

    switch (cam_type_) {
      case CameraType::Pinhole:
        // fill the left  and the right 2x3 blocks of J_proj with J_hom*K
        J_proj(0, 0) = fx * iz;
        J_proj(0, 2) = cx * iz - camera_point.x() * iz2;
        J_proj(1, 1) = fy * iz;
        J_proj(1, 2) = cy * iz - camera_point.y() * iz2;

        // add the jacobian of depth prediction to row 1.
        J_.row(1) = J_icp.row(2);

        break;
      case CameraType::Spherical: {
        const float ir    = iz;
        const float ir2   = iz2;
        const float rxy2  = transformed_point.head(2).squaredNorm();
        const float irxy2 = 1. / rxy2;
        const float rxy   = sqrt(rxy2);
        const float irxy  = 1. / rxy;

        J_proj << -fx * transformed_point.y() * irxy2, // 1st row
          fx * transformed_point.x() * irxy2, 0,
          -fy * transformed_point.x() * transformed_point.z() * irxy * ir2, // 2nd row
          -fy * transformed_point.y() * transformed_point.z() * irxy * ir2, fy * rxy * ir2;

        Matrix1_3f J_depth; // jacobian of range(x,y,z)
        J_depth << transformed_point.x() * ir, transformed_point.y() * ir,
          transformed_point.z() * ir;

        // add the jacobian of range/depth prediction to row 1
        J_.row(1) = J_depth * J_icp;

      } break;
    }

    // chain rule to get the jacobian
    J_ -= image_derivatives * J_proj * J_icp;

    // including normals
    J_.block<3, 3>(2, 3) += neg2rotSX_ * geometry3d::skew((const Vector3f) normal);

    // Omega is diagonal matrix
    // to avoid multiplications we premultiply the rows of J by sqrt of diag
    // elements
    J_.row(0) *= omega_intensity_sqrt_;
    J_.row(1) *= omega_depth_sqrt_;
    J_.block<3, 2>(2, 0) *= omega_normals_sqrt_;
    return status;
  }

  __global__ void linearize_kernel(LinearSystemEntry* ls_entry_, // dst
                                   Workspace* workspace_,        // src-dst
                                   const MDPyramidMatrix* level_matrix_,
                                   const Isometry3f SX_,
                                   const Matrix3f neg2rotSX_,
                                   const Matrix3f cam_matrix_,
                                   const CameraType cam_type_,
                                   const float kernel_chi_threshold_,
                                   const double scaling_,
                                   const float omega_intensity_sqrt_,
                                   const float omega_depth_sqrt_,
                                   const float omega_normals_sqrt_,
                                   const float depth_error_rejection_threshold_,
                                   const bool chi_only_) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= workspace_->size())
      return;

    srrg2_core::Vector5f e   = Vector5f::Zero();
    srrg2_core::Matrix5_6f J = Matrix5_6f::Zero();

    WorkspaceEntry& ws_entry = workspace_->at<1>(tid);
    const int idx            = ws_entry._index;

    if (idx == -1)
      return;

    PointStatusFlag status = ws_entry._status;
    if (status != Good) {
      return;
    }

    status = errorAndJacobian(e,
                              J,
                              ws_entry,
                              level_matrix_,
                              SX_,
                              neg2rotSX_,
                              cam_matrix_,
                              cam_type_,
                              omega_intensity_sqrt_,
                              omega_depth_sqrt_,
                              omega_normals_sqrt_,
                              depth_error_rejection_threshold_,
                              chi_only_);
    if (status != Good) {
      return;
    }
    // from now will be evaluated later
    ls_entry_[idx].is_good = 1;
    const float chi        = e.dot(e);
    ls_entry_[idx].chi = ws_entry._chi = chi;
    float lambda                       = 1.f;
    if (chi > kernel_chi_threshold_) {
      lambda = sqrt(kernel_chi_threshold_ / chi);
    } else {
      ls_entry_[idx].is_inlier = 1;
    }
    if (!chi_only_) {
      // here will add all good contribution of pixels
      // outliers contribution will be zero in the sum reduction

      // using double the precision for accumulation
      Matrix6d tmp_full_H = Matrix6d::Zero();
      computeAtxA(tmp_full_H, J, lambda);
      tmp_full_H *= scaling_;

      ls_entry_[idx].upperH = Mat6dUpperTriangularToVector(tmp_full_H);
      ls_entry_[idx].b      = (J.transpose() * e * lambda).cast<double>() * scaling_;
    }
  }

  void MDFactor::_linearize(bool chi_only_) {
    // std::cerr << "SIZE CU: " << sizeof(MDFactor) << std::endl;
    Chrono t_lin("linearize", &timings, false);
    _omega_intensity_sqrt = std::sqrt(_omega_intensity);
    _omega_depth_sqrt     = std::sqrt(_omega_depth);
    _omega_normals_sqrt   = std::sqrt(_omega_normals);

    const double scaling = 1.0 / _workspace->size();

    // compute workspace only when is needed
    if (_entry_array_size != _workspace->size()) {
      if (_entry_array) {
        CUDA_CHECK(cudaFree(_entry_array));
        _entry_array = nullptr;
      }
      if (_workspace->size())
        CUDA_CHECK(
          cudaMalloc((void**) &_entry_array, sizeof(LinearSystemEntry) * _workspace->size()));
      _entry_array_size = _workspace->size();
    }

    LinearSystemEntry sum;
    if (!_workspace->empty()) {
      // clear linear system entry buffer
      LinearSystemEntry zeroval;
      fill_kernel<<<_workspace->nBlocks(), _workspace->nThreads()>>>(
        _entry_array, zeroval, _workspace->size());
      CUDA_CHECK(cudaDeviceSynchronize());

      // linearize get solver entry from two pyramids
      linearize_kernel<<<_workspace->nBlocks(), _workspace->nThreads()>>>(
        _entry_array,
        _workspace->deviceInstance(),
        _level_ptr->matrix.deviceInstance(),
        _SX,
        _neg2rotSX,
        _camera_matrix,
        _camera_type,
        _kernel_chi_threshold,
        scaling,
        _omega_intensity_sqrt,
        _omega_depth_sqrt,
        _omega_normals_sqrt,
        _depth_error_rejection_threshold,
        chi_only_);
      CUDA_CHECK(cudaDeviceSynchronize());

      {
        Chrono("reduce", &timings, false);
        // reduce sum
        const int num_threads    = BLOCKSIZE; // macro optimized for LinearSystemEntry
        const int num_blocks     = (_workspace->size() + num_threads - 1) / num_threads;
        const int required_shmem = num_threads * sizeof(LinearSystemEntry);
        LinearSystemEntry* dsum_block;
        CUDA_CHECK(cudaMalloc((void**) &dsum_block, sizeof(LinearSystemEntry) * num_blocks));
        sum_reduce_wrapper(dsum_block, _entry_array, _workspace->size(), num_blocks, num_threads);

        // copy to host and do last reduction, useless to evocate the gpu for stupid problem
        LinearSystemEntry* hsum_block = new LinearSystemEntry[num_blocks];
        CUDA_CHECK(cudaMemcpy(
          hsum_block, dsum_block, num_blocks * sizeof(LinearSystemEntry), cudaMemcpyDeviceToHost));
        // sum latest part, buffer equal to num of blocks
        for (int i = 0; i < num_blocks; i++) {
          sum += hsum_block[i];
        }

        // free mem
        delete hsum_block;
        CUDA_CHECK(cudaFree(dsum_block));
      }
    }

    // get total linear system entry H, b and tot error
    Matrix6d tot_H = VectorToMat6dUpperTriangular(sum.upperH);
    copyLowerTriangleUp(tot_H);
    const Vector6d& tot_b    = sum.b;
    const float tot_chi      = sum.chi;
    const size_t num_inliers = sum.is_inlier;
    const size_t num_good    = sum.is_good;


    // if num good is 0 fucks chi up, if we don't have any inliers is
    // likely num good is very low
    if (!num_good || !num_inliers){
      _stats.status = srrg2_solver::FactorStats::Suppressed;
      return;
    }
    _stats.chi = tot_chi / num_good;

    if (!chi_only_) {
      //  use solver to easily handle level solving
      Eigen::Map<Matrix6f> _H(this->_H_blocks[0]->storage());
      Eigen::Map<Vector6f> _b(this->_b_blocks[0]->storage());
      _H.noalias() += tot_H.cast<float>();
      _b.noalias() -= tot_b.cast<float>();
    }
  }

} // namespace md_slam
