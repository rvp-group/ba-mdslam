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

#include "dual_matrix.cu"
#include "factor_bi.cuh"
#include "sum_reduce.cu"
#include "utils.cuh"
#include <srrg_solver/solver_core/factor_impl.cpp>

namespace md_slam {
  using namespace srrg2_core;

  __device__ __forceinline__ PointStatusFlag
  errorAndJacobian(Vector5f& e_,
                   Matrix5_6f& Ji_,
                   Matrix5_6f& Jj_,
                   WorkspaceEntry& entry_,
                   const MDPyramidMatrix* level_matrix_,
                   const Isometry3f& SX_,
                   const Isometry3f& sensor_offset_inverse_,
                   const Isometry3f& Xji_,
                   const Matrix3f& cam_matrix_,
                   const CameraType& cam_type_,
                   const float& omega_intensity_sqrt_,
                   const float& omega_depth_sqrt_,
                   const float& omega_normals_sqrt_,
                   const float& depth_error_rejection_threshold_,
                   bool chi_only_) {
    // initialization and aliasing
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

    const bool ok = getSubPixel(measurement, image_derivatives, level_matrix_, image_point);
    if (!ok) {
      return Masked;
    }

    // error calculation
    e_ = entry_._prediction - measurement;

    entry_._error = e_;
    e_(0) *= omega_intensity_sqrt_;
    e_(1) *= omega_depth_sqrt_;
    e_.tail(3) *= omega_normals_sqrt_;

    // if the distance between a point and the corresponding one is too big, we drop
    const float depth_error = e_(1) * e_(1);
    if (depth_error > depth_error_rejection_threshold_)
      return DepthError;
    if (chi_only_)
      return status;

    // for Ai and Aj we need two diff rot part
    Matrix3_6f A_j;
    A_j.block<3, 3>(0, 0) = -sensor_offset_inverse_.linear();
    A_j.block<3, 3>(0, 3) =
      2.f * sensor_offset_inverse_.linear() * geometry3d::skew((const Vector3f)(Xji_ * point));

    Matrix3_6f A_i;
    A_i.block<3, 3>(0, 0) = SX_.linear();
    A_i.block<3, 3>(0, 3) = -2.f * SX_.linear() * geometry3d::skew(point);

    // extract values from hom for readability
    const float iz2 = iz * iz;

    // extract the valiues from camera matrix
    const float& fx = cam_matrix_(0, 0);
    const float& fy = cam_matrix_(1, 1);
    const float& cx = cam_matrix_(0, 2);
    const float& cy = cam_matrix_(1, 2);

    // J proj unchanged wrt to md factor monolita
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
        Jj_.row(1) = A_j.row(2);
        Ji_.row(1) = A_i.row(2);
        break;

      case CameraType::Spherical: {
        const float ir    = iz;
        const float ir2   = iz2;
        const float rxy2  = transformed_point.head<2>().squaredNorm();
        const float irxy2 = 1. / rxy2;
        const float rxy   = sqrt(rxy2);
        const float irxy  = 1. / rxy;

        J_proj << -fx * transformed_point.y() * irxy2, // 1st row
          fx * transformed_point.x() * irxy2, 0,
          -fy * transformed_point.x() * transformed_point.z() * irxy * ir2, // 2nd row
          -fy * transformed_point.y() * transformed_point.z() * irxy * ir2, fy * rxy * ir2;

        Matrix1_3f J_range; // jacobian of range(x,y,z)
        J_range << transformed_point.x() * ir, transformed_point.y() * ir,
          transformed_point.z() * ir;

        // add the jacobian of range prediction to row 1.
        Jj_.row(1) = J_range * A_j;
        Ji_.row(1) = J_range * A_i;
      } break;
      default:
        // throw std::runtime_error("MDFactorBivariable::errorAndJacobian|unknown camera model");
    }

    Jj_ -= image_derivatives * J_proj * A_j;
    Ji_ -= image_derivatives * J_proj * A_i;

    Jj_.block<3, 3>(2, 3) += sensor_offset_inverse_.linear() * 2.f *
                             geometry3d::skew((const Vector3f)(Xji_.linear() * normal));
    //  TODO SX_.linear() * -2.f is _neg2rotSX
    Ji_.block<3, 3>(2, 3) += SX_.linear() * -2.f * geometry3d::skew((const Vector3f) normal);

    // omega is diagonal matrix
    // to avoid multiplications we premultiply the rows of J by sqrt of diag
    // elements
    Jj_.row(0) *= omega_intensity_sqrt_;
    Jj_.row(1) *= omega_depth_sqrt_;
    Jj_.block<3, 2>(2, 0) *= omega_normals_sqrt_;

    Ji_.row(0) *= omega_intensity_sqrt_;
    Ji_.row(1) *= omega_depth_sqrt_;
    Ji_.block<3, 2>(2, 0) *= omega_normals_sqrt_;

    return status;
  }

  __global__ void linearize_kernel(LinearSystemEntryBi* ls_entry_,
                                   Workspace* workspace_,
                                   const MDPyramidMatrix* level_matrix_,
                                   const Isometry3f SX_,
                                   const Isometry3f sensor_offset_inverse_,
                                   const Isometry3f Xji_,
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

    srrg2_core::Vector5f e    = Vector5f::Zero();
    srrg2_core::Matrix5_6f Ji = Matrix5_6f::Zero();
    srrg2_core::Matrix5_6f Jj = Matrix5_6f::Zero();

    WorkspaceEntry& ws_entry = workspace_->at<1>(tid);
    const int idx            = ws_entry._index;

    if (idx < 0) {
      return;
    }

    PointStatusFlag& status = ws_entry._status;
    if (status != Good) {
      return;
    }

    status = errorAndJacobian(e,
                              Ji,
                              Jj,
                              ws_entry,
                              level_matrix_,
                              SX_,
                              sensor_offset_inverse_,
                              Xji_,
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
      lambda = sqrt(kernel_chi_threshold_ / chi); // sqrt huber weigth
    } else {
      ls_entry_[idx].is_inlier = 1;
    }

    if (!chi_only_) {
      // here will add all good contribution of pixels
      // outliers contribution will be zero in the sum reduction

      // using double the precision for accumulation
      Matrix6d tmp_full_Hii = Matrix6d::Zero();
      Matrix6d tmp_full_Hjj = Matrix6d::Zero();
      computeAtxA(tmp_full_Hii, Ji, lambda);
      computeAtxA(tmp_full_Hjj, Jj, lambda);
      tmp_full_Hii *= scaling_;
      tmp_full_Hjj *= scaling_;

      // hessians ii jj, only diagonal part is stored
      ls_entry_[idx].upperH   = Mat6dUpperTriangularToVector(tmp_full_Hii);
      ls_entry_[idx].upperHjj = Mat6dUpperTriangularToVector(tmp_full_Hjj);
      // hessian ij stored fully, not diagonal
      ls_entry_[idx].Hij = (Ji.transpose() * Jj * lambda).cast<double>() * scaling_;
      // gradients
      ls_entry_[idx].b  = (Ji.transpose() * e * lambda).cast<double>() * scaling_;
      ls_entry_[idx].bj = (Jj.transpose() * e * lambda).cast<double>() * scaling_;
    }
  }

  void MDFactorBivariable::_linearize(bool chi_only_) {
    Chrono t_lin("linearize_bivariable", &timings, false);
    _omega_intensity_sqrt = std::sqrt(_omega_intensity);
    _omega_depth_sqrt     = std::sqrt(_omega_depth);
    _omega_normals_sqrt   = std::sqrt(_omega_normals);

    const double scaling = 1.0 / _workspace->size();

    // clear linear system entry buffer
    fill_kernel<<<_workspace->nBlocks(), _workspace->nThreads()>>>(
      _entry_array, LinearSystemEntryBi(), _workspace->size());
    CUDA_CHECK(cudaDeviceSynchronize());

    linearize_kernel<<<_workspace->nBlocks(), _workspace->nThreads()>>>(
      _entry_array,
      _workspace->deviceInstance(),
      _level_ptr->matrix.deviceInstance(),
      _SX,
      _sensor_offset_inverse,
      _X_ji,
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

    // entry containing final result
    LinearSystemEntryBi sum;
    {
      Chrono("reduce", &timings, false);
      // reduce sum
      const int num_threads    = BLOCKSIZE; // macro optimized for LinearSystemEntryBi
      const int num_blocks     = (_workspace->size() + num_threads - 1) / num_threads;
      const int required_shmem = num_threads * sizeof(LinearSystemEntryBi);
      LinearSystemEntryBi* dsum_block; // partial result stored into buffer big as num of blocks
      CUDA_CHECK(cudaMalloc((void**) &dsum_block, sizeof(LinearSystemEntryBi) * num_blocks));
      sum_reduce_wrapper(dsum_block, _entry_array, _workspace->size(), num_blocks, num_threads);

      // copy to host and do last reduction, useless to evocate the gpu for stupid problem
      // TODO do allocation once since is static shared mem
      LinearSystemEntryBi* hsum_block = new LinearSystemEntryBi[num_blocks];
      CUDA_CHECK(cudaMemcpy(
        hsum_block, dsum_block, num_blocks * sizeof(LinearSystemEntryBi), cudaMemcpyDeviceToHost));
      // sum latest part, buffer equal to num of blocks
      for (int i = 0; i < num_blocks; i++) {
        sum += hsum_block[i];
      }

      // free mem
      delete hsum_block;
      CUDA_CHECK(cudaFree(dsum_block));
    }

    // update chi first
    const float tot_chi      = sum.chi;
    const size_t num_good    = sum.is_good;
    const size_t num_inliers = sum.is_inlier;
    // TODO
    // if num good is 0 fucks chi up, if we don't have any inliers is
    // likely num good is very low, if this happen better not moving variable
    if (!num_good || !num_inliers) {
      _stats.status = srrg2_solver::FactorStats::Suppressed;
    }

    _stats.chi = tot_chi / num_good;

    // tmp containers for system matrices
    Matrix6d H_tmp_blocks[3];
    Vector6d b_tmp_blocks[2];

    // get total linear system entry H, b and tot error
    H_tmp_blocks[1] = sum.Hij;
    if (_H_transpose[1]) // transpose if indices in solver are flipped
      H_tmp_blocks[1].transposeInPlace();

    H_tmp_blocks[0] = VectorToMat6dUpperTriangular(sum.upperH);
    H_tmp_blocks[2] = VectorToMat6dUpperTriangular(sum.upperHjj);
    copyLowerTriangleUp(H_tmp_blocks[0]); // Hii
    copyLowerTriangleUp(H_tmp_blocks[2]); // Hjj

    b_tmp_blocks[0] = sum.b; // bi
    b_tmp_blocks[1] = sum.bj;

    if (!chi_only_) {
      // retrieve the blocks of H and b for writing (+=, noalias)
      for (int r = 0; r < 2; ++r) {
        if (!this->_b_blocks[r])
          continue;
        Eigen::Map<Vector6f> _b(this->_b_blocks[r]->storage());
        _b.noalias() -= b_tmp_blocks[r].cast<float>();
        for (int c = r; c < 2; ++c) {
          int linear_index = blockOffset(r, c);
          if (!this->_H_blocks[linear_index])
            continue;
          Eigen::Map<Matrix6f> _H(this->_H_blocks[linear_index]->storage());
          _H.noalias() += H_tmp_blocks[linear_index].cast<float>();
        }
      }
    }
  }
} // namespace md_slam
