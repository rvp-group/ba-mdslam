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
#include "factor_common.cuh"
#include "utils.cuh"

namespace md_slam {

  using namespace srrg2_core;

  __global__ void project_kernel(Workspace* workspace_,
                                 const MDMatrixCloud* cloud_,
                                 const MDPyramidMatrix* level_matrix_,
                                 const Isometry3f SX_,
                                 const CameraType cam_type_,
                                 const Matrix3f cam_mat_,
                                 const float min_depth_,
                                 const float max_depth_,
                                 const bool is_depth_buffer_) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= cloud_->size())
      return;

    const PointNormalIntensity3f full_point = cloud_->at<1>(tid);
    const Vector3f point                    = full_point.coordinates();
    const Vector3f normal                   = full_point.normal();
    const float intensity                   = full_point.intensity();
    const Vector3f transformed_point        = SX_ * point;

    float depth           = 0.f;
    Vector3f camera_point = Vector3f::Zero();
    Vector2f image_point  = Vector2f::Zero();

    const bool is_good = project(image_point,
                                 camera_point,
                                 depth,
                                 transformed_point,
                                 cam_type_,
                                 cam_mat_,
                                 min_depth_,
                                 max_depth_);

    if (!is_good)
      return;

    // equivalent of cvRound
    const int irow = (int) (image_point.y() + (image_point.y() >= 0 ? 0.5f : -0.5f));
    const int icol = (int) (image_point.x() + (image_point.x() >= 0 ? 0.5f : -0.5f));

    if (!workspace_->inside(irow, icol)) {
      return;
    }

    // check if masked
    if (level_matrix_->at<1>(irow, icol).masked()) {
      return;
    }

    WorkspaceEntry& entry = workspace_->at<1>(irow, icol);

    if (is_depth_buffer_) {
      // unsigned long long candidate_depth_idx = pack(tid, depth);
      // depth buffer implemented as comparison between two uint64
      // packing depth and tid, with depth on the first 32 bits
      // this is required to make reproducible experiment
      // in this way, even if depth is the same, the one with lower tid is preferred
      atomicMin(&(entry.depth_idx), pack(tid, depth));
      return;
    }

    unsigned long long candidate_depth_idx = pack(tid, depth);
    if (candidate_depth_idx != entry.depth_idx)
      return;

    // unpack depth and tread id from single depth_idx variable
    unpack(entry._index, depth, candidate_depth_idx);
    const Vector3f rn = SX_.linear() * normal;
    entry._prediction << intensity, depth, rn.x(), rn.y(), rn.z();
    entry._point             = point;
    entry._normal            = normal;
    entry._transformed_point = transformed_point;
    entry._camera_point      = camera_point;
    entry._image_point       = image_point;
    entry._chi               = 0.f;
  }

  void MDFactorCommon::computeProjections() {
    _workspace->resize(_rows, _cols);
    // initialize workspace
    _workspace->fill(WorkspaceEntry(), true); // reset only in device

    project_kernel<<<_workspace->nBlocks(), _workspace->nThreads()>>>(
      _workspace->deviceInstance(),
      _cloud->deviceInstance(),
      _level_ptr->matrix.deviceInstance(),
      _SX,
      _camera_type,
      _camera_matrix,
      _min_depth,
      _max_depth,
      true);
    CUDA_CHECK(cudaDeviceSynchronize());

    project_kernel<<<_workspace->nBlocks(), _workspace->nThreads()>>>(
      _workspace->deviceInstance(),
      _cloud->deviceInstance(),
      _level_ptr->matrix.deviceInstance(),
      _SX,
      _camera_type,
      _camera_matrix,
      _min_depth,
      _max_depth,
      false);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
} // namespace md_slam
