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

#include "factor_common.cuh"
// #include <md_slam/utils.cuh>
#include <srrg_solver/solver_core/solver.h>
#include <srrg_system_utils/chrono.h>

namespace md_slam {

  using namespace srrg2_core;

  MDFactorCommon::MDFactorCommon() {
    _rows                 = 0;
    _cols                 = 0;
    _min_depth            = 0.f;
    _max_depth            = 0.f;
    _omega_intensity_sqrt = 0.f;
    _omega_depth_sqrt     = 0.f;
    _omega_normals_sqrt   = 0.f;
    _camera_type          = CameraType::Pinhole;
    _camera_matrix.setIdentity();
    _sensor_offset_rotation_inverse.setIdentity();
    _sensor_offset_inverse.setIdentity();
    _SX.setIdentity();
    _cloud     = new MDMatrixCloud();
    _workspace = new Workspace();
  }

  void MDFactorCommon::setFixed(const MDPyramidLevel* pyramid_level_) {
    _level_ptr                      = pyramid_level_;
    _rows                           = _level_ptr->rows();
    _cols                           = _level_ptr->cols();
    _min_depth                      = _level_ptr->min_depth;
    _max_depth                      = _level_ptr->max_depth;
    _camera_matrix                  = _level_ptr->camera_matrix;
    _camera_type                    = _level_ptr->camera_type;
    _sensor_offset_inverse          = _level_ptr->sensor_offset.inverse();
    _sensor_offset_rotation_inverse = _sensor_offset_inverse.linear();
  }

  void MDFactorCommon::setMoving(const MDPyramidLevel& pyramid_level_) {
    // inverse projection
    pyramid_level_.toCloudDevice(_cloud);
  }

  void MDFactorCommon::setMovingInFixedEstimate(const Isometry3f& X) {
    _SX        = _sensor_offset_inverse * X;
    _neg2rotSX = -2.f * _SX.linear();
  }

  void MDFactorCommon::toTiledImage(ImageVector3f& canvas) const {
    const int rows = _workspace->rows();
    const int cols = _workspace->cols();
    canvas.resize(rows * 3, cols);
    canvas.fill(Vector3f::Zero());
    Vector3f* dest_intensity = &canvas.at(0, 0);
    Vector3f* dest_depth     = &canvas.at(rows, 0);
    Vector3f* dest_normal    = &canvas.at(2 * rows, 0);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const WorkspaceEntry& src = _workspace->at(r, c);
        if (src._status == Good) {
          float intensity = src._error(0);
          float depth     = src._error(1);
          Vector3f normal = src._error.block<3, 1>(2, 0);
          *dest_intensity = Vector3f(intensity, intensity, intensity);
          *dest_depth     = Vector3f(depth, depth, depth);
          *dest_normal    = normal;
        }
        ++dest_intensity;
        ++dest_depth;
        ++dest_normal;
      }
    }
  }

} // namespace md_slam