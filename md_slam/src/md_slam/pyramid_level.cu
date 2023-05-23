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
#include "pyramid_level.cuh"
#include "utils.cuh"
#include <srrg_system_utils/char_array.h>

namespace md_slam {
  using namespace srrg2_core;

  // TODO to be tested
  __global__ void fromCloud_kernel(MDPyramidMatrix* mat_,
                                   const MDMatrixCloud* src_cloud_,
                                   const Isometry3f inv_sensor_offset_,
                                   const CameraType cam_type_,
                                   const size_t rows_,
                                   const size_t cols_,
                                   const float fx_,
                                   const float fy_,
                                   const float cx_,
                                   const float cy_,
                                   const float min_depth_,
                                   const float max_depth_) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > src_cloud_->size())
      return;

    const PointNormalIntensity3f& src = src_cloud_->at<1>(tid);

    if (src.status != POINT_STATUS::Valid)
      return;

    Vector3f point = inv_sensor_offset_ * src.coordinates();
    const float& x = point.x();
    const float& y = point.y();
    const float& z = point.z();
    float depth    = 0.f;

    Vector2f image_point;

    switch (cam_type_) {
      case Pinhole: {
        depth = z;
        if (depth < min_depth_ || depth > max_depth_)
          return;
        image_point.x() = fx_ * x + cx_;
        image_point.y() = fy_ * y + cy_;
        image_point *= 1. / depth;
      } break;
      case Spherical: {
        depth = point.norm();
        if (depth < min_depth_ || depth > max_depth_)
          return;
        const float azimuth   = atan2f(y, x);
        const float elevation = atan2f(z, sqrtf(x * x + y * y));
        image_point.x()       = fx_ * azimuth + cx_;
        image_point.y()       = fy_ * elevation + cy_;
      } break;
      default:;
    }

    // equivalent of cvRound?
    const int r = (int) (image_point.y() + (image_point.y() >= 0 ? 0.5f : -0.5f));
    const int c = (int) (image_point.x() + (image_point.x() >= 0 ? 0.5f : -0.5f));

    if (!mat_->inside(r, c))
      return;
    MDPyramidMatrixEntry& entry = mat_->at<1>(r, c);
    // TODO cuda depth buffer
    if (depth < entry.depth()) {
      entry.setIntensity(src.intensity());
      entry.setDepth(depth);
      entry.setNormal(inv_sensor_offset_.linear() * src.normal());
#ifdef _MD_ENABLE_SUPERRES_
      entry.c = camera_point.x();
      entry.r = camera_point.y();
#endif
      entry.setMasked(false);
    }
  }

  // TODO fix redundant arguments
  __global__ void toCloud_kernel(MDMatrixCloud* target_,
                                 const MDPyramidMatrix* mat_,
                                 const Isometry3f sensor_offset_,
                                 const Matrix3f inv_K_,
                                 const CameraType cam_type_,
                                 const float ifx_,
                                 const float ify_,
                                 const float cx_,
                                 const float cy_,
                                 const float min_depth_,
                                 const float max_depth_) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (!target_->inside(row, col))
      return;

    PointNormalIntensity3f& dst    = target_->at<1>(row, col);
    const MDPyramidMatrixEntry src = mat_->at<1>(row, col);

    float w = src.depth();
    if (src.masked() || w < min_depth_ || w > max_depth_)
      return;
#ifdef _MD_ENABLE_SUPERRES_
    const float& r = src.r;
    const float& c = src.c;
#else
    const float r = row;
    const float c = col;
#endif
    dst.status = POINT_STATUS::Valid;
    switch (cam_type_) {
      case Pinhole: {
        dst.coordinates() = inv_K_ * Vector3f(c * w, r * w, w);
      } break;
      case Spherical: {
        float azimuth        = ifx_ * (c - cx_);
        float elevation      = ify_ * (r - cy_);
        float s0             = sinf(azimuth);
        float c0             = cosf(azimuth);
        float s1             = sinf(elevation);
        float c1             = cosf(elevation);
        dst.coordinates()(0) = c0 * c1 * w;
        dst.coordinates()(1) = s0 * c1 * w;
        dst.coordinates()(2) = s1 * w;
      } break;
      default:;
    }
    dst.intensity()   = src.intensity();
    dst.normal()      = sensor_offset_.linear() * src.normal();
    dst.coordinates() = sensor_offset_ * dst.coordinates();
  }

  void MDPyramidLevel::toCloudDevice(MDMatrixCloud* target_) const {
    target_->resize(rows(), cols());

    PointNormalIntensity3f p;
    p.setZero();
    p.status = POINT_STATUS::Invalid;

    // TODO only in device?
    target_->fill(p);

    const float ifx      = 1.f / camera_matrix(0, 0);
    const float ify      = 1.f / camera_matrix(1, 1);
    const float cx       = camera_matrix(0, 2);
    const float cy       = camera_matrix(1, 2);
    const Matrix3f inv_K = camera_matrix.inverse();

    // init bidimensional kernel since we move in image space
    dim3 n_blocks(16, 16);
    dim3 n_threads;
    n_threads.x = (cols() + n_blocks.x - 1) / n_blocks.x;
    n_threads.y = (rows() + n_blocks.y - 1) / n_blocks.y;

    toCloud_kernel<<<n_blocks, n_threads>>>(target_->deviceInstance(),
                                            matrix.deviceInstance(),
                                            sensor_offset,
                                            inv_K,
                                            camera_type,
                                            ifx,
                                            ify,
                                            cx,
                                            cy,
                                            min_depth,
                                            max_depth);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void MDPyramidLevel::fromCloud(MDMatrixVectorCloud& src_cloud) {
    MDPyramidMatrixEntry zero_entry;
    zero_entry.setDepth(max_depth + 1);
    matrix.fill(zero_entry);

    Isometry3f inv_sensor_offset = sensor_offset.inverse();
    Vector3f polar_point;
    Vector3f coordinates;
    Vector3f camera_point = Vector3f::Zero();
    const float& fx       = camera_matrix(0, 0);
    const float& fy       = camera_matrix(1, 1);
    const float& cx       = camera_matrix(0, 2);
    const float& cy       = camera_matrix(1, 2);
    float w               = 0;

    for (const auto& src : src_cloud) {
      if (src.status != POINT_STATUS::Valid)
        continue;
      coordinates    = inv_sensor_offset * src.coordinates();
      const float& x = coordinates.x();
      const float& y = coordinates.y();
      const float& z = coordinates.z();
      switch (camera_type) {
        case Pinhole: {
          w = coordinates(2);
          if (w < min_depth || w > max_depth)
            continue;
          camera_point = camera_matrix * coordinates;
          camera_point.block<2, 1>(0, 0) *= 1. / w;
        } break;
        case Spherical: {
          w = coordinates.norm();
          if (w < min_depth || w > max_depth)
            continue;
          polar_point.x()  = atan2(y, x);
          polar_point.y()  = atan2(coordinates.z(), sqrt(x * x + y * y));
          polar_point.z()  = z;
          camera_point.x() = fx * polar_point.x() + cx;
          camera_point.y() = fy * polar_point.y() + cy;
          camera_point.z() = w;
        } break;
        default:;
      }
      int c = cvRound(camera_point.x());
      int r = cvRound(camera_point.y());
      if (!matrix.inside(r, c))
        continue;
      MDPyramidMatrixEntry& entry = matrix.at(r, c);
      if (w < entry.depth()) {
        entry.setIntensity(src.intensity());
        entry.setDepth(w);
        entry.setNormal(inv_sensor_offset.linear() * src.normal());
#ifdef _MD_ENABLE_SUPERRES_
        entry.c = camera_point.x();
        entry.r = camera_point.y();
#endif
        entry.setMasked(false);
      }
    }

    growMask();
    updateDerivatives();
  }

  static inline Vector3f lift(const float f) {
    return Vector3f(f, f, f);
  }

  void MDPyramidLevel::toTiledImage(ImageVector3f& canvas) {
    // collage,
    //           value cloud dx dy
    // intensity
    // depth
    // normals

    canvas.resize(rows() * 3, cols() /* *3*/);
    canvas.fill(Vector3f::Zero());
    int masked        = 0;
    int non_masked    = 0;
    float depth_scale = 1. / (max_depth - min_depth);
    for (size_t r = 0; r < rows(); ++r) {
      for (size_t c = 0; c < cols(); ++c) {
        const MDPyramidMatrixEntry& entry = matrix.at(r, c);
        if (entry.masked()) {
          ++masked;
          canvas.at(r, c) = Vector3f(0, 0, 1);
          continue;
        }
        ++non_masked;
        // intensity
        canvas.at(r, c) = lift(entry.intensity());
        // depth
        canvas.at(r + rows(), c) = lift(depth_scale * (entry.depth() - min_depth));
        // normals
        canvas.at(r + 2 * rows(), c) = entry.normal();
      }
    }
  }

  void MDPyramidLevel::toCloud(MDMatrixVectorCloud& target) const {
    target.resize(rows(), cols());
    PointNormalIntensity3f p;
    p.setZero();
    p.status = POINT_STATUS::Invalid;
    target.fill(p);
    const float ifx = 1. / camera_matrix(0, 0);
    const float ify = 1. / camera_matrix(1, 1);
    const float cx  = camera_matrix(0, 2);
    const float cy  = camera_matrix(1, 2);

    Matrix3f inv_K = camera_matrix.inverse();
    for (int r = 0; r < rows(); ++r) {
      for (int c = 0; c < cols(); ++c) {
        const MDPyramidMatrixEntry& src = matrix.at(r, c);
        PointNormalIntensity3f& dest    = target.at(r, c);
        float w                         = src.depth();
        if (src.masked() || w < min_depth || w > max_depth)
          continue;
#ifdef _MD_ENABLE_SUPERRES_
        const float& row = src.r;
        const float& col = src.c;
#else
        const float row = r;
        const float col = c;
#endif
        dest.status = POINT_STATUS::Valid;
        switch (camera_type) {
          case Pinhole: {
            Vector3f p         = inv_K * Vector3f(col * w, row * w, w);
            dest.coordinates() = p;
          } break;
          case Spherical: {
            float azimuth         = ifx * (col - cx);
            float elevation       = ify * (row - cy);
            float s0              = sin(azimuth);
            float c0              = cos(azimuth);
            float s1              = sin(elevation);
            float c1              = cos(elevation);
            dest.coordinates()(0) = c0 * c1 * w;
            dest.coordinates()(1) = s0 * c1 * w;
            dest.coordinates()(2) = s1 * w;
          } break;
          default:;
        }
        dest.intensity() = src.intensity();
        dest.normal()    = src.normal();
      }
    }
    target.transformInPlace<TRANSFORM_CLASS::Isometry>(sensor_offset);
  }

  void MDPyramidLevel::scaleTo(MDPyramidLevel& dest_, int scale) const {
    assert(rows() % scale == 0 && "MDPyramidLevel::scaleTo | rows not multiple of scale");
    assert(cols() % scale == 0 && "MDPyramidLevel::scaleTo | cols not multiple of scale");

    float inv_scale = 1. / scale;

    // copy shit to scaled level
    dest_.sensor_offset = sensor_offset;
    dest_.camera_type   = camera_type;

    dest_.resize(rows() / scale, cols() / scale);
    dest_.camera_matrix = camera_matrix;
    dest_.camera_matrix.block<2, 3>(0, 0) *= inv_scale;
    dest_.min_depth = min_depth;
    dest_.max_depth = max_depth;
    memcpy(dest_.thresholds, thresholds, sizeof(thresholds));
    memcpy(dest_.policies, policies, sizeof(policies));

    MDPyramidMatrixEntry null_entry;
    null_entry.setMasked(false);
    dest_.matrix.fill(null_entry);
    Matrix_<int> counters(dest_.rows(), dest_.cols());
    counters.fill(0);
    for (size_t r = 0; r < rows(); ++r) {
      int dr = r / scale;
      for (size_t c = 0; c < cols(); ++c) {
        const auto& src = matrix.at(r, c);
        int dc          = c / scale;
        auto& dest      = dest_.matrix.at(dr, dc);
        dest.setMasked(dest.masked() | src.masked());
        if (dest.masked())
          continue;
#ifdef _MD_ENABLE_SUPERRES_
        dest.r += src.r * inv_scale;
        dest.c += src.c * inv_scale;
#endif
        auto& ctr = counters.at(dr, dc);
        dest.value += src.value;
        ++ctr;
      }
    }
    for (size_t r = 0; r < dest_.rows(); ++r) {
      for (size_t c = 0; c < dest_.cols(); ++c) {
        auto& entry = dest_.matrix.at(r, c);
        auto& value = entry.value;
        auto& ctr   = counters.at(r, c);
        if (entry.masked())
          continue;
        assert(ctr && "MDPyramidLevel::scaleTo | counters are 0");
        float inv_ctr = 1. / ctr;
        value *= inv_ctr;
#ifdef _MD_ENABLE_SUPERRES_
        entry.r *= inv_ctr;
        entry.c *= inv_ctr;
#endif
        value.block<3, 1>(2, 0).normalize();
      }
    }
    dest_.updateDerivatives();
  }

} // namespace md_slam
