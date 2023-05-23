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
#include <srrg_config/configurable.h>
#include <srrg_geometry/geometry_defs.h>
#include <srrg_pcl/point_intensity.h>
#include <srrg_property/property.h>

namespace md_slam {
  class MDLidarConfiguration : public srrg2_core::Configurable {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // standard config for OS1-64
    PARAM(srrg2_core::PropertyFloat,
          max_intensity,
          "lidar maximum reflectivity-intensity valuse used for normalization",
          500.f,
          0);
    PARAM(srrg2_core::PropertyUnsignedInt,
          image_rows,
          "projection image rows (lidar model dependent, usually number of vertical beams)",
          64,
          0);
    PARAM(srrg2_core::PropertyUnsignedInt,
          image_cols,
          "projection image cols (can vary arbitrarely)",
          1024,
          0);
    PARAM(srrg2_core::PropertyFloat,
          min_depth,
          "minimum point depth to be considered in projection, lower values are discarded",
          0.3f,
          0);
    PARAM(srrg2_core::PropertyFloat,
          max_depth,
          "maximum point depth to be considered in projection, higher values are discarded",
          120.f,
          0);
    PARAM(srrg2_core::PropertyFloat,
          vfov_max,
          "maximum vertical fov (in DEG), usually upper - if left to zero are automatically "
          "calculated -> suggested!",
          0.f,
          0);
    PARAM(srrg2_core::PropertyFloat,
          vfov_min,
          "minimum vertical fov (in DEG), usually lower - if left to zero are automatically "
          "calculated -> suggested!",
          0.f,
          0);
    PARAM(
      srrg2_core::PropertyFloat,
      hfov_max,
      "maximum vertical fov (in DEG), usually left - if left to zero are automatically calculated",
      360.f,
      0);
    PARAM(
      srrg2_core::PropertyFloat,
      hfov_min,
      "minimu vertical fov (in DEG), usually right - if left to zero are automatically calculated",
      0.f,
      0);
    PARAM(srrg2_core::PropertyString,
          point_cloud_topic,
          "name of the point cloud lidar topic",
          "",
          0);

    MDLidarConfiguration() {
    }
    ~MDLidarConfiguration() = default;

    const float& vFOVmin() const {
      return _vfov_min;
    }
    const float& vFOVmax() const {
      if (_vfov_max == 0.f)
        throw std::runtime_error("MDLidarConfiguration|vFOVmax call calculateVerticalFOV() first");
      return _vfov_max;
    }
    const float& hFOVmin() const {
      return _hfov_min;
    }
    const float& hFOVmax() const {
      return _hfov_max;
    }

    inline void calculateVerticalFOV(const srrg2_core::PointIntensity3fVectorCloud& lidar_cloud_) {
      float min_elevation = std::numeric_limits<float>::max();
      float max_elevation = std::numeric_limits<float>::min();
      for (const auto& lidar_point : lidar_cloud_) {
        const srrg2_core::Vector3f& p = lidar_point.coordinates();
        const float range             = p.norm();
        if (range < 1e-8f) { // TODO use param min depth
          continue;
        }
        const float elevation = std::asin(p.z() / range);
        if (elevation < min_elevation) {
          min_elevation = elevation;
        }
        if (elevation > max_elevation) {
          max_elevation = elevation;
        }
      }
      _vfov_min = min_elevation;
      _vfov_max = max_elevation;
    }

    inline void
    calculateHorizontalFOV(const srrg2_core::PointIntensity3fVectorCloud& lidar_cloud_) {
      float min_azimuth = std::numeric_limits<float>::max();
      float max_azimuth = std::numeric_limits<float>::min();
      for (const auto& lidar_point : lidar_cloud_) {
        const srrg2_core::Vector3f& p = lidar_point.coordinates();
        if (p.norm() < 1e-8f) {
          continue;
        }
        const float azimuth = atan2(p.y(), p.x());
        if (azimuth < min_azimuth) {
          min_azimuth = azimuth;
        }
        if (azimuth > max_azimuth) {
          max_azimuth = azimuth;
        }
      }
      _hfov_min = min_azimuth;
      _hfov_max = max_azimuth;
    }

  protected:
    float _hfov_min = 0.f;
    float _hfov_max = 0.f;
    float _vfov_min = 0.f;
    float _vfov_max = 0.f;
  };
} // namespace md_slam
