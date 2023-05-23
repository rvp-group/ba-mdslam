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
#include "frame.h"
#include "utils.cuh"
#include <srrg_config/configurable.h>
#include <srrg_config/property_configurable.h>
#include <srrg_data_structures/correspondence.h>
#include <srrg_geometry/permutation_sampler.h>
#include <srrg_image/image.h>
#include <srrg_pcl/point_projector_types.h>
#include <srrg_pcl/point_types.h>
#include <srrg_pcl/point_unprojector_types.h>

namespace md_slam_closures {

  class LoopValidator : public srrg2_core::Configurable {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LoopValidator();
    virtual ~LoopValidator();

    // clang-format off
    PARAM(srrg2_core::PropertyInt, min_ransac_correspondences, "min number of correspondences to use with SVD ICP and RANSAC", 3, nullptr);
    PARAM(srrg2_core::PropertyInt, ransac_rounds, "min number of correspondences to use with SVD ICP and RANSAC", 70, nullptr);
    PARAM(srrg2_core::PropertyInt, min_num_inliers, "min num inliers after RANSAC validation", 10, nullptr);
    PARAM(srrg2_core::PropertyFloat, max_reprojection_error, "max reprojection error when evaluating best fitting model", 20.f, nullptr);
    PARAM(srrg2_core::PropertyInt, camera_type, "if not set to -1, 0 for pinhole projection, 1 for spherical projection", md_slam::CameraType::Unknown, nullptr);
    // clang-format on

    inline const bool isLoopValid() const {
      return _is_loop_valid;
    }

    inline void setAssociations(const FramePointAssociationVector& associations_) {
      _associations = associations_;
    }

    inline void setCameraMatrix(const srrg2_core::Matrix3f& cam_matrix_) {
      _cam_matrix = cam_matrix_;
    }

    inline void setClosure(std::unique_ptr<Closure> closure_) {
      _closure   = std::move(closure_);
      _reference = _closure->reference;
      _query     = _closure->query;
    }

    inline const FramePointAssociationVector associations() const {
      return _associations;
    }

    inline const srrg2_core::Isometry3f estimate() const {
      return _estimate;
    }

    static srrg2_core::Isometry3f solveLinear(const std::vector<srrg2_core::Vector3f>& q_points,
                                              const std::vector<srrg2_core::Vector3f>& r_points,
                                              const srrg2_core::Vector3d& q_centroid_,
                                              const srrg2_core::Vector3d& r_centroid_,
                                              const size_t& num_matches_);

    void compute();

  protected:
    int _computeMatchesProjective(float& error_,
                                  srrg2_core::CorrespondenceVector& tmp_correspondences_,
                                  const srrg2_core::Point3fVectorCloud& tmp_ref_cloud_);
    int _RANSACSVDICP(srrg2_core::Isometry3f& estimate_,
                      srrg2_core::CorrespondenceVector& best_matches_,
                      float& error_,
                      const srrg2_core::Point3fVectorCloud& query_cloud_,
                      const srrg2_core::Point3fVectorCloud& ref_cloud_,
                      const int num_rounds_,
                      const std::vector<double>& weights_);
    void _refinementSVDICP(srrg2_core::Isometry3f& estimate_,
                           const srrg2_core::Point3fVectorCloud& query_cloud_,
                           const srrg2_core::Point3fVectorCloud& ref_cloud_,
                           const srrg2_core::CorrespondenceVector& matches_);

    // results
    srrg2_core::Isometry3f _estimate = srrg2_core::Isometry3f::Identity();
    srrg2_core::Matrix3f _cam_matrix = srrg2_core::Matrix3f::Zero(); // K

    // correspondences and data structures
    Frame* _reference                 = nullptr;
    Frame* _query                     = nullptr;
    std::unique_ptr<Closure> _closure = nullptr;
    FramePointAssociationVector _associations;

    // sparse projection stuff, not same as pyramid generator
    using PointUnprojectorBase = srrg2_core::PointUnprojectorBase_<srrg2_core::Point3fVectorCloud>;
    using PointUnprojectorPinhole =
      srrg2_core::PointUnprojectorPinhole_<srrg2_core::Point3fVectorCloud>;
    using PointUnprojectorPolar =
      srrg2_core::PointUnprojectorPolar_<srrg2_core::Point3fVectorCloud>;

    std::unique_ptr<PointUnprojectorBase> _unprojector;
    bool _is_loop_valid = false;
  };

  using LoopValidatorPtr = std::shared_ptr<LoopValidator>;

} // namespace md_slam_closures
