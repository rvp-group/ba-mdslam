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
#include "loop_detector_hbst.h"
#include "loop_validator.h"
#include <srrg_viewer/active_drawable.h>

#include <srrg_config/configurable.h>
#include <srrg_config/property_configurable.h>

namespace md_slam {
  using namespace srrg2_core;
  using namespace md_slam_closures;

  class MDCloser : public srrg2_core::Configurable, public srrg2_core::ActiveDrawable {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MDCloser();
    ~MDCloser();

    // clang-format off
    PARAM(srrg2_core::PropertyConfigurable_<md_slam_closures::LoopValidator>, loop_validator, "loop validator to perform geometric verification", nullptr, nullptr);
    PARAM(srrg2_core::PropertyConfigurable_<md_slam_closures::LoopDetectorHBST>, loop_det, "loop detector to detect loop, for now only HBST", nullptr, nullptr);
    PARAM(srrg2_core::PropertyInt, nms_radius, "neighboorood search radius for non max suppresion", 4, 0);
    
    bool compute();
    inline void setTimestamp(const double& timestamp_) { _timestamp = timestamp_; }
    inline void setPyramid(MDImagePyramid* pyramid_) { _pyramid = pyramid_; }
    inline void setPose(const Isometry3f& tracker_pose_) { _tracker_pose = tracker_pose_; }
    inline void setIntensity(const ImageFloat& intensity_) { _intensity = intensity_; }
    inline void setDepth(const ImageFloat& depth_) { _depth = depth_; }
    inline int closureIndex() const { return _reference_closures_idx; }
    inline Isometry3f estimate() const { return _estimate; }
    inline FramesContainerPtr framesContainer() const { return _fcontainer; }
    // clang-format on

  protected:
    void _drawImpl(ViewerCanvasPtr gl_canvas_) const override;
    void _generateFrame(FramesContainerPtr frames, const Isometry3f& curr_pose_);
    std::vector<cv::KeyPoint> _nonMaxSuppression(std::vector<cv::KeyPoint>& keypoints_,
                                                 const size_t& rows_,
                                                 const size_t& cols_,
                                                 const int& nms_dist_);

    ImageFloat _intensity;
    ImageFloat _depth;
    FramesContainerPtr _fcontainer = nullptr;
    MDImagePyramid* _pyramid       = nullptr;
    Isometry3f _estimate           = Isometry3f::Identity();
    Isometry3f _tracker_pose       = Isometry3f::Identity();
    double _timestamp              = 0.0;
    int _reference_closures_idx    = -1;
    int _total_num_closures        = 0;
  };
} // namespace md_slam