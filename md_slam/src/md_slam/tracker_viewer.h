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
#include "pyramid_message.h"
#include "tracker_status_message.h"
#include <srrg_messages/message_handlers/message_sink_base.h>
#include <srrg_viewer/active_drawable.h>

namespace md_slam {
  /**
   * Module used just for DEBUGGING
   */
  class MDTrackerViewer : public srrg2_core::MessageSinkBase, public srrg2_core::ActiveDrawable {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // clang-format off
    PARAM(srrg2_core::PropertyFloat, voxelize_coord_res, "coordinates resolution", 0.05, nullptr);
    PARAM(srrg2_core::PropertyFloat, voxelize_normal_res, "normal resolution", 0.f, nullptr);
    PARAM(srrg2_core::PropertyInt, voxelize_interval, "when to voxelize", 10, nullptr);
    PARAM(srrg2_core::PropertyString, status_topic, "tracker status", "/md_tracker_status", nullptr);
    void reset() override;
    bool putMessage(srrg2_core::BaseSensorMessagePtr msg_) override;
  
  protected:
    inline MDVectorCloud& globalCloud() { return _global_cloud[_g_idx & 0x1];}
    inline MDVectorCloud& otherCloud() { return _global_cloud[(_g_idx + 1) & 0x1]; }
    inline const MDVectorCloud& globalCloud() const { return _global_cloud[_g_idx & 0x1]; }
    inline const MDVectorCloud& otherCloud() const { return _global_cloud[(_g_idx + 1) & 0x1]; }
    // clang-format on
    inline void voxelize();
    void addCloud(const MDMatrixCloud& cloud, const Isometry3f& isometry, bool is_keyframe = false);
    void _drawImpl(srrg2_core::ViewerCanvasPtr gl_canvas_) const override;

    Isometry3f _current_pose = Isometry3f::Identity();
    MDVectorCloud _global_cloud[2];
    uint8_t _g_idx = 0;
    MDTrackerStatusMessagePtr _status_msg;
    MDImagePyramidMessagePtr _pyramid_msg;
    MDVectorCloud _current_cloud;
    int _last_time_voxelize = 0;
    std::list<Isometry3f, Eigen::aligned_allocator<Isometry3f>> _trajectory;
    std::string _pyramid_topic  = "";
    mutable bool _lists_created = false;
  };
} // namespace md_slam
