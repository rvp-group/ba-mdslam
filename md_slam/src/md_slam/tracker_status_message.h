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
#include <srrg_messages/messages/base_sensor_message.h>
#include <srrg_property/property_eigen.h>

namespace md_slam {

  class MDTrackerStatusMessage : public srrg2_core::BaseSensorMessage {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MDTrackerStatusMessage(const std::string& topic_    = "",
                           const std::string& frame_id_ = "",
                           int seq_                     = -1,
                           const double& timestamp_     = -1);
    srrg2_core::PropertyEigen_<srrg2_core::Isometry3f> global_pose;

    // local pose is the offset w.r.t. the last keyframe, before self (even if keyframe)
    srrg2_core::PropertyEigen_<srrg2_core::Isometry3f> local_pose;
    srrg2_core::PropertyEigen_<srrg2_core::Matrix6f> information_matrix;
    srrg2_core::PropertyBool is_keyframe;
    srrg2_core::PropertyString pyramid_topic;
  };
  using MDTrackerStatusMessagePtr = std::shared_ptr<MDTrackerStatusMessage>;

} // namespace md_slam
