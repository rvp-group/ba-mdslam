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
#include "image_pyramid.h"
#include <srrg_messages/messages/base_sensor_message.h>

namespace md_slam {
  class MDImagePyramidMessage : public srrg2_core::BaseSensorMessage {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MDImagePyramidMessage(const std::string& topic_    = "",
                          const std::string& frame_id_ = "",
                          int seq_                     = -1,
                          const double& timestamp_     = -1) :
      BaseSensorMessage(topic_, frame_id_, seq_, timestamp_) {
    }
    // clang-format off
    MDImagePyramid* get() { return _pyramid.get(); }
    void set(MDImagePyramid* pyr) { _pyramid.set(pyr); }
    void serialize(srrg2_core::ObjectData& odata, srrg2_core::IdContext& context);
    void deserialize(srrg2_core::ObjectData& odata, srrg2_core::IdContext& context);
    // clang-format on
  protected:
    MDImagePyramidReference _pyramid;
  };
  using MDImagePyramidMessagePtr = std::shared_ptr<MDImagePyramidMessage>;

} // namespace md_slam
