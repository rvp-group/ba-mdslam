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

#include "tracker_viewer.h"
#include <unistd.h>

namespace md_slam {
  using namespace srrg2_core;

  void MDTrackerViewer::voxelize() {
    ++_last_time_voxelize;
    if (_last_time_voxelize < param_voxelize_interval.value())
      return;
    std::cerr << "voxelize " << globalCloud().size() << " -> ";
    std::back_insert_iterator<MDVectorCloud> out_vox(otherCloud());
    float cs = param_voxelize_coord_res.value();
    float ns = param_voxelize_normal_res.value();
    PointNormalIntensity3f::PlainVectorType<float> scales;
    scales << cs, cs, cs, ns, ns, ns, 0;
    globalCloud().voxelize(out_vox, scales);
    ++_g_idx;
    otherCloud().clear();
    _last_time_voxelize = 0;
    std::cerr << globalCloud().size() << std::endl;
  }

  bool MDTrackerViewer::putMessage(srrg2_core::BaseSensorMessagePtr msg_) {
    MDImagePyramidMessagePtr pyr_msg     = std::dynamic_pointer_cast<MDImagePyramidMessage>(msg_);
    MDTrackerStatusMessagePtr status_msg = std::dynamic_pointer_cast<MDTrackerStatusMessage>(msg_);
    if (status_msg && status_msg->topic.value() == param_status_topic.value()) {
      _status_msg    = status_msg;
      _pyramid_topic = _status_msg->pyramid_topic.value();
    }
    if (pyr_msg && pyr_msg->topic.value() == _pyramid_topic) {
      _pyramid_msg = pyr_msg;
    }
    if (_pyramid_msg && _status_msg &&
        _pyramid_msg->timestamp.value() == _status_msg->timestamp.value()) {
      MDImagePyramid* current = _pyramid_msg->get();
      if (!current) {
        std::cerr << "! current" << std::endl;
        return false;
      }
      MDMatrixCloud cloud;
      // current->front()->toCloud(cloud);
      // addCloud(cloud, _status_msg->global_pose.value(), _status_msg->is_keyframe.value());
      return true;
    }
    return false;
  }

  void
  MDTrackerViewer::addCloud(const MDMatrixCloud& cloud, const Isometry3f& iso, bool is_keyframe) {
    _current_cloud.clear();
    std::back_insert_iterator<MDVectorCloud> out_current(_current_cloud);
    // cloud.copyTo(out_current);
    _current_pose = iso;
    _current_cloud.transformInPlace<TRANSFORM_CLASS::Isometry>(iso);
    if (!is_keyframe)
      return;

    // add the points to global
    std::back_insert_iterator<MDVectorCloud> out_global(globalCloud());
    _current_cloud.copyTo(out_global);
    // cerr << "add " << globalCloud().size() << endl;
    _trajectory.push_back(iso);
    voxelize();
    this->_need_redraw = true;
    ActiveDrawable::draw();
  }

  void MDTrackerViewer::reset() {
    _pyramid_msg.reset();
    _status_msg.reset();
    _pyramid_topic = "";
    _global_cloud[0].clear();
    _global_cloud[1].clear();
    _last_time_voxelize = 0;
    _current_pose.setIdentity();
    _trajectory.clear();
  }

  void MDTrackerViewer::_drawImpl(ViewerCanvasPtr gl_canvas_) const {
    if (!gl_canvas_)
      return;
    if (!_lists_created) {
      gl_canvas_->createList("cloud");
      gl_canvas_->createList("trajectory");
      _lists_created = true;
    }

    gl_canvas_->beginList("cloud");
    PointNormal3fVectorCloud drawn;
    drawn.resize(globalCloud().size());
    globalCloud().copyFieldTo<0, 0, PointNormal3fVectorCloud>(drawn);
    globalCloud().copyFieldTo<1, 1, PointNormal3fVectorCloud>(drawn);
    gl_canvas_->putPoints(drawn);

    drawn.resize(_current_cloud.size());
    _current_cloud.copyFieldTo<0, 0, PointNormal3fVectorCloud>(drawn);
    _current_cloud.copyFieldTo<1, 1, PointNormal3fVectorCloud>(drawn);
    gl_canvas_->pushColor();
    gl_canvas_->setColor(Vector3f(1, 0, 0));
    gl_canvas_->putPoints(drawn);
    gl_canvas_->popAttribute();

    gl_canvas_->pushMatrix();
    gl_canvas_->multMatrix(_current_pose.matrix());
    gl_canvas_->putReferenceSystem(3.f);
    gl_canvas_->popMatrix();
    gl_canvas_->endList();

    gl_canvas_->beginList("trajectory");
    for (const auto& iso : _trajectory) {
      Matrix4f m = iso.matrix();
      m.block<3, 3>(0, 0) *= 0.5;
      gl_canvas_->pushMatrix();
      gl_canvas_->multMatrix(m);
      gl_canvas_->putReferenceSystem(1.f);
      gl_canvas_->popMatrix();
    }
    gl_canvas_->endList();
    gl_canvas_->callList("cloud");
    gl_canvas_->callList("trajectory");
    gl_canvas_->flush();
  }

  using MDTrackerViewerPtr = std::shared_ptr<MDTrackerViewer>;
} // namespace md_slam
