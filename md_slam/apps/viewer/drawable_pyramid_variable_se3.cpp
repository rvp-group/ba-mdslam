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

#include "drawable_pyramid_variable_se3.h"

namespace srrg2_core {
  DrawablePyramidVariableSE3VBO::DrawablePyramidVariableSE3VBO(MDVariableSE3* instance_) :
    DrawableVBO_<MDVariableSE3*>(instance_) {
    if (!_instance)
      return;

    if (_cloud_vbo)
      return;

    auto pyramid = _instance->pyramid();
    if (!pyramid) {
      // cerr << "unable to load pyramid, aborting" << endl;
      return;
    }
    // get the highest level
    const auto& l = pyramid->front();
    MDMatrixVectorCloud matrix_cloud;
    l->toCloud(matrix_cloud);

    std::back_insert_iterator<PointNormalIntensity3fVectorCloud> dest(cloud);
    matrix_cloud.copyTo(dest);
    //_instance->setPyramid(0);
    _cloud_vbo.reset(new PointNormalIntensity3fVectorCloudVBO(cloud));
    // cerr << "variable_id: " << _instance->graphId() << " cloud_vbo_id: " <<
    // _cloud_vbo->_gl_vertex_buffer << " cloud_vbo_aid: " << _cloud_vbo->_gl_vertex_array << endl;
    // cerr << "current cloud size: " << cloud.size() << endl;
    std::cerr << ".";
    if (!_camera_pose_vbo) {
      _camera_pose_vbo.reset(new CameraPoseVBO(0.1, 0.1));
    }
  }

  void DrawablePyramidVariableSE3VBO::init() {
  }
  void DrawablePyramidVariableSE3VBO::update() {
  }
  void DrawablePyramidVariableSE3VBO::draw(const Eigen::Matrix4f& projection,
                                           const Eigen::Matrix4f& model_pose,
                                           const Eigen::Matrix4f& object_pose,
                                           const Eigen::Vector3f& light_direction,
                                           const CustomDraw& custom_draw) {
    if (!(_cloud_vbo && _camera_pose_vbo))
      return;

    Eigen::Matrix4f this_object_pose = object_pose * _instance->estimate().matrix();
    if (custom_draw.draw_cloud)
      _cloud_vbo->draw(projection, model_pose, this_object_pose, light_direction, custom_draw);
    if (custom_draw.draw_trajectory)
      _camera_pose_vbo->draw(
        projection, model_pose, this_object_pose, light_direction, custom_draw);
  }

  std::shared_ptr<CameraPoseVBO> DrawablePyramidVariableSE3VBO::_camera_pose_vbo;

} // namespace srrg2_core
