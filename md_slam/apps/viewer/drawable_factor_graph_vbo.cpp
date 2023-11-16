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

#include "drawable_factor_graph_vbo.h"

namespace srrg2_core {

  DrawableFactorGraphVBO::DrawableFactorGraphVBO(FactorGraph* instance_) :
    DrawableVBO_<FactorGraph*>(instance_) {
    if (!_instance)
      return;
    _factors_binary_vbo.reset(new FactorsBinaryVBO(*_instance));
  }

  void DrawableFactorGraphVBO::init() {
  }

  void DrawableFactorGraphVBO::update() {
    if (!_instance && _variables_vbo.size()) {
      _variables_vbo.clear();
      return;
    }
    if (_factors_binary_vbo)
      _factors_binary_vbo->update();
    // scan the variables in viewer, and remove the ones that are not anymore in the game
    for (auto v_it = _variables_vbo.begin(); v_it != _variables_vbo.end(); ++v_it) {
      if (!_instance->variable(v_it->first)) {
        auto v_erased = v_it;
        ++v_it;
        _variables_vbo.erase(v_erased);
      }
    }
    // scan the variables in the graph, and add those that are new
    for (auto v_it : _instance->variables()) {
      VariableBase* v_base = v_it.second;
      MDVariableSE3* v     = dynamic_cast<MDVariableSE3*>(v_base);
      if (!v)
        continue;
      if (!_variables_vbo.count(v_it.first)) {
        _variables_vbo.insert(std::make_pair(
          v_it.first,
          std::shared_ptr<DrawablePyramidVariableSE3VBO>(new DrawablePyramidVariableSE3VBO(v))));
      }
    }
  }

  void DrawableFactorGraphVBO::draw(const Eigen::Matrix4f& projection,
                                    const Eigen::Matrix4f& model_pose,
                                    const Eigen::Matrix4f& object_pose,
                                    const Eigen::Vector3f& light_direction,
                                    const CustomDraw& custom_draw) {
    for (auto v_it : _variables_vbo) {
      v_it.second->draw(projection, model_pose, object_pose, light_direction, custom_draw);
    }
    if (custom_draw.draw_trajectory) {
      if (_factors_binary_vbo)
        _factors_binary_vbo->draw(
          projection, model_pose, object_pose, light_direction, custom_draw);
    }
  }

} // namespace srrg2_core
