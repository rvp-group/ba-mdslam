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
#include "camera_pose_vbo.h"
#include "cloud_vbo.h"
#include "drawable_pyramid_variable_se3.h"
#include "factors_binary_vbo.h"

namespace srrg2_core {

  using namespace srrg2_solver;
  using namespace md_slam;

  struct DrawableFactorGraphVBO : public DrawableVBO_<FactorGraph*> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DrawableFactorGraphVBO(FactorGraph* instance_);
    void init() override;

    void update() override;

    void draw(const Eigen::Matrix4f& projection,
              const Eigen::Matrix4f& model_pose,
              const Eigen::Matrix4f& object_pose,
              const Eigen::Vector3f& light_direction,
              const CustomDraw& custom_draw) override;

    std::shared_ptr<FactorsBinaryVBO> _factors_binary_vbo;
    std::map<VariableBase::Id, std::shared_ptr<DrawablePyramidVariableSE3VBO>> _variables_vbo;
  };

} // namespace srrg2_core
