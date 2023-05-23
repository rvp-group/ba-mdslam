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
#define GL_GLEXT_PROTOTYPES 1
#include <Eigen/Core>
#include <GL/gl.h>
#include <memory>

namespace srrg2_core {

  struct CustomDraw {
    // some key bindings fields
    float m_intensity    = 0.5f;
    bool draw_cloud      = true;
    bool draw_trajectory = true;
    bool draw_intensity  = true; // false is normals

    // useful for ba viz
    float camera_distance          = 20.f;
    float camera_height            = 0.f;
    bool enable_auto_camera_motion = false;
  };

  struct ShaderBase {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    unsigned int _shader_program           = 0;
    unsigned int _model_pose_location      = 0;
    unsigned int _object_pose_location     = 0;
    unsigned int _projection_location      = 0;
    unsigned int _light_direction_location = 0;
    unsigned int _draw_intensity_location  = 0;
    unsigned int _m_intensity_location     = 0;
    ShaderBase(const char* vertex_shader_source, const char* fragment_shader_source);
    virtual ~ShaderBase();
  };

  using ShaderBasePtr = std::shared_ptr<ShaderBase>;

  struct VBOBase {
    ShaderBasePtr _shader;
    VBOBase(ShaderBasePtr shader_) : _shader(shader_) {
    }

    void callShader(const Eigen::Matrix4f& projection,
                    const Eigen::Matrix4f& model_pose,
                    const Eigen::Matrix4f& object_pose,
                    const Eigen::Vector3f& light_direction,
                    const CustomDraw& custom_draw);

    virtual void draw(const Eigen::Matrix4f& projection,
                      const Eigen::Matrix4f& model_pose,
                      const Eigen::Matrix4f& object_pose,
                      const Eigen::Vector3f& light_direction,
                      const CustomDraw& custom_draw) = 0;
  };

} // namespace srrg2_core
