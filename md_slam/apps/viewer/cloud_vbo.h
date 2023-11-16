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
#include "get_pcl_offset.h"
#include "vbo_base.h"
#include <iostream>
#include <srrg_pcl/point_types.h>

namespace srrg2_core {
  template <typename PointCloudType_>
  struct CloudVBO_ : public VBOBase {
    unsigned int _gl_vertex_buffer = 0, _gl_vertex_array = 0;
    CloudVBO_(ShaderBasePtr shader_, const PointCloudType_& cloud_) :
      VBOBase(shader_),
      _cloud(cloud_) {
      glGenVertexArrays(1, &_gl_vertex_array);
      glGenBuffers(1, &_gl_vertex_buffer);
      glBindVertexArray(_gl_vertex_array);
      glBindBuffer(GL_ARRAY_BUFFER, _gl_vertex_buffer);
      glBufferData(GL_ARRAY_BUFFER,
                   sizeof(typename PointCloudType_::PointType) * _cloud.size(),
                   &(_cloud)[0],
                   GL_STATIC_DRAW);
    }
    virtual ~CloudVBO_() {
      // cerr << __PRETTY_FUNCTION__ << this << " dtor" << endl;
      glDeleteVertexArrays(1, &_gl_vertex_array);
      glDeleteBuffers(1, &_gl_vertex_buffer);
    }

    const PointCloudType_& _cloud;
  };

  struct PointNormalIntensity3fVectorCloudVBO
    : public CloudVBO_<PointNormalIntensity3fVectorCloud> {
    PointNormalIntensity3fVectorCloudVBO(const PointNormalIntensity3fVectorCloud& cloud_);
    void draw(const Eigen::Matrix4f& projection,
              const Eigen::Matrix4f& model_pose,
              const Eigen::Matrix4f& object_pose,
              const Eigen::Vector3f& light_direction,
              const CustomDraw& custom_draw) override;

  protected:
    static ShaderBasePtr _my_shader;
    static const char* _vertex_shader_source;
    static const char* _fragment_shader_source;
    static ShaderBasePtr getShader();
  };
} // namespace srrg2_core
