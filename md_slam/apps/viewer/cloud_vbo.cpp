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

#include "cloud_vbo.h"
namespace srrg2_core {

  ShaderBasePtr PointNormalIntensity3fVectorCloudVBO::_my_shader;

  const char* PointNormalIntensity3fVectorCloudVBO::_vertex_shader_source =
    "#version 330 core\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "layout (location = 0) in vec3 coords;\n"
    "layout (location = 1) in vec3 normal;\n"
    "layout (location = 2) in float intensity;\n"
    "uniform mat4 model_pose;\n"
    "uniform mat4 object_pose;\n"
    "uniform mat4 projection;\n"
    "uniform vec3 light_direction;\n"
    "uniform float draw_intensity;\n"
    "uniform float m_intensity;\n"
    "out vec3 obj_color;\n"
    "out vec3 diffuse;\n"
    "out float scale_intensity;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = projection*model_pose*object_pose*vec4(coords.x, coords.y, coords.z, 1);\n"
    "   mat3 R = mat3(object_pose);\n"
    "   vec3 n = R*normal;\n"
    "   float diff = max(dot(-n, light_direction), 0.0);\n"
    // "   vec3 light_color = vec3(0.33, 0.42, 0.18);\n"
    "   vec3 light_color = vec3(1.0, 1.0, 1.0);\n"
    "   diffuse = diff*light_color;\n"
    "if(draw_intensity > 0.5)\n"
    "   obj_color = diffuse*vec3(intensity, intensity, intensity);\n"
    "else\n"
    "   obj_color = -n;\n"
    "   scale_intensity = m_intensity;\n"
    "}\0";

  const char* PointNormalIntensity3fVectorCloudVBO::_fragment_shader_source =
    "#version 330 core\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "out vec4 FragColor;\n"
    "in vec3  obj_color;\n"
    "in vec3 diffuse;\n"
    "in float scale_intensity;\n"
    "vec3 result = obj_color;\n" // use diffuse here
    "void main()\n"
    "{\n"
    "   FragColor = vec4(result, scale_intensity);\n"
    "}\n\0";

  ShaderBasePtr PointNormalIntensity3fVectorCloudVBO::getShader() {
    if (!_my_shader)
      _my_shader.reset(new ShaderBase(_vertex_shader_source, _fragment_shader_source));
    return _my_shader;
  }

  PointNormalIntensity3fVectorCloudVBO::PointNormalIntensity3fVectorCloudVBO(
    const PointNormalIntensity3fVectorCloud& cloud_) :
    CloudVBO_<PointNormalIntensity3fVectorCloud>(getShader(), cloud_) {
    glBindVertexArray(_gl_vertex_array);
    glVertexAttribPointer(0,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(PointNormalIntensity3fVectorCloud::PointType),
                          (const void*) field_offset_<PointNormalIntensity3f, 0>);

    glVertexAttribPointer(1,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(PointNormalIntensity3fVectorCloud::PointType),
                          (const void*) field_offset_<PointNormalIntensity3f, 1>);

    glVertexAttribPointer(2,
                          1,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(PointNormalIntensity3fVectorCloud::PointType),
                          (const void*) field_offset_<PointNormalIntensity3f, 2>);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  void PointNormalIntensity3fVectorCloudVBO::draw(const Eigen::Matrix4f& projection,
                                                  const Eigen::Matrix4f& model_pose,
                                                  const Eigen::Matrix4f& object_pose,
                                                  const Eigen::Vector3f& light_direction,
                                                  const CustomDraw& custom_draw) {
    callShader(projection, model_pose, object_pose, light_direction, custom_draw);
    glBindVertexArray(_gl_vertex_array);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glDrawArrays(GL_POINTS, 0, _cloud.size());
  }
} // namespace srrg2_core
