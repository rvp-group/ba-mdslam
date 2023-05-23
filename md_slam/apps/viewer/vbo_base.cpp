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

#include "vbo_base.h"
#include <iostream>
#include <string>

namespace srrg2_core {

  ShaderBase::ShaderBase(const char* vertex_shader_source, const char* fragment_shader_source) {
    // compile shaders
    // cerr << "ShaderBase::ctor " << this << endl;
    unsigned int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    // cerr << "_vertex_shader_id: " << vertex_shader << endl;

    glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertex_shader);
    int success;
    char info_log[512];
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(vertex_shader, 512, NULL, info_log);
      // std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << info_log << std::endl;
    }

    unsigned int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    // cerr << "_fragment_shader_id: " << fragment_shader << endl;

    glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
    glCompileShader(fragment_shader);
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(fragment_shader, 512, NULL, info_log);
      // std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << info_log << std::endl;
    }

    _shader_program = glCreateProgram();
    glAttachShader(_shader_program, vertex_shader);
    glAttachShader(_shader_program, fragment_shader);
    glLinkProgram(_shader_program);
    glGetProgramiv(_shader_program, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(_shader_program, 512, NULL, info_log);
      // std::cout << "ERROR::SHADER::PROGRAM::LINK_FAILED\n" << info_log << std::endl;
    }
    _model_pose_location      = glGetUniformLocation(_shader_program, "model_pose");
    _object_pose_location     = glGetUniformLocation(_shader_program, "object_pose");
    _projection_location      = glGetUniformLocation(_shader_program, "projection");
    _light_direction_location = glGetUniformLocation(_shader_program, "light_direction");
    _draw_intensity_location  = glGetUniformLocation(_shader_program, "draw_intensity");
    _m_intensity_location     = glGetUniformLocation(_shader_program, "m_intensity");

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
  }

  ShaderBase::~ShaderBase() {
    // cerr << "shader " << this << " dtor" << endl;
  }

  void VBOBase::callShader(const Eigen::Matrix4f& projection,
                           const Eigen::Matrix4f& model_pose,
                           const Eigen::Matrix4f& object_pose,
                           const Eigen::Vector3f& light_direction,
                           const CustomDraw& custom_draw) {
    auto ld = light_direction.normalized();
    glUseProgram(_shader->_shader_program);
    glUniformMatrix4fv(_shader->_model_pose_location, 1, GL_FALSE, model_pose.data());
    glUniformMatrix4fv(_shader->_object_pose_location, 1, GL_FALSE, object_pose.data());
    glUniformMatrix4fv(_shader->_projection_location, 1, GL_FALSE, projection.data());
    glUniform3fv(_shader->_light_direction_location, 1, ld.data());
    const float draw_intensity = (custom_draw.draw_intensity) ? 1.f : 0.f;
    glUniform1f(_shader->_draw_intensity_location, draw_intensity);
    glUniform1f(_shader->_m_intensity_location, custom_draw.m_intensity);
  }

} // namespace srrg2_core
