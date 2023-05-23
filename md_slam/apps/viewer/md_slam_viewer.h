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
#include "drawable_factor_graph_vbo.h"
#include <QGLViewer/qglviewer.h>
#include <mutex>
#include <semaphore.h>

namespace srrg2_core {

  using namespace srrg2_solver;
  using namespace md_slam;

  class MDViewer : public QGLViewer {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MDViewer(FactorGraphPtr graph_, std::mutex& proc_mutex_);
    ~MDViewer();
    void init() override;
    void draw() override;
    void keyPressEvent(QKeyEvent* e) override;

    void setBA();

    void setCamera(const Eigen::Isometry3f& camera_pose_) {
      _camera_pose = camera_pose_;
    };

    FactorGraphPtr _graph;
    CustomDraw _custom_draw;
    sem_t _sem;

  protected:
    std::shared_ptr<DrawableFactorGraphVBO> _graph_vbo;
    std::mutex& _proc_mutex;
    Eigen::Isometry3f _camera_pose = Eigen::Isometry3f::Identity();

    bool _is_ba = false;
  };

  using MDViewerPtr = std::shared_ptr<MDViewer>;

} // namespace srrg2_core
