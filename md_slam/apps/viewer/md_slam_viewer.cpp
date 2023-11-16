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

#include "md_slam_viewer.h"
#include "spline.h"
#include <QGLViewer/camera.h>
#include <QKeyEvent>

namespace srrg2_core {

  using namespace srrg2_solver;
  using namespace md_slam;

  MDViewer::MDViewer(FactorGraphPtr graph_, std::mutex& proc_mutex_) :
    _graph(graph_),
    _proc_mutex(proc_mutex_) {
    sem_init(&_sem, 0, 0);
  }

  MDViewer::~MDViewer() {
    sem_destroy(&_sem);
  }

  void MDViewer::setBA() {
    _is_ba = true;
  }

  void MDViewer::init() {
    QGLViewer::init();

    // set background color white
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // set some default settings
    // glEnable(GL_LINE_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_FLAT);
    setSceneRadius(100);
    glEnable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // don't save state
    setStateFileName(QString::null);

    // cerr << "instantiating required VBO" << endl;
    _proc_mutex.lock();
    _graph_vbo.reset(new DrawableFactorGraphVBO(_graph.get()));
    _proc_mutex.unlock();
    // mouse bindings
    setMouseBinding(Qt::NoModifier, Qt::RightButton, CAMERA, ZOOM);
    setMouseBinding(Qt::NoModifier, Qt::MidButton, CAMERA, TRANSLATE);
    setMouseBinding(Qt::ControlModifier, Qt::LeftButton, RAP_FROM_PIXEL);

    // add some new key descriptions
    setKeyDescription(Qt::Key_E, "Draw clouds attached to each pose");
    setKeyDescription(Qt::Key_J, "Draw trajectory, poses and edges");

    setKeyDescription(Qt::Key_N, "Color cloud point with intensity or normals");
    setKeyDescription(Qt::Key_U, "Makes intensity lighter");
    setKeyDescription(Qt::Key_I, "Makes intensity darker");

    setKeyDescription(Qt::Key_W, "BA | Increase gl camera distance from viewpoint");
    setKeyDescription(Qt::Key_Q, "BA | Decrease gl camera distance from viewpoint");
    setKeyDescription(Qt::Key_T, "BA | Increase gl camera height from viewpoint");
    setKeyDescription(Qt::Key_R, "BA | Decrease gl camera height from viewpoint");
    setKeyDescription(Qt::Key_M, "BA | Enable automatic gl camera motion");
    setKeyDescription(Qt::Key_P, "Trajectory Interpolation | Spawn new control-point at this view");
    setKeyDescription(Qt::Key_O, "Trajectory Interpolation | Play the interpolated trajectory");
    setKeyDescription(Qt::Key_L, "Trajectory Interpolation | Removes last control-point");
  }

  /**
   * @brief Trajectory Interpolation parameters
   */
  // HermiteSpline trajectory_spline;
  CatmullRomSpline trajectory_interpolator;
  // LinearSplineDelta trajectory_interpolator;
  bool trajectory_recorded  = false;
  bool trajectory_replay    = false;
  float trajectory_time     = 0.0;
  float trajectory_max_time = 1.0;
  Matrix3f R_replay;

  constexpr bool interpolate_camera_poses = false;
  /**
   * @brief Camera pose interpolation parameters
   * Defines the minimum translation and rotation thresholds to spawn a new Spline
   * control point
   */
  const float interpolate_threshold_t = 0.7; // [m]
  const float interpolate_threshold_r = 0.3; // [rad]

  void MDViewer::keyPressEvent(QKeyEvent* e) {
    switch (e->key()) {
      case Qt::Key_E:
        // display or not clouds
        _custom_draw.draw_cloud = !_custom_draw.draw_cloud;
        updateGL();
        break;
      case Qt::Key_J:
        _custom_draw.draw_trajectory = !_custom_draw.draw_trajectory;
        updateGL();
        break;
      case Qt::Key_N:
        // switch between normals and intensity
        _custom_draw.draw_intensity = !_custom_draw.draw_intensity;
        updateGL();
        break;
      case Qt::Key_I:
        if (_custom_draw.m_intensity <= 1.f) // make intensity lighter
          _custom_draw.m_intensity += 0.1f;
        updateGL();
        break;
      case Qt::Key_U:
        if (_custom_draw.m_intensity >= 0.f) // make intensity darker
          _custom_draw.m_intensity -= 0.1f;
        updateGL();
        break;
      case Qt::Key_W:
        _custom_draw.camera_distance += 0.01f;
        updateGL();
        break;
      case Qt::Key_Q:
        _custom_draw.camera_distance -= 0.01f;
        updateGL();
        break;
      case Qt::Key_T:
        _custom_draw.camera_height += 0.01f;
        updateGL();
        break;
      case Qt::Key_R:
        _custom_draw.camera_height -= 0.01f;
        updateGL();
        break;
      case Qt::Key_M:
        _custom_draw.enable_auto_camera_motion = !_custom_draw.enable_auto_camera_motion;
        updateGL();
        break;
      case Qt::Key_P: {
        Eigen::Matrix4f model_current;
        glGetFloatv(GL_MODELVIEW_MATRIX, model_current.data());

        if (!interpolate_camera_poses) {
          // trajectory_spline.addKeypoint(model_current);
          // trajectory_interpolator.addPoint(model_current);
          Vector3f t_current;
          Quaternionf q_current;
          t_current << camera()->position().x, camera()->position().y, camera()->position().z;
          q_current.coeffs() << camera()->orientation()[0], camera()->orientation()[1],
            camera()->orientation()[2], camera()->orientation()[3];
          trajectory_interpolator.addPoint(t_current, q_current);
        } else {
          Eigen::Isometry3f T_prev;
          for (auto var = _graph->variables().begin(); var != _graph->variables().end(); ++var) {
            MDVariableSE3* v = dynamic_cast<MDVariableSE3*>(var.value());
            if (!v)
              continue;

            const Isometry3f& T = v->estimate();
            const auto delta_T  = T_prev.inverse() * T;
            const auto t_norm   = delta_T.translation().norm();
            const auto R_norm   = Eigen::AngleAxisf(delta_T.linear()).angle();

            if (T_prev.isApprox(Eigen::Isometry3f::Identity()) || t_norm >= 1.0 ||
                fabs(R_norm) > 0.7) {
              T_prev = T;
              // trajectory_spline.addKeypoint(T.matrix().inverse());
              trajectory_interpolator.addPoint(T.matrix());
            }
          }
        }
        updateGL();
        break;
      }
      case Qt::Key_O: {
        if (trajectory_interpolator.size() < 2) {
          break;
        }
        trajectory_replay = true;
        trajectory_time   = 0.0f;
        updateGL();
        break;
      }
      case Qt::Key_L: {
        // trajectory_spline.pop();
        trajectory_interpolator.pop();
        break;
      }

      // Default calls the original method to handle standard keys
      default:
        QGLViewer::keyPressEvent(e);
    }
  }

  void MDViewer::draw() {
    QGLViewer::draw();
    _proc_mutex.lock();
    _graph_vbo->update();
    Eigen::Matrix4f model;
    Eigen::Matrix4f projection;
    Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
    Eigen::Vector3f light_direction(0.5, 0.5, -0.5);
    light_direction.normalize();
    glGetFloatv(GL_MODELVIEW_MATRIX, model.data());
    glGetFloatv(GL_PROJECTION_MATRIX, projection.data());
    if (!_is_ba) {
      Eigen::Matrix4f temp   = Eigen::Matrix4f::Identity();
      temp.block<3, 1>(0, 3) = _camera_pose.matrix().inverse().block<3, 1>(0, 3);
      _camera_pose(2, 3)     = 0;
      model                  = model * _camera_pose.matrix().inverse();
    } else {
      model = model * _camera_pose.matrix();
    }
    if (trajectory_replay) {
      Vector3f t_replay;
      Quaternionf q_replay;
      // trajectory_spline.interpolate(t_replay, q_replay, trajectory_time);
      trajectory_interpolator.interpolate(t_replay, q_replay, trajectory_time);
      std::cerr << "[t=" << trajectory_time << "] Interpolating pose: t=" << t_replay.transpose()
                << std::endl;
      camera()->setOrientation(
        qglviewer::Quaternion(q_replay.x(), q_replay.y(), q_replay.z(), q_replay.w()));
      camera()->setPosition(qglviewer::Vec(t_replay.x(), t_replay.y(), t_replay.z()));
      trajectory_time += 0.001;
    }
    if (trajectory_time >= trajectory_max_time) {
      trajectory_replay = false;
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _graph_vbo->draw(projection, model, mat, light_direction, _custom_draw);
    _proc_mutex.unlock();
  }

  std::shared_ptr<DrawableFactorGraphVBO> _graph_vbo;

} // namespace srrg2_core
