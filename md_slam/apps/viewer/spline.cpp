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

#include "spline.h"

#include <iostream>

namespace srrg2_core {

  template <typename T>
  T lerp(const T& start_, const T& end_, const float t_) {
    return start_ + t_ * (end_ - start_);
  }

  BezierCurve::BezierCurve(const PointType& p0_,
                           const PointType& p1_,
                           const PointType& p2_,
                           const PointType& p3_) :
    _p0(p0_),
    _p1(p1_),
    _p2(p2_),
    _p3(p3_) {
  }

  BezierCurve::PointType BezierCurve::interpolate(const float t_) const {
    float it = 1.0 - t_;

    return _p0 * (it * it * it) + 3 * _p1 * t_ * (it * it) + 3 * _p2 * (t_ * t_) * it +
           _p3 * (t_ * t_ * t_);
  }

  void Spline::addKeypoint(const Eigen::Matrix4f& camera_pose_) {
    // Extract t and q from camera_pose and store it into t_control and q_control
    const Eigen::Quaternionf q(camera_pose_.topLeftCorner<3, 3>());
    const Eigen::Vector3f t(camera_pose_.rightCols<1>().head<3>());
    std::cerr << "Saving KeyPoint t= " << t.transpose() << " q= " << q.coeffs().transpose()
              << std::endl;
    _t_control.push_back(t);
    _q_control.push_back(q);
  }

  size_t Spline::generateControlPoints() {
    // Create control points for t
    std::vector<Eigen::Vector3f> t_augmented_control;

    // Augment control points for t
    for (size_t i = 0; i < _t_control.size(); ++i) {
      const auto& t_current = _t_control[i];
      if (i == 0) {
        /**
         * Spawn a new control point in the direction from t0 towards t1
         */
        const auto& t_next   = _t_control[i + 1];
        const auto direction = (t_next - t_current) * 0.1;
        t_augmented_control.push_back(t_current);
        t_augmented_control.push_back(t_current + direction * 0.1);

      } else if (i == _t_control.size() - 1) {
        /**
         * @brief Spawn a new control point in the direction of t(i) towards
         * t(i-1)
         */
        const auto& t_prev   = _t_control[i - 1];
        const auto direction = (t_prev - t_current) * 0.1;
        t_augmented_control.push_back(t_current + direction * 0.1);
        t_augmented_control.push_back(t_current);
      } else {
        /**
         * @brief Spawn two control points with center t(i) and opposite
         * directions. The direction is computed by looking
         */
        const auto& t_next   = _t_control[i + 1];
        const auto direction = (t_next - t_current) * 0.1;
        t_augmented_control.push_back(t_current - direction * 0.1);
        t_augmented_control.push_back(t_current);
        t_augmented_control.push_back(t_current + direction * 0.1);
      }
    }
    std::cerr << "\nGenerated a total of " << t_augmented_control.size() << " control points.\n";
    // Generate Bezier Curves
    bezier_vector.clear();

    for (size_t i = 0; i < _t_control.size() - 1; ++i) {
      const auto& p0 = t_augmented_control[3 * i];
      const auto& p1 = t_augmented_control[3 * i + 1];
      const auto& p2 = t_augmented_control[3 * i + 2];
      const auto& p3 = t_augmented_control[3 * i + 3];
      // std::cerr << "Spline contains:"
      //           << "\np0=" << p0.transpose() << "\np1=" << p1.transpose()
      //           << "\np2=" << p2.transpose() << "\np3=" << p3.transpose()
      //           << std::endl;
      // std::cerr << "\nCreating a Spline from " << p0.transpose() << " to "
      //           << p3.transpose() << std::endl;
      bezier_vector.push_back(BezierCurve(p0, p1, p2, p3));
    }
    return bezier_vector.size();
  }

  void
  Spline::interpolate(Eigen::Vector3f& t_interp_, Eigen::Quaternionf& q_interp_, const float t_) {
    const size_t bezier_idx = (size_t) floor(t_);
    const float t_rel       = t_ - floor(t_);
    const auto& bezier      = bezier_vector[bezier_idx];
    t_interp_               = bezier.interpolate(t_rel);
    q_interp_               = _q_control[bezier_idx].slerp(t_rel, _q_control[bezier_idx + 1]);
    //   _q_control[3 * bezier_idx].slerp(t_rel, _q_control[3 * bezier_idx +
    //   3]);
  }

  void Spline::pop() {
    if (_t_control.size()) {
      _t_control.pop_back();
      _q_control.pop_back();
    }
  }

  size_t HermiteSpline::generateControlPoints() {
    // Compute velocities for every keypoint
    std::vector<Eigen::Vector3f> _v_control(_t_control.size());
    for (size_t i = 0; i < _t_control.size() - 1; ++i) {
      const auto& p0 = _t_control[i];
      const auto& p1 = _t_control[i + 1];
      _v_control[i]  = ((p1 - p0));
    }
    _v_control[_t_control.size() - 1].setZero();

    // Generate Bezier curves
    bezier_vector.clear();
    for (size_t i = 0; i < _t_control.size() - 1; ++i) {
      const auto& p0 = _t_control[i];
      const auto& p3 = _t_control[i + 1];
      const auto p1  = p0 + _v_control[i] / 3;
      const auto p2  = p3 - _v_control[i + 1] / 3;
      bezier_vector.push_back(BezierCurve(p0, p1, p2, p3));
    }

    return bezier_vector.size();
  }

  void TrajectoryInterpolatorBase::addPoint(const Eigen::Matrix4f& camera_pose_) {
    // Extract t and q from camera_pose and store it into t_control and q_control
    const Eigen::Quaternionf q(camera_pose_.topLeftCorner<3, 3>());
    const Eigen::Vector3f t(camera_pose_.rightCols<1>().head<3>());
    _t_control.push_back(t);
    _q_control.push_back(q);
  }

  void TrajectoryInterpolatorBase::addPoint(const Eigen::Vector3f& camera_t_,
                                            const Eigen::Quaternionf& camera_q_) {
    _t_control.push_back(camera_t_);
    _q_control.push_back(camera_q_);
  }

  void CatmullRomSpline::interpolate(Eigen::Vector3f& t_interp_,
                                     Eigen::Quaternionf& q_interp_,
                                     const float t_) {
    const float delta_t = 1.0 / _t_control.size();

    int p = (int) (t_ / delta_t);
#define BOUNDS(pp)                              \
  {                                             \
    if (pp < 0)                                 \
      pp = 0;                                   \
    else if (pp >= (int) _t_control.size() - 1) \
      pp = _t_control.size() - 1;               \
  }

    int p0 = p - 1;
    BOUNDS(p0);
    int p1 = p;
    BOUNDS(p1);
    int p2 = p + 1;
    BOUNDS(p2);
    int p3 = p + 2;
    BOUNDS(p3);

    float lt = (t_ - delta_t * (float) p) / delta_t;
    t_interp_ =
      CatmullRomSpline::Eq(lt, _t_control[p0], _t_control[p1], _t_control[p2], _t_control[p3]);
    // const auto q_a = _q_control[p0].slerp(lt, _q_control[p1]);
    // const auto q_b = _q_control[p1].slerp(lt, _q_control[p2]);
    // const auto q_c = _q_control[p2].slerp(lt, _q_control[p3]);
    // const auto q_d = q_a.slerp(lt, q_b);
    // const auto q_e = q_b.slerp(lt, q_c);
    // q_interp_ = q_d.slerp(lt, q_e);
    q_interp_ = _q_control[p1].slerp(lt, _q_control[p2]);
  }

  Eigen::Vector3f CatmullRomSpline::Eq(const float t_,
                                       const Eigen::Vector3f& p0_,
                                       const Eigen::Vector3f& p1_,
                                       const Eigen::Vector3f& p2_,
                                       const Eigen::Vector3f& p3_) {
    const auto t   = t_;
    const auto t2  = pow(t, 2);
    const auto t3  = pow(t, 3);
    const float b0 = 0.5 * (-t3 + 2 * t2 - t);
    const float b1 = 0.5 * (3 * t3 - 5 * t2 + 2);
    const float b2 = 0.5 * (-3 * t3 + 4 * t2 + t);
    const float b3 = 0.5 * (t3 - t2);

    return p0_ * b0 + p1_ * b1 + p2_ * b2 + p3_ * b3;
  }

  void LinearSpline::interpolate(Eigen::Vector3f& t_interp_,
                                 Eigen::Quaternionf& q_interp_,
                                 const float t_) {
    const float delta_t = 1.0 / (float) _t_control.size();

    int p = (int) (t_ / delta_t);
#define BOUNDS(pp)                              \
  {                                             \
    if (pp < 0)                                 \
      pp = 0;                                   \
    else if (pp >= (int) _t_control.size() - 1) \
      pp = _t_control.size() - 1;               \
  }

    int& p0 = p;
    BOUNDS(p0);
    int p1 = p + 1;
    BOUNDS(p1);

    float lt  = (t_ - delta_t * (float) p) / delta_t;
    t_interp_ = _t_control[p0] + lt * (_t_control[p1] - _t_control[p0]);
    q_interp_ = _q_control[p0].slerp(lt, _q_control[p1]);
  }

  void LinearSplineDelta::interpolate(Eigen::Vector3f& t_interp_,
                                      Eigen::Quaternionf& q_interp_,
                                      const float t_) {
    const float delta_t = 1.0 / (float) _t_control.size();

    int p = (int) (t_ / delta_t);
#define BOUNDS(pp)                              \
  {                                             \
    if (pp < 0)                                 \
      pp = 0;                                   \
    else if (pp >= (int) _t_control.size() - 1) \
      pp = _t_control.size() - 1;               \
  }

    int& p0 = p;
    BOUNDS(p0);
    int p1 = p0 + 1;
    BOUNDS(p1);

    float lt = (t_ - delta_t * (float) p) / delta_t;

    // Compute delta
    Eigen::Isometry3f T0, T1;
    T0.linear()             = _q_control[p0].normalized().toRotationMatrix();
    T0.translation()        = _t_control[p0];
    T1.linear()             = _q_control[p1].normalized().toRotationMatrix();
    T1.translation()        = _t_control[p1];
    Eigen::Matrix4f T_delta = (T0.inverse() * T1).matrix();

    Eigen::Vector3f t_delta = T_delta.rightCols<1>().head<3>();
    Eigen::Quaternionf q_delta(T_delta.topLeftCorner<3, 3>());

    Eigen::Isometry3f T_interp(Eigen::Isometry3f::Identity());
    T_interp.linear()      = Eigen::Quaternionf::Identity().slerp(lt, q_delta).toRotationMatrix();
    T_interp.translation() = lt * t_delta;

    t_interp_ = (T0 * T_interp).translation();
    q_interp_ = Eigen::Quaternionf((T0 * T_interp).linear()).normalized();
  }

} // namespace srrg2_core