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

#include <Eigen/Dense>
#include <vector>

namespace srrg2_core {

  class BezierCurve {
  public:
    using PointType = Eigen::Vector3f;
    BezierCurve(const PointType& p0_,
                const PointType& p1_,
                const PointType& p2_,
                const PointType& p3_);

    PointType interpolate(const float t_) const;

  protected:
    PointType _p0, _p1, _p2, _p3;
  };

  class Spline {
  public:
    void addKeypoint(const Eigen::Matrix4f& camera_pose_);
    virtual size_t generateControlPoints();
    void interpolate(Eigen::Vector3f& t_interp_, Eigen::Quaternionf& q_interp_, const float t_);
    void pop();
    inline size_t size() const {
      return _t_control.size();
    }

  protected:
    std::vector<Eigen::Quaternionf> _q_control;
    std::vector<Eigen::Vector3f> _t_control;
    std::vector<BezierCurve> bezier_vector;
  };

  class HermiteSpline : public Spline {
  public:
    size_t generateControlPoints() override;
    // void interpolate(Eigen::Vector3f& t_interp_, Eigen::Quaternionf& q_interp_,
    //                  const float t_);
  };

  class TrajectoryInterpolatorBase {
  public:
    void addPoint(const Eigen::Matrix4f& camera_pose_);
    void addPoint(const Eigen::Vector3f& camera_t_, const Eigen::Quaternionf& camera_q_);
    virtual void
    interpolate(Eigen::Vector3f& t_interp_, Eigen::Quaternionf& q_interp_, const float t_) = 0;
    inline size_t size() const {
      return _t_control.size();
    }

    inline void pop() {
      _t_control.pop_back();
      _q_control.pop_back();
    }

  protected:
    std::vector<Eigen::Vector3f> _t_control;
    std::vector<Eigen::Quaternionf> _q_control;
  };

  class CatmullRomSpline : public TrajectoryInterpolatorBase {
  public:
    void
    interpolate(Eigen::Vector3f& t_interp_, Eigen::Quaternionf& q_interp_, const float t_) override;
    static Eigen::Vector3f Eq(const float t_,
                              const Eigen::Vector3f& p0_,
                              const Eigen::Vector3f& p1_,
                              const Eigen::Vector3f& p2_,
                              const Eigen::Vector3f& p3_);
  };

  class LinearSpline : public TrajectoryInterpolatorBase {
  public:
    void
    interpolate(Eigen::Vector3f& t_interp_, Eigen::Quaternionf& q_interp_, const float t_) override;
  };

  class LinearSplineDelta : public TrajectoryInterpolatorBase {
  public:
    void
    interpolate(Eigen::Vector3f& t_interp_, Eigen::Quaternionf& q_interp_, const float t_) override;
  };

} // namespace srrg2_core