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
#include "factor_stack.h"
#include <md_slam/pyramid_message.h>
#include <srrg_solver/solver_core/solver.h>

namespace md_slam {

  class MDPairwiseAligner : public srrg2_core::Configurable {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // clang-format off
    PARAM(PropertyConfigurable_<srrg2_solver::Solver>, solver, "solver running photometric alignment", srrg2_solver::SolverPtr(new srrg2_solver::Solver), nullptr);
    PARAM(PropertyFloat, omega_depth, "omega for the depth cue", 1, nullptr);
    PARAM(PropertyFloat, omega_normal, "omega for the normal cue", 1, nullptr);
    PARAM(PropertyFloat, omega_intensity, "omega for the intensity cue", 1, nullptr);
    PARAM(PropertyFloat, depth_rejection_threshold, "points with a depth error higher than this are rejected", 0.25, nullptr);
    PARAM(PropertyFloat, kernel_chi_threshold, "above this chi2, kernel acts", 1, nullptr);
    MDPairwiseAligner();

    inline void setMoving(MDImagePyramid* moving_) { _moving = moving_; }
    inline void setFixed(MDImagePyramid* fixed_) { _fixed = fixed_; }
    inline void setEstimate(const Isometry3f estimate_) { _estimate = estimate_; }
    void compute();
    void initialize();
    inline const Isometry3f& estimate() const { return _estimate; }
    inline const Matrix6f& informationMatrix() const { return _information; }
    // clang-format on

  protected:
    Eigen::Isometry3f _estimate       = Eigen::Isometry3f::Identity(); // initial guess and estimate
    srrg2_core::Matrix6f _information = srrg2_core::Matrix6f::Identity();
    MDFactorStack _md_factors;
    srrg2_solver::FactorGraph _pairwise_graph;
    std::shared_ptr<srrg2_solver::VariableSE3QuaternionRight> _v;
    MDImagePyramid* _moving = nullptr;
    MDImagePyramid* _fixed  = nullptr;
    bool is_initialized     = false;
    int increment_it        = 10;
  };

  using MDPairwiseAlignerPtr = std::shared_ptr<MDPairwiseAligner>;
} // namespace md_slam
