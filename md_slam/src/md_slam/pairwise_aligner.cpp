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

#include "pairwise_aligner.h"
#include <srrg_system_utils/shell_colors.h>
namespace md_slam {

  using namespace srrg2_core;
  using namespace srrg2_solver;

  // prev fixed
  // curr moving

  MDPairwiseAligner::MDPairwiseAligner() {
    param_solver.setValue(SolverPtr(new Solver));
    auto term_crit = std::dynamic_pointer_cast<SimpleTerminationCriteria>(
      param_solver->param_termination_criteria.value());
    term_crit->param_epsilon.setValue(1e-5);
    param_solver->param_max_iterations.value() = vector<int>{10, 20, 50};

    _v.reset(new VariableSE3QuaternionRight);
    _v->setGraphId(0);
    _v->setEstimate(Eigen::Isometry3f::Identity());
    _pairwise_graph.addVariable(_v);
  }

  void MDPairwiseAligner::initialize() {
    for (auto& action : param_solver->param_actions.value()) {
      MDFactorShowActionPtr show_action = std::dynamic_pointer_cast<MDFactorShowAction>(action);
      if (show_action)
        show_action->setFactors(_md_factors);
    }
    _md_factors.setFixed(*_fixed); // previous in tracking
    _md_factors.makeFactors();
    _md_factors.setVariableId(_v->graphId());
    _md_factors.addFactors(_pairwise_graph);
    for (auto& f : _md_factors) {
      f->setOmegaDepth(param_omega_depth.value());
      f->setOmegaIntensity(param_omega_intensity.value());
      f->setOmegaNormals(param_omega_normal.value());
      f->setDepthRejectionThreshold(param_depth_rejection_threshold.value());
      f->setKernelChiThreshold(param_kernel_chi_threshold.value());
    }
  }

  void MDPairwiseAligner::compute() {
    _v->setEstimate(_estimate);
    _md_factors.setFixed(*_fixed);
    _md_factors.setMoving(*_moving);
    _md_factors.assignPyramids();
    param_solver->setGraph(_pairwise_graph);
    param_solver->compute();

    // check that optimization did not fail
    bool failure       = false;
    const int n_levels = _moving->_levels.size();
    for (int i = n_levels - 1; i >= 0; --i) {
      failure = failure | (_md_factors[i]->stats().status == srrg2_solver::FactorStats::Suppressed);
    }

    if (failure) {
      // change the number of iterations if opt failed
      const std::vector<int> max_iters_copy(param_solver->param_max_iterations.value());

      int increment = 0;
      for (int i = n_levels - 1; i >= 0; --i) {
        std::vector<int> iters_i(n_levels, 0);
        iters_i[i] = max_iters_copy[i] + increment;
        param_solver->param_max_iterations.setValue(iters_i);
        _v->setEstimate(_estimate);
        param_solver->compute();
        // if factor suppressed
        if (_md_factors[i]->stats().status == srrg2_solver::FactorStats::Suppressed) {
          increment = increment_it;
          std::cerr << FG_YELLOW("Warning | ") << FG_YELLOW("level ") << i
                    << FG_YELLOW(" failure, adding: ") << increment_it
                    << FG_YELLOW(" to next level, full its: ") << max_iters_copy[i - 1] + increment
                    << std::endl;
        }

        _estimate = _v->estimate();
      }

      // restore the old values
      param_solver->param_max_iterations.setValue(max_iters_copy);
    }

    // get data off
    _estimate    = _v->estimate();
    _information = param_solver->extractFisherInformationBlock(*_v);
  }
} // namespace md_slam
