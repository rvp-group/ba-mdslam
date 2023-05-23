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
#include "factor.cuh"
#include <srrg_solver/solver_core/solver_action_base.h>
#include <srrg_viewer/active_drawable.h>

namespace md_slam {
  using namespace srrg2_core;
  // holds a pool of factors
  // call setFixed, setMoving and makeFactors to populate the structure
  // afterwards add the stuff tothe graph, calling addFactors;
  struct MDFactorStack : public std::vector<MDFactorPtr> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setFixed(MDImagePyramid& pyramid);
    void setMoving(MDImagePyramid& pyramid);
    // creates a factor stack, requires fixed to be set
    void makeFactors();
    // assigns fixed and moving along the pyramid to all factors
    void assignPyramids();
    void setVariableId(srrg2_solver::VariableBase::Id id);
    void addFactors(srrg2_solver::FactorGraph& graph);
    void _fixedPyramidToDevice();

    MDImagePyramid* _fixed      = nullptr;
    MDImagePyramid* _prev_fixed = nullptr; // when fixed changes copy again to device
    MDImagePyramid* _moving     = nullptr;
  };

  struct MDFactorShowAction : public srrg2_solver::SolverActionBase, public ActiveDrawable {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setFactors(MDFactorStack& factors) {
      _md_factors = &factors;
    }
    void doAction() override;
    void _drawImpl(ViewerCanvasPtr gl_canvas_) const override;
    MDFactorStack* _md_factors = nullptr;
  };

  using MDFactorShowActionPtr = std::shared_ptr<MDFactorShowAction>;

} // namespace md_slam
