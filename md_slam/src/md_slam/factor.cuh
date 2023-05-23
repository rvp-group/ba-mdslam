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
#include "dual_matrix.cuh"
#include "factor_common.cuh"
#include "image_pyramid.h"
#include "utils.cuh"
#include <srrg_solver/solver_core/factor.h>
#include <srrg_solver/solver_core/factor_graph.h>
#include <srrg_solver/variables_and_factors/types_3d/variable_se3.h>
#include <srrg_system_utils/chrono.h>

namespace md_slam {
  using namespace srrg2_core;

  class MDFactor : public srrg2_solver::Factor_<
                     srrg2_solver::VariablePtrTuple_<srrg2_solver::VariableSE3QuaternionRight>>,
                   public MDFactorCommon {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ~MDFactor() {
      if (_workspace)
        delete _workspace;
      if (_entry_array)
        cudaFree(_entry_array);
      if (_cloud)
        delete _cloud;
    }

    //! populate H and b with measurments contributions
    void compute(bool chi_only = false, bool force = false) override;
    void serialize(ObjectData& odata, IdContext& context) override;   // serialize boss
    void deserialize(ObjectData& odata, IdContext& context) override; // deserialize boss
    // clang-format off
    inline bool isValid() const override { return true; }
    inline int measurementDim() const override { return 5; }
    // clang-format on

  protected:
    //! compute the pieces needed for the minimization
    void _linearize(bool chi_only = false);
    // Matrix3_6f _J_icp = Matrix3_6f::Zero();
    LinearSystemEntry* _entry_array = nullptr;
    int _entry_array_size           = 0;
  };
  using MDFactorPtr = std::shared_ptr<MDFactor>;

} // namespace md_slam
