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
#include "factor_common.cuh"
#include "image_pyramid.h"
#include "pyramid_variable_se3.h"
#include <srrg_solver/solver_core/factor.h>
#include <srrg_solver/solver_core/factor_graph.h>
#include <srrg_solver/variables_and_factors/types_3d/variable_se3.h>
#include <srrg_system_utils/chrono.h>
#include <srrg_viewer/active_drawable.h>

#include <cstddef>

namespace md_slam {
  using namespace srrg2_core;

  using WorkspaceSystem = std::pair<Workspace, LinearSystemEntryBi*>;

  class ALIGN(16) MDFactorBivariable
    : public MDFactorCommon,
      public srrg2_solver::Factor_<srrg2_solver::VariablePtrTuple_<MDVariableSE3, MDVariableSE3>> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using CloudMap =
      std::map<std::pair<size_t, size_t>, MDMatrixCloud>;    // std::shared_ptr<MDMatrixCloud>>
    using WorkspaceMap = std::map<uint8_t, WorkspaceSystem>; // allocate one workspace for level
    // populate H and b with measurments contributions
    MDFactorBivariable();
    ~MDFactorBivariable();
    void compute(bool chi_only = false, bool force = false) override;
    void serialize(ObjectData& odata, IdContext& context) override;
    void deserialize(ObjectData& odata, IdContext& context) override;
    static MDMatrixCloud* setMoving(MDPyramidLevelPtr, const size_t&, const size_t&);
    static WorkspaceSystem* getWorkspaceSystem(uint8_t level_, int rows_, int cols_);

    // clang-format off
    inline bool isValid() const override { return true; }
    inline int measurementDim() const override { return _rows * _cols * 5; }
    // clang-format on

  protected:
    // error and Jacobian computation
    PointStatusFlag errorAndJacobian(srrg2_core::Vector5f& e_,
                                     srrg2_core::Matrix5_6f& J_i,
                                     srrg2_core::Matrix5_6f& J_j,
                                     WorkspaceEntry& entry_,
                                     bool chi_only);
    void _linearize(bool chi_only = false);

    // static CloudMap _cloud_map;
    // static WorkspaceMap _workspace_map;
    MDPyramidLevelPtr _fixed_pyr = nullptr;
    // char pad[8];
    LinearSystemEntryBi* _entry_array = nullptr; // this needs to be here, padding issues
    int _entry_array_size             = 0;
    char pad[12];
    Isometry3f _X_ji;
    static CloudMap _cloud_map;
    static WorkspaceMap _workspace_map;
  };

  using MDFactorBivariablePtr = std::shared_ptr<MDFactorBivariable>;

} // namespace md_slam
