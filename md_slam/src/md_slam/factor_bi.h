#pragma once
#include "factor_common.h"
#include "image_pyramid.h"
#include "pyramid_variable_se3.h"
#include <srrg_solver/solver_core/factor.h>
#include <srrg_solver/solver_core/factor_graph.h>
#include <srrg_solver/variables_and_factors/types_3d/variable_se3.h>
#include <srrg_system_utils/chrono.h>
#include <srrg_viewer/active_drawable.h>

namespace md_slam {
  using namespace srrg2_core;

  class MDFactorBivariable
    : public MDFactorCommon,
      public srrg2_solver::Factor_<srrg2_solver::VariablePtrTuple_<MDVariableSE3, MDVariableSE3>> {
  public:
    using CloudMap = std::map<std::pair<size_t, size_t>, std::shared_ptr<MDMatrixCloud>>;
    // populate H and b with measurments contributions
    void compute(bool chi_only = false, bool force = false) override;
    void serialize(ObjectData& odata, IdContext& context) override;
    void deserialize(ObjectData& odata, IdContext& context) override;
    static void setMoving(MDMatrixCloud&, MDPyramidLevelPtr, const size_t&, const size_t&);
    // clang-format off
    inline bool isValid() const override { return true; }
    inline int measurementDim() const override { return _rows * _cols * 5; }
    // clang-format on
    static CloudMap _cloud_map;

  protected:
    // error and Jacobian computation
    PointStatusFlag errorAndJacobian(srrg2_core::Vector5f& e_,
                                     srrg2_core::Matrix5_6f& J_i,
                                     srrg2_core::Matrix5_6f& J_j,
                                     WorkspaceEntry& entry_,
                                     bool chi_only);
    void linearize(bool chi_only = false);
    MDPyramidLevelPtr _fixed_pyr = nullptr;
    Isometry3f _X_i, _inv_X_j, _X_j, _X_ji;
  };

  using MDFactorBivariablePtr = std::shared_ptr<MDFactorBivariable>;

} // namespace md_slam
