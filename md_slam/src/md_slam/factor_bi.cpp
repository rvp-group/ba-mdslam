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

#include "factor_bi.cuh"
#include <srrg_solver/solver_core/factor_impl.cpp>

namespace md_slam {
  using namespace srrg2_core;

  MDFactorBivariable::CloudMap MDFactorBivariable::_cloud_map;
  MDFactorBivariable::WorkspaceMap MDFactorBivariable::_workspace_map;

  MDFactorBivariable::MDFactorBivariable() {
    _X_ji.setIdentity();
  }

  MDFactorBivariable::~MDFactorBivariable() {
    for (auto& [level, ws_entry] : _workspace_map) {
      // first clean cuda stuff
      if (ws_entry.second) {
        cudaFree(ws_entry.second);
        ws_entry.second = nullptr;
      }
    }
    // clean the static rest
    _workspace_map.clear();
    _cloud_map.clear();
  }

  MDMatrixCloud* MDFactorBivariable::setMoving(MDPyramidLevelPtr pyr_level_,
                                               const size_t& var_id_,
                                               const size_t& level_) {
    // TODO once level is over, remove all clouds belonging to that level
    const auto key = std::pair<size_t, size_t>(var_id_, level_);
    if (auto it{_cloud_map.find(key)}; it != std::end(_cloud_map)) {
      // if cloud exists in map, retrieve its address
      return &it->second;
    } else {
      // if does not exist create one from pyramid level
      _cloud_map.insert(
        {key, MDMatrixCloud()}); // TODO use ptr, avoiding copy each time we call a cloud
      MDMatrixCloud* cloud = &_cloud_map.find(key)->second;
      pyr_level_->toCloudDevice(cloud); // make cloud generation on device only
      return cloud;
    }
  }

  WorkspaceSystem* MDFactorBivariable::getWorkspaceSystem(uint8_t level_, int rows_, int cols_) {
    // if workspace exists in map, retrieve, otherwise allocate right amount of space
    if (auto it{_workspace_map.find(level_)}; it != std::end(_workspace_map)) {
      return &it->second;
    } else {
      _workspace_map.insert({level_, WorkspaceSystem()});
      WorkspaceSystem* ws = &_workspace_map.find(level_)->second;
      ws->first           = Workspace(rows_, cols_);
      CUDA_CHECK(cudaMalloc((void**) &ws->second, sizeof(LinearSystemEntryBi) * ws->first.size()));
      return ws;
    }
  }

  void MDFactorBivariable::compute(bool chi_only, bool force) {
    if (!this->isActive() && !force)
      return;

    if (level() != this->currentLevel()) {
      _stats.status = srrg2_solver::FactorStats::Suppressed;
      return;
    }
    _stats.status = srrg2_solver::FactorStats::Inlier;

    // retrieve the variables
    auto& v_i = _variables.at<0>();
    auto& v_j = _variables.at<1>();

    // retrieve the pyramid level from the variables
    auto level_i            = v_i->pyramid()->at(this->level());
    MDPyramidLevel* level_j = v_j->pyramid()->at(this->level()).get();

    // retrieve level to inverse project from factory, do this operation only once
    _cloud = setMoving(level_i, v_i->graphId(), this->level());
    // retrieve other pyr level
    setFixed(level_j);

    // store current estimate
    const Isometry3f& X_i = v_i->estimate();
    const Isometry3f& X_j = v_j->estimate();
    _X_ji                 = X_j.inverse() * X_i;

    setMovingInFixedEstimate(_X_ji);

    // get right address from factory
    WorkspaceSystem* system = getWorkspaceSystem(this->level(), _rows, _cols);
    _workspace              = &system->first;
    _entry_array            = system->second;
    // always compute projections, even if variable is not updated, ba works in batch
    computeProjections();

    // for each point, compute the jacobians, check the state of the point
    // fill H and b to get new estimate
    _linearize(chi_only);
  }

  void MDFactorBivariable::serialize(ObjectData& odata, IdContext& context) {
    Identifiable::serialize(odata, context);
    if (level() != 0) // TODO serialize only finest level
      return;
    odata.setInt("graph_id", graphId());
    odata.setBool("enabled", enabled());
    odata.setInt("level", level());
    ArrayData* adata = new ArrayData;
    for (int pos = 0; pos < NumVariables; ++pos) {
      adata->add((int) variableId(pos));
    }
    odata.setField("variables", adata);
    odata.setFloat("omega_intensity", _omega_intensity);
    odata.setFloat("omega_depth", _omega_depth);
    odata.setFloat("omega_normals", _omega_normals);
    odata.setFloat("dept_rejection", _depth_error_rejection_threshold);
    odata.setFloat("kernel_chi", _kernel_chi_threshold);
  }

  void MDFactorBivariable::deserialize(ObjectData& odata, IdContext& context) {
    Identifiable::deserialize(odata, context);
    _graph_id = odata.getInt("graph_id");
    if (odata.getField("enabled")) {
      FactorBase::_enabled = odata.getBool("enabled");
    }
    if (odata.getField("level")) {
      FactorBase::setLevel(odata.getInt("level"));
    }
    ArrayData* adata = dynamic_cast<ArrayData*>(odata.getField("variables"));
    int pos          = 0;
    for (auto it = adata->begin(); it != adata->end(); ++it) {
      ThisType::_variables.setGraphId(pos, (*it)->getInt());
      ++pos;
    }
    _omega_intensity                 = odata.getFloat("omega_intensity");
    _omega_depth                     = odata.getFloat("omega_depth");
    _omega_normals                   = odata.getFloat("omega_normals");
    _depth_error_rejection_threshold = odata.getFloat("dept_rejection");
    _kernel_chi_threshold            = odata.getFloat("kernel_chi");
  }

} // namespace md_slam
