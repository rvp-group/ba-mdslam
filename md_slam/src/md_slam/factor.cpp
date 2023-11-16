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

#include "factor.cuh"
#include <srrg_system_utils/chrono.h>

namespace md_slam {

  using namespace srrg2_core;

  void MDFactor::compute(bool chi_only, bool force) {
    Chrono time("compute", &timings, false);

    // some preliminary checks
    if (!this->isActive() && !force) {
      return;
    }

    if (level() != this->currentLevel()) {
      _stats.status = srrg2_solver::FactorStats::Suppressed;
      return;
    }

    // this factor is seen by the solver as a single contribution for the solver
    _stats.status = srrg2_solver::FactorStats::Inlier;

    // checks if needed variables are correctly set
    assert((_level_ptr->rows() != 0 || _level_ptr->cols() != 0) &&
           "MDFactor::compute|level rows or columns set to zero");
    assert((_level_ptr->max_depth != 0.f || _level_ptr->min_depth != 0.f) &&
           "MDFactor::compute|level level max_depth or min_depth set to zero");

    setMovingInFixedEstimate(_variables.at<0>()->estimate());
    // for chi only calculation
    if (this->_variables.updated()) {
      Chrono time("projections", &timings, false);
      computeProjections();
    }
    // performs linearization through jacobian computation
    _linearize(chi_only);
  }

  void MDFactor::serialize(ObjectData& odata, IdContext& context) {
    Identifiable::serialize(odata, context);
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

  void MDFactor::deserialize(ObjectData& odata, IdContext& context) {
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
