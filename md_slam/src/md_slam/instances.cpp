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

#include "instances.h"
#include "factor.cuh"
#include "graph_manager.h"
#include "orientation_estimator.h"
#include "pairwise_aligner.h"
#include "tracker.h"
#include "tracker_status_message.h"
#include "tracker_viewer.h"

#include "pyramid_generator.h"
#include "pyramid_message.h"
#include "pyramid_variable_se3.h"

#include "loop_closer.h"
#include "loop_detector_base.h"
#include "loop_detector_hbst.h"
#include "loop_validator.h"

#include "factor_bi.cuh"

// sick we need to register this for dataset manip
#include "../../apps/dataset_manipulators/lidar_config_object.h"

namespace md_slam {

  void md_registerTypes() {
    // basic stuff
    BOSS_REGISTER_CLASS(MDPyramidGenerator);
    BOSS_REGISTER_CLASS(MDNormalComputator2DCrossProduct);
    BOSS_REGISTER_CLASS(MDImagePyramidReference);
    BOSS_REGISTER_CLASS(MDImagePyramidMessage);
    BOSS_REGISTER_CLASS(MDVariableSE3);
    // tracker stuff
    BOSS_REGISTER_CLASS(MDFactor);
    BOSS_REGISTER_CLASS(MDFactorShowAction);
    BOSS_REGISTER_CLASS(MDPairwiseAligner);
    BOSS_REGISTER_CLASS(MDTrackerStandalone);
    BOSS_REGISTER_CLASS(MDTrackerStatusMessage);
    BOSS_REGISTER_CLASS(MDTrackerViewer);
    BOSS_REGISTER_CLASS(MDGraphManager);
    BOSS_REGISTER_CLASS(MDOrientationEstimator);
    // loop closure stuff
    BOSS_REGISTER_CLASS(LoopDetectorBase);
    BOSS_REGISTER_CLASS(LoopDetectorHBST);
    BOSS_REGISTER_CLASS(LoopValidator);
    BOSS_REGISTER_CLASS(MDCloser);
    // dataset manip
    BOSS_REGISTER_CLASS(MDLidarConfiguration);
    // motion only ba
    BOSS_REGISTER_CLASS(MDFactorBivariable);
  }
} // namespace md_slam
