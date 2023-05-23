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
#include "loop_closer.h"
#include "pairwise_aligner.h"
#include "pyramid_message.h"
#include "pyramid_variable_se3.h"
#include "tracker_status_message.h"
#include <mutex>
#include <srrg_messages/message_handlers/message_sink_base.h>
#include <srrg_solver/solver_core/factor_graph.h>
#include <srrg_solver/solver_core/solver.h>
#include <srrg_viewer/active_drawable.h>
#include <thread>

namespace md_slam {

  class MDGraphManager : public srrg2_core::MessageSinkBase, public srrg2_core::ActiveDrawable {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MDGraphManager();
    ~MDGraphManager();

    // clang-format off
    PARAM(PropertyConfigurable_<MDPairwiseAligner>, pairwise_aligner, "photometric pairwise aligner, takes 2 pyramids gives you a pose",nullptr, nullptr);
    PARAM(srrg2_core::PropertyConfigurable_<srrg2_solver::Solver>, solver, "pose-graph solver", nullptr, nullptr);
    PARAM(srrg2_core::PropertyConfigurable_<MDCloser>, closer, "closer manager for md slam", nullptr, nullptr);
    PARAM(srrg2_core::PropertyString, status_topic, "tracker status", "/md_tracker_status", nullptr);
    PARAM(PropertyString, tf_dyn_topic, "topic where to push the dynamic transforms", "/tf", nullptr);
    PARAM(PropertyString, map_frame_id, "map frame id of the graph", "/md_map", nullptr);
    PARAM(srrg2_core::PropertyBool, enable_closures, "enable closures", true, nullptr);
    PARAM(PropertyFloat, angle_check, "loop clousure neighboroud check on angle", 7e-2, nullptr);
    PARAM(PropertyFloat, translation_check, "loop clousure neighboroud check on norm of translation", 7e-2, nullptr);

    void reset() override;
    bool putMessage(srrg2_core::BaseSensorMessagePtr msg_) override;
    void closureCallback();
    void publishTransform();

    inline void quitClosureThread() { _quit_closure_thread = true; }
    inline std::thread& closureThread() { return _closure_thread; }
    Matrix6f photometricClosure(Isometry3f& estimate_, MDImagePyramid* fixed_, MDImagePyramid* moving_);
    inline srrg2_solver::FactorGraphPtr graph() { return _graph; }
    inline std::mutex& graphMutex() { return _mutex; }
    // clang-format on

  protected:
    bool cmdSaveGraph(std::string& response, const std::string& filename);
    void _drawImpl(srrg2_core::ViewerCanvasPtr gl_canvas_) const override;
    mutable srrg2_solver::FactorGraphPtr _graph;

    MDTrackerStatusMessagePtr _status_msg;
    MDImagePyramidMessagePtr _pyramid_msg;
    MDVariableSE3Ptr _previous_variable;
    std::unique_ptr<std::queue<MDVariableSE3Ptr>> _variable_buffer_queue;
    std::thread _closure_thread;
    std::mutex _mutex;

    int _max_id                = 0;
    std::string _pyramid_topic = "";
    // closure thread stuff
    bool _initial_var                 = true;
    mutable bool _quit_closure_thread = false;
    mutable bool _lists_created       = false;
    mutable bool _is_closure_valid    = false;
  };

} // namespace md_slam
