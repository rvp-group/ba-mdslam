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
#include "md_slam_viewer.h"
#include <chrono>
#include <srrg_solver/solver_core/solver.h>
#include <srrg_solver/solver_core/solver_action_base.h>

namespace srrg2_core {

  using namespace srrg2_solver;

  // action merd
  class SolverEndOptimizationAction : public SolverActionBase {
  public:
    void doAction() override {
      _optimization_ended = true;
    }

    bool optimizationEnded() {
      return _optimization_ended;
    }

    std::atomic<bool> _optimization_ended = false;
  };
  using SolverEndOptimizationActionPtr = std::shared_ptr<SolverEndOptimizationAction>;

  // action merd
  class SolverShowGraphSyncAction : public SolverActionBase {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void doAction() override {
      if (param_event.value() == Solver::SolverEvent::IterationStart) {
        if (_viewer) {
          sem_wait(&_viewer->_sem);
        }
      }

      if (param_event.value() == Solver::SolverEvent::IterationEnd) {
        if (_viewer)
          sem_post(&_viewer->_sem);
      }
    }

    void setViewer(MDViewer* viewer_) {
      _viewer = viewer_;
    }

    MDViewer* _viewer = nullptr;
  };

  using SolverShowGraphSyncActionPtr = std::shared_ptr<SolverShowGraphSyncAction>;

  class MDBAViewer {
  public:
    void compute(int argc, char** argv) {
      // setup solver with actions
      SolverEndOptimizationActionPtr end_action(new SolverEndOptimizationAction);
      end_action->param_event.setValue(Solver::SolverEvent::ComputeEnd);
      _solver->param_actions.pushBack(end_action);

      SolverShowGraphSyncActionPtr action_trig_sem(new SolverShowGraphSyncAction);
      action_trig_sem->param_event.setValue(Solver::SolverEvent::IterationEnd);
      _solver->param_actions.pushBack(action_trig_sem);

      SolverShowGraphSyncActionPtr action_release_sem(new SolverShowGraphSyncAction);
      action_release_sem->param_event.setValue(Solver::SolverEvent::IterationStart);
      _solver->param_actions.pushBack(action_release_sem);

      // start solver computation on different thread
      // std::thread compute_thread(std::bind(&Solver::compute, _solver.get()));
      std::thread compute_thread([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cerr << "\npress any key to start optimization" << std::endl;
        char symbol;
        std::cin.get(symbol);
        _solver->compute();
      });

      // enable viewer to show progress of BA optimization
      std::mutex mut;
      QApplication app(argc, argv);
      MDViewerPtr viewer(new MDViewer(_graph, mut));
      viewer->setWindowTitle("ba_md_slam_viewer");
      viewer->show();
      viewer->setBA();

      // loop to view graph
      while (true) {
        viewer->update();
        app.processEvents();

        if (end_action->optimizationEnded() && _solver->currentLevel() == 0)
          break; // if optimization ended after last level exit
      }

      // when optimization is finished join thread
      compute_thread.join();
    }

    void setSolver(SolverPtr solver_) {
      _solver = solver_;
    }

    void setGraph(FactorGraphPtr graph_) {
      _graph = graph_;
    }

  protected:
    FactorGraphPtr _graph;
    SolverPtr _solver;
  };

  using MDBAViewerPtr = std::shared_ptr<MDBAViewer>;

} // namespace srrg2_core