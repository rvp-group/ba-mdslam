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

#include <bits/stdc++.h>
#include <srrg_data_structures/matrix.h>
#include <srrg_pcl/point_types.h>

#include <srrg_system_utils/shell_colors.h>
#include <srrg_system_utils/system_utils.h>

#include <srrg_image/image.h>

#include <iostream>
#include <unistd.h>
#include <vector>

#include <md_slam/factor_bi.cuh>

#include <md_slam/pyramid_variable_se3.h>
#include <md_slam/utils.cuh>
#include <srrg_solver/solver_core/internals/linear_solvers/sparse_block_linear_solver_cholesky_csparse.h>
// #include <srrg_solver/solver_core/iteration_algorithm_dl.h>
#include <srrg_solver/solver_core/iteration_algorithm_gn.h>
#include <srrg_solver/solver_core/iteration_algorithm_lm.h>
#include <srrg_solver/solver_core/solver.h>
#include <srrg_solver/solver_core/solver_action_base.h>
#include <srrg_solver/variables_and_factors/types_3d/se3_pose_pose_geodesic_error_factor.h>
#include <srrg_system_utils/parse_command_line.h>
#include <srrg_system_utils/system_utils.h>

// #include "viewer/md_slam_viewer.h"
// #include <qapplication.h>

using namespace srrg2_core;
using namespace srrg2_solver;
using namespace md_slam;

const char* banner[] = {"syncronize and makes map using groundtruth, only for ncd for now", 0};

struct GTPose {
  Isometry3f pose  = Isometry3f::Identity();
  double timestamp = 0.0;
};

using GTPoses = std::vector<GTPose>;

void readTUMGroundtruth(GTPoses& gt_poses_, const std::string& filename_) {
  std::string line;
  std::ifstream fi(filename_.c_str());
  getline(fi, line, '\n'); // ignore the first line
  double time = 0., tx = 0., ty = 0., tz = 0., qx = 0., qy = 0., qz = 0., qw = 0.;
  while (!fi.eof()) {
    getline(fi, line, '\n');
    sscanf(
      line.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf", &time, &tx, &ty, &tz, &qx, &qy, &qz, &qw);
    GTPose gt_pose;
    gt_pose.timestamp          = time;
    gt_pose.pose.translation() = Vector3f(tx, ty, tz);
    Quaternionf q(qw, qx, qy, qz);
    gt_pose.pose.linear() = q.toRotationMatrix();
    gt_poses_.emplace_back(gt_pose);
  }
  std::cerr << "read number of gt poses [ " << gt_poses_.size() << " ]" << std::endl;
}

int main(int argc, char** argv) {
  // clang-format off
  ParseCommandLine cmd_line(argv, banner);
  ArgumentString input_graph(&cmd_line, "i", "input", "path to input graph", "");
  ArgumentString groundtruth(&cmd_line, "gt", "groundtruth", "path to the gt file < timestamp tx ty tz qx qy qz qw > ", "");
  ArgumentString output_graph(&cmd_line, "o", "output", "path to output graph (optimized)", "");
  ArgumentDouble max_diff(&cmd_line, "m", "max-diff", "max time difference for an association to be made", 0.2);
  cmd_line.parse();
  // clang-format on

  if (!input_graph.isSet()) {
    std::cerr << "no input provided, aborting" << std::endl;
    return 0;
  }

  if (!output_graph.isSet()) {
    std::cerr << "no output provided, aborting" << std::endl;
    return 0;
  }

  FactorGraphPtr graph = FactorGraph::read(input_graph.value());
  if (!graph) {
    std::cerr << "unable to load graph" << std::endl;
  }

  std::cerr << "input graph loaded, n vars: " << graph->variables().size()
            << "| n factors: " << graph->factors().size() << std::endl;

  std::cerr << "loading pyramids, all as an example" << std::endl;
  int num_levels = 0;
  for (auto v_pair : graph->variables()) {
    VariableBase* v_base = v_pair.second;
    MDVariableSE3* v     = dynamic_cast<MDVariableSE3*>(v_base);
    if (!v)
      continue;
    MDImagePyramid* pyr = v->pyramid();
    std::cerr << "v: " << v->graphId() << " pyramid: " << pyr << " levels: " << pyr->numLevels()
              << " [ ";
    num_levels = pyr->numLevels();
    for (size_t i = 0; i < pyr->numLevels(); ++i) {
      auto& l = pyr->at(i);
      std::cerr << l->rows() << "x" << l->cols() << " ";
    }
    std::cerr << " ]" << std::endl;
  }

  // read groundtruth
  GTPoses gt_poses;
  readTUMGroundtruth(gt_poses, groundtruth.value());

  Isometry3f twoworldsT = Isometry3f::Identity();
  // twoworldsT.matrix() << 0.796443, 0.603779, -0.0336122, -3.44173, 0.603893, -0.797027,
  // -0.00779997,
  //   0.260206, -0.0314993, -0.014086, -0.999404, -0.0785324, 0, 0, 0, 1;

  Isometry3f sensorT = Isometry3f::Identity();
  // sensorT.matrix() << 0.999864, -0.0156887, 0.0050239, 0.269586, -0.0157907, -0.999656,
  // 0.0209417,
  //   0.12728, 0.00469362, -0.0210182, -0.999768, -0.619249, 0, 0, 0, 1;

  // substitute pose
  int variables_to_keep = 0;
  for (auto& gt_pose : gt_poses) {
    double best_diff      = std::numeric_limits<double>::max();
    double best_var_time  = std::numeric_limits<double>::max();
    const double gt_time  = gt_pose.timestamp;
    MDVariableSE3* best_v = nullptr;
    for (auto var = graph->variables().begin(); var != graph->variables().end(); ++var) {
      MDVariableSE3* v = dynamic_cast<MDVariableSE3*>(var.value());
      if (!v)
        continue;
      const double var_time  = v->timestamp();
      const double curr_diff = fabs(var_time - gt_time);
      // std::cerr << "\t\t curr diff: " << curr_diff << std::endl;
      if (curr_diff < best_diff) { //&& best_diff < max_diff.value()) {
        // std::cerr << std::setprecision(12) << " associated: " << var_time << " " << gt_time
        //           << std::endl;
        best_diff     = curr_diff;
        best_var_time = var_time;
        best_v        = v;
      }
    }
    if (best_diff < max_diff.value()) {
      std::cerr << std::setprecision(12) << " associated: " << best_var_time << " " << gt_time
                << " | best diff: " << best_diff << std::endl;
      best_v->setEstimate(twoworldsT * gt_pose.pose * sensorT);
      variables_to_keep++;
    }
  }

  std::cerr << "original number of variables: " << graph->variables().size() << std::endl;
  std::cerr << "processed number of variables: " << variables_to_keep << std::endl; // TODO +1

  // if (variables_to_keep != graph->variables().size()) {
  //   std::cerr << std::string(environ[0]) +
  //                  "|ERROR, number of variables different, be more large with -m param"
  //             << std::endl;
  //   return -1;
  // }

  graph->write(output_graph.value());
  std::cerr << "graph written successfully : " << output_graph.value()
            << " | n variables: " << graph->variables().size()
            << " | n factors: " << graph->factors().size() << std::endl;

  // enable viewer
  // QApplication app(argc, argv);
  // // instantiate the viewer
  // std::mutex proc_mutex;
  // MDViewerPtr viewer(new MDViewer(graph, proc_mutex));
  // viewer->setWindowTitle("graph");
  // // make the viewer window visible on screen
  // viewer->show();
  // return app.exec();
}