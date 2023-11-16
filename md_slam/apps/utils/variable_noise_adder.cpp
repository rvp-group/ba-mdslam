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

using namespace srrg2_core;
using namespace srrg2_solver;
using namespace md_slam;

const char* banner[] = {"loads a graph and inject noise on variables", 0};

int main(int argc, char** argv) {
  // clang-format off
  ParseCommandLine cmd_line(argv, banner);
  ArgumentString input_graph(&cmd_line, "i", "input", "path to input graph", "");
  ArgumentString output_graph(&cmd_line, "o", "output", "path to output graph (optimized)", "");
  // both uniform real distribution or rectangular distribution
  ArgumentFloat max_noise_rot(&cmd_line, "nr", "max-noise-rotation", "max value of noise rotation for yaw, instead roll, pitch 1/10 yaw [deg]", 10);
  ArgumentFloat max_noise_trans(&cmd_line, "nt", "max-noise-translation", "max value of noise translation for x and y, instead z is 1/5 of x, y [m]", 0.1);
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

  std::cerr << "injecting noise into variables..." << std::endl;

  // uniform real distribution or rectangular distribution, value between min and max
  std::default_random_engine generator;
  std::uniform_real_distribution<float> rot_dist(-max_noise_rot.value(), max_noise_rot.value());
  std::uniform_real_distribution<float> t_dist(-max_noise_trans.value(), max_noise_trans.value());

  for (auto v_pair : graph->variables()) {
    VariableBase* v_base = v_pair.second;
    MDVariableSE3* v     = dynamic_cast<MDVariableSE3*>(v_base);
    if (!v)
      continue;

    // x and y entire noise, z 1/5 of x, y noise
    const Vector3f t_noise =
      Vector3f(t_dist(generator), t_dist(generator), t_dist(generator) * 0.5f);

    // yaw entire noise, roll and pitch 1/10 of yaw noise
    const float yaw_noise   = rot_dist(generator);
    const float pitch_noise = rot_dist(generator) * 0.1f;
    const float roll_noise  = rot_dist(generator) * 0.1f;

    std::cerr << "angles noise [deg] (yaw : " << yaw_noise << " | pitch: " << pitch_noise
              << " | roll: " << roll_noise << ")" << std::endl;

    const AngleAxisf yaw   = AngleAxisf(yaw_noise * M_PI / 180.f, Vector3f::UnitZ());
    const AngleAxisf pitch = AngleAxisf(pitch_noise * M_PI / 180.f, Vector3f::UnitY());
    const AngleAxisf roll  = AngleAxisf(roll_noise * M_PI / 180.f, Vector3f::UnitX());

    const Matrix3f rotation_noise =
      yaw.toRotationMatrix() * pitch.toRotationMatrix() * roll.toRotationMatrix();

    std::cerr << "translation noise [m] " << t_noise.transpose() << std::endl;
    // std::cerr << "rotation noise: \n" << rotation_noise << std::endl;

    Isometry3f noisy_estimate    = Isometry3f::Identity();
    noisy_estimate.translation() = v->estimate().translation() + t_noise;
    noisy_estimate.linear()      = v->estimate().linear() * rotation_noise;
    v->setEstimate(noisy_estimate);
  }

  FactorGraph outgraph = *graph;
  outgraph.write(output_graph.value());
  std::cerr << "graph written successfully : " << output_graph.value()
            << " | n variables: " << outgraph.variables().size()
            << " | n factors: " << outgraph.factors().size() << std::endl;
}
