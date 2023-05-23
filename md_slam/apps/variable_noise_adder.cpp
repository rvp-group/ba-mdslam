#include <bits/stdc++.h>
#include <srrg_data_structures/matrix.h>
#include <srrg_pcl/point_types.h>

#include <srrg_system_utils/shell_colors.h>
#include <srrg_system_utils/system_utils.h>

#include <srrg_image/image.h>

#include <iostream>
#include <vector>

#include <md_slam/factor_bi.h>

#include <md_slam/pyramid_variable_se3.h>
#include <md_slam/utils.h>
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

const char* banner[] = {"loads a graph with pyramids attached and performs motion only ba", 0};

// local implementation of from cloud of pyramid level
void fromCloud(MDPyramidMatrix& matrix_,
               const MDMatrixCloud& src_cloud_,
               const Isometry3f& sensor_offset_,
               const Matrix3f& camera_mat_,
               const CameraType& camera_type_,
               const size_t rows_,
               const size_t cols_,
               const float min_depth_,
               const float max_depth_);

void visualizeCorrespondingImages(MDPyramidLevelPtr li_, MDPyramidLevelPtr lj_) {
  ImageFloat ij;
  lj_->getIntensity(ij);
  cv::Mat cvimgj;
  ij.toCv(cvimgj);

  ImageFloat ii;
  li_->getIntensity(ii);
  cv::Mat cvimgi;
  ii.toCv(cvimgi);

  // concat and output image
  cv::Mat cvout;
  cv::vconcat(cvimgi, cvimgj, cvout);
  cv::imshow("matches_image", cvout);
  cv::waitKey(0);
}

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

  graph->write(output_graph.value());
  std::cerr << "graph written successfully : " << output_graph.value()
            << " | n variables: " << graph->variables().size()
            << " | n factors: " << graph->factors().size() << std::endl;
}
