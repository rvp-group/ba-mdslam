#include <bits/stdc++.h>
#include <random>
#include <srrg_data_structures/matrix.h>
#include <srrg_pcl/point_types.h>

#include <srrg_system_utils/shell_colors.h>
#include <srrg_system_utils/system_utils.h>

#include <srrg_image/image.h>

#include <iostream>
#include <vector>

#include <md_slam/factor_bi.h>
#include <md_slam/pyramid_generator.h>
#include <md_slam/utils.h>
#include <srrg_solver/solver_core/iteration_algorithm_gn.h>
#include <srrg_solver/solver_core/solver.h>

#include <gtest/gtest.h>
#include <srrg_test/test_helper.hpp>

#ifndef MD_TEST_DATA_FOLDER
#error "NO TEST DATA FOLDER"
#endif

using namespace srrg2_core;
using namespace srrg2_solver;
using namespace md_slam;
using namespace std;

using PointVector = PointNormalIntensity3fVectorCloud;

using MDPyramidGeneratorPtr = std::shared_ptr<MDPyramidGenerator>;

const std::string test_path          = MD_TEST_DATA_FOLDER;
const std::string depth_filename     = test_path + "camera.depth.image_raw_00002102.pgm";
const std::string intensity_filename = test_path + "camera.rgb.image_raw_00002102.png";

TEST(DUMMY, MDFactor) {
  // we load 2 imayges
  ImageUInt16 depth_fixed;
  ImageVector3uc intensity_fixed;
  loadImage(depth_fixed, depth_filename);
  loadImage(intensity_fixed, intensity_filename);

  // we set the same as moving, the aligner will start from a wrong initial guess
  ImageUInt16 depth_moving        = depth_fixed;
  ImageVector3uc intensity_moving = intensity_fixed;

  // bdc camera matrix for Xtion test image, offset of the sensor, depth cropping parameters
  Matrix3f pinhole_camera_matrix;
  pinhole_camera_matrix << 481.2f, 0.f, 319.5f, 0.f, 481.f, 239.5f, 0.f, 0.f, 1.f;
  float min_depth             = 0.3;
  float max_depth             = 5.0;
  Isometry3f forced_offset    = Isometry3f::Identity();
  forced_offset.translation() = Vector3f(0.2, 0.3, 0.1);
  forced_offset.linear()      = AngleAxisf(1, Vector3f(1, 0, 0)).toRotationMatrix();

  // configure the pyrgen
  MDPyramidGeneratorPtr pyrgen(new MDPyramidGenerator);
  pyrgen->param_min_depth.setValue(min_depth);
  pyrgen->param_max_depth.setValue(max_depth);
  pyrgen->setSensorOffset(forced_offset);
  pyrgen->setDepthScale(0.001);
  pyrgen->param_normals_policy.setValue(0); //.pushBack(4);
  pyrgen->setCameraMatrix(pinhole_camera_matrix);
  pyrgen->param_scales.value() = vector<int>{1, 2, 4};

  // bdc compute previous pyramid
  pyrgen->setImages(depth_fixed, intensity_fixed);
  {
    Chrono timing("pyrgen1", nullptr, true);
    pyrgen->compute();
  }
  MDImagePyramidMessagePtr pyramid_fixed = pyrgen->pyramidMessage();
  // bdc compute current pyramid
  pyrgen->setImages(depth_moving, intensity_moving);
  {
    Chrono timing("pyrgen2", nullptr, true);
    pyrgen->compute();
  }
  MDImagePyramidMessagePtr pyramid_moving = pyrgen->pyramidMessage();

  // create graph
  FactorGraph graph;

  // add variable, with id 0
  std::shared_ptr<MDVariableSE3> vi(new MDVariableSE3);
  vi->setGraphId(0);
  Vector6f guess;
  guess << 0.01, 0.01, 0.01, 0.01, 0.01, 0.02;
  Isometry3f guess_T = geometry3d::ta2t(guess);
  vi->setEstimate(guess_T);
  vi->setPyramid(new MDImagePyramid(*pyramid_moving->get()));
  vi->setStatus(VariableBase::Fixed);
  graph.addVariable(vi);

  std::shared_ptr<MDVariableSE3> vj(new MDVariableSE3);
  vj->setGraphId(1);
  vj->setEstimate(Isometry3f::Identity());
  // vj->setStatus(VariableBase::Fixed);
  vj->setPyramid(new MDImagePyramid(*pyramid_fixed->get()));
  graph.addVariable(vj);

  uint8_t id           = 0;
  const uint8_t levels = vi->pyramid()->numLevels() - 1;
  for (int l = levels; l >= 0; --l) { // we start opt from higher level
    // create factor for level
    MDFactorBivariablePtr factor(new MDFactorBivariable);
    factor->setVariableId(0, 0);
    factor->setVariableId(1, 1);
    factor->setLevel(l);
    factor->setGraphId(id++);
    graph.addFactor(factor);
  }


  // TEST SOLVER
  Solver solver;
  std::shared_ptr<IterationAlgorithmGN> algorithm(new IterationAlgorithmGN);
  algorithm->param_damping.setValue(0.1f);
  solver.param_algorithm.setValue(algorithm);
  solver.setGraph(graph);
  solver.param_max_iterations.value() = vector<int>{10, 20, 50};
  solver.param_termination_criteria.setValue(0);
  std::cerr << "before compute" << std::endl;
  {
    Chrono timings("solve: ", nullptr, true);
    solver.compute();
  }
  std::cerr << "after compute" << std::endl;

  std::cerr << " ========= iteration stats =========" << std::endl;
  std::cerr << solver.iterationStats() << std::endl;
  std::cerr << " ===================================" << std::endl;

  std::cerr << "guess   T: " << FG_GREEN(geometry3d::t2v(guess_T).transpose()) << std::endl;
  std::cerr << "GT      T: " << FG_GREEN(Vector6f::Zero().transpose()) << std::endl;
  std::cerr << "Solver  T: " << FG_GREEN(geometry3d::t2v(vi->estimate()).transpose()) << std::endl;

  Vector6f estimate = geometry3d::t2v(vi->estimate());

  ASSERT_LT(estimate(0), 1e-7);
  ASSERT_LT(estimate(1), 1e-7);
  ASSERT_LT(estimate(2), 1e-7);
  ASSERT_LT(estimate(3), 1e-8);
  ASSERT_LT(estimate(4), 1e-8);
  ASSERT_LT(estimate(5), 1e-8);
}

int main(int argc, char** argv) {
  return srrg2_test::runTests(argc, argv, true /*use test folder*/);
}