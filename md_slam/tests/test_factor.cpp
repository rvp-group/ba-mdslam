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
#include <random>
#include <srrg_data_structures/matrix.h>
#include <srrg_pcl/point_types.h>

#include <srrg_system_utils/shell_colors.h>
#include <srrg_system_utils/system_utils.h>

#include <srrg_image/image.h>

#include <iostream>
#include <vector>

#include <md_slam/factor.cuh>
#include <md_slam/factor_stack.h>
#include <md_slam/pyramid_generator.h>
#include <md_slam/utils.cuh>
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

using PointVector           = PointNormalIntensity3fVectorCloud;
using MDPyramidGeneratorPtr = std::shared_ptr<MDPyramidGenerator>;

constexpr int NUM_EXPERIMENTS        = 5;
const std::string test_path          = MD_TEST_DATA_FOLDER;
const std::string depth_filename     = test_path + "camera.depth.image_raw_00002102.pgm";
const std::string intensity_filename = test_path + "camera.rgb.image_raw_00002102.png";

TEST(DUMMY, MDFactor) {
  // reproducible results
  Vector6f results[NUM_EXPERIMENTS];
  for (int i = 0; i < NUM_EXPERIMENTS; ++i) {
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
    std::shared_ptr<VariableSE3QuaternionRight> v(new VariableSE3QuaternionRight);
    v->setGraphId(0);
    Vector6f guess;
    guess << 0.01, 0.01, 0.01, 0.01, 0.01, 0.02;
    Isometry3f guess_T = geometry3d::ta2t(guess);
    v->setEstimate(guess_T);
    graph.addVariable(v);

    MDFactorStack md_factors;
    {
      Chrono timings("copy_pyramids: ", nullptr, true);
      // create factor stack
      md_factors.setFixed(*pyramid_fixed->get());
      md_factors.makeFactors();
      md_factors.setMoving(*pyramid_moving->get());
      md_factors.setVariableId(v->graphId());
      md_factors.addFactors(graph);
      md_factors.assignPyramids();
    }

    // TEST SOLVER
    Solver solver;
    std::shared_ptr<IterationAlgorithmGN> algorithm(new IterationAlgorithmGN);
    algorithm->param_damping.setValue(0.1f);
    solver.param_algorithm.setValue(algorithm);
    solver.setGraph(graph);
    solver.param_max_iterations.value() = vector<int>{10, 20, 50};
    solver.param_termination_criteria.setValue(0);
    {
      Chrono timings("solve: ", nullptr, true);
      solver.compute();
    }
    std::cerr << " ========= iteration stats =========" << std::endl;
    std::cerr << solver.iterationStats() << std::endl;
    std::cerr << " ===================================" << std::endl;

    const Vector6f estimate = geometry3d::t2v(v->estimate());
    // store if results are equal over experiments
    results[i] = estimate;

    std::cerr << "guess   T: " << FG_GREEN(geometry3d::t2v(guess_T).transpose()) << std::endl;
    std::cerr << "GT      T: " << FG_GREEN(Vector6f::Zero().transpose()) << std::endl;
    std::cerr << "Solver  T: " << FG_GREEN(estimate.transpose()) << std::endl;

    for (auto& f : md_factors) {
      std::cerr << "level num: " << f->level() << std::endl;
      Chrono::printReport(f->timings);
    }

    const float chi2 = solver.iterationStats().back().chi_inliers;

    ASSERT_LT(chi2, 1e-11);
    ASSERT_LT(estimate(0), 1e-7);
    ASSERT_LT(estimate(1), 1e-7);
    ASSERT_LT(estimate(2), 1e-7);
    ASSERT_LT(estimate(3), 1e-8);
    ASSERT_LT(estimate(4), 1e-8);
    ASSERT_LT(estimate(5), 1e-8);
  }

  // check if results are equal over experiments
  std::cerr << std::endl;
  std::cerr << "check for non-deterministic illnesses..\n";
  for (int i = 1; i < NUM_EXPERIMENTS; ++i) {
    std::cerr << results[0].transpose() << "\n";
    std::cerr << results[i].transpose() << "\n";
    std::cerr << std::endl;
  }
}

int main(int argc, char** argv) {
  return srrg2_test::runTests(argc, argv, true /*use test folder*/);
}