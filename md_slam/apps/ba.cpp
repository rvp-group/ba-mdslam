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

// viewer stuff
#include <qapplication.h>
#define GL_GLEXT_PROTOTYPES 1

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

#include <md_slam/pairwise_aligner.h>
#include <md_slam/pyramid_variable_se3.h>
#include <md_slam/utils.cuh>

#include "utils/standard_ba_configuration.h"
#include "viewer/md_ba_viewer.h"

#include <srrg_solver/solver_core/internals/linear_solvers/sparse_block_linear_solver_cholesky_csparse.h>
#include <srrg_solver/solver_core/iteration_algorithm_gn.h>
#include <srrg_solver/solver_core/iteration_algorithm_lm.h>
#include <srrg_solver/solver_core/solver.h>
#include <srrg_solver/solver_core/solver_action_base.h>
#include <srrg_solver/variables_and_factors/types_3d/se3_pose_pose_geodesic_error_factor.h>
#include <srrg_system_utils/parse_command_line.h>
#include <srrg_system_utils/system_utils.h>

#include <md_slam/instances.h>

#ifndef MD_DL_FOLDER
#error "NO DL FOLDER"
#endif

using namespace srrg2_core;
using namespace srrg2_solver;
using namespace md_slam;

const char* banner[] = {"loads a graph with pyramids attached and performs bundle adjustment", 0};

// global solver instance
SolverPtr solver;
constexpr int max_iterations = 50;

using OverlapMap = std::map<std::pair<size_t, size_t>, float>;

/* TODO remove */
void visualizeCorrespondingImages(MDPyramidLevel& li_,
                                  MDPyramidLevelPtr lj_,
                                  const std::string img_name_);
void fromCloud(MDPyramidLevel& level_,
               const MDMatrixCloud& src_cloud_,
               const Isometry3f& sensor_offset_,
               const Matrix3f& camera_mat_,
               const CameraType& camera_type_,
               const size_t rows_,
               const size_t cols_,
               const float min_depth_,
               const float max_depth_);
void transform(MDMatrixCloud& src_, const Isometry3f& iso_);
OverlapMap getOverlappingVariables(srrg2_solver::FactorGraph& graph_,
                                   const float dr_,
                                   const float dt_,
                                   const int overlap_threshold_,
                                   const bool visualize_ = false);

int main(int argc, char** argv) {
  srrgInit(argc, argv, "ba");
  md_registerTypes();
  // clang-format off
  ParseCommandLine cmd_line(argv, banner);
  ArgumentString input_graph(&cmd_line, "i", "input", "path to input graph", "");
  ArgumentString output_graph(&cmd_line, "o", "output", "path to output graph (optimized)", "");
  ArgumentString sensor(&cmd_line, "s", "sensor", "set to [rgbd, lidar] to use default sensor configuration, otherwise use default parameters", "");
  ArgumentFloat dr(&cmd_line, "dr", "delta-rotation", "dr between poses to put a factor [rad]", 0.2);
  ArgumentFloat dt(&cmd_line, "dt", "delta-translation", "dt between poses to put a factor [m]", 0.6);
  ArgumentFloat overlap_threshold(&cmd_line, "ov", "overlap", "overalp threshold", 0.4);
  // start optimization parameters
  ArgumentIntVector_<3> iterations(&cmd_line, "its", "iterations", "num iterations from coarse to fine level, default termination based on difference in residual", {0, 0, 0});
  // ArgumentFloat gradient_norm(&cmd_line, "g", "gradient-norm", "if gradient norm is below this threshold exit optimization", 1e-6f);
  ArgumentFloat residual_diff(&cmd_line, "g", "chi2-norm", "if chi2 diff between last two iterations is below this threshold exit optimization", 1e-3f);
  ArgumentFloat omega_intensity(&cmd_line, "oi", "omega-intensity", "information for intensity, higher more weight in opt", 1.f);
  ArgumentFloat omega_depth(&cmd_line, "od", "omega-depth", "information for depth or range cue, higher more weight in opt", 5.f);
  ArgumentFloat omega_normals(&cmd_line, "on", "omega-normal", "information for normals, higher more weight in opt", 1.f);
  ArgumentFloat huber_threshold(&cmd_line, "ht", "huber-threshold", "if residual higher then this value sqrt(huber) is applied", 1.f);
  ArgumentFloat distance(&cmd_line, "dd", "distance", "distance between two points in opt, if bigger then this threshold opt is droppped for the point", 0.25f);
  // end optimization parameters
  ArgumentFlag viz_overlap(&cmd_line, "vo", "visualize-overlap", "if set visualize additional images overlap factors", false);
  ArgumentFlag viz_opt(&cmd_line, "v", "visualize-optimization", "if set visualize graph during optimization", false);
  cmd_line.parse();
  // clang-format on

  // checking device
  std::cerr << std::endl
            << "_______________________________________________" << std::endl
            << std::endl;
  const int n_devices = getDeviceInfo();
  if (!n_devices) {
    std::cerr << std::string(environ[0]) + "|ERROR GPU not found, if you can't fix the problem "
                                           "checkout to our cpu branch version aborting"
              << std::endl;
    return -1;
  }

  if (!input_graph.isSet()) {
    std::cerr << std::string(environ[0]) + "|ERROR no input provided, aborting" << std::endl;
    return -1;
  }

  if (!output_graph.isSet()) {
    std::cerr << std::string(environ[0]) + "|ERROR no output provided, aborting" << std::endl;
    return -1;
  }

  FactorGraphPtr graph = FactorGraph::read(input_graph.value());
  if (!graph) {
    std::cerr << std::string(environ[0]) + "|ERROR unable to load graph" << std::endl;
    return -1;
  }

  if (graph->variables().empty()) {
    std::cerr << "loaded empty graph, aborting" << std::endl;
    return -1;
  }

  std::cerr << "successfully input graph loaded, n vars: " << graph->variables().size()
            << " | n factors: " << graph->factors().size() << std::endl;

  // some horrible configuration stuff
  if (sensor.isSet()) {
    if (sensor.value() == "rgbd") {
      distance.setValue(RGBD_DD);
      dr.setValue(RGBD_DR);
      dt.setValue(RGBD_DT);
      overlap_threshold.setValue(RGBD_OV);
      residual_diff.setValue(RGBD_G);
      huber_threshold.setValue(RGBD_HT);
      omega_depth.setValue(RGBD_OD);
      omega_intensity.setValue(RGBD_OI);
      omega_normals.setValue(RGBD_ON);
    } else if (sensor.value() == "lidar") {
      distance.setValue(LIDAR_DD);
      dr.setValue(LIDAR_DR);
      dt.setValue(LIDAR_DT);
      overlap_threshold.setValue(LIDAR_OV);
      residual_diff.setValue(LIDAR_G);
      huber_threshold.setValue(LIDAR_HT);
      omega_depth.setValue(LIDAR_OD);
      omega_intensity.setValue(LIDAR_OI);
      omega_normals.setValue(LIDAR_ON);
    } else {
      std::cerr << std::string(environ[0]) + "|ERROR sensor unknown, aborting" << std::endl;
      return -1;
    }
  }
  std::cerr << "_______________________________________________" << std::endl << std::endl;
  std::cerr << "running ba with thresholds ";
  if (sensor.isSet())
    std::cerr << "| overriding user inserted params, using default sensor params: "
              << sensor.value();
  std::cerr << "\n\t distance in opt: " << distance.value();
  std::cerr << "\n\t rotation: " << dr.value();
  std::cerr << "\n\t translation: " << dt.value();
  std::cerr << "\n\t overlap: " << overlap_threshold.value();
  std::cerr << "\n\t err diff: " << residual_diff.value();
  std::cerr << "\n\t huber: " << huber_threshold.value();
  std::cerr << "\n\t omega depth: " << omega_depth.value();
  std::cerr << "\n\t omega intensity: " << omega_intensity.value();
  std::cerr << "\n\t omega normals: " << omega_normals.value();
  std::cerr << std::endl;
  std::cerr << "_______________________________________________" << std::endl << std::endl;

  std::cerr << "loading pyramids ..." << std::endl;
  int num_levels    = 0;
  bool print_levels = true;
  for (auto v_pair : graph->variables()) {
    VariableBase* v_base = v_pair.second;
    MDVariableSE3* v     = dynamic_cast<MDVariableSE3*>(v_base);
    if (!v)
      continue;
    MDImagePyramid* pyr = v->pyramid();
    // std::cerr << "v: " << v->graphId() << " pyramid: " << pyr << " levels: " << pyr->numLevels()
    //           << " [ ";
    if (print_levels) {
      num_levels = pyr->numLevels();
      std::cerr << "levels resolution [ ";
      for (size_t i = 0; i < pyr->numLevels(); ++i) {
        auto& l = pyr->at(i);
        std::cerr << l->rows() << "x" << l->cols() << " ";
      }
      std::cerr << " ]" << std::endl;
      print_levels = false;
    }
  }

  // copy bkp factor graph
  FactorGraph mba_graph = *graph;
  std::vector<FactorBasePtr> factor_vec;
  factor_vec.reserve(graph->factors().size());
  for (auto it = graph->factors().begin(); it != graph->factors().end(); ++it) {
    FactorBase* f = it.value();
    if (!f)
      continue;
    FactorBasePtr factor(new MDFactorBivariable);
    factor->setVariableId(0, f->variableId(0));
    factor->setVariableId(1, f->variableId(1));
    factor_vec.push_back(factor);
  }
  mba_graph.factors().clear();
  // TODO this can be just a 2 int struct
  // move to a sortable container

  // TODO optimize this
  // copy pyramids to device before proceeding
  std::cerr << "copying all pyramids to device ..." << std::endl;
  for (auto var = mba_graph.variables().begin(); var != mba_graph.variables().end(); ++var) {
    MDVariableSE3* v = dynamic_cast<MDVariableSE3*>(var.value());
    // copy variables to gpu
    for (int i = 0; i < num_levels; ++i) {
      v->pyramid()->at(i)->matrix.toDevice();
    }
  }

  const int graph_size = graph->factors().size();

  std::cerr << "_______________________________________________" << std::endl << std::endl;
  std::cerr << "original number of factors: " << graph_size * num_levels << " | " << graph_size
            << " x " << num_levels << std::endl;
  std::cerr << "checking for graph augmentation ..." << std::endl;

  // lock first variable (should be already locked)
  graph->variables().begin().value()->setStatus(VariableBase::Fixed);

  // check overlap between images
  OverlapMap overlap_map = getOverlappingVariables(
    mba_graph, dr.value(), dt.value(), overlap_threshold.value(), viz_overlap.isSet());
  if (overlap_map.empty()) {
    std::cerr << "WARNING graph not augmented, optimization might fail!" << std::endl;
  }

  // iterate through map and append into factor vector
  // add to tmp container
  for (auto& overlap : overlap_map) {
    FactorBasePtr factor(new MDFactorBivariable);
    factor->setVariableId(0, overlap.first.first);  // v_i
    factor->setVariableId(1, overlap.first.second); // v_j
    factor_vec.emplace_back(factor);
  }

  // sort (based on first var) to avoid reading and processing alla cieca
  std::sort(
    factor_vec.begin(), factor_vec.end(), [](FactorBasePtr fac_p, FactorBasePtr fac_n) -> bool {
      return fac_p->variableId(0) < fac_n->variableId(0);
    });

  // once sorted by var0, insert factors in graph
  int factor_id = 0;
  for (int l = num_levels; l > 0; --l) { // we start opt from higher level
    for (size_t i = 0; i < factor_vec.size(); ++i) {
      auto tmp_fac = factor_vec[i];
      // std::cerr << "level " << l - 1 << " : [ " << tmp_fac->variableId(0) << "->"
      //           << tmp_fac->variableId(1) << " ]" << std::endl;
      MDFactorBivariablePtr factor_mba(new MDFactorBivariable);
      factor_mba->setVariableId(0, tmp_fac->variableId(0));
      factor_mba->setVariableId(1, tmp_fac->variableId(1));
      factor_mba->setLevel(l - 1);
      factor_mba->setGraphId(factor_id);

      // propagate settings to factors
      factor_mba->setOmegaDepth(omega_depth.value());
      factor_mba->setOmegaIntensity(omega_intensity.value());
      factor_mba->setOmegaNormals(omega_normals.value());
      factor_mba->setKernelChiThreshold(huber_threshold.value());
      factor_mba->setDepthRejectionThreshold(distance.value());

      mba_graph.addFactor(factor_mba);
      factor_id++;
    }
  }

  factor_vec.clear();
  std::cerr << "final number of factors: " << mba_graph.factors().size() << std::endl;
  std::cerr << "_______________________________________________" << std::endl << std::endl;

  std::cerr << "init optimization ... termination criteria: |r_curr - r_prev| < "
            << residual_diff.value() << std::endl;
  solver = SolverPtr(new Solver);
  // set LM as iteration strategy
  std::shared_ptr<IterationAlgorithmLM> algorithm(new IterationAlgorithmLM);
  algorithm->param_variable_damping.setValue(false);
  solver->param_algorithm.setValue(algorithm);
  solver->setGraph(mba_graph);
  SolverActionBasePtr parla(new SolverVerboseAction);
  parla->param_event.setValue(Solver::SolverEvent::IterationEnd);
  solver->param_actions.pushBack(parla);
  solver->param_verbose.setValue(true);
  solver->param_max_iterations.value() =
    std::vector<int>{max_iterations, max_iterations, max_iterations};
  // std::shared_ptr<PerturbationNormTerminationCriteria> termination_criteria(new
  // PerturbationNormTerminationCriteria);
  std::shared_ptr<SimpleTerminationCriteria> termination_criteria(new SimpleTerminationCriteria);
  termination_criteria->param_epsilon.setValue(residual_diff.value());
  // if num iterations has been set, override!
  if (iterations.isSet()) {
    solver->param_max_iterations.value() = std::vector<int>{
      iterations.value()[2], iterations.value()[1], iterations.value()[0]}; // low to high
    solver->param_termination_criteria.setValue(nullptr);
  }

  solver->param_linear_solver.setValue(
    SparseBlockLinearSolverPtr(new SparseBlockLinearSolverCholeskyCSparse));
  // solver.param_linear_solver.setValue(
  //   SparseBlockLinearSolverPtr(new SparseBlockLinearSolverCholmodFull));

  // this enable viewer and calls solver compute
  if (viz_opt.isSet()) {
    MDBAViewerPtr viewer(new MDBAViewer);
    viewer->setGraph(std::make_shared<FactorGraph>(mba_graph));
    viewer->setSolver(solver);
    viewer->compute(argc, argv);
  } else {
    Chrono timings("total: ", nullptr, true);
    solver->compute();
  }

  // check only chi at higher resolution
  float chi_init = 0.f;
  for (auto iter : solver->iterationStats()) {
    if (iter.level == 0) {
      chi_init = iter.chi_inliers;
      break;
    }
  }
  auto chi_final = solver->iterationStats().back().chi_inliers;
  std::cerr << "r init: " << chi_init << " r final: " << chi_final;
  if (chi_final > chi_init)
    std::cerr << "\nWARNING optimization likely failed | residual increased!" << std::endl;
  else
    std::cerr << " | r decreased: " << chi_init - chi_final << std::endl;

  // check overlap after optimization
  // OverlapMap overlap_map_end = getOverlappingVariables(
  //   mba_graph, dr.value(), dt.value(), overlap_threshold.value(), viz_overlap.isSet());

  // if (overlap_map_end.size() > overlap_map.size()) {
  //   for (auto& overlap : overlap_map_end) {
  //     if (auto it{overlap_map.find(overlap.first)}; it != std::end(overlap_map)) {
  //       std::cerr << "overlap before optimization: " << overlap.second
  //                 << " | overlap after optimization: " << it->second;
  //       if (overlap.second > it->second) {
  //         std::cerr << " - overlap decresead!";
  //       }
  //       std::cerr << std::endl;
  //     }
  //   }
  // } else {
  //   std::cerr << "warning, less overlap between variables in optimized graph!" << std::endl;
  // }
  std::cerr << "_______________________________________________" << std::endl << std::endl;

  mba_graph.write(output_graph.value());
  std::cerr << "graph written successfully : " << output_graph.value()
            << " | n variables: " << mba_graph.variables().size()
            << " | n factors: " << mba_graph.factors().size() << std::endl
            << std::endl;
}

/* TODO remove */
void visualizeCorrespondingImages(MDPyramidLevel& li_,
                                  MDPyramidLevelPtr lj_,
                                  const std::string img_name_) {
  ImageFloat ii;
  li_.getIntensity(ii);
  cv::Mat cvimgi;
  ii.toCv(cvimgi);

  ImageFloat ij;
  lj_->getIntensity(ij);
  cv::Mat cvimgj;
  ij.toCv(cvimgj);

  // concat and output image
  cv::Mat cvout;
  cv::vconcat(cvimgi, cvimgj, cvout);
  cv::imshow(img_name_, cvout);
}

void fromCloud(MDPyramidLevel& level_,
               const MDMatrixCloud& src_cloud_,
               const Isometry3f& sensor_offset_,
               const Matrix3f& camera_mat_,
               const CameraType& camera_type_,
               const size_t rows_,
               const size_t cols_,
               const float min_depth_,
               const float max_depth_) {
  MDPyramidMatrixEntry zero_entry;
  zero_entry.setDepth(max_depth_ + 1);
  level_.matrix.resize(rows_, cols_);
  level_.matrix.fill(zero_entry);

  Isometry3f inv_sensor_offset = sensor_offset_.inverse();
  Vector3f polar_point;
  Vector3f coordinates;
  Vector3f camera_point = Vector3f::Zero();
  const float& fx       = camera_mat_(0, 0);
  const float& fy       = camera_mat_(1, 1);
  const float& cx       = camera_mat_(0, 2);
  const float& cy       = camera_mat_(1, 2);
  float w               = 0;
  for (int i = 0; i < src_cloud_.size(); ++i) {
    const auto& src = src_cloud_.at(i);
    // std::cerr << src_cloud_.at(i).coordinates().transpose() << std::endl;
    // for (const auto& src : src_cloud_) {
    if (src.status != POINT_STATUS::Valid)
      continue;
    coordinates    = inv_sensor_offset * src.coordinates();
    const float& x = coordinates.x();
    const float& y = coordinates.y();
    const float& z = coordinates.z();
    switch (camera_type_) {
      case Pinhole: {
        w = coordinates(2);
        if (w < min_depth_ || w > max_depth_)
          continue;
        camera_point = camera_mat_ * coordinates;
        camera_point.block<2, 1>(0, 0) *= 1. / w;
      } break;
      case Spherical: {
        w = coordinates.norm();
        if (w < min_depth_ || w > max_depth_)
          continue;
        polar_point.x()  = atan2(y, x);
        polar_point.y()  = atan2(coordinates.z(), sqrt(x * x + y * y));
        polar_point.z()  = z;
        camera_point.x() = fx * polar_point.x() + cx;
        camera_point.y() = fy * polar_point.y() + cy;
        camera_point.z() = w;
      } break;
      default:;
    }
    int c = cvRound(camera_point.x());
    int r = cvRound(camera_point.y());
    if (!level_.matrix.inside(r, c))
      continue;
    MDPyramidMatrixEntry& entry = level_.matrix.at(r, c);
    if (w < entry.depth()) {
      entry.setIntensity(src.intensity());
      entry.setDepth(w);
      entry.setNormal(inv_sensor_offset.linear() * src.normal());
#ifdef _MD_ENABLE_SUPERRES_
      entry.c = camera_point.x();
      entry.r = camera_point.y();
#endif
      entry.setMasked(false);
    }
  }
}

void transform(MDMatrixCloud& src_, const Isometry3f& iso_) {
  for (int i = 0; i < src_.size(); ++i) {
    src_.at(i).coordinates() = iso_ * src_.at(i).coordinates();
    src_.at(i).normal()      = iso_.linear() * src_.at(i).normal();
  }
}

OverlapMap getOverlappingVariables(srrg2_solver::FactorGraph& graph_,
                                   const float dr_,
                                   const float dt_,
                                   const int overlap_threshold_,
                                   const bool visualize_) {
  // add additional factors
  // iterate through all the variables checking overlap between clouds
  OverlapMap overlap_map;
  for (auto var = graph_.variables().begin(); var != graph_.variables().end(); ++var) {
    MDVariableSE3* v_i = dynamic_cast<MDVariableSE3*>(var.value());
    if (!v_i)
      continue;
    // get next
    auto var_next = var;
    ++var_next;
    while (var_next != graph_.variables().end()) {
      ++var_next;
      MDVariableSE3* v_j = dynamic_cast<MDVariableSE3*>(var_next.value());
      if (!v_j)
        continue;

      const Isometry3f Xji = v_j->estimate().inverse() * v_i->estimate();

      // check that difference between poses is human
      Eigen::AngleAxisf aa(Xji.linear());
      if (fabs(aa.angle()) > dr_)
        continue;

      if (Xji.translation().norm() > dt_)
        continue;

      // TODO for fusion
      // use offset premultiply this matrix Xji
      // retrieve from precomputed cloud map the needed cloud with max resolution, level 0

      // operations on cloud i
      auto finest_level_i = v_i->pyramid()->at(0);

      MDMatrixCloud* cloud_i = MDFactorBivariable::setMoving(finest_level_i, v_i->graphId(), 0);
      cloud_i->toHost(); // TODO this can be done in device
      // cloud_i->transformInPlace<Isometry>(Xji);
      transform(*cloud_i, Xji);
      MDPyramidLevel level_ji; // expresses the cloud i in frame j
      fromCloud(level_ji,
                *cloud_i,
                finest_level_i->sensor_offset,
                finest_level_i->camera_matrix,
                finest_level_i->camera_type,
                finest_level_i->rows(),
                finest_level_i->cols(),
                finest_level_i->min_depth,
                finest_level_i->max_depth);

      // operation on cloud j
      auto finest_level_j = v_j->pyramid()->at(0);

      MDMatrixCloud* cloud_j = MDFactorBivariable::setMoving(finest_level_j, v_j->graphId(), 0);
      cloud_j->toHost(); // TODO this can be done in device
      // cloud_j->transformInPlace<Isometry>(Xji.inverse());
      transform(*cloud_j, Xji.inverse());
      MDPyramidLevel level_ij; // expresses the cloud j in frame i
      fromCloud(level_ij,
                *cloud_j,
                finest_level_j->sensor_offset,
                finest_level_j->camera_matrix,
                finest_level_j->camera_type,
                finest_level_j->rows(),
                finest_level_j->cols(),
                finest_level_j->min_depth,
                finest_level_j->max_depth);

      // bijective check
      int counter_ji = 0;
      for (size_t i = 0; i < level_ji.matrix.size(); ++i)
        if (!level_ji.matrix.at(i).masked())
          counter_ji++;

      int counter_ij = 0;
      for (size_t j = 0; j < level_ij.matrix.size(); ++j)
        if (!level_ij.matrix.at(j).masked())
          counter_ij++;

      const float overlap_ji = (float) counter_ji / cloud_i->size();
      const float overlap_ij = (float) counter_ij / cloud_j->size();
      if (overlap_ji < overlap_threshold_ || overlap_ij < overlap_threshold_)
        continue;

      std::cerr << "\tchecking: " << v_i->graphId() << " " << v_j->graphId()
                << " -> \toverlap ji: " << overlap_ji << "\toverlap ij: " << overlap_ij
                << std::endl;
      if (visualize_) {
        visualizeCorrespondingImages(level_ji, v_j->pyramid()->at(0), "vi_vj");
        visualizeCorrespondingImages(level_ij, v_i->pyramid()->at(0), "vj_vi");
        cv::waitKey(0);
      }
      // insert into map
      overlap_map.insert({std::make_pair<size_t, size_t>(v_i->graphId(), v_j->graphId()),
                          (overlap_ji + overlap_ij) / 2});
    }
  }

  if (visualize_)
    cv::destroyAllWindows();

  return overlap_map;
}
