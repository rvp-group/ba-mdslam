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

#include <md_slam/loop_validator.h>
#include <md_slam/pyramid_variable_se3.h>

#include <srrg_system_utils/parse_command_line.h>
#include <srrg_system_utils/system_utils.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl_conversions/pcl_conversions.h>

#include <srrg_solver/solver_core/factor_graph.h>
#include <srrg_solver/solver_core/solver.h>

#include <pcl/features/normal_3d.h>
#include <srrg_data_structures/correspondence.h>
#include <srrg_data_structures/kd_tree.h>
#include <srrg_pcl/point_types.h>

#include <srrg_solver/solver_core/solver.h>

#include <fstream>
#include <iostream>

using namespace srrg2_solver;
using namespace md_slam;

using Cloud = PointCloud_<std::vector<srrg2_core::PointNormalIntensity3f,
                                      Eigen::aligned_allocator<PointNormalIntensity3f>>>;

const char* banner[] = {"loads 2 graph, performs alignment and creates a global map encoding "
                        "euclidean distance between points ",
                        0};

struct Color {
  float r, g, b;
};

Isometry3f refinementSVDICP(const std::vector<std::pair<Vector3f, Vector3f>>& pose_pairs_);
void generateGlobalCloud(Cloud& gcloud_, FactorGraphPtr graph_);
pcl::visualization::PCLVisualizer::Ptr
visualizeRGBCloudPCL(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_,
                     const std::string viewer_name_);
template <typename Cloud>
void writePointCloudToPCDFile(const Cloud& cloud_, const std::string filename_);

#include <chrono>
using namespace std::chrono;

template <typename Cloud>
void visualizeCloudOverlap(const Cloud& ref_cloud_,
                           const Cloud& tar_cloud_,
                           const std::string name_);

template <typename Cloud>
int computeMatches(float& error,
                   CorrespondenceVector& matches_,
                   const Cloud& ref_cloud_,
                   const Cloud& tar_cloud_,
                   const float max_distance_) {
  const float max_distance2 = max_distance_ * max_distance_;
  std::unordered_map<size_t, std::pair<size_t, float>> tar_map;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < ref_cloud_.size(); ++i) {
    const Vector3f& p_ref = ref_cloud_[i].coordinates();
    float best_distance   = std::numeric_limits<float>::max();
    for (int j = 0; j < tar_cloud_.size(); ++j) {
      // printProgress((double) i*j / (float) num_combinations);
      // if( ((i+1)*(j+1)) % num_print == 0){
      //   // std::cerr << i*j << std::endl;
      //   std::cerr << i*j/(float)num_combinations << " ";
      // }
      const Vector3f& p_tar = tar_cloud_[j].coordinates();
      const float d_euc     = (p_ref - p_tar).squaredNorm();
      if (d_euc > max_distance2)
        continue;
      // std::cerr << i << " " << j << std::endl;
      if (d_euc < best_distance) {
        // if index j already contained
        if (auto it{tar_map.find(j)}; it != std::end(tar_map)) {
          if (d_euc < it->second.second) { // check if need to be replaced with new one
            tar_map[j]    = std::make_pair(i, sqrt(d_euc));
            best_distance = it->second.second;
          }
        } else {
          tar_map.insert({j, std::make_pair(i, sqrt(d_euc))});
          best_distance = d_euc;
        }
      }
    }
  }
  auto end                                      = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time                          = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished exaustive search at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  int idx    = 0;
  double sum = 0.0;
  matches_.resize(tar_map.size());
  for (auto& e : tar_map) {
    matches_[idx].fixed_idx  = e.second.first;
    matches_[idx].moving_idx = e.first;
    matches_[idx].response   = e.second.second;
    sum += matches_[idx].response;
    idx++;
  }

  error = sum / matches_.size();
  return matches_.size();
}

Point3fVectorCloud convertPointCloud(const Cloud& incloud_) {
  Point3fVectorCloud outcloud;
  outcloud.resize(incloud_.size());
  for (int i = 0; i < incloud_.size(); ++i) {
    outcloud[i].coordinates() = incloud_[i].coordinates();
  }
  return outcloud;
}

int main(int argc, char** argv) {
  // clang-format off
  ParseCommandLine cmd_line(argv, banner);
  ArgumentString reference_graph(&cmd_line, "rg", "reference-graph", "path to reference graph file, usually groundtruth", "");
  ArgumentString target_graph(&cmd_line, "tg", "target-graph", "path to target grapg file, usually estimate", "");
  ArgumentString output_pcd(&cmd_line, "o", "output-pcd", "path to output pcd files aligned, prefix name", "");
  ArgumentFloat max_diff(&cmd_line, "m", "max-timestamp", "max error between timestmap to associate poses", 0.02f);
  ArgumentFloatVector_<3> voxel(&cmd_line, "v", "voxel", "voxel size [m]", {0.05f, 0.05f, 0.05f}); 
  cmd_line.parse();
  // clang-format on

  FactorGraphPtr ref_graph = FactorGraph::read(reference_graph.value());
  if (!ref_graph) {
    std::cerr << "unable to load ref graph" << std::endl;
  }

  FactorGraphPtr tar_graph = FactorGraph::read(target_graph.value());
  if (!tar_graph) {
    std::cerr << "unable to load tar graph" << std::endl;
  }

  // associating translations with timestamps
  std::vector<std::pair<Vector3f, Vector3f>> poses;
  poses.reserve(ref_graph->variables().size());
  for (int i = 0; i < ref_graph->variables().size(); ++i) {
    VariableBase* vi_base = ref_graph->variable(i);
    MDVariableSE3* vi     = dynamic_cast<MDVariableSE3*>(vi_base);
    if (!vi)
      continue;

    const double ref_timestamp     = vi->timestamp();
    const Isometry3f& ref_estimate = vi->estimate();

    double best_diff     = std::numeric_limits<double>::max();
    double best_var_time = std::numeric_limits<double>::max();
    Isometry3f best_var  = Isometry3f::Identity();
    std::pair<Vector3f, Vector3f> pair_translation;
    pair_translation.first = ref_estimate.translation();

    for (int j = 0; j < tar_graph->variables().size(); ++j) {
      VariableBase* vj_base = tar_graph->variable(j);
      MDVariableSE3* vj     = dynamic_cast<MDVariableSE3*>(vj_base);
      if (!vj)
        continue;

      const double tar_timestamp     = vj->timestamp();
      const Isometry3f& tar_estimate = vj->estimate();

      const double curr_diff = fabs(tar_timestamp - ref_timestamp);
      if (curr_diff < best_diff) { //&& best_diff < max_diff.value()) {
        // std::cerr << std::setprecision(12) << " associated: " << var_time << " " << gt_time
        //           << std::endl;
        best_diff     = curr_diff;
        best_var_time = tar_timestamp;
        best_var      = tar_estimate;
      }
    }
    if (best_diff < max_diff.value()) {
      std::cerr << std::setprecision(12) << " associated: " << best_var_time << " " << ref_timestamp
                << " | best diff: " << best_diff << std::endl;
      pair_translation.second = best_var.translation();
      poses.emplace_back(pair_translation);
    }
  }

  std::cerr << "associated num of poses: " << poses.size() << std::endl;
  const Isometry3f estimate = refinementSVDICP(poses);
  std::cerr << "estimate\n" << estimate.matrix() << std::endl;
  std::vector<double> t_errors(poses.size());
  for (const auto& p : poses) {
    t_errors.emplace_back((estimate * p.second - p.first).squaredNorm());
    // std::cerr << t_errors[t_errors.size() - 1] << " ";
  }

  auto acc_lambda = [&](const double& a, const double& b) { return a + b; };

  double ate_sum  = std::accumulate(t_errors.begin(), t_errors.end(), 0.0, acc_lambda);
  const float ate = (float) ate_sum / t_errors.size();
  std::cerr << "absolute translation error | " << ate << std::endl;

  // generate global clouds
  Cloud ref_cloud_full, tar_cloud_full;
  generateGlobalCloud(ref_cloud_full, ref_graph);
  std::cerr << "reference global cloud size: " << ref_cloud_full.size() << std::endl;
  generateGlobalCloud(tar_cloud_full, tar_graph);
  std::cerr << "target global cloud size: " << tar_cloud_full.size() << std::endl;

  const std::string ref_filename = output_pcd.value() + "_ref.pcd";
  writePointCloudToPCDFile(ref_cloud_full, ref_filename);
  const std::string tar_filename = output_pcd.value() + "_tar.pcd";
  tar_cloud_full.transformInPlace<Isometry>(estimate);
  writePointCloudToPCDFile(tar_cloud_full, tar_filename);

  return 0;
}

void generateGlobalCloud(Cloud& gcloud_, FactorGraphPtr graph_) {
  for (auto v_pair : graph_->variables()) {
    VariableBase* v_base = v_pair.second;
    MDVariableSE3* v     = dynamic_cast<MDVariableSE3*>(v_base);
    if (!v)
      continue;

    const Isometry3f& T      = v->estimate();
    const auto pyramid_level = v->pyramid()->at(0);

    MDMatrixVectorCloud cloud;
    pyramid_level->toCloud(cloud);
    cloud.transformInPlace<Isometry>(T);

    gcloud_.insert(gcloud_.end(), cloud.begin(), cloud.end());
    // std::cerr << "local cloud: " << cloud.size() << " | global cloud: " << gcloud_.size()
    //           << std::endl;
  }
}

pcl::visualization::PCLVisualizer::Ptr
visualizeRGBCloudPCL(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_,
                     const std::string viewer_name_) {
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(1, 1, 1);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud_, rgb, viewer_name_);
  // viewer->setPointCloudRenderingProperties(
  //   pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, viewer_name_);
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  return (viewer);
}

Isometry3f refinementSVDICP(const std::vector<std::pair<Vector3f, Vector3f>>& pose_pairs_) {
  const size_t tot_num_inliers = pose_pairs_.size();
  std::vector<Vector3f> query_current(tot_num_inliers);
  std::vector<Vector3f> ref_current(tot_num_inliers);
  const float inv_matches = 1.f / tot_num_inliers;
  Vector3d query_centroid = Vector3d::Zero();
  Vector3d ref_centroid   = Vector3d::Zero();
  for (size_t i = 0; i < tot_num_inliers; ++i) {
    ref_current[i]   = pose_pairs_[i].first;
    query_current[i] = pose_pairs_[i].second;
    ref_centroid.noalias() += ref_current[i].cast<double>();
    query_centroid.noalias() += query_current[i].cast<double>();
  }
  ref_centroid *= inv_matches;
  query_centroid *= inv_matches;

  // solve with svd and get transformation mat
  return md_slam_closures::LoopValidator::solveLinear(
    ref_current, query_current, ref_centroid, query_centroid, tot_num_inliers);
}

template <typename Cloud>
void writePointCloudToPCDFile(const Cloud& cloud_, const std::string filename_) {
  // find matches in whole cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>());
  // fill in the cloud data
  cloud_pcd->points.reserve(cloud_.size());
  for (auto& p : cloud_) {
    pcl::PointXYZ p_pcl;
    if (p.coordinates().norm() < 1e-8f)
      continue;
    p_pcl.x = p.coordinates().x();
    p_pcl.y = p.coordinates().y();
    p_pcl.z = p.coordinates().z();
    cloud_pcd->points.emplace_back(p_pcl);
  }
  cloud_pcd->width    = cloud_pcd->points.size();
  cloud_pcd->height   = 1;
  cloud_pcd->is_dense = true;

  // deserializing pcd
  pcl::io::savePCDFileASCII(filename_, *cloud_pcd);
  std::cerr << "pcd written successfully | size: " << cloud_pcd->points.size()
            << " | path: " << filename_ << std::endl;
}

template <typename Cloud>
void visualizeCloudOverlap(const Cloud& ref_cloud_,
                           const Cloud& tar_cloud_,
                           const std::string viewer_name_) {
  // find matches in whole cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  // fill in the cloud data
  // cloud->points.reserve(ref_cloud.size()*tar_cloud.size());
  for (auto& r : ref_cloud_) {
    pcl::PointXYZRGB p_pcl;
    p_pcl.x = r.coordinates().x();
    p_pcl.y = r.coordinates().y();
    p_pcl.z = r.coordinates().z();
    // create a jet color map for each distance value
    // const Color color = getColor(m.response, 0.f, max_error.value());
    p_pcl.r = 0;
    p_pcl.g = 0;
    p_pcl.b = 255;
    cloud->points.push_back(p_pcl);
  }

  for (auto& t : tar_cloud_) {
    pcl::PointXYZRGB p_pcl;
    p_pcl.x = t.coordinates().x();
    p_pcl.y = t.coordinates().y();
    p_pcl.z = t.coordinates().z();
    // create a jet color map for each distance value
    // const Color color = getColor(m.response, 0.f, max_error.value());
    p_pcl.r = 255;
    p_pcl.g = 124;
    p_pcl.b = 0;
    cloud->points.push_back(p_pcl);
  }
  cloud->width    = cloud->points.size();
  cloud->height   = 1;
  cloud->is_dense = true;

  // // pcl visualization
  pcl::visualization::PCLVisualizer::Ptr viewer = visualizeRGBCloudPCL(cloud, viewer_name_);
  // //--------------------
  // // -----Main loop-----
  // //--------------------
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(100ms);
  }
}