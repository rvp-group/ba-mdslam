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

#include <md_slam/pyramid_variable_se3.h>

#include <srrg_system_utils/parse_command_line.h>
#include <srrg_system_utils/system_utils.h>

#include <pcl_conversions/pcl_conversions.h>
#include <srrg_solver/solver_core/solver.h>

#include <srrg_solver/solver_core/factor_graph.h>

#include <fstream>
#include <iostream>

using namespace srrg2_solver;
using namespace md_slam;

using Cloud =
  std::vector<srrg2_core::PointNormalIntensity3f, Eigen::aligned_allocator<PointNormalIntensity3f>>;

const char* banner[] = {
  "loads a graph with pyramids attached and write a global point cloud in pcd format",
  0};

void get_pose_stream(std::ofstream* stream_,
                     const Isometry3f& T_,
                     const double& timestamp_,
                     const bool is_tum_) {
  if (is_tum_) {
    const auto trans = geometry3d::t2tnqw(T_).head(3);
    const auto quat  = geometry3d::t2tnqw(T_).tail(4);
    *stream_ << std::fixed << std::setprecision(6) << timestamp_ << " " << trans.transpose() << " "
             << quat(1) << " " << quat(2) << " " << quat(3) << " " << quat(0);
  } else {
    // dump in sick matrix format
    // clang-format off
    *stream_ << T_(0, 0) << "," << T_(0, 1) << "," << T_(0, 2) << "," << T_(0, 3) << "\n" <<
                T_(1, 0) << "," << T_(1, 1) << "," << T_(1, 2) << "," << T_(1, 3) << "\n" <<
                T_(2, 0) << "," << T_(2, 1) << "," << T_(2, 2) << "," << T_(2, 3) << "\n" <<
                T_(3, 0) << "," << T_(3, 1) << "," << T_(3, 2) << "," << 
                std::fixed << std::setprecision(6) << timestamp_;
    // clang-format on
  }
  *stream_ << std::endl;
}

int main(int argc, char** argv) {
  // clang-format off
  ParseCommandLine cmd_line(argv, banner);
  ArgumentString input_graph(&cmd_line, "i", "input", "path to input graph", "");
  ArgumentString output(&cmd_line, "o", "output", "path to output graph ", "");
  ArgumentFlag is_tum(&cmd_line, "tum", "tum", "dump trajectory in tum format (timestamp tx ... qw)", true);
  cmd_line.parse();
  // clang-format on

  if (!input_graph.isSet()) {
    std::cerr << "no input provided, aborting" << std::endl;
    return 0;
  }

  if (!output.isSet()) {
    std::cerr << "no output provided, aborting" << std::endl;
    return 0;
  }

  FactorGraphPtr graph = FactorGraph::read(input_graph.value());
  if (!graph) {
    std::cerr << "unable to load graph" << std::endl;
  }

  std::cerr << "input graph loaded, n vars: " << graph->variables().size()
            << "| n factors: " << graph->factors().size() << std::endl;

  // strip extension from filename and set up writer
  size_t lastindex            = output.value().find_last_of(".");
  const std::string base_name = output.value().substr(0, lastindex);
  std::ofstream os(base_name + "_trajectory.txt");

  int counter = 0;
  for (auto v_pair : graph->variables()) {
    VariableBase* v_base = v_pair.second;
    MDVariableSE3* v     = dynamic_cast<MDVariableSE3*>(v_base);
    if (!v)
      continue;

    const Isometry3f& T      = v->estimate();
    const auto pyramid_level = v->pyramid()->at(0);

    // getting cloud in md slam matrix format
    MDMatrixVectorCloud cloud;
    pyramid_level->toCloud(cloud);

    // converting srrg pcl into standard pcl
    pcl::PointCloud<pcl::PointXYZINormal> pcl_cloud;
    // Fill in the cloud data
    pcl_cloud.width    = cloud.size();
    pcl_cloud.height   = 1;
    pcl_cloud.is_dense = false;
    pcl_cloud.points.resize(pcl_cloud.width * pcl_cloud.height);
    for (int i = 0; i < cloud.size(); ++i) {
      pcl::PointXYZINormal p_pcl;
      p_pcl.x             = cloud[i].coordinates().x();
      p_pcl.y             = cloud[i].coordinates().y();
      p_pcl.z             = cloud[i].coordinates().z();
      p_pcl.normal_x      = cloud[i].normal().x();
      p_pcl.normal_y      = cloud[i].normal().y();
      p_pcl.normal_z      = cloud[i].normal().z();
      p_pcl.intensity     = cloud[i].intensity();
      pcl_cloud.points[i] = p_pcl;
    }
    // write pcd
    const std::string pcd_filename = base_name + "_" + std::to_string(counter++) + ".pcd";
    pcl::io::savePCDFileASCII(pcd_filename, pcl_cloud);

    // write pose
    get_pose_stream(&os, T, v->timestamp(), is_tum.isSet());

    std::cerr << "saved " << pcd_filename << " size: " << pcl_cloud.size() << std::endl;
  }
  os.close();
}
