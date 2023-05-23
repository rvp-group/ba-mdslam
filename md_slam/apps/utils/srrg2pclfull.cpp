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

void write_pcd(const Cloud& cloud_, const std::string filename_) {
  // converting srrg pcl into standard pcl
  pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
  // fill in the cloud data
  pcl_cloud.points.reserve(pcl_cloud.width * pcl_cloud.height);
  for (int i = 0; i < cloud_.size(); ++i) {
    if (cloud_[i].status != Valid)
      continue;
    if (cloud_[i].coordinates().norm() < 1e-8f)
      continue;
    pcl::PointXYZRGB p_pcl;
    p_pcl.x                       = cloud_[i].coordinates().x();
    p_pcl.y                       = cloud_[i].coordinates().y();
    p_pcl.z                       = cloud_[i].coordinates().z();
    const uint8_t intensity_value = (uint8_t)(cloud_[i].intensity() * 255);
    p_pcl.r                       = intensity_value;
    p_pcl.g                       = intensity_value;
    p_pcl.b                       = intensity_value;
    pcl_cloud.points.emplace_back(p_pcl);
  }
  pcl_cloud.width    = pcl_cloud.points.size();
  pcl_cloud.height   = 1;
  pcl_cloud.is_dense = true;
  // write pcd
  pcl::io::savePCDFileASCII(filename_, pcl_cloud);
  std::cerr << "global pcd written successfully | size: " << pcl_cloud.points.size()
            << " | path: " << filename_ << std::endl;
}

int main(int argc, char** argv) {
  // clang-format off
  ParseCommandLine cmd_line(argv, banner);
  ArgumentString input_graph(&cmd_line, "i", "input", "path to input graph", "");
  ArgumentString output_pcd(&cmd_line, "o", "output", "path to global pcd file", "");
  cmd_line.parse();
  // clang-format on

  if (!input_graph.isSet()) {
    std::cerr << "no input provided, aborting" << std::endl;
    return 0;
  }

  if (!output_pcd.isSet()) {
    std::cerr << "no output provided, aborting" << std::endl;
    return 0;
  }

  FactorGraphPtr graph = FactorGraph::read(input_graph.value());
  if (!graph) {
    std::cerr << "unable to load graph" << std::endl;
  }

  std::cerr << "input graph loaded, n vars: " << graph->variables().size()
            << "| n factors: " << graph->factors().size() << std::endl;

  Cloud global_cloud;

  for (auto v_pair : graph->variables()) {
    VariableBase* v_base = v_pair.second;
    MDVariableSE3* v     = dynamic_cast<MDVariableSE3*>(v_base);
    if (!v)
      continue;

    // std::cerr << "variable!\n";

    const Isometry3f& T      = v->estimate();
    const auto pyramid_level = v->pyramid()->at(0);

    MDMatrixVectorCloud cloud;
    pyramid_level->toCloud(cloud);
    cloud.transformInPlace<Isometry>(T);

    global_cloud.insert(global_cloud.end(), cloud.begin(), cloud.end());
    std::cerr << "local cloud: " << cloud.size() << " | global cloud: " << global_cloud.size()
              << std::endl;
  }

  write_pcd(global_cloud, output_pcd.value());
}
