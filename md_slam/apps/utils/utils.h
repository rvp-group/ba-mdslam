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
#include <iostream>
#include <srrg_geometry/geometry3d.h>

using namespace srrg2_core;

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