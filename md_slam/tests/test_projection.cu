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

#include <gtest/gtest.h>
#include <iostream>
#include <md_slam/dual_matrix.cuh>
#include <md_slam/factor.cuh>
#include <md_slam/factor_common.cu>
#include <md_slam/pyramid_level.cuh>
#include <random>
#include <srrg_pcl/point_types.h>
#include <srrg_test/test_helper.hpp>
#include <stdlib.h>

// clock stuff
#include <chrono>
#include <ctime>
#include <ratio>

using namespace std::chrono;

#ifndef MD_TEST_DATA_FOLDER
#error "NO TEST DATA FOLDER"
#endif

using namespace md_slam;
using namespace srrg2_core;

void computeProjectionsHost(Workspace& workspace_,
                            const MDMatrixCloud& cloud_,
                            const MDPyramidMatrix& level_,
                            const Isometry3f& SX_,
                            const CameraType& cam_type_,
                            const Matrix3f& cam_mat_,
                            const float min_depth_,
                            const float max_depth_) {
  workspace_.fill(WorkspaceEntry());

  for (int i = 0; i < cloud_.size(); ++i) {
    const PointNormalIntensity3f& full_point = cloud_.at(i);
    const Vector3f& point                    = full_point.coordinates();
    const Vector3f& normal                   = full_point.normal();
    const float intensity                    = full_point.intensity();
    const Vector3f transformed_point         = SX_ * point;

    float depth           = 0.f;
    Vector3f camera_point = Vector3f::Zero();
    Vector2f image_point  = Vector2f::Zero();

    const bool& is_good = project(image_point,
                                  camera_point,
                                  depth,
                                  transformed_point,
                                  cam_type_,
                                  cam_mat_,
                                  min_depth_,
                                  max_depth_);
    if (!is_good)
      continue;

    const int irow = cvRound(image_point.y());
    const int icol = cvRound(image_point.x());

    if (!workspace_.inside(irow, icol)) {
      continue;
    }

    // check if masked
    if (level_.at(irow, icol).masked()) {
      continue;
    }

    WorkspaceEntry& entry = workspace_.at(irow, icol);

    if (depth > entry.depth())
      continue;

    const Vector3f rn = SX_.linear() * normal;
    entry._prediction << intensity, depth, rn.x(), rn.y(), rn.z();
    entry._point             = point;
    entry._normal            = normal;
    entry._transformed_point = transformed_point;
    entry._camera_point      = camera_point;
    entry._image_point       = image_point;
    entry._index             = i;
    entry._chi               = 0;
    entry._status            = Good;
  }
}

TEST(DUMMY, DummyProject) {
  const int rows = 700;
  const int cols = 700;
  // generate random image point cloud
  MDMatrixCloud cloud(rows, cols);
  cloud.fill(PointNormalIntensity3f());
  for (int i = 0; i < cloud.size(); ++i) {
    cloud.at(i).coordinates() = Vector3f::Random() * 50;
    cloud.at(i).normal()      = Vector3f::Random();
    cloud.at(i).intensity()   = rand() / (RAND_MAX + 1.);
  }
  cloud.toDevice();

  Workspace dworkspace(rows, cols);
  dworkspace.fill(WorkspaceEntry());

  // create a dummy pyramid and set all point to not masked
  MDPyramidMatrix mat(rows, cols);
  mat.fill(MDPyramidMatrixEntry());
  for (int i = 0; i < mat.size(); ++i) {
    mat.at(i).setMasked(false);
  }
  mat.toDevice();

  Isometry3f offset     = Isometry3f::Identity();
  Matrix3f K            = Matrix3f::Identity();
  CameraType cam_type   = CameraType::Pinhole;
  const float max_depth = 100.f;
  const float min_depth = 0.3;

  project_kernel<<<cloud.nBlocks(), cloud.nThreads()>>>(dworkspace.deviceInstance(),
                                                        cloud.deviceInstance(),
                                                        mat.deviceInstance(),
                                                        offset,
                                                        cam_type,
                                                        K,
                                                        min_depth,
                                                        max_depth,
                                                        true);
  cudaDeviceSynchronize();
  project_kernel<<<cloud.nBlocks(), cloud.nThreads()>>>(dworkspace.deviceInstance(),
                                                        cloud.deviceInstance(),
                                                        mat.deviceInstance(),
                                                        offset,
                                                        cam_type,
                                                        K,
                                                        min_depth,
                                                        max_depth,
                                                        false);
  cudaDeviceSynchronize();
  dworkspace.toHost();

  // do projection on host
  Workspace hworkspace(rows, cols);
  hworkspace.fill(WorkspaceEntry());
  computeProjectionsHost(hworkspace, cloud, mat, offset, cam_type, K, min_depth, max_depth);

  // compare
  for (int r = 0; r < hworkspace.rows(); ++r) {
    for (int c = 0; c < hworkspace.cols(); ++c) {
      // clang-format off
      ASSERT_EQ(hworkspace.at(r, c)._point(0), dworkspace.at(r, c)._point(0));
      ASSERT_EQ(hworkspace.at(r, c)._point(1), dworkspace.at(r, c)._point(1));
      ASSERT_EQ(hworkspace.at(r, c)._point(2), dworkspace.at(r, c)._point(2));
      ASSERT_EQ(hworkspace.at(r, c)._transformed_point(0), dworkspace.at(r, c)._transformed_point(0));
      ASSERT_EQ(hworkspace.at(r, c)._transformed_point(1), dworkspace.at(r, c)._transformed_point(1));
      ASSERT_EQ(hworkspace.at(r, c)._transformed_point(2), dworkspace.at(r, c)._transformed_point(2));
      ASSERT_EQ(hworkspace.at(r, c)._camera_point(0), dworkspace.at(r, c)._camera_point(0));
      ASSERT_EQ(hworkspace.at(r, c)._camera_point(1), dworkspace.at(r, c)._camera_point(1));
      ASSERT_EQ(hworkspace.at(r, c)._camera_point(2), dworkspace.at(r, c)._camera_point(2));
      ASSERT_EQ(hworkspace.at(r, c)._normal(0), dworkspace.at(r, c)._normal(0));
      ASSERT_EQ(hworkspace.at(r, c)._normal(1), dworkspace.at(r, c)._normal(1));
      ASSERT_EQ(hworkspace.at(r, c)._normal(2), dworkspace.at(r, c)._normal(2));
      ASSERT_EQ(hworkspace.at(r, c)._image_point(0), dworkspace.at(r, c)._image_point(0));
      ASSERT_EQ(hworkspace.at(r, c)._image_point(1), dworkspace.at(r, c)._image_point(1));
      ASSERT_EQ(hworkspace.at(r, c).intensity(), dworkspace.at(r, c).intensity());
      ASSERT_EQ(hworkspace.at(r, c).depth(), dworkspace.at(r, c).depth());
      ASSERT_EQ(hworkspace.at(r, c)._index, dworkspace.at(r, c)._index);
      // clang-format on
    }
  }
}

TEST(DUMMY, DummyStressDepthBuffer) {
  Isometry3f offset     = Isometry3f::Identity();
  Matrix3f K            = Matrix3f::Identity();
  CameraType cam_type   = CameraType::Pinhole;
  const float max_depth = 200.f;
  const float min_depth = 0.01f;

  const int rows = 1000;
  const int cols = 1000;
  // generate random image point cloud
  MDMatrixCloud cloud(rows, cols);
  cloud.fill(PointNormalIntensity3f());

  const int p_row = rows / 2;
  const int p_col = cols / 2;

  const float step_depth = (max_depth - min_depth) / (rows * cols);
  float d                = min_depth; // current depth to be incremented
  std::cerr << "step depth: " << step_depth << " | among: " << rows * cols << std::endl;
  for (int i = 0; i < cloud.size(); ++i) {
    // all points have to fall in the same pixel to stress atomic cuda operations
    cloud.at(i).coordinates() = K.inverse() * Vector3f(p_col * d, p_row * d, d);
    d += step_depth;
    // std::cerr << cloud.at(i).coordinates().transpose() << std::endl;
  }
  cloud.toDevice();

  Workspace dworkspace(rows, cols);
  dworkspace.fill(WorkspaceEntry());

  // create a dummy pyramid and set all point to not masked
  MDPyramidMatrix mat(rows, cols);
  mat.fill(MDPyramidMatrixEntry());
  for (int i = 0; i < mat.size(); ++i) {
    mat.at(i).setMasked(false);
  }
  mat.toDevice();

  project_kernel<<<cloud.nBlocks(), cloud.nThreads()>>>(dworkspace.deviceInstance(),
                                                        cloud.deviceInstance(),
                                                        mat.deviceInstance(),
                                                        offset,
                                                        cam_type,
                                                        K,
                                                        min_depth,
                                                        max_depth,
                                                        true);
  cudaDeviceSynchronize();
  project_kernel<<<cloud.nBlocks(), cloud.nThreads()>>>(dworkspace.deviceInstance(),
                                                        cloud.deviceInstance(),
                                                        mat.deviceInstance(),
                                                        offset,
                                                        cam_type,
                                                        K,
                                                        min_depth,
                                                        max_depth,
                                                        false);
  cudaDeviceSynchronize();
  dworkspace.toHost();

  ASSERT_EQ(min_depth, dworkspace.at(p_row, p_col).depth());

  // std::cerr << dworkspace.at(p_row, p_col).depth() << std::endl;
  // std::cerr << min_depth << std::endl;
}

int main(int argc, char** argv) {
  return srrg2_test::runTests(argc, argv, true /*use test folder*/);
}
