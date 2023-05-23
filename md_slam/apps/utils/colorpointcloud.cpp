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
#include <unistd.h>
#include <vector>

#include <srrg_solver/solver_core/solver.h>
#include <srrg_solver/variables_and_factors/types_3d/se3_pose_pose_geodesic_error_factor.h>

#include <md_slam/factor_bi.cuh>
#include <md_slam/pyramid_generator.h>
#include <md_slam/pyramid_variable_se3.h>
#include <md_slam/utils.cuh>

#include <srrg_config/configurable_manager.h>
#include <srrg_config/pipeline_runner.h>
#include <srrg_messages/message_handlers/message_sorted_sink.h>
#include <srrg_messages/message_handlers/message_synchronized_sink.h>

#include <srrg_messages_ros/instances.h>
#include <srrg_pcl/point_color.h>
#include <srrg_system_utils/parse_command_line.h>
#include <srrg_system_utils/system_utils.h>

#include <pcl_conversions/pcl_conversions.h>

#include "utils.h"

using namespace srrg2_core;
using namespace srrg2_core_ros;
using namespace srrg2_solver;
using namespace md_slam;

const char* banner[] = {"", 0};
using PointColor3fVector =
  std::vector<srrg2_core::PointColor3f, Eigen::aligned_allocator<PointColor3f>>;
using MDMatrixColorVectorCloud = srrg2_core::PointCloud_<
  srrg2_core::Matrix_<srrg2_core::PointColor3f, Eigen::aligned_allocator<PointColor3f>>>;
using PointColorUnprojectorPinhole =
  srrg2_core::PointUnprojectorPinhole_<srrg2_core::PointColor3fVectorCloud>;
using TrajectorySE3 = std::vector<Isometry3f, Eigen::aligned_allocator<Isometry3f>>;

void prepareImageRGB(ImageVector3f& dest,
                     const ImageVector3f& src,
                     int max_scale,
                     int row_scale,
                     int col_scale,
                     bool suppress_zero) {
  int d_rows = src.rows() / row_scale;
  int d_cols = src.cols() / col_scale;
  d_rows     = (d_rows / max_scale) * max_scale;
  d_cols     = (d_cols / max_scale) * max_scale;

  int s_rows = d_rows * row_scale;
  int s_cols = d_cols * col_scale;
  // cerr << "d_rows: " << d_rows << " d_cols: " << d_cols << endl;
  // cerr << "s_rows: " << s_rows << " s_cols: " << s_cols << endl;

  dest.resize(d_rows, d_cols);
  dest.fill(Vector3f::Zero());
  ImageInt counts(d_rows, d_cols);
  counts.fill(0);
  for (int r = 0; r < s_rows; ++r) {
    int dr = r / row_scale;
    for (int c = 0; c < s_cols; ++c) {
      int dc = c / col_scale;
      auto v = src.at(r, c);
      if (suppress_zero && v.isZero())
        continue;
      dest.at(dr, dc) += src.at(r, c);
      ++counts.at(dr, dc);
    }
  }
  for (int r = 0; r < d_rows; ++r) {
    for (int c = 0; c < d_cols; ++c) {
      int cnt = counts.at(r, c);
      if (cnt)
        dest.at(r, c) *= (1. / cnt);
    }
  }
}

class MDColorStorage : public MDPyramidGenerator {
public:
  void setGraph(const FactorGraphPtr graph_) {
    _graph = graph_;
  }

  void setImgScale(const int img_scale_) {
    _img_scale     = img_scale_;
    _inv_img_scale = 1.f / (float) img_scale_;
  }

  void setMaxDepth(const float max_depth_) {
    _max_depth = max_depth_;
  }

  void setImagesColor(const ImageUInt16& raw_depth, const BaseImage& raw_intensity) {
    assert((raw_depth.rows() == raw_intensity.rows() || raw_depth.cols() == raw_intensity.cols()) &&
           "MDColorStorage::setImages | input image size mismatch");

    // scale fuckin depth
    _full_depth.resize(raw_depth.rows(), raw_depth.cols());
    for (size_t i = 0; i < _full_depth.size(); ++i) {
      _full_depth[i] = _depth_scale * raw_depth[i];
      if (_full_depth[i] > _max_depth)
        _full_depth = 0.f;
    }
    prepareImage(_depth, _full_depth, _img_scale, _img_scale, _img_scale, true);
    // allocatePyramids();

    const ImageVector3uc* rgb_image = dynamic_cast<const ImageVector3uc*>(&raw_intensity);
    if (rgb_image) {
      cv::Mat cv_rgb, cv_intensity;
      rgb_image->toCv(cv_rgb);
      rgb_image->convertTo(_full_rgb, 1.f / 255);
      prepareImageRGB(_rgb, _full_rgb, _img_scale, _img_scale, _img_scale, false);
      cv::cvtColor(cv_rgb, cv_intensity, CV_RGB2GRAY);
      ImageUInt8 raw_intensity;
      raw_intensity.fromCv(cv_intensity);
      raw_intensity.convertTo(_full_intensity, 1. / 255);
      return;
    }
    const ImageUInt8* intensity_image = dynamic_cast<const ImageUInt8*>(&raw_intensity);
    if (intensity_image) {
      intensity_image->convertTo(_full_intensity, 1. / 255);
      throw std::runtime_error(
        "MDColorStorage::setImages | cannot read color, image read as intensity!");
    }
    throw std::runtime_error("MDColorStorage::setImages | unknown intensity");
  }

  // overriding put message
  bool putMessage(srrg2_core::BaseSensorMessagePtr msg_) override {
    MessagePackPtr pack = std::dynamic_pointer_cast<MessagePack>(msg_);
    if (!pack) {
      std::cerr << "MDColorStorage::putMessage | no pack received" << std::endl;
      return false;
    }

    ImageMessagePtr depth;
    ImageMessagePtr intensity;
    CameraInfoMessagePtr camera_info;
    for (auto& m : pack->messages) {
      if (m->topic.value() == param_depth_topic.value()) {
        depth = std::dynamic_pointer_cast<ImageMessage>(m);
        assert(depth && "MDColorStorage::putMessage|depth, invalid type");
      }

      if (m->topic.value() == param_intensity_topic.value()) {
        intensity = std::dynamic_pointer_cast<ImageMessage>(m);
        assert(intensity && "MDColorStorage::putMessage|intensity, invalid type");
      }

      if (m->topic.value() == param_camera_info_topic.value()) {
        camera_info = std::dynamic_pointer_cast<CameraInfoMessage>(m);
        assert(camera_info && "MDColorStorage::putMessage|camera_info, invalid type");
      }
    }

    assert(depth->image()->rows() == intensity->image()->rows() &&
           "MDColorStorage::putMessage|depth and intensity rows size mismatch");
    assert(depth->image()->cols() == intensity->image()->cols() &&
           "MDColorStorage::putMessage|depth and intensity cols size mismatch");

    // TODO set sensor offset for LiDAR-RGBD mapping

    // set to pinhole since rgb channels are not for lidar
    camera_info->projection_model.setValue("pinhole");
    CameraType camera_type = Pinhole;
    setCameraType(camera_type);
    setDepthScale(camera_info->depth_scale.value());
    if (param_depth_scale_override.value() > 0)
      setDepthScale(param_depth_scale_override.value());

    Matrix3f K_scaled = camera_info->camera_matrix.value();
    K_scaled.block<2, 3>(0, 0) *= _inv_img_scale;
    setCameraMatrix(K_scaled);

    ImageUInt16* depth_16 = dynamic_cast<ImageUInt16*>(depth->image());
    assert(depth_16 && "MDColorStorage::putMessage|depth not uint16");

    setImagesColor(*depth_16, *intensity->image());

    _rgb_matrix_cloud.resize(_rgb.rows(), _rgb.cols());
    _rgb_matrix_cloud.fill(PointColor3f());
    _mask = (_depth == 0);

    PointColorUnprojectorPinhole pinhole_unprojector;
    pinhole_unprojector.setCameraMatrix(K_scaled);

    int unp_mat_valid = pinhole_unprojector.computeMatrix(_rgb_matrix_cloud, _depth, _rgb);
    assert(unp_mat_valid && "MDColorStorage::compute|invalid projection");

    // TODO this is never populated
    _rgb_matrix_cloud.transformInPlace<TRANSFORM_CLASS::Isometry>(_sensor_offset);

    // cv::Mat cv_rgb, cv_intensity, cv_depth;
    // _rgb.toCv(cv_rgb);
    // // _full_intensity.toCv(cv_intensity);
    // _depth.toCv(cv_depth);
    // cv::imshow("rgb", cv_rgb);
    // // cv::imshow("intensity", cv_intensity);
    // cv::imshow("depth", cv_depth);
    // cv::waitKey(0);

    // _full_intensity, _full_depth, _full_rgb populated

    const double imgs_timestamp = depth->timestamp.value();

    // syncronize pose with images
    double best_diff           = std::numeric_limits<double>::max();
    double best_posegraph_time = std::numeric_limits<double>::max();
    Isometry3f best_isometry   = Isometry3f::Identity();
    for (auto var = _graph->variables().begin(); var != _graph->variables().end(); ++var) {
      MDVariableSE3* v       = dynamic_cast<MDVariableSE3*>(var.value());
      const double curr_diff = fabs(imgs_timestamp - v->timestamp());
      if (curr_diff == 0.f) {
        best_diff           = curr_diff;
        best_posegraph_time = v->timestamp();
        best_isometry       = v->estimate();

        std::cerr << ".";
        _rgb_matrix_cloud.transformInPlace<TRANSFORM_CLASS::Isometry>(best_isometry);
        _full_rgb_cloud.insert(
          _full_rgb_cloud.end(), _rgb_matrix_cloud.begin(), _rgb_matrix_cloud.end());

        // std::back_insert_iterator<PointColor3fVectorCloud> dest(_rgb_cloud);
        // _rgb_matrix_cloud.copyTo(dest);

        // // just remove redundant points
        // Vector6f scales = Vector6f::Zero();
        // scales << 0.01, 0.01, 0.01, -1.0, -1.0, -1.0;
        // std::back_insert_iterator<PointColor3fVectorCloud> voxelized_iterator(_full_rgb_cloud);
        // _rgb_cloud.voxelize(voxelized_iterator, scales);

        _size++;
      }
    }

    _rgb_matrix_cloud.clear();
    return true;
  }

  void writePCD(const std::string& filename_) {
    std::cerr << "\npostprocessed num poses: " << _size
              << " | vs num graph variables: " << _graph->variables().size() << std::endl;
    if (_size != _graph->variables().size()) {
      throw std::runtime_error("");
    }

    // converting srrg pcl into standard pcl
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    // Fill in the cloud data
    pcl_cloud.points.reserve(pcl_cloud.width * pcl_cloud.height);
    for (int i = 0; i < _full_rgb_cloud.size(); ++i) {
      if (_full_rgb_cloud[i].coordinates().norm() < 1e-8f) // if this is 0
        continue;
      pcl::PointXYZRGB p_pcl;
      p_pcl.x = _full_rgb_cloud[i].coordinates().x();
      p_pcl.y = _full_rgb_cloud[i].coordinates().y();
      p_pcl.z = _full_rgb_cloud[i].coordinates().z();
      p_pcl.r = (uint8_t)(_full_rgb_cloud[i].color()(0) * 255);
      p_pcl.g = (uint8_t)(_full_rgb_cloud[i].color()(1) * 255);
      p_pcl.b = (uint8_t)(_full_rgb_cloud[i].color()(2) * 255);
      pcl_cloud.points.emplace_back(p_pcl);
    }
    _full_rgb_cloud.clear();
    pcl_cloud.width    = pcl_cloud.points.size();
    pcl_cloud.height   = 1;
    pcl_cloud.is_dense = true;

    pcl::io::savePCDFileASCII(filename_, pcl_cloud);
  }

protected:
  FactorGraphPtr _graph;
  ImageVector3f _full_rgb, _rgb;
  MDMatrixColorVectorCloud _rgb_matrix_cloud;
  PointColor3fVectorCloud _rgb_cloud;
  PointColor3fVectorCloud _full_rgb_cloud;
  float _inv_img_scale = 1.f;
  float _max_depth     = 5.f;
  int _img_scale       = 1;
  int _size            = 0;
};

using MDColorStoragePtr = std::shared_ptr<MDColorStorage>;
BOSS_REGISTER_CLASS(MDColorStorage)

int main(int argc, char** argv) {
  srrgInit(argc, argv, "colorpointcloud");

  // clang-format off
  ParseCommandLine cmd(argv, banner);
  ArgumentString input_graph(&cmd, "ig", "input-graph", "path to the input graph", "");
  ArgumentString input_rosbag(&cmd, "ir", "input-rosbag", "path to input rosbag with rgb images", "");
  ArgumentString output_pcd(&cmd, "o", "output-pcd", "path to output pcd file", "");
  ArgumentString arg_topic_rgb(&cmd, "toi", "topic-intensity", "name of topic intensity image", "/rgbd/image_rgb");
  ArgumentString arg_topic_depth(&cmd, "tod", "topic-depth", "name of topic depth image", "/rgbd/image_depth");
  ArgumentString arg_topic_camera_info(&cmd, "tk", "topic-camera-info", "name of topic camera info", "/rgbd/camera_info");
  ArgumentFloat arg_depth_scale(&cmd, "d", "depth-scale", "depth scale used", 0.0002);
  ArgumentFloat arg_max_depth(&cmd, "m", "max-depth", "max depth to chop inverse projection", 3.f);
  ArgumentInt arg_scale_img(&cmd, "s", "scale-img", "scale factor for img", 2);
  cmd.parse();
  // clang-format on

  if (!input_graph.isSet()) {
    std::cerr << "no input graph provided, aborting" << std::endl;
    return -1;
  }

  if (!input_rosbag.isSet()) {
    std::cerr << "no input rosbag provided, aborting" << std::endl;
    return -1;
  }

  if (!output_pcd.isSet()) {
    std::cerr << "no output filename provided, aborting" << std::endl;
    return -1;
  }

  FactorGraphPtr graph = FactorGraph::read(input_graph.value());
  if (!graph) {
    std::cerr << "unable to load graph" << std::endl;
  }

  std::cerr << "input graph loaded, n vars: " << graph->variables().size() << std::endl;

  ConfigurableManager manager;

  const std::vector<std::string> t_vec = {
    arg_topic_rgb.value(), arg_topic_depth.value(), arg_topic_camera_info.value()};

  auto colorator = manager.create<MDColorStorage>("colorator");
  colorator->param_intensity_topic.setValue(arg_topic_rgb.value());
  colorator->param_depth_topic.setValue(arg_topic_depth.value());
  colorator->param_camera_info_topic.setValue(arg_topic_camera_info.value());
  colorator->param_depth_scale_override.setValue(arg_depth_scale.value());
  colorator->setImgScale(arg_scale_img.value());
  colorator->setMaxDepth(arg_max_depth.value());

  auto sync = manager.create<MessageSynchronizedSink>("sync");
  sync->param_topics.setValue(t_vec);
  sync->param_push_sinks.value().emplace_back(colorator);

  auto sorter = manager.create<MessageSortedSink>("sorter");
  sorter->param_push_sinks.value().emplace_back(sync);

  auto runner = manager.create<PipelineRunner>("runner");
  runner->param_push_sinks.value().emplace_back(sorter);

  // creating and setup source for reading
  MessageROSBagSourcePtr source = std::make_shared<MessageROSBagSource>();
  source->param_topics.setValue(sync->param_topics.value());
  runner->param_source.setValue(source);

  source->open(input_rosbag.value());
  colorator->setGraph(graph);
  runner->compute();

  colorator->writePCD(output_pcd.value());
}
