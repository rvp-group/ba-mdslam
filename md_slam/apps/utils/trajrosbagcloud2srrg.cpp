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
#include <srrg_system_utils/parse_command_line.h>
#include <srrg_system_utils/system_utils.h>

#include "utils.h"

using namespace srrg2_core;
using namespace srrg2_core_ros;
using namespace srrg2_solver;
using namespace md_slam;

const char* banner[] = {"", 0};

class MDPyramidGraphGenerator : public MDPyramidGenerator {
public:
  using PyrQueue    = std::queue<MDImagePyramidMessagePtr>;
  using PyrQueuePtr = std::unique_ptr<PyrQueue>;
  MDPyramidGraphGenerator() {
    // creating factor graph for serialization
    _graph    = FactorGraphPtr(new FactorGraph);
    _prev_var = MDVariableSE3Ptr(new MDVariableSE3);
    _init_T.setIdentity();
    _offset.setIdentity();
    _max_diff        = 0.0;
    _max_angle       = 0.f;
    _max_translation = 0.f;
    _graph_id        = 0;
  }

  void setPoses(const GTPoses& gt_poses_) {
    _gt_poses = gt_poses_;
  }

  void setTimeMaxDiff(const double& max_diff_) {
    _max_diff = max_diff_;
  }

  FactorGraphPtr graph() {
    return _graph;
  }

  void setDownsamplingAngle(const float max_angle_) {
    _max_angle = max_angle_;
  }

  void setDownsamplingTranslation(const float max_translation_) {
    _max_translation = max_translation_;
  }

  void forceSensorOffset(const Isometry3f& offset_) {
    _offset = offset_;
  }

  // overriding put message
  bool putMessage(srrg2_core::BaseSensorMessagePtr msg) override {
    bool is_good = MDPyramidGenerator::putMessage(msg);
    if (is_good) {
      if (_pyramid_msg) { // if pyramid msg is valid
                          // initializing some stuff
        MDVariableSE3Ptr var(new MDVariableSE3);
        double best_diff      = std::numeric_limits<double>::max();
        double best_gt_time   = std::numeric_limits<double>::max();
        Isometry3f best_pose  = Isometry3f::Identity();
        const double pyr_time = _pyramid_msg->get()->timestamp();
        // find best pose associated (close in time)
        for (auto& gt_pose : _gt_poses) {
          const double gt_time   = gt_pose.timestamp;
          const double curr_diff = fabs(pyr_time - gt_time);
          if (curr_diff < best_diff) {
            best_diff    = curr_diff;
            best_gt_time = gt_time;
            best_pose    = gt_pose.pose;
            var->setPyramid(new MDImagePyramid(*_pyramid_msg->get()));
            var->setTimestamp(pyr_time);
          }
        }
        if (best_diff < _max_diff) {
          std::cerr << "graph id: " << _graph_id << std::fixed << std::setprecision(10)
                    << " associated: " << best_gt_time << " " << var->timestamp()
                    << " | best diff: " << best_diff << std::endl;

          // first pose is locked
          if (!_graph_id) {
            var->setStatus(VariableBase::Fixed);
            var->setGraphId(_graph_id++);
            _init_T = best_pose;
            var->setEstimate(_init_T.inverse() * best_pose * _offset);
            _graph->addVariable(var);
            _prev_var = var;
            return is_good;
          }

          const Isometry3f curr_T = _init_T.inverse() * best_pose * _offset;
          const Isometry3f rel_T  = _prev_var->estimate().inverse() * curr_T;

          // poses downsampling
          Eigen::AngleAxisf aa(rel_T.linear());
          if ((fabs(aa.angle()) < _max_angle) && (rel_T.translation().norm() < _max_translation))
            return is_good;

          // std::cerr << "T\n" << curr_T.matrix() << std::endl;
          // std::cerr << "prev T\n" << _prev_var->estimate().matrix() << std::endl;
          // std::cerr << "rel T\n" << rel_T.matrix() << std::endl;
          // std::cerr << "===================" << std::endl;
          // usleep(100);

          var->setGraphId(_graph_id);
          // twoworldsT * best_pose * sensorT
          var->setEstimate(curr_T);
          // var->setEstimate(curr_T);
          // var->setEstimate(best_pose);
          _graph->addVariable(var);

          if (_graph_id) {
            // this is just required to be consistent with BA input
            // requires a full factor-graph
            FactorBasePtr factor(new MDFactorBivariable);
            factor->setVariableId(0, _prev_var->graphId());
            factor->setVariableId(1, var->graphId());
            _graph->addFactor(factor);
          }

          _prev_var = var;
          _graph_id++;
        }
      }
    }
    return is_good;
  }

protected:
  FactorGraphPtr _graph;
  MDVariableSE3Ptr _prev_var;
  Isometry3f _init_T, _offset;
  GTPoses _gt_poses;
  double _max_diff;
  float _max_angle;
  float _max_translation;
  int _graph_id;
};

using MDPyramidGraphGeneratorPtr = std::shared_ptr<MDPyramidGraphGenerator>;
BOSS_REGISTER_CLASS(MDPyramidGraphGenerator)

void generateConfig(ConfigurableManager& manager_, std::string config_filename_) {
  // auto source = manager_.create<MessageROSBagSource>("source");
  // source->param_topics.setValue({"/os/image_depth", "/os/image_intensity", "/os/camera_info"});

  const std::string t_intensity = "/os/image_intensity";
  const std::string t_depth     = "/os/image_depth";
  const std::string t_cam_info  = "/os/camera_info";

  const std::vector<std::string> t_vec = {t_intensity, t_depth, t_cam_info};

  auto pyrgen = manager_.create<MDPyramidGraphGenerator>("pyrgen");
  pyrgen->param_intensity_topic.setValue(t_intensity);
  pyrgen->param_depth_topic.setValue(t_depth);
  pyrgen->param_camera_info_topic.setValue(t_cam_info);

  auto sync = manager_.create<MessageSynchronizedSink>("sync");
  sync->param_topics.setValue(t_vec);
  sync->param_push_sinks.value().emplace_back(pyrgen);

  auto sorter = manager_.create<MessageSortedSink>("sorter");
  sorter->param_push_sinks.value().emplace_back(sync);

  auto runner = manager_.create<PipelineRunner>("runner");
  // runner->param_source.setValue(source);
  runner->param_push_sinks.value().emplace_back(sorter);

  std::cerr
    << "writing template configuration to use this executable [ " << FG_YELLOW(config_filename_)
    << " ] \n ... modify this configuration and pass this with -c flag as input to the program!"
    << std::endl;
  manager_.write(config_filename_);
  std::cerr << "done!\n";
}

int main(int argc, char** argv) {
  srrgInit(argc, argv, "trajrosbagcloud2srrg");

  // clang-format off
  ParseCommandLine cmd(argv, banner);
  ArgumentString trajectory(&cmd, "traj", "trajectory", "path to the traj file < timestamp tx ty tz qx qy qz qw > ", "");
  ArgumentString config_file(&cmd, "c", "config", "config file to load", "");
  ArgumentString output_graph(&cmd, "o", "output", "path to output graph", "");
  ArgumentFloat dr(&cmd, "dr", "delta-rotation", "dr between poses for downsampling [rad], if less then this threshold is ignored - 0.0 gets all the poses from traj file", 0.1);
  ArgumentFloat dt(&cmd, "dt", "delta-translation", "dt between poses downsampling [m], if less then this threshold is ignored - 0.0 gets all the poses from traj file", 0.5);
  ArgumentDouble max_diff(&cmd, "m", "max-diff", "max time difference for an association to be made", 0.02);
  ArgumentFlag is_suma(&cmd, "suma", "suma", "if is suma transform using sensor offset, 180 deg around z-axis", false);
  ArgumentString input_configuration_filename(&cmd, "j", "write-configuration", "if output configuration set, write configuration to process dataset", "data_configuration.json");
  cmd.parse();
  // clang-format on

  ConfigurableManager manager;

  // writes configuration on file (configuration has to be given as input to process data)
  if (input_configuration_filename.isSet()) {
    generateConfig(manager, input_configuration_filename.value());
    return 0;
  }

  if (!trajectory.isSet()) {
    std::cerr << "no input trajectory provided, aborting" << std::endl;
    return -1;
  }

  if (cmd.lastParsedArgs().empty()) {
    std::cerr << std::string(environ[0]) +
                   "|ERROR, input dataset(s) <rosbag> not set correctly, aborting\n";
    return -1;
  }

  if (!output_graph.isSet()) {
    std::cerr << "no output provided, aborting" << std::endl;
    return -1;
  }

  if (!config_file.isSet()) {
    std::cerr << std::string(environ[0]) + "|ERROR, no config file provided, aborting" << std::endl;
    return -1;
  }

  // serializing configuration
  std::cerr << "opening configuration [ " << FG_YELLOW(config_file.value()) << " ]\n";
  manager.read(config_file.value());
  // clang-format off
  auto sync = manager.getByName<MessageSynchronizedSink>("sync");
  if (!sync) { std::cerr << std::string(environ[0]) + "|ERROR, cannot find sync, maybe wrong configuration path" << std::endl; return -1;}
  auto pyrgen = manager.getByName<MDPyramidGraphGenerator>("pyrgen");
  if (!pyrgen) { std::cerr << std::string(environ[0]) + "|ERROR, cannot find pyrgen, maybe wrong configuration path" << std::endl;  return -1;}
  auto sorter = manager.getByName<MessageSortedSink>("sorter");
  if (!sorter) { std::cerr << std::string(environ[0]) + "|ERROR, cannot find sorter, maybe wrong configuration path!" << std::endl; return -1;}
  auto runner = manager.getByName<PipelineRunner>("runner");
  if (!runner) { std::cerr << std::string(environ[0]) + "|ERROR, cannot find runner, maybe wrong configuration path" << std::endl; return -1;}
  // clang-format on

  // creating and setup source for reading
  MessageROSBagSourcePtr source = std::make_shared<MessageROSBagSource>();
  source->param_topics.setValue(sync->param_topics.value());
  runner->param_source.setValue(source);

  // read trajectory
  GTPoses gt_poses;
  readTUMGroundtruth(gt_poses, trajectory.value());

  Isometry3f sensorT = Isometry3f::Identity();
  if (is_suma.isSet()) {
    Eigen::AngleAxisf yaw_rot(M_PI, Vector3f::UnitZ());
    sensorT.linear() = yaw_rot.toRotationMatrix();
  }

  pyrgen->setDownsamplingAngle(dr.value());
  pyrgen->setDownsamplingTranslation(dt.value());
  pyrgen->setTimeMaxDiff(max_diff.value());
  pyrgen->forceSensorOffset(sensorT);
  pyrgen->setPoses(gt_poses);

  for (const std::string& bag_name : cmd.lastParsedArgs()) {
    std::cerr << "processing dataset [ " << FG_GREEN(bag_name) << " ]\n";
    std::cerr << std::endl;
    source->open(bag_name);
    runner->compute();
  }

  // write output graph
  pyrgen->graph()->write(output_graph.value());
  std::cerr << "graph written successfully : " << output_graph.value()
            << " | n variables: " << pyrgen->graph()->variables().size() << std::endl;
}
