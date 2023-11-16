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

#define GL_GLEXT_PROTOTYPES 1
#include <bits/stdc++.h>
#include <srrg_data_structures/matrix.h>
#include <srrg_pcl/point_types.h>
#include <unistd.h>

#include <srrg_image/image.h>
#include <srrg_system_utils/shell_colors.h>
#include <srrg_system_utils/system_utils.h>

#include <iostream>
#include <vector>

#include "viewer/md_slam_viewer.h"
#include <md_slam/graph_manager.h>
#include <md_slam/tracker.h>
#include <md_slam/utils.cuh>
#include <qapplication.h>
#include <srrg_config/configurable_manager.h>
#include <srrg_config/pipeline_runner.h>
#include <srrg_messages/message_handlers/message_file_source.h>
#include <srrg_messages/message_handlers/message_sorted_sink.h>
#include <srrg_solver/solver_core/solver.h>
#include <srrg_system_utils/parse_command_line.h>
#include <srrg_system_utils/system_utils.h>
#include <thread>

#ifndef MD_DL_FOLDER
#error "NO DL FOLDER"
#endif

using namespace srrg2_core;
using namespace srrg2_solver;
using namespace md_slam;

const char* banner[] = {"run md slam", 0};

ConfigurableManager manager;
std::shared_ptr<PipelineRunner> runner;

void computeThread() {
  runner->compute();
}

int main(int argc, char** argv) {
  srrgInit(argc, argv, "slam");
  // clang-format off
  ParseCommandLine cmd_line(argv, banner);
  ArgumentString config_file(&cmd_line, "c", "config", "config file to load", "");
  ArgumentString bag_file(&cmd_line, "i", "input", "input bag file to start", "");
  ArgumentString output_graph(&cmd_line, "o", "ouput", "output filename for graph and pyramid serialization", "");
  ArgumentFlag visualize(&cmd_line, "e", "enable-viewer", "if set enables viewer, otherwise just runs", false);
  ArgumentFlag perspective_view(&cmd_line, "p", "perspective", "if set enables perspective view, i.e. gl camera follows sensor", false);
  ArgumentFlag verbose(&cmd_line, "v", "verbose", "if set enables cerr and cout streams", false);
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

  if (!config_file.isSet()) {
    std::cerr << std::string(environ[0]) + "|ERROR, no config file provided, aborting" << std::endl;
    return -1;
  }

  if (!bag_file.isSet()) {
    std::cerr << std::string(environ[0]) + "|ERROR, no bag (dataset) file provided, aborting"
              << std::endl;
    return -1;
  }

  // load libraries from dl.conf inside md_slam
  const std::string dl_folder_path = MD_DL_FOLDER;
  const std::string dl_stub_path   = crawlForFile("dl.conf", dl_folder_path);
  ConfigurableManager::initFactory(dl_stub_path);
  manager.read(config_file.value());

  // retrieve a runner
  runner = manager.getByName<PipelineRunner>("runner");
  if (!runner) {
    std::cerr << std::string(environ[0]) +
                   "|ERROR, cannot find runner, maybe wrong configuration path!"
              << std::endl;
  }

  // retrieve the source from the runner
  auto source = dynamic_pointer_cast<MessageFileSourceBase>(runner->param_source.value());
  if (!source) {
    std::cerr << std::string(environ[0]) +
                   "|ERROR, cannot find source, maybe wrong configuration path!"
              << std::endl;
  }

  // retrieve the factor graph from the manager
  auto graph_manager = manager.getByName<md_slam::MDGraphManager>("graph_manager");
  if (!graph_manager) {
    std::cerr << std::string(environ[0]) +
                   "|ERROR, cannot find graph_manager, maybe wrong configuration path!"
              << std::endl;
  }

  // retrieve the tracker from the manager
  auto tracker = manager.getByName<md_slam::MDTrackerStandalone>("tracker");
  if (!tracker) {
    std::cerr << std::string(environ[0]) +
                   "|ERROR, cannot find tracker, maybe wrong configuration path!"
              << std::endl;
  }

  auto sink = manager.getByName<MessageSortedSink>("sink");
  if (!sink) {
    std::cerr << std::string(environ[0]) +
                   "|ERROR, cannot find sink, maybe wrong configuration path!"
              << std::endl;
  }

  // getting ptr to graph for eventual viz
  FactorGraphPtr graph = graph_manager->graph();

  std::cerr << "running md slam ... " << std::endl;

  // enable printing in console
  if (!verbose.isSet()) {
    std::cout.rdbuf(NULL);
    std::cerr.rdbuf(NULL);
  }

  source->open(bag_file.value());
  if (!visualize.isSet()) {
    // run without viewing
    runner->compute();
  } else {
    std::thread compute_thread(computeThread);
    // viewer on main thread
    QApplication app(argc, argv);
    // instantiate the viewer
    MDViewerPtr viewer(new MDViewer(graph, graph_manager->graphMutex()));
    viewer->setWindowTitle("md_slam viewer");
    // make the viewer window visible on screen
    viewer->show();
    Eigen::Isometry3f sensor_pose_in_world = Eigen::Isometry3f::Identity();
    while (1) {
      // if input source if completed
      if (sink->isFlushed()) {
        break;
      }
      // this enable perspective view
      if (!graph->variables().empty() && perspective_view.isSet()) {
        MDVariableSE3* X =
          static_cast<MDVariableSE3*>(graph->variable(graph->variables().lastKey()));
        sensor_pose_in_world = X->estimate(); // increment in the keyframe * tracker->_local_t;
      }
      viewer->setCamera(sensor_pose_in_world);
      viewer->update();
      app.processEvents();
      usleep(30000);
    }
    compute_thread.join();
  }

  // write output graph
  if (output_graph.isSet()) {
    graph->write(output_graph.value());
    std::cerr << "graph written successfully : " << output_graph.value()
              << " | n variables: " << graph->variables().size()
              << " | n factors: " << graph->factors().size() << std::endl;
  }

  // safely clear srrg-ros stuff
  manager.erase(runner);
  manager.erase(source);
  runner.reset();
  source.reset();
}
