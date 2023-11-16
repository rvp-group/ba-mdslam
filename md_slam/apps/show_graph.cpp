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

#include <srrg_image/image.h>
#include <srrg_system_utils/shell_colors.h>
#include <srrg_system_utils/system_utils.h>

#include <iostream>
#include <vector>

#include "viewer/md_slam_viewer.h"
#include <md_slam/utils.cuh>
#include <qapplication.h>
#include <srrg_system_utils/parse_command_line.h>
#include <srrg_system_utils/system_utils.h>

#include <unistd.h>

using namespace srrg2_core;
using namespace srrg2_solver;
using namespace md_slam;

const char* banner[] = {"loads a graph with pyramids attached and displays everyhting", 0};

int main(int argc, char** argv) {
  ParseCommandLine cmd_line(argv, banner);
  ArgumentString graph_name(&cmd_line, "i", "input", "graph name to load", "");
  cmd_line.parse();
  if (!graph_name.isSet()) {
    std::cerr << std::string(environ[0]) + "|ERROR, no input provided, aborting" << std::endl;
    return -1;
  }
  FactorGraphPtr graph = FactorGraph::read(graph_name.value());
  if (!graph) {
    std::cerr << std::string(environ[0]) + "|ERROR, unable to load graph, aborting" << std::endl;
    return -1;
  }
  std::cerr << std::string(environ[0]) + "|ERROR, graph loaded, n vars: "
            << graph->variables().size() << " factors:" << graph->factors().size() << std::endl;

  QApplication app(argc, argv);
  // instantiate the viewer
  std::mutex proc_mutex;
  MDViewerPtr viewer(new MDViewer(graph, proc_mutex));
  viewer->setWindowTitle("graph");
  // make the viewer window visible on screen
  viewer->show();
  return app.exec();
}
