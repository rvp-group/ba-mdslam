# dataset manipulat
add_executable(ncd_manipulator
dataset_manipulators/newer_college_dataset_manipulator.cpp)

target_link_libraries(ncd_manipulator
  ${catkin_LIBRARIES}
  srrg2_data_structures_library
  srrg2_point_cloud_library
  srrg2_image_library
  md_slam_library
)

add_executable(eth_dataset_manipulator
dataset_manipulators/eth_dataset_manipulator.cpp)

target_link_libraries(eth_dataset_manipulator
  ${catkin_LIBRARIES}
  srrg2_data_structures_library
  srrg2_point_cloud_library
  srrg2_image_library
  md_slam_library
)

# utils 
add_executable(graph_converter 
  graph_converter.cpp
)

target_link_libraries(graph_converter
  srrg2_data_structures_library
  srrg2_point_cloud_library
  srrg2_image_library
  md_slam_library
  ${catkin_LIBRARIES}
)


add_executable(show_graph show_graph.cpp viewer/vbo_base.cpp viewer/cloud_vbo.cpp viewer/camera_pose_vbo.cpp viewer/factors_binary_vbo.cpp viewer/drawable_factor_graph_vbo.cpp viewer/drawable_pyramid_variable_se3.cpp viewer/md_slam_viewer.cpp)

target_link_libraries(show_graph
  md_slam_library
  srrg2_solver_core_library
  srrg2_point_cloud_library
  srrg2_messages_library
  srrg2_data_structures_library
  srrg2_solver_types3d_library
  srrg2_image_library
  srrg2_system_utils_library
  srrg2_viewer_library
  srrg2_boss_library
  srrg2_property_library
  ${OpenCV_LIBS}
  ${QGLViewer_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${SRRG_QT_LIBRARIES}
  ${CATKIN_libraries}
  -lGL
)


add_executable(slam slam.cpp viewer/vbo_base.cpp viewer/cloud_vbo.cpp viewer/camera_pose_vbo.cpp viewer/factors_binary_vbo.cpp viewer/drawable_factor_graph_vbo.cpp viewer/drawable_pyramid_variable_se3.cpp viewer/md_slam_viewer.cpp)

target_link_libraries(slam
  md_slam_library
  srrg2_solver_core_library
  srrg2_point_cloud_library
  srrg2_messages_library
  srrg2_data_structures_library
  srrg2_solver_types3d_library
  srrg2_image_library
  srrg2_system_utils_library
  srrg2_viewer_library
  srrg2_boss_library
  srrg2_property_library
  srrg2_messages_library
  srrg2_configurable_shell_library
  srrg2_config_library
  srrg2_data_structures_library
  ${OpenCV_LIBS}
  ${QGLViewer_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${SRRG_QT_LIBRARIES}
  ${CATKIN_libraries}
  -lGL
)

add_executable(ba ba.cpp)

target_link_libraries(ba
md_slam_library
srrg2_solver_core_library
srrg2_point_cloud_library
srrg2_messages_library
srrg2_data_structures_library
srrg2_solver_types3d_library
srrg2_image_library
srrg2_viewer_library
srrg2_system_utils_library
srrg2_boss_library
srrg2_property_library
srrg2_messages_library
srrg2_configurable_shell_library
srrg2_config_library
srrg2_data_structures_library
srrg2_solver_linear_solvers_library
srrg2_solver_sparse_block_matrix_library
${OpenCV_LIBS}
${CATKIN_libraries}
)


add_executable(gt_mapper gt_mapper.cpp) #viewer/vbo_base.cpp viewer/cloud_vbo.cpp viewer/camera_pose_vbo.cpp viewer/factors_binary_vbo.cpp viewer/drawable_factor_graph_vbo.cpp viewer/drawable_pyramid_variable_se3.cpp viewer/md_slam_viewer.cpp)

target_link_libraries(gt_mapper
md_slam_library
srrg2_solver_core_library
srrg2_point_cloud_library
srrg2_messages_library
srrg2_data_structures_library
srrg2_solver_types3d_library
srrg2_image_library
srrg2_viewer_library
srrg2_system_utils_library
srrg2_boss_library
srrg2_property_library
srrg2_messages_library
srrg2_configurable_shell_library
srrg2_config_library
srrg2_data_structures_library
srrg2_solver_linear_solvers_library
srrg2_solver_sparse_block_matrix_library
${OpenCV_LIBS}
${QGLViewer_LIBRARIES}
# ${GLUT_LIBRARIES}
# ${SRRG_QT_LIBRARIES}
# ${CATKIN_libraries}
# -lGL
)


add_executable(variable_noise_adder variable_noise_adder.cpp) #viewer/vbo_base.cpp viewer/cloud_vbo.cpp viewer/camera_pose_vbo.cpp viewer/factors_binary_vbo.cpp viewer/drawable_factor_graph_vbo.cpp viewer/drawable_pyramid_variable_se3.cpp viewer/md_slam_viewer.cpp)

target_link_libraries(variable_noise_adder
md_slam_library
srrg2_solver_core_library
srrg2_point_cloud_library
srrg2_messages_library
srrg2_data_structures_library
srrg2_solver_types3d_library
srrg2_image_library
srrg2_viewer_library
srrg2_system_utils_library
srrg2_boss_library
srrg2_property_library
srrg2_messages_library
srrg2_configurable_shell_library
srrg2_config_library
srrg2_data_structures_library
srrg2_solver_linear_solvers_library
srrg2_solver_sparse_block_matrix_library
${OpenCV_LIBS}
${QGLViewer_LIBRARIES}
# ${GLUT_LIBRARIES}
# ${SRRG_QT_LIBRARIES}
# ${CATKIN_libraries}
# -lGL
)





