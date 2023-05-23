# dataset manipulat
add_executable(ncd_manipulator
  dataset_manipulators/newer_college_dataset_manipulator.cpp)

target_link_libraries(ncd_manipulator
  ${catkin_LIBRARIES}
  srrg2_data_structures_library
  srrg2_messages_library
  srrg2_point_cloud_library
  srrg2_image_library
  md_slam_library
  ${CUDA_LIBRARIES}
)

add_executable(eth_dataset_manipulator
  dataset_manipulators/eth_dataset_manipulator.cpp)

target_link_libraries(eth_dataset_manipulator
  ${catkin_LIBRARIES}
  srrg2_data_structures_library
  srrg2_point_cloud_library
  srrg2_image_library
  md_slam_library
  ${CUDA_LIBRARIES}
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
  ${CUDA_LIBRARIES}
)

add_executable(show_graph show_graph.cpp viewer/vbo_base.cpp viewer/cloud_vbo.cpp viewer/camera_pose_vbo.cpp viewer/factors_binary_vbo.cpp viewer/drawable_factor_graph_vbo.cpp viewer/drawable_pyramid_variable_se3.cpp viewer/md_slam_viewer.cpp viewer/spline.cpp)

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
  ${CUDA_LIBRARIES}
  -lGL
)

add_executable(slam slam.cpp viewer/vbo_base.cpp viewer/cloud_vbo.cpp viewer/camera_pose_vbo.cpp viewer/factors_binary_vbo.cpp viewer/drawable_factor_graph_vbo.cpp viewer/drawable_pyramid_variable_se3.cpp viewer/md_slam_viewer.cpp viewer/spline.cpp)

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
  ${CUDA_LIBRARIES}
  -lGL
)

# motion only BA
add_executable(ba ba.cpp viewer/vbo_base.cpp viewer/cloud_vbo.cpp viewer/camera_pose_vbo.cpp viewer/factors_binary_vbo.cpp viewer/drawable_factor_graph_vbo.cpp viewer/drawable_pyramid_variable_se3.cpp viewer/md_slam_viewer.cpp viewer/spline.cpp)

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
  ${GLUT_LIBRARIES}
  ${QGLViewer_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${SRRG_QT_LIBRARIES}
  -lGL
)

add_executable(gt_mapper utils/gt_mapper.cpp) # viewer/vbo_base.cpp viewer/cloud_vbo.cpp viewer/camera_pose_vbo.cpp viewer/factors_binary_vbo.cpp viewer/drawable_factor_graph_vbo.cpp viewer/drawable_pyramid_variable_se3.cpp viewer/md_slam_viewer.cpp)

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
  ${CUDA_LIBRARIES}

  # ${GLUT_LIBRARIES}
  # ${SRRG_QT_LIBRARIES}
  # ${CATKIN_libraries}
  # -lGL
)

add_executable(variable_noise_adder utils/variable_noise_adder.cpp) # viewer/vbo_base.cpp viewer/cloud_vbo.cpp viewer/camera_pose_vbo.cpp viewer/factors_binary_vbo.cpp viewer/drawable_factor_graph_vbo.cpp viewer/drawable_pyramid_variable_se3.cpp viewer/md_slam_viewer.cpp)

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
  ${CUDA_LIBRARIES}

  # ${GLUT_LIBRARIES}
  # ${SRRG_QT_LIBRARIES}
  # ${CATKIN_libraries}
  # -lGL
)

if(PCL_FOUND)
  add_executable(srrg2pcl utils/srrg2pcl.cpp)
  target_link_libraries(srrg2pcl
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
    ${PCL_LIBRARIES}
  )
endif()

add_executable(trajrosbagcloud2srrg utils/trajrosbagcloud2srrg.cpp)
target_link_libraries(trajrosbagcloud2srrg
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

  # ${PCL_LIBRARIES}
  ${catkin_LIBRARIES}
)

if(PCL_FOUND)
  add_executable(colorpointcloud utils/colorpointcloud.cpp)
  target_link_libraries(colorpointcloud
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
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
  )
endif()

if(PCL_FOUND)
  add_executable(srrg2pclfull utils/srrg2pclfull.cpp)
  target_link_libraries(srrg2pclfull
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
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
  )
endif()

if(PCL_FOUND)
  add_executable(pcldistancemaptraj utils/pcldistancemaptraj.cpp)
  target_link_libraries(pcldistancemaptraj
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
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
  )
endif()
