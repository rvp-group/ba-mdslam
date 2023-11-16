catkin_add_gtest(test_factor test_factor.cpp)

target_link_libraries(test_factor
  ${catkin_LIBRARIES}
  srrg2_data_structures_library
  srrg2_point_cloud_library
  srrg2_image_library
  md_slam_library
  ${SRRG_QT_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${QGLViewer_LIBRARIES}
  pthread
)

catkin_add_gtest(test_factor_bi test_factor_bi.cpp)

target_link_libraries(test_factor_bi
  ${catkin_LIBRARIES}
  srrg2_data_structures_library
  srrg2_point_cloud_library
  srrg2_image_library
  md_slam_library
  ${SRRG_QT_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${QGLViewer_LIBRARIES}
  pthread
)

catkin_add_gtest(test_dual_buffer test_dual_buffer.cu) 

target_link_libraries(test_dual_buffer
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  md_slam_library
)

catkin_add_gtest(test_sum_reduction test_sum_reduction.cu) 

target_link_libraries(test_sum_reduction
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  md_slam_library
)


catkin_add_gtest(test_projection test_projection.cu) 

target_link_libraries(test_projection
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  md_slam_library
)
