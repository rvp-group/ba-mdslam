add_library(md_slam_library SHARED
  # basic stuff
  dual_matrix.cu
  utils.cpp
  pyramid_level.cu
  # pyramid_level.cpp
  image_pyramid.cpp   
  pyramid_level.cpp 
  pyramid_message.cpp 
  pyramid_generator.cpp 
  pyramid_variable_se3.cpp   
  # pairwise alignment stuff
  factor_common.cu
  factor_common.cpp  
  factor.cpp
  factor.cu
  factor_stack.cpp 
  pairwise_aligner.cpp
  tracker_status_message.cpp 
  tracker.cpp 
  graph_manager.cpp 
  tracker_viewer.cpp # TODO to be removed
  # closure stuff 
  frame.cpp
  loop_closer.cpp
  loop_detector_base.cpp
  loop_detector_hbst.cpp
  loop_validator.cpp
  instances.cpp 
  # motion only ba
  factor_bi.cu
  factor_bi.cpp
)

target_link_libraries(md_slam_library
  srrg2_solver_core_library
  srrg2_point_cloud_library
  srrg2_messages_library
  srrg2_data_structures_library
  srrg2_solver_types3d_library
  srrg2_image_library
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

set_target_properties( md_slam_library 
 PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)