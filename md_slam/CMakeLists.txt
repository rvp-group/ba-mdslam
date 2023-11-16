cmake_minimum_required(VERSION 2.8.3)
project(md_slam LANGUAGES CXX CUDA)

find_package(catkin REQUIRED COMPONENTS
  srrg_cmake_modules
  srrg_hbst
  srrg2_core
  srrg2_core_ros
  srrg2_solver
)

set(SRRG_ENABLE_CUDA ON)
  
include(${srrg_cmake_modules_INCLUDE_DIRS}/CMakeCompileOptions.txt)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=pedantic -fopenmp")
message("CMAKE FLAGS SET TO ${CMAKE_CXX_FLAGS}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(QGLViewer REQUIRED)
find_package(OpenCV REQUIRED)
message(STATUS "${PROJECT_NAME}: OpenCV version ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")


# CUDA stuff, automatically detect arch
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
set(CUDA_ARCH "${ARCH_FLAGS}" CACHE STRING "Value of the NVCC -arch option." FORCE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -lineinfo")
message("CUDA ARCH FLAGS: ${ARCH_FLAGS}")
message("CUDA FLAGS: ${CUDA_NVCC_FLAGS}")
message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")

# hbst stuff
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSRRG_MERGE_DESCRIPTORS -DSRRG_HBST_HAS_OPENCV")

# optional
# PCL only required read or dump data in universal way
find_package(PCL REQUIRED)
if(PCL_FOUND)
message("PCL FOUND: building apps to dump point clouds in pcd format")
endif()

catkin_package(
  INCLUDE_DIRS
    src
  LIBRARIES
    md_slam
  CATKIN_DEPENDS
    srrg_cmake_modules
    srrg_hbst
    srrg2_core
    srrg2_core_ros
    srrg2_solver
  DEPENDS
    SRRG_QT
    OpenCV
    QGLViewer
)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${QGLVIEWER_INCLUDE_DIR}
  ${SRRG_QT_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS} # only required for some dumping stuff
  ${PROJECT_SOURCE_DIR}/src
)

#ia help catkin build
link_directories(${catkin_LIBRARY_DIRS})

# path to damn dynamic loader
set(DL_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/../")
add_definitions(-DMD_DL_FOLDER="${DL_FOLDER}")
message("${PROJECT_NAME}|dl_dir dir = ${DL_FOLDER}")

# srrg preprocessor definition used for tests (and apps)
set(TEST_DATA_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/../test_data/")
add_definitions(-DMD_TEST_DATA_FOLDER="${TEST_DATA_FOLDER}")
message("${PROJECT_NAME}|test_data dir = ${TEST_DATA_FOLDER}")

set(CONFIG_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/../configurations/")
add_definitions(-DMD_CONFIG_FOLDER="${CONFIG_FOLDER}")
message("${PROJECT_NAME}|configurations dir = ${CONFIG_FOLDER}")

# sources folder
add_subdirectory(src)
add_subdirectory(apps)

# unittest folder
add_subdirectory(tests)
