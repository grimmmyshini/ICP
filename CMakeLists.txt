cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(iterative_closest_point)

find_package(PCL 1.2 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set(DEBUG_BUILD FALSE)

include_directories(${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} src/include/icp)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(icp icp.cpp include/icp/icp.h)
target_link_libraries (icp ${PCL_LIBRARIES})
