cmake_minimum_required(VERSION 2.8.3)
PROJECT(cudahog_ros)

## Find catkin macros and libraries
## if COMPONENTS list like FIND_PACKAGE(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
FIND_PACKAGE(PkgConfig REQUIRED)
FIND_PACKAGE(catkin REQUIRED COMPONENTS roscpp rospy std_msgs sensor_msgs MESSAGE_filters image_transport MESSAGE_generation cv_bridge)
FIND_PACKAGE(OpenCV REQUIRED)

PKG_CHECK_MODULES(CUDAHOG cudaHOG)

FIND_PACKAGE(CUDA)
if (CUDA_FOUND)
  FIND_PACKAGE(CUDA REQUIRED)
  MESSAGE(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
  MESSAGE(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
  set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_20,code=[sm_20,sm_21]
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
  )
else()
  MESSAGE(ERROR "CUDA Libararies NOT found. :|")
endif()

## Include Qt
FIND_PACKAGE(Qt4 REQUIRED COMPONENTS QtCore QtGui)
include(${QT_USE_FILE})

set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O2")
endif()

SET(ROS_BUILD_TYPE Release)
SET(CMAKE_BUILD_TYPE Release)

if(CUDAHOG_FOUND)
    add_definitions(-DWITH_CUDA)
else(CUDAHOG_FOUND)
    MESSAGE(WARNING "libcudaHOG not found: Compiling cudahog_ros without any functionality.
    This is just a convenience function to not break compilation on systems without NVIDIA graphics cards.
    People detection will work without the ground_hog but won't be able to detect people far away from the robot.
     Please install libcudaHOG from the rwth_perception_people/3rd_party directory if possible.")
endif(CUDAHOG_FOUND)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

#######################################
## Declare ROS MESSAGEs and services ##
#######################################

## Generate MESSAGEs in the 'msg' folder
add_MESSAGE_files(
   FILES
   CudaHogDetections.msg
)

## Generate added MESSAGEs and services with any dependencies listed here
generate_MESSAGEs(
   DEPENDENCIES
   std_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${CUDAHOG_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${CUDAHOG_LIBRARY_DIRS} ${CUDA_LIBRARIES})

## Declare a cpp executable
add_executable(${PROJECT_NAME} src/main.cpp)

## Add cmake target dependencies of the executable/library
## as an example, MESSAGE headers may need to be generated before nodes
ADD_DEPENDENCIES(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
ADD_DEPENDENCIES(${PROJECT_NAME} ${PROJECT_NAME}_generate_MESSAGEs_cpp ${catkin_EXPORTED_TARGETS})


## Specify libraries to link a library or executable target against
 target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${QT_LIBRARIES}
   ${QT_QTCORE_LIBRARY}
   ${QT_QTGUI_LIBRARY}
   ${CUDAHOG_LIBRARIES}
   ${OpenCV_LIBS}
   ${CUDA_LIBRARIES}
 )

#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".git" EXCLUDE
)

install(DIRECTORY model/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/model
  PATTERN ".git" EXCLUDE
)
