cmake_minimum_required(VERSION 2.8.3)
PROJECT(cudahog_ros)

## Find catkin macros and libraries
## if COMPONENTS list like FIND_PACKAGE(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
FIND_PACKAGE(PkgConfig REQUIRED)
FIND_PACKAGE(catkin REQUIRED COMPONENTS roscpp rospy std_msgs sensor_msgs message_filters image_transport message_generation cv_bridge)
FIND_PACKAGE(OpenCV REQUIRED)

PKG_CHECK_MODULES(CUDAHOG cudaHOG)

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
    MESSAGE(FATAL_ERROR "libcudaHOG not found. Please build this library first.")
endif(CUDAHOG_FOUND)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

#######################################
## Declare ROS MESSAGEs and services ##
#######################################

## Generate MESSAGEs in the 'msg' folder
add_message_files(
   FILES
   CudaHogDetections.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
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
ADD_DEPENDENCIES(${PROJECT_NAME} ${PROJECT_NAME}_generate_messages_cpp ${catkin_EXPORTED_TARGETS})


## Specify libraries to link a library or executable target against
 target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${QT_LIBRARIES}
   ${QT_QTCORE_LIBRARY}
   ${QT_QTGUI_LIBRARY}
   ${CUDAHOG_LIBRARIES}
   ${OpenCV_LIBS}
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
