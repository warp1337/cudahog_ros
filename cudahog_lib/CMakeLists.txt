cmake_minimum_required(VERSION 2.8.3)
project(3rd_party)

##################################################################################################################
### CUDA HOG ###
message(STATUS "Preparing libcudaHOG")
message(STATUS "Created pkg-config file: ${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG/pkgconfig/cudaHOG.pc")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/pkgconfig/cudaHOG.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/cudaHOG.pc
    @ONLY
)

FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/libcudaHOG_src" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/pkgconfig" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

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

IF(CUDA_HOME)
    MESSAGE("-- YOU SET CUDA_HOME TO: " ${CUDA_HOME})
    MESSAGE("-- PATCHING *.PRO FILES NOW")

    ### Find all *.pro files
    FILE(GLOB_RECURSE PRO_FILES "${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG_src/*.pro")

    FOREACH(item ${PRO_FILES})
        MESSAGE("-- PATCHING *.PRO FILE: " ${item})
        FILE(READ ${item} Contents)
        STRING(REPLACE "/usr/local/cuda/" "${CUDA_HOME}/" replacement "${Contents}")
        FILE(WRITE ${item} ${replacement})
    ENDFOREACH(item)
ENDIF()

include(ExternalProject)
ExternalProject_Add(
    libcudaHOG
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG_src
    CONFIGURE_COMMAND qmake -makefile <SOURCE_DIR>
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make
    BUILD_COMMAND ""
    INSTALL_COMMAND "" #Skipp install step. See below
)

### Install files ###
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG_src/src/libcudaHOG/cudaHOG
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  FILES_MATCHING PATTERN "*.h"
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG_src/src/libcudaHOG-build/cudaHOG/libcudaHOG.so
  ${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG_src/src/libcudaHOG-build/cudaHOG/libcudaHOG.so.1
  ${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG_src/src/libcudaHOG-build/cudaHOG/libcudaHOG.so.1.0
  ${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG_src/src/libcudaHOG-build/cudaHOG/libcudaHOG.so.1.0.0
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
) 

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/cudaHOG.pc
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig/
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG_src/src/libcudaHOG/model/svm_model
  ${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG_src/src/libcudaHOG/model/config
  DESTINATION ${CMAKE_INSTALL_PREFIX}/share/cudahog
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG_src/src/libcudaHOG-build/bin/cudaHOGDump
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/libcudaHOG_src/src/libcudaHOG-build/bin/cudaHOGDetect
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

##################################################################################################################
