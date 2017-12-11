#!/bin/bash

sed -i s/sm_11/sm_20/g ./build/libcudaHOG/src/libcudaHOG/cudaHOG/cudaHOG.pro
sed -i s#/usr/local/cuda/lib64#/home/fl/citk/systems/pepper-robocup-nightly/lib64#g ./build/libcudaHOG/src/libcudaHOG/cudaHOG/cudaHOG.pro

sed -i s/sm_11/sm_20/g ./build/libcudaHOG/src/libcudaHOG-build/cudaHOG/Makefile
sed -i s#/usr/local/cuda/lib64#/home/fl/citk/systems/pepper-robocup-nightly/lib64#g ./build/libcudaHOG/src/libcudaHOG-build/cudaHOG/Makefile

sed -i s/boost_program_options-mt/boost_program_options/g ./build/libcudaHOG/src/libcudaHOG/src/cudaHOGDump/cudaHOGDump.pro
sed -i s/boost_program_options-mt/boost_program_options/g ./build/libcudaHOG/src/libcudaHOG-build/src/cudaHOGDump/Makefile
sed -i s#/usr/local/cuda/lib64#/home/fl/citk/systems/pepper-robocup-nightly/lib64#g ./build/libcudaHOG/src/libcudaHOG/src/cudaHOGDump/cudaHOGDump.pro

sed -i s/boost_program_options-mt/boost_program_options/g ./build/libcudaHOG/src/libcudaHOG/src/cudaHOGDetect/cudaHOGDetect.pro
sed -i s#/usr/local/cuda/lib64#/home/fl/citk/systems/pepper-robocup-nightly/lib64#g ./build/libcudaHOG/src/libcudaHOG/src/cudaHOGDetect/cudaHOGDetect.pro
sed -i s#/usr/local/cuda/lib64#/home/fl/citk/systems/pepper-robocup-nightly/lib64#g ./build/libcudaHOG/src/libcudaHOG-build/src/cudaHOGDetect/Makefile

# /home/fl/dev/cudaHOG/build/libcudaHOG/src/libcudaHOG-build/src/cudaHOGDetect at qtcore
