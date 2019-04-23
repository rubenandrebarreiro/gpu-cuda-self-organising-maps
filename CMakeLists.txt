cmake_minimum_required(VERSION 3.12)

set (ENV{CUDA_TOOLKIT_ROOT_DIR} "/usr/local/cuda")
set (CMAKE_CUDA_COMPILER $ENV{CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

project(gpu_self_organising_maps LANGUAGES CXX CUDA)

########### Cmake config

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 14)

########### CUDA

find_package(CUDA REQUIRED)

############## Add labs

include_directories(include)
enable_testing()

add_executable(gpu-self-organising-maps src/cpu_som_sequential.cpp)
# target_link_libraries(som_cuda ${GTEST_BOTH_LIBRARIES}) # link with google test if you want