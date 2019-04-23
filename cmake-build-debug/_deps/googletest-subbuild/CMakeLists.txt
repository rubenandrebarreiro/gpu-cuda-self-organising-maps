# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.13.2)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(googletest-populate NONE)

include(ExternalProject)
ExternalProject_Add(googletest-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://github.com/google/googletest" "GIT_TAG" "origin/master"
                    SOURCE_DIR          "/home/rubenandrebarreiro/Desktop/Projects/Repositories/gpu-self-organising-maps/cmake-build-debug/_deps/googletest-src"
                    BINARY_DIR          "/home/rubenandrebarreiro/Desktop/Projects/Repositories/gpu-self-organising-maps/cmake-build-debug/_deps/googletest-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
