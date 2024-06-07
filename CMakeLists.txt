cmake_minimum_required(VERSION 3.2)

# set the project name and version
project(RaptorCodes4GPU LANGUAGES CXX CUDA)

# check cuda
include(CheckLanguage)
check_language(CUDA)

add_subdirectory(src/cpuRaptor10)
add_subdirectory(src/common)
add_subdirectory(src/cudaRaptor10)
add_subdirectory(example)