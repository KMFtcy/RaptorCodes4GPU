cmake_minimum_required(VERSION 3.2)

# set the project name and version
project(RaptorCodes4GPU LANGUAGES CUDA CXX)

# check cuda
include(CheckLanguage)
check_language(CUDA)

aux_source_directory(./src DIR_SRC)
include_directories(./include)

add_subdirectory(src/cudaRaptorCodes)

add_executable(cuTest ${DIR_SRC})

#Link static and dynamic libraries in other directories
target_link_libraries(cuTest cudaR10)