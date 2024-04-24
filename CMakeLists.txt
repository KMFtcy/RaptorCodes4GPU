cmake_minimum_required(VERSION 3.2)

# set the project name and version
project(RaptorCodes4GPU LANGUAGES CUDA CXX)

# check cuda
include(CheckLanguage)
check_language(CUDA)

# add code directory
aux_source_directory(./src DIR_SRC)
add_subdirectory(src/cudaRaptorCodes)

# add library
include_directories(./include)

# add executable
add_executable(cuTest ${DIR_SRC})

#Link static and dynamic libraries in other directories
target_link_libraries(cuTest cudaR10)