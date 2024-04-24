cmake_minimum_required(VERSION 3.2)

# set the project name and version
project(RaptorCodes4GPU)

# specify the C++ standard
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)


aux_source_directory(./src DIR_SRC)
include_directories(./include)

add_executable(test ${DIR_SRC})