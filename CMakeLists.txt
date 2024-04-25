cmake_minimum_required(VERSION 3.2)

# set the project name and version
project(RaptorCodes4GPU LANGUAGES CXX CUDA)

# check cuda
include(CheckLanguage)
check_language(CUDA)

# add code directory
aux_source_directory(./src DIR_SRC)
# add_subdirectory(src/cudaRaptorCodes)

# add library
include_directories(./include)

# add CPU library
add_library(libCpuR10
    ./src/raptor10.c
    ./src/raptor_consts.c
)
# Set link language for libCpuR10
set_target_properties(libCpuR10 PROPERTIES LINKER_LANGUAGE CXX)

# add GPU library
add_library(libCudaR10
    ./src/cuda_raptor_10.cu
)
# Set link language for libCudaR10
set_target_properties(libCudaR10 PROPERTIES LINKER_LANGUAGE CUDA)

# add executable
add_executable(cuTest ${DIR_SRC})

#Link static and dynamic libraries in other directories
target_link_libraries(cuTest libCudaR10 libCpuR10)