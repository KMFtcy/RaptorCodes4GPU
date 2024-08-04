# check cuda
include(CheckLanguage)
check_language(CUDA)

# add library
include_directories(../../include)

# Set CUDA RDC (Relocatable Device Code) to true
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --relocatable-device-code=true")

add_library(gpu_r10_static cuda_raptor_10.cu)

# Ensure RDC is enabled for the target
set_target_properties(gpu_r10_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(gpu_r10_static PROPERTIES LINKER_LANGUAGE CUDA)
