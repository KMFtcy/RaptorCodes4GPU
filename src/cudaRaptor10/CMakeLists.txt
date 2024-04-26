# check cuda
include(CheckLanguage)
check_language(CUDA)

# add library
include_directories(../../include)

add_library(gpu_r10_static cuda_raptor_10.cu ../raptorConsts/raptor_consts.cpp)

set_target_properties(gpu_r10_static PROPERTIES LINKER_LANGUAGE CUDA)
