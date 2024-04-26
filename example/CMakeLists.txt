# check cuda
include(CheckLanguage)
check_language(CUDA)

# add_subdirectory(../src/cpuRaptor10)
# add_subdirectory(../src/cudaRaptor10)
# add_subdirectory(../src/raptorConsts)

# add library
include_directories(../include)

add_executable(cuTest test.cpp)

#Link static and dynamic libraries in other directories
target_link_libraries(cuTest gpu_r10_static)