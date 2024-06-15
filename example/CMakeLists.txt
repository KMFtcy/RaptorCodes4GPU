# check cuda
set(CMAKE_VERBOSE_MAKEFILE ON)

include(CheckLanguage)
check_language(CUDA)

# add library
include_directories(../include)

add_executable(cuTest gpu_test.cu)
add_executable(cpuTest cpu_test.cpp)

#Link static and dynamic libraries in other directories
target_link_libraries(cpuTest cpu_r10_static common_static)
target_link_libraries(cuTest cpu_r10_static gpu_r10_static common_static)