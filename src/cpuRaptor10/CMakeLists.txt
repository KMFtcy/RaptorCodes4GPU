# add library
include_directories(../../include)

add_library(cpu_r10_static raptor10.cpp)

set_target_properties(cpu_r10_static PROPERTIES LINKER_LANGUAGE CXX)