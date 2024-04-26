# add library
include_directories(../../include)

add_library(r10_consts_static raptor_consts.cpp)

set_target_properties(r10_consts_static PROPERTIES LINKER_LANGUAGE CXX)