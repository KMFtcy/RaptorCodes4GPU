# add library
include_directories(../../include)

add_library(common_static gf2matrix.cpp)

set_target_properties(common_static PROPERTIES LINKER_LANGUAGE CXX)