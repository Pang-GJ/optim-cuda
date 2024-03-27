file(GLOB_RECURSE SOURCES *.cpp *.cu)
add_library(common ${SOURCES})
target_include_directories(common PUBLIC ${CMAKE_SOURCE_DIR})