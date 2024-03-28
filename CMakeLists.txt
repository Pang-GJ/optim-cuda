cmake_minimum_required(VERSION 3.10)
project(optim-CUDA LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_FLAGS "-O2 -fopenmp -arch=compute_89 -code=sm_89")

# custom function to add CUDA executable
function(add_cuda_executable EXECUTABLE_NAME SOURCE_FILE)
    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})
    target_include_directories(${EXECUTABLE_NAME} PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(${EXECUTABLE_NAME} common nvToolsExt cublas)
endfunction()

add_subdirectory(common)
add_subdirectory(reduce)
add_subdirectory(sgemm)