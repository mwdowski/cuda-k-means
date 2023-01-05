cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(cuda-k-means
        VERSION 1.0
        LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CUDA_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
#set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(DEFAULT_CUDA_ARCHITECTURE "72")

add_executable(main_gpu gpu/main.cu)
set_target_properties(main_gpu PROPERTIES CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCHITECTURE})
target_link_libraries(main_gpu ${CUDA_RUNTIME_LIBRARY})