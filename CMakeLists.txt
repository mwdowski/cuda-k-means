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

# options
add_library(options options/options.hpp options/options.cpp)

# macros
add_library(macros macros/macros.cuh)
set_target_properties(macros PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(macros PROPERTIES CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCHITECTURE})
target_link_libraries(macros ${CUDA_RUNTIME_LIBRARY})

# kmeans_kernels
add_library(kmeans_kernels kmeans/kmeans_kernels.cuh)
target_link_libraries(kmeans_kernels ${CUDA_RUNTIME_LIBRARY})
set_target_properties(kmeans_kernels PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(kmeans_kernels PROPERTIES CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCHITECTURE})

# kmeans
add_library(kmeans kmeans/kmeans.cuh)
target_include_directories(kmeans PUBLIC ${CUDA_INCLUDE_DIRS} macros kmeans_kernels)
target_link_libraries(kmeans ${CUDA_RUNTIME_LIBRARY})
set_target_properties(kmeans PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(kmeans PROPERTIES CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCHITECTURE})

# main
add_executable(main_gpu gpu/main.cu)
set_target_properties(main_gpu PROPERTIES CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCHITECTURE})
target_link_libraries(main_gpu ${CUDA_RUNTIME_LIBRARY} options)