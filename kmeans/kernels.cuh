#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <limits>
#include <algorithm>

namespace kernels
{
    const int THREADS_PER_BLOCK = 128;
    __device__ const int CUDA_CONSTANT_MEMORY_SIZE = 65536;
    __device__ const int CENTROIDS_ARRAY_SIZE = CUDA_CONSTANT_MEMORY_SIZE / sizeof(int);

    __constant__ __device__ float centroids[CENTROIDS_ARRAY_SIZE];

    template <int DIMENSIONS_COUNT>
    __device__ float distance_squared(float x[DIMENSIONS_COUNT], float y[DIMENSIONS_COUNT])
    {
        float result = 0.0f;

        for (int i = 0; i < DIMENSIONS_COUNT; i++)
        {
            result += x[i] * x[i] + y[i] * y[i];
        }

        return result;
    }

    template <int DIMENSIONS_COUNT>
    __global__ void assign_nearest_cluster_kernel(float *points_data[DIMENSIONS_COUNT], int clusters_count, int rows_count, int *cluster_assignments, int *changed_assignments)
    {
        float point[DIMENSIONS_COUNT];
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < rows_count)
        {
            // get point data
            for (int i = 0; i < DIMENSIONS_COUNT; i++)
            {
                point[i] = points_data[DIMENSIONS_COUNT][index];
            }

            // find closest cluster center
            float min_distance_squared = distance_squared(centroids, point);
            int min_cluster = 0;

            for (int k = 1; k < clusters_count; k++)
            {
                float current_distance_squared = distance_squared(centroids + k * DIMENSIONS_COUNT, point);
                if (current_distance_squared < min_distance_squared)
                {
                    min_distance_squared = current_distance_squared;
                    min_cluster = k;
                }
            }

            int old_assignment = cluster_assignments[index];
            cluster_assignments[index] = min_cluster;
            changed_assignments[index] = min_cluster != old_assignment;
        }
    }
}