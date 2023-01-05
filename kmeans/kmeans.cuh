#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../macros/macros.cuh"

template <int DIMENSION_COUNT>
class kmeans
{
    float *dev_data[DIMENSION_COUNT]{nullptr};
    const int rows_number;

    cudaError_t allocate_device_data()
    {
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            cuda_try_or_return(cudaMalloc(&dev_data[i], rows_number * sizeof(float)));
        }

        return cudaDeviceSynchronize();
    }

    cudaError_t free_device_data()
    {
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            cuda_try_or_return(cudaFree(dev_data[i]));
            dev_data[i] = nullptr;
        }

        return cudaDeviceSynchronize();
    }

public:
    kmeans(int rows_number) : rows_number{rows_number}
    {
        cuda_try_or_exit(allocate_device_data());
    }

    ~kmeans()
    {
        cuda_try_or_exit(free_device_data());
    }
};
