#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../macros/macros.cuh"
#include "kernels.cuh"

template <int DIMENSION_COUNT>
class kmeans
{
    float *dev_points_data[DIMENSION_COUNT]{nullptr};
    int *dev_cluster_assignments = nullptr;
    int *dev_changed_assignment = nullptr;
    const int rows_count;
    const int clusters_count;

private:
    cudaError_t allocate_device_data()
    {
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            cuda_try_or_return(cudaMalloc(&dev_points_data[i], rows_count * sizeof(float)));
        }

        cuda_try_or_return(cudaMalloc(&dev_cluster_assignments, rows_count * sizeof(int)));

        return cudaDeviceSynchronize();
    }

    cudaError_t free_device_data()
    {
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            cuda_try_or_return(cudaFree(dev_points_data[i]));
            dev_points_data[i] = nullptr;
        }

        cuda_try_or_return(cudaFree(dev_cluster_assignments));
        dev_cluster_assignments = nullptr;

        return cudaDeviceSynchronize();
    }

    cudaError_t set_first_points_as_centroids()
    {
        size_t current_offset = 0U;

        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            cuda_try_or_return(cudaMemcpyToSymbol(
                kernels::centroids,
                dev_points_data[i],
                clusters_count * sizeof(float),
                current_offset));
            current_offset += clusters_count * sizeof(float);
        }

        return cudaDeviceSynchronize();
    }

    cudaError_t assign_nearest_clusters()
    {
        kernels::assign_nearest_cluster_kernel<DIMENSION_COUNT><<<rows_count / kernels::THREADS_PER_BLOCK + 1, kernels::THREADS_PER_BLOCK>>>(
            dev_points_data, clusters_count, rows_count, dev_cluster_assignments, dev_changed_assignments);

        return cudaDeviceSynchronize();
    }

public:
    kmeans(int rows_count, int clusters_count) : rows_count{rows_count}, clusters_count{clusters_count}
    {
        cuda_try_or_exit(allocate_device_data());
    }

    ~kmeans()
    {
        cuda_try_or_exit(free_device_data());
    }

    cudaError_t load_points_data(float *host_points_data[DIMENSION_COUNT])
    {
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            cuda_try_or_return(cudaMemcpy(dev_points_data[i], host_points_data[i], rows_count, cudaMemcpyHostToDevice));
            dev_points_data[i] = nullptr;
        }

        return cudaDeviceSynchronize();
    }

    cudaError_t compute()
    {

        return cudaDeviceSynchronize();
    }

    cudaError_t get_points_assignments(int *host_cluster_assignments)
    {
        cuda_try_or_return(cudaMemcpy(host_cluster_assignments, dev_cluster_assignments, rows_count, cudaMemcpyDeviceToHost));

        return cudaDeviceSynchronize();
    }
};
