#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../macros/macros_cuda.cuh"
#include "../csv_reader/csv_columnwise_data.hpp"
#include "kernels.cuh"
#include <vector>
#include "../helpers/call_function.hpp"
#include <array>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

template <int DIMENSION_COUNT>
class kmeans
{
    float *dev_points_data[DIMENSION_COUNT];
    int *dev_cluster_assignments = nullptr;
    int *dev_changed_assignments = nullptr;
    const int rows_count;
    const int clusters_count;

private:
    cudaError_t allocate_device_data()
    {
        size_t offset = 0U;
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            cuda_try_or_return(cudaMalloc(&dev_points_data[i], rows_count * sizeof(float)));
            cuda_try_or_exit(cudaMemcpyToSymbol(
                kernels::dev_points,
                &dev_points_data[i],
                sizeof(float *),
                offset,
                cudaMemcpyHostToDevice));
            offset += sizeof(float *);
        }

        cuda_try_or_return(cudaMalloc(&dev_cluster_assignments, rows_count * sizeof(int)));
        cuda_try_or_return(cudaMalloc(&dev_changed_assignments, rows_count * sizeof(int)));

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

        cuda_try_or_return(cudaFree(dev_changed_assignments));
        dev_changed_assignments = nullptr;

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
                sizeof(float),
                current_offset,
                cudaMemcpyDeviceToDevice));
            current_offset += clusters_count * sizeof(float);
        }

        return cudaDeviceSynchronize();
    }

    cudaError_t assign_nearest_clusters()
    {
        kernels::assign_nearest_cluster_kernel<DIMENSION_COUNT><<<rows_count / kernels::THREADS_PER_BLOCK + 1, kernels::THREADS_PER_BLOCK>>>(
            clusters_count, rows_count, dev_cluster_assignments, dev_changed_assignments);

        return cudaDeviceSynchronize();
    }

    cudaError_t recalculate_centroids()
    {
        thrust::device_ptr<float> data[DIMENSION_COUNT];
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            data[i] = thrust::device_ptr<float>(dev_points_data[i]);
        }

        // auto as_std_array = std::array<thrust::device_ptr<float>, DIMENSION_COUNT>();

        // TODO: make_zip_iterator with arguments from array
        auto zipped_data = helpers::call_function(thrust::make_zip_iterator, data);

        thrust::device_ptr<int> clusters(dev_cluster_assignments);

        thrust::sort_by_key(clusters, clusters + rows_count, zipped_data);

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

    cudaError_t load_points_data(const csv_columnwise_data<DIMENSION_COUNT> &data)
    {
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            cuda_try_or_return(cudaMemcpy(dev_points_data[i], data.data[i].data(), rows_count * sizeof(float), cudaMemcpyHostToDevice));
        }

        return cudaDeviceSynchronize();
    }

    cudaError_t compute()
    {
        cuda_try_or_return(set_first_points_as_centroids());
        cuda_try_or_return(assign_nearest_clusters());
        cuda_try_or_return(recalculate_centroids());

        return cudaDeviceSynchronize();
    }

    cudaError_t get_points_assignments(int *host_cluster_assignments)
    {
        cuda_try_or_return(cudaMemcpy(host_cluster_assignments, dev_cluster_assignments, rows_count * sizeof(float), cudaMemcpyDeviceToHost));

        return cudaDeviceSynchronize();
    }
};
