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
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

template <int DIMENSION_COUNT>
class kmeans
{
    float *dev_points_data[DIMENSION_COUNT];
    int *dev_cluster_assignments = nullptr;
    int *dev_changed_assignments = nullptr;
    const int rows_count;
    const int clusters_count;

private:
    cudaError_t
    allocate_device_data()
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

    cudaError_t set_first_points_as_centroids(const csv_columnwise_data<DIMENSION_COUNT> &data)
    {
        std::vector<float> pos(DIMENSION_COUNT * clusters_count);
        for (int i = 0; i < clusters_count; i++)
        {
            for (int j = 0; j < DIMENSION_COUNT; j++)
            {
                pos[j + clusters_count * i] = data.data[j][i];
            }
        }
        cuda_try_or_return(cudaMemcpyToSymbol(
            kernels::centroids,
            pos.data(),
            sizeof(float) * DIMENSION_COUNT * clusters_count,
            0U,
            cudaMemcpyHostToDevice));

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
        // auto zipped_data = helpers::call_function(thrust::make_zip_iterator, data);
        thrust::device_ptr<int> clusters(dev_cluster_assignments);
        auto zipped_data = thrust::make_zip_iterator(data[0], data[1]);

        thrust::sort_by_key(clusters, clusters + rows_count, zipped_data);

        thrust::device_vector<int> new_keys(clusters_count);
        thrust::device_vector<float> sums[DIMENSION_COUNT];
        thrust::device_vector<int> counts[DIMENSION_COUNT];
        std::vector<float> averages(DIMENSION_COUNT * clusters_count);

        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            sums[i] = thrust::device_vector<float>(clusters_count);
            counts[i] = thrust::device_vector<int>(clusters_count);
        }

        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            thrust::reduce_by_key(
                clusters,
                clusters + rows_count,
                data[i],
                new_keys.begin(),
                sums[i].begin(),
                thrust::equal_to<int>(),
                thrust::plus<float>());

            thrust::reduce_by_key(
                clusters,
                clusters + rows_count,
                thrust::make_constant_iterator<int>(1),
                new_keys.begin(),
                counts[i].begin(),
                thrust::equal_to<int>(),
                thrust::plus<int>());
        }

        for (int i = 0; i < clusters_count; i++)
        {
            for (int j = 0; j < DIMENSION_COUNT; j++)
            {
                averages[i * DIMENSION_COUNT + j] = sums[j][i] / counts[j][i];
            }
        }

        cuda_try_or_return(cudaMemcpyToSymbol(
            kernels::centroids,
            averages.data(),
            sizeof(float) * DIMENSION_COUNT * clusters_count,
            0U,
            cudaMemcpyHostToDevice));

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

        cuda_try_or_return(set_first_points_as_centroids(data));

        return cudaDeviceSynchronize();
    }

    cudaError_t compute()
    {
        // TODO: do until no changes in assignments
        for (int i = 0; i < 10; i++)
        {
            cuda_try_or_return(assign_nearest_clusters());
            cuda_try_or_return(recalculate_centroids());
        }

        cuda_try_or_return(assign_nearest_clusters());

        return cudaDeviceSynchronize();
    }

    cudaError_t get_points_assignments(int *host_cluster_assignments)
    {
        cuda_try_or_return(cudaMemcpy(host_cluster_assignments, dev_cluster_assignments, rows_count * sizeof(int), cudaMemcpyDeviceToHost));

        return cudaDeviceSynchronize();
    }
};
