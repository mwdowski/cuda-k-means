#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../macros/macros_cuda.cuh"
#include "../csv_reader/csv_columnwise_data.hpp"
#include "kernels.cuh"
#include <vector>
#include <array>
#include <tuple>
#include <utility>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/logical.h>

template <int DIMENSION_COUNT>
class kmeans
{
    float *dev_points_data[DIMENSION_COUNT];
    int *dev_cluster_assignments = nullptr;
    int *dev_changed_assignments = nullptr;
    thrust::device_vector<int> permutation;
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
        cuda_try_or_return(cudaMemset(dev_changed_assignments, 1, rows_count * sizeof(int)));

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
                pos[j + DIMENSION_COUNT * i] = data.data[j][i];
            }
        }

        cuda_try_or_return(cudaMemcpyToSymbol(kernels::centroids, pos.data(), sizeof(float) * DIMENSION_COUNT * clusters_count));

        return cudaDeviceSynchronize();
    }

    cudaError_t assign_nearest_clusters()
    {
        kernels::assign_nearest_cluster_kernel<DIMENSION_COUNT><<<rows_count / kernels::THREADS_PER_BLOCK + 1, kernels::THREADS_PER_BLOCK>>>(
            clusters_count, rows_count, dev_cluster_assignments, dev_changed_assignments);

        return cudaDeviceSynchronize();
    }

    template <std::size_t... I, std::size_t N, typename T, typename U>
    constexpr auto _make_tuple_with_permutation(const U &perm, const T (&arr)[N], std::index_sequence<I...>)
    {
        return thrust::make_tuple(perm, arr[I]...);
    }

    template <std::size_t N, typename T, typename U>
    constexpr auto make_tuple_with_permutation(const U &perm, const T (&arr)[N])
    {
        return _make_tuple_with_permutation(perm, arr, std::make_index_sequence<N>{});
    }

    cudaError_t recalculate_centroids()
    {
        thrust::device_ptr<float> data[DIMENSION_COUNT];
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            data[i] = thrust::device_ptr<float>(dev_points_data[i]);
        }

        thrust::device_ptr<int> clusters(dev_cluster_assignments);
        auto ptr_tuple = make_tuple_with_permutation(permutation.begin(), data);

        auto zipped_data = thrust::make_zip_iterator(ptr_tuple);
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
                float x = sums[j][i];
                float y = counts[j][i];
                averages[j + DIMENSION_COUNT * i] = x / y;
            }
        }

        cuda_try_or_return(cudaMemcpyToSymbol(kernels::centroids, averages.data(), sizeof(float) * DIMENSION_COUNT * clusters_count));

        return cudaDeviceSynchronize();
    }

    bool is_finished()
    {
        thrust::device_ptr<int> a(dev_changed_assignments);
        return thrust::none_of(a, a + rows_count, thrust::identity<int>());
    }

public:
    kmeans(int rows_count, int clusters_count) : rows_count{rows_count}, clusters_count{clusters_count}, permutation{(size_t)rows_count}
    {
        thrust::sequence(permutation.begin(), permutation.end());
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

    cudaError_t compute(int iteration_limit)
    {
        int counter = 0;
        while (!is_finished() && counter < iteration_limit)
        {
            cuda_try_or_return(assign_nearest_clusters());
            cuda_try_or_return(recalculate_centroids());

            counter++;
        }

        cuda_try_or_return(assign_nearest_clusters());

        return cudaDeviceSynchronize();
    }

    cudaError_t get_points_assignments(int *host_cluster_assignments, float *centroids)
    {
        thrust::device_ptr<int> aaaa(dev_cluster_assignments);
        thrust::device_vector<int> sss(rows_count);
        thrust::scatter(aaaa, aaaa + rows_count, permutation.begin(), sss.begin());

        cuda_try_or_return(cudaMemcpy(host_cluster_assignments, sss.data().get(), rows_count * sizeof(int), cudaMemcpyDeviceToHost));

        cuda_try_or_return(cudaMemcpyFromSymbol(centroids, kernels::centroids, DIMENSION_COUNT * clusters_count * sizeof(float)));
        return cudaDeviceSynchronize();
    }
};
