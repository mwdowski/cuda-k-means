#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../macros/macros_cuda.cuh"
#include "../csv_reader/csv_columnwise_data.hpp"
#include "kernels.cuh"
#include "kmeans.cuh"
#include <vector>
#include <array>
#include <tuple>
#include <utility>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
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
class gpu_kmeans : public kmeans<DIMENSION_COUNT>
{
    typedef kmeans<DIMENSION_COUNT> base;

protected:
    float *dev_points_data[DIMENSION_COUNT];
    int *dev_cluster_assignments = nullptr;
    int *dev_changed_assignments = nullptr;
    thrust::device_vector<int> permutation;

private:
    cudaError_t allocate_device_data()
    {
        size_t offset = 0U;
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            cuda_try_or_return(cudaMalloc(&dev_points_data[i], base::rows_count * sizeof(float)));
            cuda_try_or_exit(cudaMemcpyToSymbol(
                kernels::dev_points,
                &dev_points_data[i],
                sizeof(float *),
                offset,
                cudaMemcpyHostToDevice));
            offset += sizeof(float *);
        }

        cuda_try_or_return(cudaMalloc(&dev_cluster_assignments, base::rows_count * sizeof(int)));
        cuda_try_or_return(cudaMalloc(&dev_changed_assignments, base::rows_count * sizeof(int)));
        cuda_try_or_return(cudaMemset(dev_changed_assignments, 1, base::rows_count * sizeof(int)));

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
        std::vector<float> pos(DIMENSION_COUNT * base::clusters_count);

        for (int i = 0; i < base::clusters_count; i++)
        {
            for (int j = 0; j < DIMENSION_COUNT; j++)
            {
                pos[j + DIMENSION_COUNT * i] = data.data[j][i];
            }
        }

        cuda_try_or_return(cudaMemcpyToSymbol(kernels::centroids, pos.data(), sizeof(float) * DIMENSION_COUNT * base::clusters_count));

        return cudaDeviceSynchronize();
    }

    cudaError_t assign_nearest_clusters()
    {
        kernels::assign_nearest_cluster_kernel<DIMENSION_COUNT><<<base::rows_count / kernels::THREADS_PER_BLOCK + 1, kernels::THREADS_PER_BLOCK>>>(
            base::clusters_count, base::rows_count, dev_cluster_assignments, dev_changed_assignments);

        return cudaDeviceSynchronize();
    }

    bool is_finished()
    {
        thrust::device_ptr<int> a(dev_changed_assignments);
        return thrust::none_of(a, a + base::rows_count, thrust::identity<int>());
    }

protected:
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

    virtual cudaError_t recalculate_centroids()
    {
        thrust::device_ptr<float> data[DIMENSION_COUNT];
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            data[i] = thrust::device_ptr<float>(dev_points_data[i]);
        }

        thrust::device_ptr<int> clusters(dev_cluster_assignments);
        auto ptr_tuple = make_tuple_with_permutation(permutation.begin(), data);

        auto zipped_data = thrust::make_zip_iterator(ptr_tuple);
        thrust::sort_by_key(clusters, clusters + base::rows_count, zipped_data);

        thrust::device_vector<int> new_keys(base::clusters_count);
        thrust::device_vector<float> sums[DIMENSION_COUNT];
        thrust::device_vector<int> counts(base::clusters_count);
        std::vector<float> averages(DIMENSION_COUNT * base::clusters_count);

        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            sums[i] = thrust::device_vector<float>(base::clusters_count);
        }

        thrust::reduce_by_key(
            clusters,
            clusters + base::rows_count,
            thrust::make_constant_iterator<int>(1),
            new_keys.begin(),
            counts.begin(),
            thrust::equal_to<int>(),
            thrust::plus<int>());

        thrust::host_vector<int> h_counts(counts);

        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            thrust::reduce_by_key(
                clusters,
                clusters + base::rows_count,
                data[i],
                new_keys.begin(),
                sums[i].begin(),
                thrust::equal_to<int>(),
                thrust::plus<float>());
        }

        for (int i = 0; i < base::clusters_count; i++)
        {
            float y = h_counts[i];
            for (int j = 0; j < DIMENSION_COUNT; j++)
            {
                float x = sums[j][i];
                averages[j + DIMENSION_COUNT * i] = x / y;
            }
        }

        cuda_try_or_return(cudaMemcpyToSymbol(kernels::centroids, averages.data(), sizeof(float) * DIMENSION_COUNT * base::clusters_count));

        return cudaDeviceSynchronize();
    }

public:
    gpu_kmeans(int rows_count, int clusters_count) : kmeans<DIMENSION_COUNT>(rows_count, clusters_count), permutation{(size_t)rows_count}
    {
        thrust::sequence(permutation.begin(), permutation.end());
        cuda_try_or_exit(allocate_device_data());
    }

    ~gpu_kmeans()
    {
        cuda_try_or_exit(free_device_data());
    }

    virtual cudaError_t load_points_data(const csv_columnwise_data<DIMENSION_COUNT> &data) override
    {
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            cuda_try_or_return(cudaMemcpy(dev_points_data[i], data.data[i].data(), base::rows_count * sizeof(float), cudaMemcpyHostToDevice));
        }

        cuda_try_or_return(set_first_points_as_centroids(data));

        return cudaDeviceSynchronize();
    }

    virtual cudaError_t compute(int iteration_limit) override
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

    virtual cudaError_t get_points_assignments(int *host_cluster_assignments, float *centroids) override
    {
        thrust::device_ptr<int> aaaa(dev_cluster_assignments);
        thrust::device_vector<int> sss(base::rows_count);
        thrust::scatter(aaaa, aaaa + base::rows_count, permutation.begin(), sss.begin());

        cuda_try_or_return(cudaMemcpy(host_cluster_assignments, sss.data().get(), base::rows_count * sizeof(int), cudaMemcpyDeviceToHost));

        cuda_try_or_return(cudaMemcpyFromSymbol(centroids, kernels::centroids, DIMENSION_COUNT * base::clusters_count * sizeof(float)));
        return cudaDeviceSynchronize();
    }
};

template <int DIMENSION_COUNT>
class gpu_kmedians_with_partition : public gpu_kmeans<DIMENSION_COUNT>
{
    typedef gpu_kmeans<DIMENSION_COUNT> base;

protected:
    thrust::device_vector<float> partition_buffer;

    virtual float find_median(thrust::device_ptr<float> begin, thrust::device_ptr<float> end, size_t size)
    {
        thrust::copy(begin, end, partition_buffer.begin());
        thrust::device_ptr<float> partition_buffer_ptr = partition_buffer.data();
        thrust::device_ptr<float> partition_point;
        thrust::device_ptr<float> buffer_begin = partition_buffer_ptr;
        thrust::device_ptr<float> buffer_end = buffer_begin + size;
        float pivot_value;
        thrust::device_ptr<float> middle = partition_buffer_ptr + size / 2;

        while (size > 1)
        {
            int index = rand() % size;
            thrust::copy(buffer_begin + index, buffer_begin + index + 1, &pivot_value);
            partition_point = thrust::partition(buffer_begin, buffer_end, thrust::placeholders::_1 <= pivot_value);
            if (partition_point <= middle)
            {
                buffer_begin = partition_point;
            }
            else
            {
                buffer_end = partition_point - 1;
            }
            size = thrust::distance(buffer_begin, buffer_end);
        }

        thrust::copy(buffer_begin, buffer_begin + 1, &pivot_value);
        return pivot_value;
    }

    virtual cudaError_t recalculate_centroids() override
    {
        thrust::device_ptr<float> data[DIMENSION_COUNT];
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            data[i] = thrust::device_ptr<float>(base::dev_points_data[i]);
        }

        thrust::device_ptr<int> clusters(base::dev_cluster_assignments);
        auto ptr_tuple = base::make_tuple_with_permutation(base::permutation.begin(), data);

        auto zipped_data = thrust::make_zip_iterator(ptr_tuple);
        thrust::sort_by_key(clusters, clusters + base::rows_count, zipped_data);

        thrust::device_vector<int> new_keys(base::clusters_count);
        thrust::device_vector<int> counts(base::clusters_count);

        thrust::reduce_by_key(
            clusters,
            clusters + base::rows_count,
            thrust::make_constant_iterator<int>(1),
            new_keys.begin(),
            counts.begin(),
            thrust::equal_to<int>(),
            thrust::plus<int>());

        thrust::host_vector<int> h_counts(counts);

        thrust::host_vector<float> medians(DIMENSION_COUNT * base::clusters_count);

        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            int begin = 0;
            int end = 0;
            for (int j = 0; j < base::clusters_count; j++)
            {
                begin = end;
                end += h_counts[j];
                medians[i + j * DIMENSION_COUNT] = find_median(data[i] + begin, data[i] + end, h_counts[j]);
            }
        }

        cuda_try_or_return(cudaMemcpyToSymbol(kernels::centroids, medians.data(), sizeof(float) * DIMENSION_COUNT * base::clusters_count));

        return cudaDeviceSynchronize();
    }

public:
    gpu_kmedians_with_partition(int rows_count, int clusters_count) : base(rows_count, clusters_count), partition_buffer(rows_count) {}
};

template <int DIMENSION_COUNT>
class gpu_kmedians_with_sort : public gpu_kmedians_with_partition<DIMENSION_COUNT>
{
    typedef gpu_kmedians_with_partition<DIMENSION_COUNT> base;

    float find_median(thrust::device_ptr<float> begin, thrust::device_ptr<float> end, size_t size) override
    {
        thrust::copy(begin, end, base::partition_buffer.begin());
        thrust::device_ptr<float> partition_buffer_ptr = base::partition_buffer.data();
        thrust::device_ptr<float> partition_point;
        thrust::device_ptr<float> buffer_begin = partition_buffer_ptr;
        thrust::device_ptr<float> buffer_end = buffer_begin + size;
        float pivot_value;
        thrust::device_ptr<float> middle = partition_buffer_ptr + size / 2;

        thrust::sort(buffer_begin, buffer_end);
        if (size % 2 == 0)
        {
            thrust::copy(middle, middle + 1, &pivot_value);
            return pivot_value;
        }
        else
        {
            thrust::host_vector<float> pivot_values(2);
            thrust::copy(middle, middle + 2, pivot_values.begin());
            return (pivot_values[0] + pivot_values[1]) * 0.5f;
        }
    }

public:
    gpu_kmedians_with_sort(int rows_count, int clusters_count) : base(rows_count, clusters_count) {}
};
