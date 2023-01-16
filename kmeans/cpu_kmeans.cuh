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
#include <algorithm>
#include <functional>
#include <ranges>

template <int DIMENSION_COUNT>
class cpu_kmeans : public kmeans<DIMENSION_COUNT>
{
    typedef kmeans<DIMENSION_COUNT> base;

protected:
    float *host_points_data[DIMENSION_COUNT];
    int *host_cluster_assignments = nullptr;
    int *host_changed_assignments = nullptr;
    int *permutation;
    std::vector<float> centroids = std::vector<float>(base::clusters_count * DIMENSION_COUNT);

private:
    cudaError_t
    allocate_data()
    {
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            host_points_data[i] = new float[base::rows_count];
        }

        host_cluster_assignments = new int[base::rows_count];
        host_changed_assignments = new int[base::rows_count];
        permutation = new int[base::rows_count];
        for (int i = 0; i < base::rows_count; i++)
        {
            permutation[i] = i;
        }

        memset(host_changed_assignments, 1, base::rows_count * sizeof(int));

        return cudaSuccess;
    }

    cudaError_t free_device_data()
    {
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            delete[] host_points_data[i];
            host_points_data[i] = nullptr;
        }

        delete[] host_cluster_assignments;
        host_cluster_assignments = nullptr;

        delete[] host_changed_assignments;
        host_changed_assignments = nullptr;

        delete[] permutation;
        permutation = nullptr;

        return cudaSuccess;
    }

    cudaError_t set_first_points_as_centroids(const csv_columnwise_data<DIMENSION_COUNT> &data)
    {
        for (int i = 0; i < base::clusters_count; i++)
        {
            for (int j = 0; j < DIMENSION_COUNT; j++)
            {
                centroids[j + DIMENSION_COUNT * i] = data.data[j][i];
            }
        }

        return cudaSuccess;
    }

    float distance_squared(float *x, float *y)
    {
        float result = 0.0f;
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            result += (x[i] - y[i]) * (x[i] - y[i]);
        }

        return result;
    }

    cudaError_t assign_nearest_clusters()
    {
        for (int index = 0; index < base::rows_count; index++)
        {
            float point[DIMENSION_COUNT];

            for (int i = 0; i < DIMENSION_COUNT; i++)
            {
                point[i] = host_points_data[i][index];
            }

            // find closest cluster center
            float min_distance_squared = distance_squared(centroids.data(), point);
            int min_cluster = 0;

            for (int k = 1; k < base::clusters_count; k++)
            {
                float current_distance_squared = distance_squared(&centroids.data()[k * DIMENSION_COUNT], point);
                if (current_distance_squared < min_distance_squared)
                {
                    min_distance_squared = current_distance_squared;
                    min_cluster = k;
                }
            }

            int old_assignment = host_cluster_assignments[index];
            host_cluster_assignments[index] = min_cluster;
            host_changed_assignments[index] = min_cluster != old_assignment;
        }

        return cudaSuccess;
    }

    bool is_finished()
    {
        // return thrust::none_of(a, a + base::rows_count, thrust::identity<int>());
        return std::none_of(host_changed_assignments, host_changed_assignments + base::rows_count, [](int &x)
                            { return x != 0; });
    }

protected:
    virtual cudaError_t recalculate_centroids()
    {
        std::vector<float> sums[DIMENSION_COUNT];
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            sums[i] = std::vector<float>(base::clusters_count);
        }
        std::vector<int> counts(base::clusters_count);
        std::fill(counts.begin(), counts.end(), 0);

        for (int i = 0; i < base::rows_count; i++)
        {
            counts[host_cluster_assignments[i]]++;
        }

        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            std::fill(sums[i].begin(), sums[i].end(), 0);

            for (int j = 0; j < base::rows_count; j++)
            {
                sums[i][host_cluster_assignments[j]] += host_points_data[i][j];
            }
        }

        for (int i = 0; i < base::clusters_count; i++)
        {
            float y = counts[i];
            for (int j = 0; j < DIMENSION_COUNT; j++)
            {
                float x = sums[j][i];
                centroids[j + DIMENSION_COUNT * i] = x / y;
            }
        }

        return cudaSuccess;
    }

public:
    cpu_kmeans(int rows_count, int clusters_count) : kmeans<DIMENSION_COUNT>(rows_count, clusters_count)
    {
        cuda_try_or_exit(allocate_data());
    }

    ~cpu_kmeans()
    {
        cuda_try_or_exit(free_device_data());
    }

    virtual cudaError_t load_points_data(const csv_columnwise_data<DIMENSION_COUNT> &data) override
    {
        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            memcpy(host_points_data[i], data.data[i].data(), base::rows_count * sizeof(float));
        }

        cuda_try_or_return(set_first_points_as_centroids(data));

        return cudaDeviceSynchronize();
    }

    virtual cudaError_t compute(int iteration_limit, int &interation_count) override
    {
        int counter = 0;
        while (!is_finished() && counter < iteration_limit)
        {
            cuda_try_or_return(assign_nearest_clusters());
            cuda_try_or_return(recalculate_centroids());

            counter++;
        }
        interation_count = counter;
        cuda_try_or_return(assign_nearest_clusters());

        return cudaSuccess;
    }

    virtual cudaError_t get_points_assignments(int *cluster_assignments_out, float *centroids_out) override
    {
        std::vector<int> result(base::rows_count);

        for (int i = 0; i < base::rows_count; i++)
        {
            result[i] = host_cluster_assignments[permutation[i]];
        }

        memcpy(cluster_assignments_out, result.data(), base::rows_count * sizeof(int));

        memcpy(centroids_out, centroids.data(), DIMENSION_COUNT * base::clusters_count * sizeof(float));
        return cudaSuccess;
    }
};

template <int DIMENSION_COUNT>
class cpu_kmedians : public cpu_kmeans<DIMENSION_COUNT>
{
    typedef cpu_kmeans<DIMENSION_COUNT> base;
    float *median_buffer;

protected:
    virtual float find_median(std::vector<float> data)
    {
        std::nth_element(data.begin(), data.begin() + (data.size() / 2), data.end());
        return data[data.size() / 2];
    }

    virtual cudaError_t recalculate_centroids() override
    {
        std::vector<int> counts(base::clusters_count);
        std::fill(counts.begin(), counts.end(), 0);

        for (int i = 0; i < base::rows_count; i++)
        {
            counts[base::host_cluster_assignments[i]]++;
        }

        std::vector<std::vector<int>> buckets(base::clusters_count);
        for (int i = 0; i < base::clusters_count; i++)
        {
            buckets[i] = std::vector<int>();
        }

        for (int i = 0; i < base::rows_count; i++)
        {
            buckets[base::host_cluster_assignments[i]].push_back(i);
        }

        for (int i = 0; i < DIMENSION_COUNT; i++)
        {
            for (int j = 0; j < base::clusters_count; j++)
            {
                std::vector<float> data;
                for (size_t k = 0; k < buckets[j].size(); k++)
                {
                    data.push_back(base::host_points_data[i][buckets[j][k]]);
                }
                base::centroids[j * DIMENSION_COUNT + i] = find_median(data);
            }
        }

        return cudaSuccess;
    }

public:
    cpu_kmedians(int rows_count, int clusters_count) : base(rows_count, clusters_count)
    {
        median_buffer = new float[rows_count];
    }

    ~cpu_kmedians()
    {
        delete[] median_buffer;
    }
};
