#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../csv_reader/csv_columnwise_data.hpp"

template <int DIMENSION_COUNT>
class kmeans
{
protected:
    const int rows_count;
    const int clusters_count;

public:
    kmeans(int rows_count, int clusters_count) : rows_count{rows_count}, clusters_count{clusters_count} {};
    virtual cudaError_t load_points_data(const csv_columnwise_data<DIMENSION_COUNT> &data) = 0;
    virtual cudaError_t compute(int iteration_limit, int &interation_count) = 0;
    virtual cudaError_t get_points_assignments(int *host_cluster_assignments, float *centroids) = 0;
    virtual ~kmeans(){};
};
