#pragma once

#include <string>

struct options
{
    enum class kmeans_centroid_algorithm
    {
        gpu_mean = 0,
        gpu_median_partition = 1,
        gpu_median_sort = 2,
        cpu_mean = 3,
        cpu_median = 4,
        min = gpu_mean,
        max = cpu_median
    };

    int cluster_count = CLUSTER_COUNT_NOT_DECLARED;
    int dimension_count = CLUSTER_COUNT_NOT_DECLARED;
    unsigned int random_seed = time(nullptr);
    int iteration_limit = 40;
    bool visualize = false;
    bool print_centroids = false;
    kmeans_centroid_algorithm centroid_algorithm = kmeans_centroid_algorithm::gpu_mean;
    std::string input_file_name = FILE_NAME_NOT_DECLARED;
    std::string output_file_name = FILE_NAME_NOT_DECLARED;

public:
    static const int CLUSTER_COUNT_NOT_DECLARED = -1;
    static const int GETOPT_FINISHED = -1;
    static const int MIN_CLUSTER_COUNT = 2;
    static const int MAX_CLUSTER_COUNT = 18;
    static const int MIN_DIMENSIONS_COUNT = 2;
    static const int MAX_DIMENSIONS_COUNT = 8;
    static const std::string FILE_NAME_NOT_DECLARED;
    static const std::string HELP_MESSAGE;

    options(){};
    bool is_valid();

public:
    static options from_commandline_arguments(const int argc, char **argv);
};
