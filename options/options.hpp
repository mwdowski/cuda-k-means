#pragma once

#include <string>

struct options
{
    enum kmeans_centroid_algorithm
    {
        gpu_mean = 0,
        gpu_median_partition = 1,
        gpu_median_sort = 2,
        cpu_mean = 3,
        cpu_median = 4,
    };

    int cluster_count = CLUSTER_COUNT_NOT_DECLARED;
    int dimension_count = 2;
    int iteration_limit = 20;
    bool visualize = false;
    kmeans_centroid_algorithm centroid_algorithm = gpu_mean;
    std::string input_file_name = FILE_NAME_NOT_DECLARED;
    std::string output_file_name = FILE_NAME_NOT_DECLARED;

private:
    inline static const int CLUSTER_COUNT_NOT_DECLARED = -1;
    inline static const int GETOPT_FINISHED = -1;
    inline static const int MIN_CLUSTER_COUNT = 2;
    inline static const int MIN_DIMENSIONS_COUNT = 2;
    inline static const int MAX_DIMENSIONS_COUNT = 9;
    inline static const std::string FILE_NAME_NOT_DECLARED = "";
    inline static const std::string HELP_MESSAGE =
        "Required arguments:\n"
        " -i [file path]: Input file path.\n"
        " -k [integer]: Desired cluster count. Must be an integer equal or higher than 2.\n"
        " -n [integer]: Data dimension (column) count. Must be an integer in range <2, 9>.\n"
        "Optional arguments:\n"
        " -v: Visualize results. Works only for 2 and 3-dimensional data.\n"
        " -o [file path]: Output file path.\n"
        " -l [integer]: Desired iteration limit. Must be non-negative. Default value is 20.\n"
        " -h: Display help.\n"
        " -a: Algorithm\n"
        "    -a 0: k-means on GPU (default);\n"
        "    -a 1: k-medians with partition on GPU (default);\n"
        "    -a 2: k-medians with sorting on GPU (default);\n";

    options(){};
    bool is_valid();

public:
    static options from_commandline_arguments(const int argc, char **argv);
};
