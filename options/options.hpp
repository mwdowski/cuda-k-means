#pragma once

#include <string>

struct options
{
    enum kmeans_centroid_algorithm
    {
        not_set,
        alg1,
        alg2
    };

    int cluster_count = CLUSTER_COUNT_NOT_DECLARED;
    int dimension_count = 2;
    bool visualize = false;
    kmeans_centroid_algorithm centroid_algorithm = not_set;
    std::string input_file_name = FILE_NAME_NOT_DECLARED;
    std::string output_file_name = FILE_NAME_NOT_DECLARED;

private:
    inline static const int CLUSTER_COUNT_NOT_DECLARED = -1;
    inline static const int GETOPT_FINISHED = -1;
    inline static const int MIN_CLUSTER_COUNT = 2;
    inline static const std::string FILE_NAME_NOT_DECLARED = "";
    inline static const std::string HELP_MESSAGE =
        "Required arguments:\n"
        " -i [file path]: Input file path.\n"
        " -o [file path]: Output file path.\n"
        " -k [integer]: Desired cluster count. Must be an integer higher than 1\n"
        "Optional arguments:\n"
        " -v: Visualize results. Works only for 2 and 3-dimensional data.\n"
        " -h: Display help.\n"
        " -a: Algorithm - TODO.\n";

    options(){};
    bool is_valid();

public:
    static options from_commandline_arguments(const int argc, char **argv);
};
