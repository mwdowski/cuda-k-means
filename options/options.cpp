#include "options.hpp"

#ifdef __WIN32__
#define __GNU_LIBRARY__
#include "../getopt/getopt.h"
#else
#include <getopt.h>
#include <unistd.h>
#endif

#include <stdexcept>
#include <cstdlib>

const std::string options::FILE_NAME_NOT_DECLARED = "";
const std::string options::HELP_MESSAGE =
    "Required arguments:\n"
    " -i [file path]: Input file path.\n"
    " -k [integer]: Desired cluster count. Must be an integer in range <2, 18>.\n"
    " -n [integer]: Data dimension (column) count. Must be an integer in range <2, 9>.\n"
    "Optional arguments:\n"
    " -v: Visualize results. Works only for 2 and 3-dimensional data. Changed to \"-c\" when used with higher number of dimensions.\n"
    " -c: Print out calculated centroids.\n"
    " -o [file path]: Output file path.\n"
    " -l [integer]: Desired iteration limit. Must be non-negative. Default value is 40.\n"
    " -h: Display help.\n"
    " -r: [integer] Seed for randomizer. It's value of time(NULL) by default.\n"
    " -a: Algorithm:\n"
    "    -a 0: k-means on GPU (default);\n"
    "    -a 1: k-medians with partition on GPU (default);\n"
    "    -a 2: k-medians with sorting on GPU (default);\n"
    "    -a 3: k-means on CPU ;\n"
    "    -a 4: k-medians on CPU.\n";

bool options::is_valid()
{
    return this->input_file_name.length() > 0 && this->cluster_count != CLUSTER_COUNT_NOT_DECLARED;
}

options options::from_commandline_arguments(const int argc, char **argv)
{
    options result;
    int current_option = 0;
    while (current_option != GETOPT_FINISHED)
    {
        switch (current_option = getopt(argc, argv, "i:o:a:k:n:l:hvcr:"))
        {
        case GETOPT_FINISHED:
            break;
        case 'i':
            result.input_file_name = std::string(optarg);
            break;
        case 'r':
            try
            {
                result.random_seed = std::stoi(std::string(optarg));
            }
            catch (...)
            {
                fprintf(stderr, "Invalid seed \"%s\". Use option \"h\" for help.\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'o':
            result.output_file_name = std::string(optarg);
            break;
        case 'k':
            try
            {
                result.cluster_count = std::stoi(std::string(optarg));

                if (result.cluster_count < MIN_CLUSTER_COUNT || result.cluster_count > MAX_CLUSTER_COUNT)
                {
                    throw nullptr;
                }
            }
            catch (...)
            {
                fprintf(stderr, "Invalid cluster count \"%s\". Use option \"h\" for help.\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'l':
            try
            {
                result.iteration_limit = std::stoi(std::string(optarg));

                if (result.iteration_limit < 0)
                {
                    throw nullptr;
                }
            }
            catch (...)
            {
                fprintf(stderr, "Invalid iteration limit \"%s\". Use option \"h\" for help.\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'n':
            try
            {
                result.dimension_count = std::stoi(std::string(optarg));

                if (result.dimension_count < MIN_DIMENSIONS_COUNT || result.dimension_count > MAX_DIMENSIONS_COUNT)
                {
                    throw nullptr;
                }
            }
            catch (...)
            {
                fprintf(stderr, "Invalid dimensions count \"%s\". Use option \"h\" for help.\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'h':
            fprintf(stderr, "%s", HELP_MESSAGE.c_str());
            exit(EXIT_FAILURE);
            break;
        case 'v':
            result.visualize = true;
            break;
        case 'c':
            result.print_centroids = true;
            break;
        case 'a':
            try
            {
                int value = std::stoi(std::string(optarg));
                if (value < 0 || value > 4)
                {
                    throw nullptr;
                }
                result.centroid_algorithm = static_cast<kmeans_centroid_algorithm>(value);
            }
            catch (...)
            {
                fprintf(stderr, "Invalid algoritm \"%s\". Use option \"h\" for help.\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case ':':
            fprintf(stderr, "Missing argument for option \"%s\". Use option \"h\" for help.\n", optarg);
            exit(EXIT_FAILURE);
            break;
        default: /* ? */
            fprintf(stderr, "Unrecognized option \"%s\". Use option \"h\" for help.\n", optarg);
            exit(EXIT_FAILURE);
            break;
        }
    }

    if (!result.is_valid())
    {
        fprintf(stderr, "Specify correct input file and specify cluster number. Use option \"h\" for help.\n");
        exit(EXIT_FAILURE);
    }

    return result;
}