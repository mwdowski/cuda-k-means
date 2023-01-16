#include "options.hpp"
#include <getopt.h>
#include <unistd.h>
#include <stdexcept>
#include <cstdlib>

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
        switch (current_option = getopt(argc, argv, ":i:o:a:k:n:l:hv"))
        {
        case GETOPT_FINISHED:
            break;
        case 'i':
            result.input_file_name = std::string(optarg);
            break;
        case 'o':
            result.output_file_name = std::string(optarg);
            break;
        case 'k':
            try
            {
                result.cluster_count = std::stoi(std::string(optarg));

                if (result.cluster_count < MIN_CLUSTER_COUNT)
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
            fprintf(stderr, "Missing argument for option \"%c\". Use option \"h\" for help.\n", optopt);
            exit(EXIT_FAILURE);
            break;
        default: /* ? */
            fprintf(stderr, "Unrecognized option \"%c\". Use option \"h\" for help.\n", (char)optopt);
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