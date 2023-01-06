#include <cstdlib>
#include "../options/options.hpp"
#include "../application/application.cuh"

int main(int argc, char *argv[])
{
    options opts = options::from_commandline_arguments(argc, argv);
    application::run(opts);

    return EXIT_SUCCESS;
}