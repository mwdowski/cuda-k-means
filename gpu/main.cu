#include <cstdlib>
#include "../options/options.hpp"
#include <iostream>
int main(int argc, char *argv[])
{
    options opts = options::from_commandline_arguments(argc, argv);

    return EXIT_SUCCESS;
}