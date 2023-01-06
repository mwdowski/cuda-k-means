#pragma once

#include "../options/options.hpp"

class application
{
    application() = delete;

private:
    template <int DIMENSIONS_COUNT>
    static void run_for_one_dimensions_count(options &options);

public:
    static void run(options &options);
};
