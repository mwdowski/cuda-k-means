#include "application.cuh"
#include "../kmeans/kmeans.cuh"

template <int DIMENSIONS_COUNT>
void application::run_for_one_dimensions_count(options &options)
{
    kmeans<DIMENSIONS_COUNT> kmeans(2137, options.cluster_count);
}

void application::run(options &options)
{
    switch (options.dimension_count)
    {
    case 2:
        run_for_one_dimensions_count<2>(options);
        break;
    case 3:
        run_for_one_dimensions_count<3>(options);
        break;
    case 4:
        run_for_one_dimensions_count<4>(options);
        break;
    case 5:
        run_for_one_dimensions_count<5>(options);
        break;
    case 6:
        run_for_one_dimensions_count<6>(options);
        break;
    case 7:
        run_for_one_dimensions_count<7>(options);
        break;
    default:
        break;
    }
}