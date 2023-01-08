#include "application.cuh"
#include "../kmeans/kmeans.cuh"
#include "../csv_reader/csv_reader.hpp"
#include "../csv_reader/csv_columnwise_data.hpp"
#include "../macros/macros.hpp"

template <int DIMENSIONS_COUNT>
void application::run_for_one_dimensions_count(options &options)
{
    csv_columnwise_data<DIMENSIONS_COUNT> data = csv_reader<DIMENSIONS_COUNT>::from_file(options.input_file_name.c_str());
    if (!data.is_correct())
    {
        fprintf_error_and_exit("Invalid data.\n");
    }

    kmeans<DIMENSIONS_COUNT> kmeans((int)data.size(), options.cluster_count);

    kmeans.load_points_data(data.data);
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