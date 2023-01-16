#include "application.cuh"
#include "../kmeans/gpu_kmeans.cuh"
#include "../csv_reader/csv_reader.hpp"
#include "../csv_reader/csv_columnwise_data.hpp"
#include "../macros/macros.hpp"
#include "../data_visualizer/data_visualizer_2d.hpp"
#include <vector>
#include <boost/preprocessor.hpp>
#include "../macros/macros.hpp"
#include "application_timer.hpp"
#include <ctime>
#include <chrono>

#ifndef DIMENSION_TOP_LIMIT
#define DIMENSION_TOP_LIMIT 3
#endif
#ifndef DIMENSION_BOTTOM_LIMIT
#define DIMENSION_BOTTOM_LIMIT 2
#endif

template <int DIMENSIONS_COUNT>
kmeans<DIMENSIONS_COUNT> *get_kmeans(int rows_count, int clusters_count, options::kmeans_centroid_algorithm alg)
{
    kmeans<DIMENSIONS_COUNT> *result = nullptr;

    switch (alg)
    {
    case options::kmeans_centroid_algorithm::gpu_mean:
        result = new gpu_kmeans<DIMENSIONS_COUNT>(rows_count, clusters_count);
        break;
    case options::kmeans_centroid_algorithm::gpu_median_partition:
        result = new gpu_kmedians_with_partition<DIMENSIONS_COUNT>(rows_count, clusters_count);
        break;
    case options::kmeans_centroid_algorithm::gpu_median_sort:
        result = new gpu_kmedians_with_sort<DIMENSIONS_COUNT>(rows_count, clusters_count);
        break;
    default:
        break;
    }

    return result;
}

template <int DIMENSIONS_COUNT>
void application::run_for_one_dimensions_count(options &options)
{
    application_timer timer;
    timer.start = std::chrono::high_resolution_clock::now();

    csv_columnwise_data<DIMENSIONS_COUNT> data = csv_reader<DIMENSIONS_COUNT>::from_file(options.input_file_name.c_str());
    if (!data.is_correct())
    {
        fprintf_error_and_exit("Invalid data.\n");
    }
    timer.file_loading = std::chrono::high_resolution_clock::now();

    kmeans<DIMENSIONS_COUNT> *kmeans = get_kmeans<DIMENSIONS_COUNT>(data.size(), options.cluster_count, options.centroid_algorithm);
    cuda_try_or_exit(kmeans->load_points_data(data));
    timer.copying_to_gpu = std::chrono::high_resolution_clock::now();

    cuda_try_or_exit(kmeans->compute(options.iteration_limit, timer.iterations));
    timer.computation = std::chrono::high_resolution_clock::now();

    int *colors_p = new int[data.size()];
    float *centroids_p = new float[options.cluster_count * DIMENSIONS_COUNT];
    cuda_try_or_exit(kmeans->get_points_assignments(colors_p, centroids_p));

    delete kmeans;

    std::vector<int> colors;
    colors.assign(colors_p, colors_p + data.size());
    colors_p = nullptr;

    timer.copying_from_gpu = std::chrono::high_resolution_clock::now();

    if (options.output_file_name.length() > 0)
    {
        csv_reader<1>::to_file(options.output_file_name.c_str(), colors);
    }
    timer.file_saving = std::chrono::high_resolution_clock::now();

    if (options.visualize)
    {

        std::vector<float> clusters;
        clusters.assign(centroids_p, centroids_p + options.cluster_count * DIMENSIONS_COUNT);
        centroids_p = nullptr;

        if (DIMENSIONS_COUNT == 2)
        {
            data_visualizer_2d visualizer(data.data[0], data.data[1], colors, clusters, options.cluster_count);
            visualizer.show_plot();
        }
        /*
        else if (DIMENSIONS_COUNT == 3)
        {
            data_visualizer_3d visualizer(data.data[0], data.data[1], data.data[2], vector<float>());
            visualizer.show_plot();
        }
        */
    }

    timer.end = std::chrono::high_resolution_clock::now();
    std::cout << timer;
}

void application::run(options &options)
{
#define CODE_DIMENSION_SWITCH_PASTE(rep, n, _)    \
    case n:                                       \
        run_for_one_dimensions_count<n>(options); \
        break;

    switch (options.dimension_count)
    {
        BOOST_PP_REPEAT_FROM_TO(DIMENSION_BOTTOM_LIMIT, DIMENSION_TOP_LIMIT, CODE_DIMENSION_SWITCH_PASTE, _)
    default:
        fprintf_error_and_exit("Not supported number of dimensions.\n")
    }
}