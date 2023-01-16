#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std::chrono;

struct application_timer
{
    typedef _V2::system_clock::time_point moment;
    moment start;
    moment file_loading;
    moment copying_to_gpu;
    moment computation;
    moment copying_from_gpu;
    moment file_saving;
    moment end;
    int iterations;

public:
    friend std::ostream &operator<<(std::ostream &out, const application_timer &timer)
    {
        out << "Finished execution.\n";
        out << std::fixed;
        out << std::setw(8) << std::setprecision(3) << duration_cast<microseconds>(timer.file_loading - timer.start).count() / 1000.0f
            << "ms - File loading time\n";
        out << std::setw(8) << std::setprecision(3) << duration_cast<microseconds>(timer.copying_to_gpu - timer.file_loading).count() / 1000.0f
            << "ms - Allocating and copying to GPU time.\n";
        out << std::setw(8) << std::setprecision(3) << duration_cast<microseconds>(timer.computation - timer.copying_to_gpu).count() / 1000.0f
            << "ms - Computation time. (";
        out << std::setw(8) << std::setprecision(3) << duration_cast<microseconds>(timer.computation - timer.copying_to_gpu).count() / 1000.0f / timer.iterations
            << "ms per iteration. There were " << timer.iterations << " iterations.)\n";
        out << std::setw(8) << std::setprecision(3) << duration_cast<microseconds>(timer.copying_from_gpu - timer.computation).count() / 1000.0f
            << "ms - Copying from GPU time.\n";
        out << std::setw(8) << std::setprecision(3) << duration_cast<microseconds>(timer.file_saving - timer.copying_from_gpu).count() / 1000.0f
            << "ms - File saving time.\n";
        out << std::setw(8) << std::setprecision(3) << duration_cast<microseconds>(timer.end - timer.start).count() / 1000.0f
            << "ms - Summarized run time.\n";

        return out;
    }
};
