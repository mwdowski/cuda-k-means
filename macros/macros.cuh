#pragma once

#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define cuda_try_or_exit(result)    \
    if (result != cudaSuccess)      \
    {                               \
        fprintf_cuda_error(result); \
        exit(1);                    \
    }

#define cuda_try_or_return(result)  \
    if (result != cudaSuccess)      \
    {                               \
        fprintf_cuda_error(result); \
        return result;              \
    }

#define fprintf_error_and_exit(error_string) \
    fprintf(error_string);                   \
    exit(EXIT_FAILURE);

#define fprintf_cuda_error(error_code) fprintf(              \
    stderr,                                                  \
    "%s: line %d - CUDA action failed! Error %d (%s): %s\n", \
    __FILE__, __LINE__, error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code));

#define GET_VARIABLE_NAME(Variable) (#Variable)