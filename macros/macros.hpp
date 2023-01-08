#pragma once

#include <cstdlib>

#define fprintf_error_and_exit(error_string) \
    fprintf(stderr, error_string);           \
    exit(EXIT_FAILURE);

#define variable_name(Variable) (#Variable)