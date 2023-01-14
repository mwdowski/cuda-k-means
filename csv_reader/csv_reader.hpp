#pragma once

#include "csv_columnwise_data.hpp"
#include <fstream>
#include <iostream>
#include "../macros/macros.hpp"
#include <string>
#include <vector>

template <int COLUMNS_COUNT>
class csv_reader
{
public:
    static csv_columnwise_data<COLUMNS_COUNT> from_file(const char *file_name)
    {
        std::ifstream file;
        std::string line;
        std::string delimeter = " ";
        std::string current_string;
        csv_columnwise_data<COLUMNS_COUNT> result;

        file.open(file_name);

        if (!file.is_open())
        {
            fprintf_error_and_exit("Could not open input file.\n");
        }

        while (getline(file, line))
        {
            int current_index = 0;
            float current_float;
            size_t start = 0U;
            size_t end = line.find(delimeter);
            while (end != std::string::npos)
            {
                current_string = line.substr(start, end - start);
                current_float = std::stof(current_string);
                result.data[current_index].push_back(current_float);
                current_index++;

                start = end + delimeter.length();
                end = line.find(delimeter, start);
            }
            current_string = line.substr(start, end - start);
            current_float = std::stof(current_string);
            result.data[current_index].push_back(current_float);
        }

        file.close();

        return result;
    }

    static void to_file(const char *file_name, const std::vector<int> &data)
    {
        std::ofstream file;
        std::string line;
        std::string delimeter = " ";
        std::string current_string;

        file.open(file_name, std::ios_base::trunc);

        if (!file.is_open())
        {
            fprintf_error_and_exit("Could not open output file.\n");
        }

        for (size_t i = 0; i < data.size(); i++)
        {
            file << data[i] << "\n";
        }
    }
};
