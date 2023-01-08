#pragma once

#include <vector>

template <int COLUMN_COUNT>
class csv_columnwise_data
{
public:
    std::vector<float> data[COLUMN_COUNT]{std::vector<float>()};

    bool is_correct()
    {
        size_t size = data[0].size();

        for (int i = 1; i < COLUMN_COUNT; i++)
        {
            if (size != data[i].size())
            {
                return false;
            }
            size = data[i].size();
        }

        return true;
    }

    size_t size()
    {
        return data[0].size();
    }
};
