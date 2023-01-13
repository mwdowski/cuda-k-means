#pragma once

#include <vector>

template <int COLUMN_COUNT>
class csv_columnwise_data
{
public:
    inline static const int DEFAULT_ROWS_NUMBER = 1'000'000;

    std::vector<float> data[COLUMN_COUNT]{std::vector<float>()};

    csv_columnwise_data()
    {
        for (int i = 0; i < COLUMN_COUNT; i++)
        {
            data[i].reserve(DEFAULT_ROWS_NUMBER);
        }
    }

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
