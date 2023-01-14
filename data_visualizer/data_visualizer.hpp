#pragma once

class data_visualizer
{
public:
    virtual void show_plot() = 0;

protected:
    inline static const unsigned int COLORS[] = {
        0xFF4900E6U,
        0xFFFFB40BU,
        0xFF91E950U,
        0xFF00D8E6U,
        0xFFF5199BU,
        0xFF00A3FFU,
        0xFFB40ADCU,
        0xFFFFD4B3U,
        0xFFA0BF00U,
    };
};