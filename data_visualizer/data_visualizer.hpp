#pragma once

class data_visualizer
{
public:
    virtual void show_plot() = 0;

protected:
    inline static const unsigned int COLORS[3] = {
        0xFF4D4DFFU,
        0xFF4DFF4DU,
        0xFFFF4D4DU,
    };
};
