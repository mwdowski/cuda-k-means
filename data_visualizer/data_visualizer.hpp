#pragma once

#include <cmath>
#include <GLFW/glfw3.h>

class data_visualizer
{
public:
    virtual void show_plot() = 0;

    inline static void draw_circle(float cx, float cy, float r, int num_segments)
    {
        glBegin(GL_LINE_LOOP);
        for (int ii = 0; ii < num_segments; ii++)
        {
            float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments); // get the current angle
            float x = r * cosf(theta);                                         // calculate the x component
            float y = r * sinf(theta);                                         // calculate the y component
            glVertex2f(x + cx, y + cy);                                        // output vertex
        }
        glEnd();
    }

public:
    const unsigned int COLORS[18] = {
        0xFF4900E6U,
        0xFFFFB40BU,
        0xFF91E950U,
        0xFF00D8E6U,
        0xFFF5199BU,
        0xFF00A3FFU,
        0xFFB40ADCU,
        0xFFFFD4B3U,
        0xFFA0BF00U,
        0xFF6F7FFDU,
        0xFFD5B07EU,
        0xFF61E0B2U,
        0xFFBE7EBDU,
        0xFF5AB5FFU,
        0xFF65EEFFU,
        0xFFDBB9BEU,
        0xFFE5CCFDU,
        0xFFC7D38BU,
    };
};