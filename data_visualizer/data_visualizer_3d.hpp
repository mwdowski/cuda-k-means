#pragma once

#include "data_visualizer.hpp"
#include <vector>

#include <GLFW/glfw3.h>

class data_visualizer_3d : public data_visualizer
{
    const std::vector<float> &x;
    const std::vector<float> &y;
    const std::vector<float> &z;
    const std::vector<int> &color;
    const int cluster_count;
    float y_current_angle = 0.0f;
    float x_current_angle = 0.0f;
    float old_x = -1.0f;
    float old_y = -1.0f;

private:
    void display(GLFWwindow *window);
    void reshape(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max);
    static void cursor_position_callback(GLFWwindow *window, double xpos, double ypos);

public:
    data_visualizer_3d(const std::vector<float> &x, const std::vector<float> &y, const std::vector<float> &z, const std::vector<int> &color, int cluster_count);
    virtual void show_plot();
};
