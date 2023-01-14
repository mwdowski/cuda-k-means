#pragma once

#include "data_visualizer.hpp"
#include <vector>

#include <GLFW/glfw3.h>

class data_visualizer_2d : public data_visualizer
{
    const std::vector<float> &x;
    const std::vector<float> &y;
    const std::vector<int> &color;
    const std::vector<float> &centroids;
    const int cluster_count;

private:
    void display(GLFWwindow *window);
    void reshape(float x_min, float x_max, float y_min, float y_max);

public:
    data_visualizer_2d(const std::vector<float> &x, const std::vector<float> &y, const std::vector<int> &color, const std::vector<float> &centroids, int cluster_count);
    virtual void show_plot() override;
};
