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

private:
    void display(GLFWwindow *window);
    void reshape(int width, int height);

public:
    data_visualizer_2d(const std::vector<float> &x, const std::vector<float> &y, const std::vector<int> &color, const std::vector<float> &centroids);
    virtual void show_plot() override;
};
