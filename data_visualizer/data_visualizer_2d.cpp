#include "data_visualizer_2d.hpp"

#include <math.h>
#include "../macros/macros.hpp"

data_visualizer_2d::data_visualizer_2d(
    const std::vector<float> &x,
    const std::vector<float> &y,
    const std::vector<int> &color,
    const std::vector<float> &centroids,
    int cluster_count)
    : x{x}, y{y}, color{color}, centroids{centroids}, cluster_count{cluster_count}
{
}

void data_visualizer_2d::reshape(float x_min, float x_max, float y_min, float y_max)
{
    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();
    glOrtho(x_min, x_max, y_min * (x_max - x_min) / (y_max - y_min), y_max * (x_max - x_min) / (y_max - y_min), -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
}

void data_visualizer_2d::display(GLFWwindow *window)
{
    glClear(GL_COLOR_BUFFER_BIT);

    float x_min, x_max, y_min, y_max;

    glBegin(GL_POINTS);

    glColor4ubv((GLubyte *)&COLORS[color[0]]);
    glVertex2f(x[0], y[0]);

    x_min = x_max = x[0];
    y_min = y_max = y[0];

    for (size_t i = 1; i < x.size(); i++)
    {
        glColor4ubv((GLubyte *)&COLORS[color[i]]);
        glVertex2f(x[i], y[i]);
        x_min = std::min(x_min, x[i]);
        y_min = std::min(y_min, y[i]);
        x_max = std::max(x_max, x[i]);
        y_max = std::max(y_max, y[i]);
    }
    glEnd();

    GLubyte white[4];
    *((unsigned int *)white) = 0xFFFFFFFFU;

    glColor4ubv(white);
    for (int i = 0; i < cluster_count; i++)
    {
        data_visualizer::draw_circle(centroids[i * 2], centroids[i * 2 + 1], 0.1, 100);
    }
    glfwSwapBuffers(window);

    reshape(
        x_min - (x_max - x_min) * 0.05f,
        x_max + (x_max - x_min) * 0.05f,
        y_min - (y_max - y_min) * 0.05f,
        y_max + (y_max - y_min) * 0.05f);
}

void data_visualizer_2d::show_plot()
{
    if (!glfwInit())
    {
        fprintf_error_and_exit("Could not initialize GLFW.\n");
    }

    GLFWwindow *window = glfwCreateWindow(800, 600, "GPU project", NULL, NULL);
    if (!window)
    {
        return;
    }

    glfwMakeContextCurrent(window);

    while (!glfwWindowShouldClose(window))
    {
        display(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}
