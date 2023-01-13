#include "data_visualizer_2d.hpp"

#include <math.h>
#include "../macros/macros.hpp"

data_visualizer_2d::data_visualizer_2d(
    const std::vector<float> &x,
    const std::vector<float> &y,
    const std::vector<int> &color,
    const std::vector<float> &centroids)
    : x{x}, y{y}, color{color}, centroids{centroids} {}

void data_visualizer_2d::reshape(int width, int height)
{
    glMatrixMode(GL_PROJECTION);

    glViewport(0, 0, width, height);

    glLoadIdentity();
    glOrtho(-10, 10, -5, 15, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
}

void DrawCircle(float cx, float cy, float r, int num_segments)
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

void data_visualizer_2d::display(GLFWwindow *window)
{
    glClear(GL_COLOR_BUFFER_BIT);

    int ctr[5] = {0};
    glBegin(GL_POINTS);
    for (size_t i = 0; i < x.size(); i++)
    {
        glColor4ubv((GLubyte *)&COLORS[color[i]]);
        glVertex2f(x[i], y[i]);
        ctr[color[i]]++;
    }
    glEnd();

    GLubyte white[4];
    *((unsigned int *)white) = 0xFFFFFFFFU;

    glColor4ubv(white);
    for (int i = 0; i < 3; i++)
    {
        DrawCircle(centroids[i * 2], centroids[i * 2 + 1], 0.1, 100);
    }

    glfwSwapBuffers(window);
}

void data_visualizer_2d::show_plot()
{
    if (!glfwInit())
    {
        fprintf_error_and_exit("Could not initialize GLFW.\n");
    }
    // glfwSetErrorCallback(error_callback);

    GLFWwindow *window = glfwCreateWindow(800, 600, "GPU project", NULL, NULL);
    if (!window)
    {
        return;
    }

    glfwMakeContextCurrent(window);

    // glEnable(GL_TEXTURE_2D);
    // glfwSetKeyCallback(window, key_callback);
    // glfwSetMouseButtonCallback(window, mouse_button_callback);
    while (!glfwWindowShouldClose(window))
    {
        reshape(800, 800);
        display(window);
        glfwPollEvents();
    }

    // onexit();
    glfwDestroyWindow(window);
    glfwTerminate();
}
