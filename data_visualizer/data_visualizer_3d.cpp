#include "data_visualizer_3d.hpp"

#include <math.h>
#include "../macros/macros.hpp"

data_visualizer_3d::data_visualizer_3d(
    const std::vector<float> &x,
    const std::vector<float> &y,
    const std::vector<float> &z,
    const std::vector<int> &color,
    int cluster_count)
    : x{x}, y{y}, z{z}, color{color}, cluster_count{cluster_count}
{
}

void data_visualizer_3d::reshape(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max)
{
    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();
    glOrtho(
        x_min,
        x_max,
        y_min * (x_max - x_min) / (y_max - y_min),
        y_max * (x_max - x_min) / (y_max - y_min),
        z_min * (x_max - x_min) / (z_max - z_min), z_max * (x_max - x_min) / (z_max - z_min));

    glMatrixMode(GL_MODELVIEW);
}

void data_visualizer_3d::display(GLFWwindow *window)
{
    glClear(GL_COLOR_BUFFER_BIT);

    float x_min, x_max, y_min, y_max, z_min, z_max;

    glPushMatrix();
    glRotatef(y_current_angle, 0.0f, 1.0f, 0.0f);
    glRotatef(x_current_angle, 1.0f, 0.0f, 0.0f);
    glBegin(GL_POINTS);

    glColor4ubv((GLubyte *)&COLORS[color[0]]);
    glVertex3f(x[0], y[0], z[0]);

    x_min = x_max = x[0];
    y_min = y_max = y[0];
    z_min = z_max = z[0];

    for (size_t i = 1; i < x.size(); i++)
    {
        glColor4ubv((GLubyte *)&COLORS[color[i]]);
        glVertex3f(x[i], y[i], z[i]);
        x_min = std::min(x_min, x[i]);
        y_min = std::min(y_min, y[i]);
        z_min = std::min(z_min, z[i]);
        x_max = std::max(x_max, x[i]);
        y_max = std::max(y_max, y[i]);
        z_max = std::max(z_max, z[i]);
    }
    glEnd();
    glPopMatrix();

    glfwSwapBuffers(window);

    reshape(
        x_min - (x_max - x_min) * 0.05f,
        x_max + (x_max - x_min) * 0.05f,
        y_min - (y_max - y_min) * 0.05f,
        y_max - (y_max - y_min) * 0.05f,
        z_min - (z_max - z_min) * 0.05f,
        z_max + (z_max - z_min) * 0.05f);
}

void data_visualizer_3d::cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    data_visualizer_3d *handler = reinterpret_cast<data_visualizer_3d *>(glfwGetWindowUserPointer(window));
    if (handler->old_x < 0)
    {
        handler->old_x = xpos;
    }
    else
    {
        handler->y_current_angle += (xpos - handler->old_x);
        handler->old_x = xpos;
    }

    if (handler->old_y < 0)
    {
        handler->old_y = ypos;
    }
    else
    {
        handler->x_current_angle += (ypos - handler->old_y);
        if (handler->x_current_angle > 90.0f)
        {
            handler->x_current_angle = 90.0f;
        }
        if (handler->x_current_angle < -90.0f)
        {
            handler->x_current_angle = -90.0f;
        }
        handler->old_y = ypos;
    }
}

void data_visualizer_3d::show_plot()
{
    if (!glfwInit())
    {
        fprintf_error_and_exit("Could not initialize GLFW.\n");
    }

    GLFWwindow *window = glfwCreateWindow(800, 800, "GPU project", NULL, NULL);
    if (!window)
    {
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, reinterpret_cast<void *>(this));
    glfwSetCursorPosCallback(window, cursor_position_callback);

    while (!glfwWindowShouldClose(window))
    {
        display(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}
