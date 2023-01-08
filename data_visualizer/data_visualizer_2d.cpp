#include "data_visualizer_2d.hpp"

#include "../macros/macros.hpp"

data_visualizer_2d::data_visualizer_2d(const std::vector<float> &x, const std::vector<float> &y, const std::vector<int> &color)
    : x{x}, y{y}, color{color} {}

void data_visualizer_2d::reshape(int width, int height)
{
    glMatrixMode(GL_PROJECTION);

    glViewport(0, 0, width, height);

    glLoadIdentity();
    glOrtho(-7, 13, -6, 14, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
}

void data_visualizer_2d::display(GLFWwindow *window)
{
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_POINTS);
    for (size_t i = 0; i < x.size(); i++)
    {
        // TODO: take color from array
        glColor4ubv((GLubyte *)&COLORS[0]);
        glVertex2f(x[i], y[i]);
    }
    glEnd();

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
