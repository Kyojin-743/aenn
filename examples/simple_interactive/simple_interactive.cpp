#include <iostream>
#include <vector>
#include <tuple>
#include <memory>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <thread>
#include <cstdint>
#include <atomic>

#include "simple_model.h"
#include "simple_gui.h"

using namespace AennTensor;
using namespace SimpleModels;
using namespace SimpleGui;

class App {
public:

    //Data held for the lifetime of the application
    GLFWwindow* window;
    GuiState guistate;
    SimpleModel model;
    Tensor train_images, test_images;
    Tensor train_labels, test_labels;

    //Synchronization
    std::thread worker_thread;  //all tasks will be ran async (detached)
    std::mutex gui;
    std::atomic<bool> worker_active;
    std::atomic<bool> worker_stop;

    App() : model({
        {28 * 28, std::make_shared<Linear>()},
        {128,     std::make_shared<ReLU>()},
        {64,      std::make_shared<ReLU>()},
        {10,      std::make_shared<Linear>()}
    }) {
        //setup glfw, opengl, imgui, implot 
        //load mnist data(all)  (happen in worker thread )
    }

    ~App() {
        //signal worker to stop
        //wait for it
        //shut down gl
        //shutdown imgui, implot
    }


    void run() {
        while(!glfwWindowShouldClose(window)) {

        }
    }

};

int main() {
    GLFWwindow* window = nullptr;
    ImGuiIO io;
    if(!initialize_gui(window, io)) return 1;

    srand(42);
    TrainingState state;

    while (!glfwWindowShouldClose(window)) {
        draw_gui(window, state);
    }
    deinitialize_gui(window);

    return 0;
}

