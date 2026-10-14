#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <queue>

#include "simple_model.h"
#include "simple_gui.h"

using namespace AennTensor;
using namespace SimpleModels;
using namespace SimpleGui;

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> stop{false};

public:
    ThreadPool(size_t threads) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->cv.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        cv.notify_one();
    }

    ~ThreadPool() {
        stop = true;
        cv.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }
};

static const auto thread_count = (20 > std::thread::hardware_concurrency()) ? std::thread::hardware_concurrency() : 20; 

class App {
public:

    //Data held for the lifetime of the application
    GLFWwindow* window;
    GuiState gui_state; //what gets displayed
    SimpleModel model;
    Tensor train_images, test_images;
    Tensor train_labels, test_labels;

    //Buffers to swap out when displaying
    bool can_use_a{0};
    Tensor images_a, images_b;

    //Synchronization
    ThreadPool pool{thread_count};
    std::mutex gui;
    std::atomic<bool> worker_active{false};
    std::atomic<bool> worker_stop{false};

    App() : model({
        {28 * 28, std::make_shared<Linear>()},
        {128,     std::make_shared<ReLU>()},
        {64,      std::make_shared<ReLU>()},
        {10,      std::make_shared<Linear>()}
        }) {

        initialize_gui("Simple MNIST Demo", window);
        
        //initialize gui to point to one of the buffers
        gui_state.test_images = std::make_shared<Tensor>(images_b);

        pool.enqueue([this]() {
            std::vector<std::thread> threads;
            threads.emplace_back([this]() {
                if(!load_mnist_images("../data/MNIST/train-images-idx3-ubyte", train_images, 1000)) {
                    //do something, idk
                }
            });
            
            threads.emplace_back([this]() {
                if(!load_mnist_labels("../data/MNIST/train-labels-idx1-ubyte", train_labels, 1000)) {
                    //do something, idk
                }
            });
            
            threads.emplace_back([this]() {
                if(!load_mnist_images("../data/MNIST/t10k-images-idx3-ubyte", test_images, 1000)) {
                    //do something, idk
                }
            });
            
            threads.emplace_back([this]() {
                if(!load_mnist_labels("../data/MNIST/t10k-labels-idx1-ubyte", test_labels, 1000)) {
                    //do something, idk
                }
            });

            //waits for them all to finish
            for(auto& t : threads) {
                if (t.joinable()) 
                    t.join();
            }

            gui_state.images_loaded = true;
        });

    }

    ~App() {
        //signal worker to stop
        //wait for it
        deinitialize_gui(window);
    }

    void run() {
        while(!glfwWindowShouldClose(window)) {
            draw_gui(window, gui_state);
        }
    }

};

int main() {
    GLFWwindow* window = nullptr;
    App app = App();
    app.run();
    return 0;
}

