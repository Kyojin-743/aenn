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
public:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> stop{false};

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

static const auto thread_count = std::thread::hardware_concurrency(); 

class App {
public:

    //Data held for the lifetime of the application
    GLFWwindow* window;
    GuiState gui_state; //what gets displayed
    SimpleModel model;
    std::shared_ptr<Tensor> train_images, test_images;
    std::shared_ptr<Tensor> train_labels, test_labels;

    //Buffers to swap out when displaying
    bool can_use_a{0};
    std::shared_ptr<Tensor> images_a, images_b;

    //Synchronization
    ThreadPool pool{thread_count};

    App() : model({
        {28 * 28, std::make_shared<Linear>()},
        {128,     std::make_shared<ReLU>()},
        {64,      std::make_shared<ReLU>()},
        {10,      std::make_shared<Linear>()}
        }) {

        initialize_gui("Simple MNIST Demo", window);
        
        //initialize gui to point to one of the buffers
        gui_state.test_images = images_b;

        //Default configuration for training (decent results)
        gui_state.epochs        = 5;
        gui_state.learn_rate    = .02;
        gui_state.batch_size    = 20;
        gui_state.num_test      = 100;
        gui_state.num_train     = 1000;

        pool.enqueue([this]() {
            train_images      = std::make_shared<Tensor>();
            train_labels      = std::make_shared<Tensor>();
            auto loaded_test_images = std::make_shared<Tensor>();
            auto loaded_test_labels = std::make_shared<Tensor>();
            
            bool ok = true;
            ok &= load_mnist_images("../data/MNIST/train-images-idx3-ubyte", *train_images, 10000);
            ok &= load_mnist_labels("../data/MNIST/train-labels-idx1-ubyte", *train_labels, 10000);
            ok &= load_mnist_images("../data/MNIST/t10k-images-idx3-ubyte", *loaded_test_images, 10000);
            ok &= load_mnist_labels("../data/MNIST/t10k-labels-idx1-ubyte", *loaded_test_labels, 10000);

            ASSERT(ok) << "Problem Loading MNIST Data";

            test_images = loaded_test_images;
            test_labels = loaded_test_labels;

            gui_state.test_images = test_images; 
            gui_state.images_loaded.store(true);

            printf("Done Loading Test Dataset!\n");
        });
    }

    ~App() {
        pool.stop = true;
        deinitialize_gui(window);
    }

    void run() {
        while(!glfwWindowShouldClose(window)) {
            gui_step(window, gui_state);

            if(gui_state.start_training) {
                //reset statistics
                gui_state.start_training = false;
                printf("Started Training!\n");
                _start_training();
            }

            if(gui_state.stop_training) {
                gui_state.stop_training = false;
                _stop_training();
            }

            if(gui_state.load_file) {
                gui_state.load_file = false;
                //keep the image viewer from reading the tensor
                gui_state.images_loaded = false;
                pool.enqueue([this](){
                    if(!_load_training_data(gui_state.file)) {
                        //dont signal that the images were loaded
                        return;
                    }
                    gui_state.images_loaded = true;
                });
                
            }

            if(gui_state.save_file) {
                gui_state.save_file = false;
                pool.enqueue([this](){
                    if(!_save_training_data(gui_state.file)) {

                    }
                });
            }    
        }
    }

    bool _load_training_data(std::string file_path) {
        std::ifstream in_file(file_path, std::ios::binary);
        if (!in_file) return false;
        in_file.read(reinterpret_cast<char*>(&gui_state.epochs),        sizeof(gui_state.epochs));
        in_file.read(reinterpret_cast<char*>(&gui_state.batch_size),    sizeof(gui_state.batch_size));
        in_file.read(reinterpret_cast<char*>(&gui_state.num_train),     sizeof(gui_state.num_train));
        in_file.read(reinterpret_cast<char*>(&gui_state.num_test),      sizeof(gui_state.num_test));
        in_file.read(reinterpret_cast<char*>(&gui_state.learn_rate),    sizeof(gui_state.learn_rate));
        in_file.read(reinterpret_cast<char*>(&gui_state.images_loaded), sizeof(gui_state.images_loaded));
        in_file.read(reinterpret_cast<char*>(&gui_state.training),      sizeof(gui_state.training));
        in_file.read(reinterpret_cast<char*>(&gui_state.training_progress), sizeof(gui_state.training_progress));

        auto read_vector = [&in_file](std::vector<float>& vec) {
            size_t size = 0;
            in_file.read(reinterpret_cast<char*>(&size), sizeof(size));
            vec.resize(size);
            if (size > 0) {
                in_file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
            }
        };

        auto read_vector_int = [&in_file](std::vector<int>& vec) {
            size_t size = 0;
            in_file.read(reinterpret_cast<char*>(&size), sizeof(size));
            vec.resize(size);
            if (size > 0) {
                in_file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(int));
            }
        };
        
        auto read_tensor = [&in_file](AennTensor::Tensor& tensor) {
            bool has_data = false;
            in_file.read(reinterpret_cast<char*>(&has_data), sizeof(has_data));
            if (!has_data) {
                tensor = AennTensor::Tensor();
                return;
            }

            size_t shape_size = 0;
            in_file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
            tensor.shape.resize(shape_size);
            if (shape_size > 0) {
                in_file.read(reinterpret_cast<char*>(tensor.shape.data()), shape_size * sizeof(size_t));
            }

            size_t strides_size = 0;
            in_file.read(reinterpret_cast<char*>(&strides_size), sizeof(strides_size));
            tensor.strides.resize(strides_size);
            if (strides_size > 0) {
                in_file.read(reinterpret_cast<char*>(tensor.strides.data()), strides_size * sizeof(size_t));
            }

            in_file.read(reinterpret_cast<char*>(&tensor.offset), sizeof(tensor.offset));

            size_t data_size = 0;
            in_file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
            
            tensor.data = std::make_shared<AennTensor::Data>(data_size);
            if (data_size > 0) {
                in_file.read(reinterpret_cast<char*>(tensor.data->data.data()), data_size * sizeof(AennTensor::data_type));
            }
        };

        read_vector(gui_state.loss_history);
        read_vector(gui_state.acc_history);
        read_vector_int(gui_state.current_guesses);

        size_t num_weights = 0;
        in_file.read(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
        model.weights.resize(num_weights);
        for (size_t i = 0; i < num_weights; ++i) {
            read_tensor(model.weights[i]);
        }
        return true;
    }

    bool _save_training_data(std::string file_path) {
        std::ofstream out_file(file_path, std::ios::binary);
        if (!out_file) return false;
        out_file.write(reinterpret_cast<const char*>(&gui_state.epochs),        sizeof(gui_state.epochs));
        out_file.write(reinterpret_cast<const char*>(&gui_state.batch_size),    sizeof(gui_state.batch_size));
        out_file.write(reinterpret_cast<const char*>(&gui_state.num_train),     sizeof(gui_state.num_train));
        out_file.write(reinterpret_cast<const char*>(&gui_state.num_test),      sizeof(gui_state.num_test));
        out_file.write(reinterpret_cast<const char*>(&gui_state.learn_rate),    sizeof(gui_state.learn_rate));
        out_file.write(reinterpret_cast<const char*>(&gui_state.images_loaded), sizeof(gui_state.images_loaded));
        out_file.write(reinterpret_cast<const char*>(&gui_state.training),      sizeof(gui_state.training));
        out_file.write(reinterpret_cast<const char*>(&gui_state.training_progress), sizeof(gui_state.training_progress));

        auto write_vector = [&out_file](const std::vector<float>& vec) {
            size_t size = vec.size();
            out_file.write(reinterpret_cast<const char*>(&size), sizeof(size));
            if (size > 0) {
                out_file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
            }
        };

        auto write_vector_int = [&out_file](const std::vector<int>& vec) {
            size_t size = vec.size();
            out_file.write(reinterpret_cast<const char*>(&size), sizeof(size));
            if (size > 0) {
                out_file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(int));
            }
        };

        auto write_tensor = [&out_file](const AennTensor::Tensor& tensor) {
            bool has_data = (tensor.data != nullptr);
            out_file.write(reinterpret_cast<const char*>(&has_data), sizeof(has_data));
            if (!has_data) return;

            size_t shape_size = tensor.shape.size();
            out_file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
            if (shape_size > 0) {
                out_file.write(reinterpret_cast<const char*>(tensor.shape.data()), shape_size * sizeof(size_t));
            }

            size_t strides_size = tensor.strides.size();
            out_file.write(reinterpret_cast<const char*>(&strides_size), sizeof(strides_size));
            if (strides_size > 0) {
                out_file.write(reinterpret_cast<const char*>(tensor.strides.data()), strides_size * sizeof(size_t));
            }

            out_file.write(reinterpret_cast<const char*>(&tensor.offset), sizeof(tensor.offset));

            size_t data_size = tensor.data->data.size();
            out_file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
            if (data_size > 0) {
                out_file.write(reinterpret_cast<const char*>(tensor.data->data.data()), data_size * sizeof(AennTensor::data_type));
            }
        };

        write_vector(gui_state.loss_history);
        write_vector(gui_state.acc_history);
        write_vector_int(gui_state.current_guesses);

        size_t num_weights = model.weights.size();
        out_file.write(reinterpret_cast<const char*>(&num_weights), sizeof(num_weights));
        for (const auto& w : model.weights) {
            write_tensor(w);
        }
        return true;
    }

    void _start_training() {
        gui_state.training = true;
        gui_state.training_progress = 0;
        {
            std::lock_guard<std::mutex> lock(gui_state.lock);
            gui_state.acc_history.clear();
            gui_state.loss_history.clear();
        }

        pool.enqueue([this]() {
            //create copy of configuration
            size_t epochs = gui_state.epochs;
            size_t batch_size = gui_state.batch_size;
            size_t num_train = gui_state.num_train;
            size_t num_test = gui_state.num_test;
            float learn_rate = gui_state.learn_rate;

            //create smaller subset 
            auto train_images_sub = train_images->slice(Ranges{Range{0, static_cast<ptrdiff_t>(num_train)}, Range{-1, -1}}).contiguous();
            auto train_labels_sub = train_labels->slice(Ranges{Range{0, static_cast<ptrdiff_t>(num_train)}, Range{-1, -1}}).contiguous();     
            auto test_images_sub = train_images->slice(Ranges{Range{0, static_cast<ptrdiff_t>(num_test)}, Range{-1, -1}}).contiguous();
            auto test_labels_sub = train_labels->slice(Ranges{Range{0, static_cast<ptrdiff_t>(num_test)}, Range{-1, -1}}).contiguous();            

            //main training loop
            for(size_t epoch = 0; epoch < epochs ; ++epoch) {
                printf("Training Epoch %d\n", epoch, epochs);
                if(!gui_state.training || pool.stop) return;
                float epoch_loss = train_one_epoch(model, train_images_sub, train_labels_sub, batch_size, learn_rate);
                printf("Evaluating the model\n");
                auto [epoch_accuracy, guesses] = evaluate_model(model, test_images_sub, test_labels_sub);
                for(const auto& g : guesses) {
                    auto& [ind, pred, exp] = g;

                }

                //update gui (either atomically, or under a mutex)
                gui_state.training_progress = (float)epoch / epochs;
                {
                    std::lock_guard<std::mutex> lock(gui_state.lock);
                    gui_state.loss_history.push_back(epoch_loss);
                    gui_state.acc_history.push_back(epoch_accuracy);
                }
            }
            printf("Ended Training!\n");
            gui_state.training = false;
        });
    }

    void _stop_training() {
        gui_state.training = false;
    }
};


int main() {
    GLFWwindow* window = nullptr;
    App app = App();
    app.run();
    return 0;
}

