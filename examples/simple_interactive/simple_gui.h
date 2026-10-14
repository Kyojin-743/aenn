#pragma once

#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <atomic>
#include <optional>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#if defined(_WIN32)
    #include <backends/imgui_impl_opengl3_loader.h>
#endif
#include <backends/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include <implot.h>
#include <ImGuiFileDialog.h>

#include "simple_model.h"

namespace SimpleGui {
    using namespace AennTensor;
    using namespace SimpleModels;

    using ints = std::vector<int>;
    using floats = std::vector<float>;
    struct GuiState;

    //Gui State Managers
    bool initialize_gui(const char* title, GLFWwindow*& glfw_window);
    void deinitialize_gui(GLFWwindow*& glfw_window);

    //Gui Displaying
    std::optional<std::string> show_file_dialog(bool& open_dialog);
    void show_training_config(GuiState& gui);
    void show_training_graph(const GuiState& gui);
    void show_prediciton_image(const GuiState& gui);
    //Helper Functions
    ImTextureID GetMNISTTexture(const float* data784);
    inline void set_center_ofs(float needed_width);

    //stores all data needed to display the gui
    struct GuiState {

        //Data Managed by the Gui and forwarded out 
        float   learn_rate{0.0f};
        int     epochs{0};
        int     batch_size{0};
        int     num_test{0};
        int     num_train{0};

        ImTextureID texture_id{0ULL};

        //Signals used by background tasks
        bool start_training{false};
        bool stop_training{false};
        bool load_file{false};
        bool save_file{false};    

        //Passed in data from tasks (atomic/mutex guarded) 
        std::atomic<bool>  images_loaded{false};
        std::atomic<bool>  training{false};
        std::atomic<float> training_progress{0.0f};
        floats loss_history{0};
        floats acc_history{0};
        ints   current_guesses{0};
        std::shared_ptr<Tensor> test_images{nullptr};//shared_ptr for larger data
    };

    bool initialize_gui(const char* title, GLFWwindow*& glfw_window) {
        if (!glfwInit()) return false;
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        glfw_window = glfwCreateWindow(1280, 720, title, NULL, NULL);
        if (!glfw_window) { glfwTerminate(); return false; }
        glfwMakeContextCurrent(glfw_window);
        glfwSwapInterval(1);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGuiIO& imgui_io =  ImGui::GetIO(); (void)imgui_io;

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(glfw_window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
        return true;
    }

    void deinitialize_gui(GLFWwindow*& glfw_window) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        ImPlot::DestroyContext();

        glfwDestroyWindow(glfw_window);
        glfwTerminate();
    }

    inline void set_center_ofs(float needed_width) {
        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - needed_width) * 0.5f);
    }

    ImTextureID GetMNISTTexture(const float* data784) {
        static GLuint textureID = 0;
        const int width = 28;
        const int height = 28;

        // Initialize the texture handle on the first call
        if (textureID == 0) {
            glGenTextures(1, &textureID);
            glBindTexture(GL_TEXTURE_2D, textureID);
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            
            // Allocate storage
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        }

        // Convert [0, 1] floats to 32-bit RGBA bytes
        std::vector<uint8_t> rgbaPixels(width * height * 4);
        for (int i = 0; i < 784; ++i) {
            uint8_t val = static_cast<uint8_t>(std::clamp(data784[i], 0.0f, 1.0f) * 255.0f);
            rgbaPixels[i * 4 + 0] = val; // Red
            rgbaPixels[i * 4 + 1] = val; // Green
            rgbaPixels[i * 4 + 2] = val; // Blue
            rgbaPixels[i * 4 + 3] = 255; // Alpha
        }

        // Update texture data on the GPU
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgbaPixels.data());

        return (ImTextureID)(intptr_t)textureID;
    }

    void draw_gui (GLFWwindow*& glfw_window, GuiState& gui) {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImVec2 viewportSize = ImGui::GetMainViewport()->Size;
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
            ImGui::SetNextWindowSize(viewportSize);
            ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | 
                                    ImGuiWindowFlags_NoMove | 
                                    ImGuiWindowFlags_NoSavedSettings |
                                    ImGuiWindowFlags_MenuBar;
            //===================================================
            //Start Drawing Gui
            //===================================================
            if(ImGui::Begin("MNIST Control Panel", nullptr, flags)) {

            static bool open_dialog = false;
            static bool is_loading = false;

            if(ImGui::BeginMenuBar()) {
                if(ImGui::BeginMenu("File")){
                    if(ImGui::BeginMenu("Load")) { 
                        if(ImGui::MenuItem("Default")) {  }
                        if(ImGui::MenuItem("Current Selection")) { }
                        if(ImGui::MenuItem("From File")) { open_dialog = true; is_loading = true; }
                        ImGui::EndMenu();
                    }
                    if(ImGui::MenuItem("Save")) { open_dialog = true; is_loading = false; }
                    ImGui::EndMenu();
                }
                if(ImGui::MenuItem("Clear")) { }

                ImGui::EndMenuBar();
            }

            if(ImGui::BeginChild("Training Config", ImVec2(350,0), true)) {
                show_training_config(gui);
                ImGui::EndChild();
            }

            ImGui::SameLine();

            if(ImGui::BeginChild("Training Visualizer", ImVec2(0,0), true)) {
                show_training_graph(gui);
                show_prediciton_image(gui);
                ImGui::EndChild();
            }   

            if(auto fp = show_file_dialog(open_dialog)) {
                std::cout << fp.value() << std::endl;
            }

                ImGui::End();
            }
            //===================================================
            //End Drawing Gui
            //===================================================
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(glfw_window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(glfw_window);
    }

// void load_new_state(TrainingState& state) {
//     //quick mnist that gives 90% accuracy

//     state.epochs = 5;
//     state.batch_size = 10;
//     state.learning_rate = 0.05f;
//     state.num_train = 500; //60k max
//     state.num_test  = 50;  //10k max

//     load_with_current(state);
// }

// void load_with_current(TrainingState& state) {
//     state.is_ready = true;
//     std::cout << "Loading dataset..." << std::endl;
//     if (!load_mnist_images("../data/train-images.idx3-ubyte", state.train_images, state.num_train) ||
//         !load_mnist_labels("../data/train-labels.idx1-ubyte", state.train_labels, state.num_train)) {
//         std::cerr << "Failed to load MNIST training files from ../data/" << std::endl;
//         exit(1);
//     }
//     if(!load_mnist_images("../data/t10k-images.idx3-ubyte", state.test_images, state.num_test) ||
//        !load_mnist_labels("../data/t10k-labels.idx1-ubyte", state.test_labels, state.num_test)) {
//         std::cerr << "Failed to load MNIST test files from ../data/" << std::endl;
//         exit(1);
//     }
// }

// void clear_training_state(TrainingState& state) {
//     if(!state.is_ready) return;
//     state.stop_next = true;
//     while(state.in_progress);

//     state.epochs = 0;
//     state.batch_size = 0;
//     state.num_train = 0; 
//     state.num_test = 0;
//     state.learning_rate = 0;

//     state.is_ready = false;
//     state.progress_fraction = 0;
//     state.is_training = false;
//     state.stop_next = false;

//     state.loss_history.clear();
//     state.acc_history.clear();  
//     state.current_guesses.clear();

//     state.train_images.clear(); 
//     state.train_labels.clear();
//     state.test_images.clear();
//     state.test_labels.clear();
// }

// void save_training_state(const std::string& filepath, const TrainingState& state) {
//     std::ofstream out(filepath, std::ios::binary);
//     if (!out) return;

//     out.write(reinterpret_cast<const char*>(&state.epochs), sizeof(state.epochs));
//     out.write(reinterpret_cast<const char*>(&state.batch_size), sizeof(state.batch_size));
//     out.write(reinterpret_cast<const char*>(&state.num_train), sizeof(state.num_train));
//     out.write(reinterpret_cast<const char*>(&state.num_test), sizeof(state.num_test));
//     out.write(reinterpret_cast<const char*>(&state.learning_rate), sizeof(state.learning_rate));

//     float progress = state.progress_fraction.load();
//     out.write(reinterpret_cast<const char*>(&progress), sizeof(progress));

//     auto write_vector = [&out](const std::vector<float>& vec) {
//         size_t size = vec.size();
//         out.write(reinterpret_cast<const char*>(&size), sizeof(size));
//         if (size > 0) {
//             out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
//         }
//     };
//     write_vector(state.loss_history);
//     write_vector(state.acc_history);

//     auto write_vector_int = [&out](const std::vector<int>& vec) {
//         size_t size = vec.size();
//         out.write(reinterpret_cast<const char*>(&size), sizeof(size));
//         if (size > 0) {
//             out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(int));
//         }
//     };
//     write_vector_int(state.current_guesses);

//     auto write_tensor = [&out](const AennTensor::Tensor& tensor) {
//         bool has_data = (tensor.data != nullptr);
//         out.write(reinterpret_cast<const char*>(&has_data), sizeof(has_data));
//         if (!has_data) return;

//         size_t shape_size = tensor.shape.size();
//         out.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
//         if (shape_size > 0) {
//             out.write(reinterpret_cast<const char*>(tensor.shape.data()), shape_size * sizeof(size_t));
//         }

//         size_t strides_size = tensor.strides.size();
//         out.write(reinterpret_cast<const char*>(&strides_size), sizeof(strides_size));
//         if (strides_size > 0) {
//             out.write(reinterpret_cast<const char*>(tensor.strides.data()), strides_size * sizeof(size_t));
//         }

//         out.write(reinterpret_cast<const char*>(&tensor.offset), sizeof(tensor.offset));

//         size_t data_size = tensor.data->data.size();
//         out.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
//         if (data_size > 0) {
//             out.write(reinterpret_cast<const char*>(tensor.data->data.data()), data_size * sizeof(AennTensor::data_type));
//         }
//     };

//     write_tensor(state.train_images);
//     write_tensor(state.train_labels);
//     write_tensor(state.test_images);
//     write_tensor(state.test_labels);

//     // --- Save Model Weights ---
//     size_t num_weights = state.model.weights.size();
//     out.write(reinterpret_cast<const char*>(&num_weights), sizeof(num_weights));
//     for (const auto& w : state.model.weights) {
//         write_tensor(w);
//     }
// }

// bool load_training_state(const std::string& filepath, TrainingState& state) {
//     std::ifstream in(filepath, std::ios::binary);
//     if (!in) return false;

//     in.read(reinterpret_cast<char*>(&state.epochs), sizeof(state.epochs));
//     in.read(reinterpret_cast<char*>(&state.batch_size), sizeof(state.batch_size));
//     in.read(reinterpret_cast<char*>(&state.num_train), sizeof(state.num_train));
//     in.read(reinterpret_cast<char*>(&state.num_test), sizeof(state.num_test));
//     in.read(reinterpret_cast<char*>(&state.learning_rate), sizeof(state.learning_rate));
//     in.read(reinterpret_cast<char*>(&state.progress_fraction), sizeof(state.progress_fraction));

//     auto read_vector = [&in](std::vector<float>& vec) {
//         size_t size = 0;
//         in.read(reinterpret_cast<char*>(&size), sizeof(size));
//         vec.resize(size);
//         if (size > 0) {
//             in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
//         }
//     };
//     read_vector(state.loss_history);
//     read_vector(state.acc_history);

//     auto read_vector_int = [&in](std::vector<int>& vec) {
//         size_t size = 0;
//         in.read(reinterpret_cast<char*>(&size), sizeof(size));
//         vec.resize(size);
//         if (size > 0) {
//             in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(int));
//         }
//     };
//     read_vector_int(state.current_guesses);

//     auto read_tensor = [&in](AennTensor::Tensor& tensor) {
//         bool has_data = false;
//         in.read(reinterpret_cast<char*>(&has_data), sizeof(has_data));
//         if (!has_data) {
//             tensor = AennTensor::Tensor();
//             return;
//         }

//         size_t shape_size = 0;
//         in.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
//         tensor.shape.resize(shape_size);
//         if (shape_size > 0) {
//             in.read(reinterpret_cast<char*>(tensor.shape.data()), shape_size * sizeof(size_t));
//         }

//         size_t strides_size = 0;
//         in.read(reinterpret_cast<char*>(&strides_size), sizeof(strides_size));
//         tensor.strides.resize(strides_size);
//         if (strides_size > 0) {
//             in.read(reinterpret_cast<char*>(tensor.strides.data()), strides_size * sizeof(size_t));
//         }

//         in.read(reinterpret_cast<char*>(&tensor.offset), sizeof(tensor.offset));

//         size_t data_size = 0;
//         in.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        
//         tensor.data = std::make_shared<AennTensor::Data>(data_size);
//         if (data_size > 0) {
//             in.read(reinterpret_cast<char*>(tensor.data->data.data()), data_size * sizeof(AennTensor::data_type));
//         }
//     };

//     read_tensor(state.train_images);
//     read_tensor(state.train_labels);
//     read_tensor(state.test_images);
//     read_tensor(state.test_labels);

//     // --- Load Model Weights ---
//     size_t num_weights = 0;
//     in.read(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
//     state.model.weights.resize(num_weights);
//     for (size_t i = 0; i < num_weights; ++i) {
//         read_tensor(state.model.weights[i]);
//     }

//     // Re-initialize gradient buffers to match the loaded weight dimensions
//     state.model.grads.resize(num_weights);
//     for (size_t i = 0; i < num_weights; ++i) {
//         state.model.grads[i] = Tensor::Zeroes(state.model.weights[i].shape);
//     }

//     state.is_ready.store(true);
//     return true;
// }

// void evaluate_and_store(TrainingState& state) {
//     std::cout << "\nEvaluating model performance..." << std::endl;
//     Tensor test_logits = state.model.forward(state.test_images);
//     Tensor test_probs = softmax(test_logits);

//     size_t correct = 0;
//     state.current_guesses.clear();
//     for (size_t i = 0; i < state.num_test; ++i) {
//         size_t pred = 0;
//         float max_p = test_probs[i * 10];
//         for (size_t c = 1; c < 10; ++c) {
//             if (test_probs[i * 10 + c] > max_p) {
//                 max_p = test_probs[i * 10 + c];
//                 pred = c;
//             }
//         }

//         size_t actual = 0;
//         for (size_t c = 0; c < 10; ++c) {
//             if (state.test_labels[i * 10 + c] > 0.5f) {
//                 actual = c;
//                 break;
//             }
//         }

//         // std::cout << "Expected " << actual << ". Got " << pred << "." << std::endl;
//         if (pred == actual) correct++;
//         state.current_guesses.push_back(pred);
//     }

//     float accuracy = (100.0f * correct / state.num_test);
//     std::cout << "Test Accuracy: " << accuracy << "% (" 
//               << correct << "/" << state.num_test << ")" << std::endl;
//     state.acc_history.push_back(accuracy);
// }

// void training_loop(TrainingState& state) {
//     std::cout << "Starting Training (" << state.epochs << " Epochs, Batch Size " << state.batch_size << ")..." << std::endl;

//     size_t epoch = 0;
//     while(state.is_training) {
//         if(epoch == state.epochs) {
//             state.is_training = false;
//             break;
//         }
        
//         float epoch_loss = 0.0f;
//         size_t batches = 0;
//         float batch_progress = 0;
//         state.in_progress = true;
//         for (size_t i = 0; i < state.num_train; i += state.batch_size) {
//             if(state.stop_next) break;
//             size_t current_batch = std::min((unsigned long)state.batch_size, state.num_train - i);
            
//             batch_progress = (float)i/state.num_train;
//             state.progress_fraction = (epoch + batch_progress) / state.epochs;

//             auto start = i;
//             auto stop  = i + current_batch;

//             Tensor x_batch = state.train_images[{{start, stop}, {-1,-1}}];
//             Tensor y_batch = state.train_labels[{{start, stop}, {-1,-1}}];

//             state.model.zero_grad();
//             Tensor logits = state.model.forward(x_batch);
//             Tensor loss = cross_entropy_loss(logits, y_batch);

//             backward_step(state.model, logits, y_batch);
//             state.model.update_weights(state.learning_rate);

//             epoch_loss += loss[0];
//             batches++;
//         }
//         state.in_progress = false;
//         if(state.stop_next) {
//             state.is_training = false;
//             return;
//         }

//         state.loss_history.push_back(epoch_loss);
//         evaluate_and_store(state);

//         std::cout << "Epoch " << (epoch + 1) << "/" << state.epochs << " - Loss: " << (epoch_loss / batches) << std::endl;
//         epoch++;
//     }
// }

// void start_training(TrainingState& state) {
//     if(state.epochs == 0) return;

//     state.is_training = true;
//     state.acc_history.clear();
//     state.loss_history.clear();
//     state.progress_fraction = 0;

//     if(!state.is_ready) load_with_current(state);
    
//     std::thread training_thread([&state] {
//         training_loop(state);
//     });
//     training_thread.detach();
// }

// void stop_training(TrainingState& state) {
//     state.stop_next = true;
//     state.in_progress = false;
// }

    std::optional<std::string> show_file_dialog(bool& open_dialog) {
        if (open_dialog) {
            IGFD::FileDialogConfig config;
            config.path = ".";
            ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".tsrd", config);
            open_dialog = false; // Reset trigger so it only opens once
        }

        // Display must be called EVERY frame to keep the popup alive and interactive
        if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                return filePathName;
            }
            ImGuiFileDialog::Instance()->Close();
        }
        return std::nullopt;
    }

    void show_training_config(GuiState& gui) {
        static const char* button_text = gui.training? "Training" : "Train";
        static const char* progress_text = "Training Progress";

        ImGui::DragInt("Epochs", &gui.epochs, .2, 1, 50);
        ImGui::DragFloat("Learn Rate", &gui.learn_rate, .001, 0.001, .5);
        ImGui::DragInt("Batch Size", &gui.batch_size, 1, 1, gui.num_train);
        ImGui::DragInt("Train Size", &gui.num_train, 100, 1, 60000);
        ImGui::DragInt("Test Size", &gui.num_test, 20, 1, 10000);
        ImGui::ProgressBar(gui.training_progress, ImVec2(-1.0f, 0.0f), progress_text);

        float buttonWidth = ImGui::CalcTextSize(button_text).x; 
        set_center_ofs(buttonWidth);
        if (ImGui::Button(button_text, ImVec2(buttonWidth + 10, 0.0f))) {
            if(!gui.training) {
            } else {
            }
        }
    }

    void show_training_graph(const GuiState& gui) {
        const size_t l_hist = gui.loss_history.size();
        const size_t a_hist = gui.acc_history.size();

        floats xs1(l_hist), xs2(a_hist);
        for(size_t i(0); i < l_hist; ++i) { xs1[i] = i; }
        for(size_t i(0); i < a_hist; ++i) { xs2[i] = i; }

        if (ImPlot::BeginPlot("Line Plots")) {
            ImPlot::SetupAxes("Epochs","Loss", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxesLimits(0, gui.epochs, 0, 10);
            ImPlot::SetupAxis(ImAxis_Y2, "Accuracy (%)", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_Y2, 0, 120);

            //plot loss
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
            ImPlot::PlotLine("Loss", xs1.data(), gui.loss_history.data(), l_hist);

            //plot accuracy
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("Accuracy", xs2.data(), gui.acc_history.data(), a_hist);

            ImPlot::EndPlot();
        }
    }

    void show_prediciton_image(const GuiState& gui) {
        if (!gui.images_loaded || gui.test_images->shape.size() != 2) {
            //show progress bar of the images loading?
            return;
        }
        
        static int image = 0;
        ImGui::DragInt("Test Image #", &image, 1, 0, gui.num_test-1);
        int guess;
        if(image >= gui.current_guesses.size()) guess = -1;
        else guess = (gui.current_guesses)[image];
        std::string guess_str = (guess == -1)? "Not Guessed Yet" : std::to_string(guess);
        ImGui::Text("Current Guess: %s", guess_str.c_str());

        Tensor img = gui.test_images->slice({{image, image}, {-1,-1}});
        auto raw = img.data->data.data() + img.offset;
        auto image_id = GetMNISTTexture(raw);
        ImGui::Image(image_id, {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().x});
    }
}