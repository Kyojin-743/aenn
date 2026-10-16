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
        std::string file;

        //Passed in data from tasks (atomic/mutex guarded) 
        mutable std::mutex lock;
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

    void gui_step (GLFWwindow*& glfw_window, GuiState& gui) {
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
                gui.file = fp.value();
                if(is_loading) {
                    gui.load_file = true;
                } else {
                    gui.save_file = true;
                }
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
                gui.start_training = true;
            } else {
                gui.stop_training = true;
            }
        }
    }

    void show_training_graph(const GuiState& gui) {
        floats local_loss;
        floats local_acc;
        int local_epochs = gui.epochs;

        {
            // std::lock_guard<std::mutex> lock(gui.lock);
            local_loss = gui.loss_history;
            local_acc = gui.acc_history;
        }

        const size_t l_hist = local_loss.size();
        const size_t a_hist = local_acc.size();

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
            ImPlot::PlotLine("Loss", xs1.data(), local_loss.data(), l_hist);

            //plot accuracy
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("Accuracy", xs2.data(), local_acc.data(), a_hist);

            ImPlot::EndPlot();
        }
    }

    void show_prediciton_image(const GuiState& gui) {
        if (!gui.images_loaded) {
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