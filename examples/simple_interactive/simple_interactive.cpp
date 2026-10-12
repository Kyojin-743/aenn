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

#include <smartassert.h>
#include <tensor.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include <implot.h>
#include <ImGuiFileDialog.h>

using namespace AennTensor;

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    
    // Enables ANSI escape code processing on modern Windows consoles
    static void enable_windows_ansi(void) {
        static int enabled = 0;
        if (!enabled) {
            HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
            if (hOut != INVALID_HANDLE_VALUE) {
                DWORD dwMode = 0;
                if (GetConsoleMode(hOut, &dwMode)) {
                    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                    SetConsoleMode(hOut, dwMode);
                }
            }
            enabled = 1;
        }
    }
#else
    static inline void enable_windows_ansi(void) {}
    #include <stdarg.h>
#endif

int pip(const char* format, ...) {
    enable_windows_ansi();

    printf("\r\033[2K");
    va_list args;
    va_start(args, format);
    int written = vprintf(format, args);
    va_end(args);

    fflush(stdout);
    return written;
}

Tensor matmul_thr(const Tensor& a, const Tensor& b) {
    ASSERT(a.shape.size() == 2) << "Expected Degree 2. Got: " << a.shape.size();
    ASSERT(b.shape.size() == 2) << "Expected Degree 2. Got: " << b.shape.size();
    ASSERT(a.shape[1] == b.shape[0]) << "Inner dimensions must match.";

    size_t M(a.shape[0]);
    size_t K(a.shape[1]);
    size_t N(b.shape[1]);

    Tensor t({M, N}, static_cast<data_type>(0));

    // Determine number of threads based on hardware concurrency
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = num_threads == 0 ? 4 : num_threads;
    
    // Don't spawn more threads than rows
    if (num_threads > M) num_threads = M;

    std::vector<std::thread> threads;
    size_t rows_per_thread = M / num_threads;

    for (unsigned int t_idx = 0; t_idx < num_threads; ++t_idx) {
        size_t start_row = t_idx * rows_per_thread;
        size_t end_row = (t_idx == num_threads - 1) ? M : start_row + rows_per_thread;

        threads.emplace_back([start_row, end_row, M, K, N, &a, &b, &t]() {
            for (size_t i = start_row; i < end_row; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    data_type val_a = a[{i, k}];
                    for (size_t j = 0; j < N; ++j) {
                        t[{i, j}] += val_a * b[{k, j}];
                    }
                }
            }
        });
    }

    // Join threads
    for (auto& th : threads) {
        th.join();
    }

    return t;
}

class Activation {
public:
    virtual ~Activation() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::unique_ptr<Activation> clone() const = 0;
};

class Linear : public Activation {
public:
    Tensor forward(const Tensor& input) override { return input; }
    Tensor backward(const Tensor& grad_output) override { return grad_output; }
    std::unique_ptr<Activation> clone() const override { return std::make_unique<Linear>(*this); }
};

class ReLU : public Activation {
private:
    Tensor cached_input;

public:
    Tensor forward(const Tensor& input) override {
        this->cached_input = input;
        Tensor result(input.shape);
        size_t total = numel(input.shape);
        for (size_t i = 0; i < total; ++i) {
            result[i] = std::max(0.0f, input[i]);
        }
        return result;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(grad_output.shape);
        size_t total = numel(grad_output.shape);
        for (size_t i = 0; i < total; ++i) {
            grad_input[i] = (cached_input[i] > 0.0f) ? grad_output[i] : 0.0f;
        }
        return grad_input;
    }

    std::unique_ptr<Activation> clone() const override { return std::make_unique<ReLU>(*this); }
};

class Sigmoid : public Activation {
private:
    Tensor cached_output;

public:
    Tensor forward(const Tensor& input) override {
        Tensor result(input.shape);
        size_t total = numel(input.shape);
        for (size_t i = 0; i < total; ++i) {
            result[i] = 1.0f / (1.0f + std::exp(-input[i]));
        }
        this->cached_output = result;
        return result;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(grad_output.shape);
        size_t total = numel(grad_output.shape);
        for (size_t i = 0; i < total; ++i) {
            float y = cached_output[i];
            grad_input[i] = grad_output[i] * (y * (1.0f - y));
        }
        return grad_input;
    }

    std::unique_ptr<Activation> clone() const override { return std::make_unique<Sigmoid>(*this); }
};

Tensor append_bias_column(const Tensor& input) {
    size_t batch_size = input.shape[0];
    size_t features   = input.shape[1];

    Tensor augmented({batch_size, features + 1});
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < features; ++f) {
            augmented[{b, f}] = input[{b, f}];
        }
        augmented[{b, features}] = 1.0f;
    }
    return augmented;
}

// Numerically stable Softmax across axis 1
Tensor softmax(const Tensor& logits) {
    size_t batch_size = logits.shape[0];
    size_t num_classes = logits.shape[1];

    Tensor probs({batch_size, num_classes});

    for (size_t b = 0; b < batch_size; ++b) {
        float max_val = logits[b * num_classes];
        for (size_t c = 1; c < num_classes; ++c) {
            max_val = std::max(max_val, logits[b * num_classes + c]);
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            float exp_val = std::exp(logits[b * num_classes + c] - max_val);
            probs[b * num_classes + c] = exp_val;
            sum_exp += exp_val;
        }

        for (size_t c = 0; c < num_classes; ++c) {
            probs[b * num_classes + c] /= sum_exp;
        }
    }
    return probs;
}

// ============================================================================
// Simple Model
// ============================================================================

class SimpleModel {
public:
    using LayerSpec = std::tuple<size_t, std::shared_ptr<Activation>>;
    using Description = std::vector<LayerSpec>;

    std::vector<std::shared_ptr<Activation>> activations;
    std::vector<Tensor> weights; // Shape: [inp_size + 1, outp_size] (includes bias row)
    std::vector<Tensor> grads;
    std::vector<Tensor> pre_activated;
    std::vector<Tensor> activated;

    SimpleModel(Description desc) {
        for (size_t i = 1; i < desc.size(); ++i) {
            size_t inp_size  = std::get<0>(desc[i - 1]);
            size_t outp_size = std::get<0>(desc[i]);

            // He/Xavier Normal Initialization 
            float stddev = std::sqrt(2.0f / static_cast<float>(inp_size));
            Tensor weight = Tensor::Random({inp_size + 1, outp_size}, 0.0f, stddev, Distribution::Normal);
            Tensor grad   = Tensor::Zeroes({inp_size + 1, outp_size});

            weights.push_back(weight);
            grads.push_back(grad);
            activations.push_back(std::get<1>(desc[i]));
        }
    }

    Tensor forward(const Tensor& input) {
        pre_activated.clear();
        activated.clear();

        Tensor current_input = input;
        activated.push_back(current_input); // activated[0] is the input batch

        for (size_t i = 0; i < weights.size(); ++i) {
            Tensor aug_input = append_bias_column(current_input);
            Tensor pre_act = matmul_thr(aug_input, (weights[i]));

            pre_activated.push_back(pre_act);
            current_input = activations[i]->forward(pre_act);
            activated.push_back(current_input);
        }

        return current_input;
    }

    void zero_grad() {
        for (auto& g : grads) {
            size_t total = numel(g.shape);
            for (size_t i = 0; i < total; ++i) g[i] = 0.0f;
        }
    }

    void update_weights(float lr) {
        for (size_t i = 0; i < weights.size(); ++i) {
            size_t total = numel(weights[i].shape);
            for (size_t k = 0; k < total; ++k) {
                weights[i][k] -= lr * grads[i][k];
            }
        }
    }
};

// Combined Softmax Cross-Entropy Loss
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& labels) {
    Tensor probs = softmax(logits);
    size_t batch_size = logits.shape[0];
    size_t num_classes = logits.shape[1];

    float total_loss = 0.0f;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < num_classes; ++c) {
            if (labels[b * num_classes + c] > 0.5f) {
                float prob = std::max(probs[b * num_classes + c], 1e-7f);
                total_loss -= std::log(prob);
            }
        }
    }

    Tensor loss({1});
    loss[0] = total_loss / static_cast<float>(batch_size);
    return loss;
}

void backward_step(SimpleModel& model, const Tensor& logits, const Tensor& labels) {
    size_t batch_size = logits.shape[0];
    Tensor probs = softmax(logits);

    // Initial gradient w.r.t logits: (Softmax - OneHot) / BatchSize
    Tensor dL_dZ(logits.shape);
    size_t total_logits = numel(logits.shape);
    for (size_t i = 0; i < total_logits; ++i) {
        dL_dZ[i] = (probs[i] - labels[i]) / static_cast<float>(batch_size);
    }

    Tensor dL_dA = dL_dZ;
    int num_layers = static_cast<int>(model.weights.size());

    for (int l = num_layers - 1; l >= 0; --l) {
        Tensor dL_dPre = (l == num_layers - 1) ? dL_dA : model.activations[l]->backward(dL_dA);

        // Previous activation with appended 1.0 for bias
        Tensor prev_act_aug = append_bias_column(model.activated[l]);

        // Weight gradient: [InpSize + 1, Batch] x [Batch, OutpSize] = [InpSize + 1, OutpSize]
        Tensor dW = matmul_thr(prev_act_aug.transpose(), dL_dPre);
        
        // Accumulate gradient
        size_t total_w = numel(model.grads[l].shape);
        for (size_t k = 0; k < total_w; ++k) {
            model.grads[l][k] += dW[k];
        }

        // Backpropagate to previous layer's activation gradient (excluding bias column)
        if (l > 0) {
            Tensor dL_dAug = matmul_thr(dL_dPre, model.weights[l].transpose());
            size_t prev_features = model.activated[l].shape[1];
            dL_dA = Tensor({batch_size, prev_features});

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t f = 0; f < prev_features; ++f) {
                    dL_dA[b * prev_features + f] = dL_dAug[b * (prev_features + 1) + f];
                }
            }
        }
    }
}

uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xFF000000) | ((val << 8) & 0x00FF0000) |
           ((val >> 8) & 0x0000FF00)  | ((val >> 24) & 0x000000FF);
}

bool load_mnist_images(const std::string& path, Tensor& images, size_t max_samples = 0) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t magic = 0, num_items = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_items), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    num_items = swap_endian(num_items);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (max_samples > 0 && max_samples < num_items) num_items = max_samples;
    size_t img_size = rows * cols;
    size_t total_bytes = num_items * img_size;
    std::vector<uint8_t> file_buffer(total_bytes);
    file.read(reinterpret_cast<char*>(file_buffer.data()), total_bytes);
    if (!file) return false;
    images = Tensor({num_items, img_size});

    size_t thread_count = std::thread::hardware_concurrency();
    if (thread_count == 0) thread_count = 4;
    if (thread_count > num_items) thread_count = num_items;

    size_t items_per_thread = num_items / thread_count;
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    // Launch threads to normalize and populate the tensor in parallel
    for (size_t t = 0; t < thread_count; ++t) {
        size_t start_idx = t * items_per_thread;
        size_t end_idx = (t == thread_count - 1) ? num_items : start_idx + items_per_thread;

        threads.emplace_back([start_idx, end_idx, img_size, &file_buffer, &images]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                size_t offset = i * img_size;
                for (size_t j = 0; j < img_size; ++j) {
                    images[offset + j] = static_cast<float>(file_buffer[offset + j]) / 255.0f;
                }
            }
        });
    }

    // Join all worker threads
    for (auto& th : threads) {
        th.join();
    }

    return true;
}

bool load_mnist_labels(const std::string& path, Tensor& labels, size_t max_samples = 0) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t magic = 0, num_items = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_items), 4);

    num_items = swap_endian(num_items);

    if (max_samples > 0 && max_samples < num_items) num_items = max_samples;
    std::vector<uint8_t> buffer(num_items);
    file.read(reinterpret_cast<char*>(buffer.data()), num_items);
    if(!file) return false;
    labels = Tensor::Zeroes({num_items, 10});


    size_t thread_count = std::thread::hardware_concurrency();
    if (thread_count == 0) thread_count = 4;
    if (thread_count > num_items) thread_count = num_items;

    size_t items_per_thread = num_items / thread_count;
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (size_t t = 0; t < thread_count; ++t) {
        size_t start_idx = t * items_per_thread;
        size_t end_idx = (t == thread_count - 1) ? num_items : start_idx + items_per_thread;

        threads.emplace_back([start_idx, end_idx, &buffer, &labels]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                uint8_t label = buffer[i];
                if (label < 10) {
                    labels[i * 10 + label] = 1.0f;
                }
            }
        });
    }

    // Join all worker threads
    for (auto& th : threads) {
        th.join();
    }

    for (size_t i = 0; i < num_items; ++i) {
    }
    return true;
}

int randint(int min, int max) {
    return ((rand() / RAND_MAX)* max-min) + min;
}

ImTextureID GetMNISTTexture(const float* data784) {
    static GLuint textureID = 0;
    const int width = 28;
    const int height = 28;

    // static int  call_count = 0;
    // printf("\n===start of new MnistTexture===\n");
    // for(int i = 0; i < 28; i++){
    //     for(int j = 0; j < 28; j++) {
    //         printf("%5.2f ", data784[i * 28 + j]);
    //     }
    //     printf("\n");
    // }
    // if(call_count ++ >= 10) exit(0);

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

void set_center_ofs(float needed_width) {
    ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - needed_width) * 0.5f);
}

struct TrainingState{
    int epochs {0};
    int batch_size{0};
    int num_train{0}; 
    int num_test{0};
    float learning_rate{0};

    std::atomic<bool> is_training{false};
    std::atomic<bool> is_ready{false};
    std::atomic<int> current_epoch{0};
    std::atomic<float> current_loss{0.0f};
    std::atomic<float> current_accuracy{0.0f};
    std::atomic<float> progress_fraction{0.0f};

    std::vector<float> loss_history;
    std::vector<float> acc_history;    
    Tensor train_images, train_labels, test_images, test_labels;
};

// ============================================================================
//  Main Routine
// ============================================================================

void main_window(TrainingState& state);

int main() {
    if (!glfwInit()) return 1;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "MNIST Live Training GUI", NULL, NULL);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    TrainingState state;
    srand(42);

    // // Build neural network model architecture
    // SimpleModel model({
    //     {28 * 28, std::make_shared<Linear>()},
    //     {128,     std::make_shared<ReLU>()},
    //     {64,      std::make_shared<ReLU>()},
    //     {10,      std::make_shared<Linear>()}
    // });


    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start the Dear ImGui frame
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

        if(ImGui::Begin("MNIST Control Panel", nullptr, flags)) {
            main_window(state);
            ImGui::End();
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    ImPlot::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
    // std::cout << "Starting Training (" << state.epochs << " Epochs, Batch Size " << state.batch_size << ")..." << std::endl;

    // for (size_t epoch = 0; epoch < state.epochs; ++epoch) {
    //     float epoch_loss = 0.0f;
    //     size_t batches = 0;

    //     for (size_t i = 0; i < state.num_train; i += state.batch_size) {
    //         size_t current_batch = std::min((unsigned long)state.batch_size, state.num_train - i);
    //         pip("Progress: %.2f%%", (float)i/state.num_train*100);
    //         auto start = i;
    //         auto stop  = i + current_batch;

    //         Tensor x_batch = state.train_images[{{start, stop}, {-1,-1}}];
    //         Tensor y_batch = state.train_labels[{{start, stop}, {-1,-1}}];

    //         model.zero_grad();
    //         Tensor logits = model.forward(x_batch);
    //         Tensor loss = cross_entropy_loss(logits, y_batch);

    //         backward_step(model, logits, y_batch);
    //         model.update_weights(state.learning_rate);

    //         epoch_loss += loss[0];
    //         batches++;
    //     }

    //     std::cout << "\n" << "Epoch " << (epoch + 1) << "/" << state.epochs << " - Loss: " << (epoch_loss / batches) << std::endl;
    // }

    // // Evaluation on Test Dataset
    // std::cout << "\nEvaluating model performance..." << std::endl;
    // Tensor test_logits = model.forward(state.test_images);
    // Tensor test_probs = softmax(test_logits);

    // size_t correct = 0;
    // for (size_t i = 0; i < state.num_test; ++i) {
    //     size_t pred = 0;
    //     float max_p = test_probs[i * 10];
    //     for (size_t c = 1; c < 10; ++c) {
    //         if (test_probs[i * 10 + c] > max_p) {
    //             max_p = test_probs[i * 10 + c];
    //             pred = c;
    //         }
    //     }

    //     size_t actual = 0;
    //     for (size_t c = 0; c < 10; ++c) {
    //         if (state.test_labels[i * 10 + c] > 0.5f) {
    //             actual = c;
    //             break;
    //         }
    //     }

    //     std::cout << "Expected " << actual << ". Got " << pred << "." << std::endl;
    //     if (pred == actual) correct++;
    // }

    // std::cout << "Test Accuracy: " << (100.0f * correct / state.num_test) << "% (" 
    //           << correct << "/" << state.num_test << ")" << std::endl;
}

void load_new_state(TrainingState& state);
void load_with_current(TrainingState& state);
void load_training_state(const std::string& filepath, const TrainingState& state);
void save_training_state(const std::string& filepath, const TrainingState& state);

void start_training(TrainingState& state);
void stop_training(TrainingState& state);

void show_training_graph(const TrainingState& state);
void show_prediciton_image(const TrainingState& state);
void show_training_config(TrainingState& state);
void show_file_loader(TrainingState& state, bool& open_dialog);
void show_file_saver(TrainingState& state);

void main_window(TrainingState& state) {
    std::thread train_thread;
    static bool open_dialog = false;

    if(ImGui::BeginMenuBar()) {
        if(ImGui::BeginMenu("File")){
            if(ImGui::BeginMenu("Load")) { 
                if(ImGui::MenuItem("Default")) { load_new_state(state); }
                if(ImGui::MenuItem("Current Selection")) {load_with_current(state); }
                if(ImGui::MenuItem("From File")) { open_dialog = true; }
                ImGui::EndMenu();
            }
            if(ImGui::MenuItem("Save")) {  }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    if(ImGui::BeginChild("Training Config", ImVec2(350,0), true)) {
        show_training_config(state);
        ImGui::EndChild();
    }

    ImGui::SameLine();

    if(ImGui::BeginChild("Training Visualizer", ImVec2(0,0), true)) {
        show_training_graph(state);
        show_prediciton_image(state);
        ImGui::EndChild();
    }   

    show_file_loader(state, open_dialog);
}

void load_new_state(TrainingState& state) {
    //quick mnist that gives 90% accuracy

    state.epochs = 5;
    state.batch_size = 10;
    state.learning_rate = 0.05f;
    state.num_train = 500; //60k max
    state.num_test  = 50;  //10k max

    load_with_current(state);
}

void load_with_current(TrainingState& state) {
    state.is_ready = true;
    std::cout << "Loading dataset..." << std::endl;
    if (!load_mnist_images("../data/train-images.idx3-ubyte", state.train_images, state.num_train) ||
        !load_mnist_labels("../data/train-labels.idx1-ubyte", state.train_labels, state.num_train)) {
        std::cerr << "Failed to load MNIST training files from ../data/" << std::endl;
        exit(1);
    }
    if(!load_mnist_images("../data/t10k-images.idx3-ubyte", state.test_images, state.num_test) ||
       !load_mnist_labels("../data/t10k-labels.idx1-ubyte", state.test_labels, state.num_test)) {
        std::cerr << "Failed to load MNIST test files from ../data/" << std::endl;
        exit(1);
    }
}

void save_training_state(const std::string& filepath, const TrainingState& state) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) return;

    out.write(reinterpret_cast<const char*>(&state.epochs), sizeof(state.epochs));
    out.write(reinterpret_cast<const char*>(&state.batch_size), sizeof(state.batch_size));
    out.write(reinterpret_cast<const char*>(&state.num_train), sizeof(state.num_train));
    out.write(reinterpret_cast<const char*>(&state.num_test), sizeof(state.num_test));
    out.write(reinterpret_cast<const char*>(&state.learning_rate), sizeof(state.learning_rate));

    int cur_epoch = state.current_epoch.load();
    float cur_loss = state.current_loss.load();
    float cur_acc = state.current_accuracy.load();
    float progress = state.progress_fraction.load();
    out.write(reinterpret_cast<const char*>(&cur_epoch), sizeof(cur_epoch));
    out.write(reinterpret_cast<const char*>(&cur_loss), sizeof(cur_loss));
    out.write(reinterpret_cast<const char*>(&cur_acc), sizeof(cur_acc));
    out.write(reinterpret_cast<const char*>(&progress), sizeof(progress));

    auto write_vector = [&out](const std::vector<float>& vec) {
        size_t size = vec.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        if (size > 0) {
            out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
        }
    };
    write_vector(state.loss_history);
    write_vector(state.acc_history);

auto write_tensor = [&out](const AennTensor::Tensor& tensor) {
        // Check if tensor is valid/initialized
        bool has_data = (tensor.data != nullptr);
        out.write(reinterpret_cast<const char*>(&has_data), sizeof(has_data));
        if (!has_data) return;

        // Save shape dimensions
        size_t shape_size = tensor.shape.size();
        out.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
        if (shape_size > 0) {
            out.write(reinterpret_cast<const char*>(tensor.shape.data()), shape_size * sizeof(size_t));
        }

        // Save strides dimensions
        size_t strides_size = tensor.strides.size();
        out.write(reinterpret_cast<const char*>(&strides_size), sizeof(strides_size));
        if (strides_size > 0) {
            out.write(reinterpret_cast<const char*>(tensor.strides.data()), strides_size * sizeof(size_t));
        }

        // Save offset
        out.write(reinterpret_cast<const char*>(&tensor.offset), sizeof(tensor.offset));

        // Save raw backing data buffer vector
        size_t data_size = tensor.data->data.size();
        out.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        if (data_size > 0) {
            out.write(reinterpret_cast<const char*>(tensor.data->data.data()), data_size * sizeof(AennTensor::data_type));
        }
    };

    write_tensor(state.train_images);
    write_tensor(state.train_labels);
    write_tensor(state.test_images);
    write_tensor(state.test_labels);
}

bool load_training_state(const std::string& filepath, TrainingState& state) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) return false;

    // 1. Load scalar metadata & primitives
    in.read(reinterpret_cast<char*>(&state.epochs), sizeof(state.epochs));
    in.read(reinterpret_cast<char*>(&state.batch_size), sizeof(state.batch_size));
    in.read(reinterpret_cast<char*>(&state.num_train), sizeof(state.num_train));
    in.read(reinterpret_cast<char*>(&state.num_test), sizeof(state.num_test));
    in.read(reinterpret_cast<char*>(&state.learning_rate), sizeof(state.learning_rate));

    int cur_epoch = 0;
    float cur_loss = 0.0f, cur_acc = 0.0f, progress = 0.0f;
    in.read(reinterpret_cast<char*>(&cur_epoch), sizeof(cur_epoch));
    in.read(reinterpret_cast<char*>(&cur_loss), sizeof(cur_loss));
    in.read(reinterpret_cast<char*>(&cur_acc), sizeof(cur_acc));
    in.read(reinterpret_cast<char*>(&progress), sizeof(progress));

    state.current_epoch.store(cur_epoch);
    state.current_loss.store(cur_loss);
    state.current_accuracy.store(cur_acc);
    state.progress_fraction.store(progress);

    // 2. Load vectors
    auto read_vector = [&in](std::vector<float>& vec) {
        size_t size = 0;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        if (size > 0) {
            in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
        }
    };
    read_vector(state.loss_history);
    read_vector(state.acc_history);

    // 3. Load Tensors
    auto read_tensor = [&in](AennTensor::Tensor& tensor) {
        bool has_data = false;
        in.read(reinterpret_cast<char*>(&has_data), sizeof(has_data));
        if (!has_data) {
            tensor = AennTensor::Tensor();
            return;
        }

        // Load shape
        size_t shape_size = 0;
        in.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
        tensor.shape.resize(shape_size);
        if (shape_size > 0) {
            in.read(reinterpret_cast<char*>(tensor.shape.data()), shape_size * sizeof(size_t));
        }

        // Load strides
        size_t strides_size = 0;
        in.read(reinterpret_cast<char*>(&strides_size), sizeof(strides_size));
        tensor.strides.resize(strides_size);
        if (strides_size > 0) {
            in.read(reinterpret_cast<char*>(tensor.strides.data()), strides_size * sizeof(size_t));
        }

        // Load offset
        in.read(reinterpret_cast<char*>(&tensor.offset), sizeof(tensor.offset));

        // Load raw backing data buffer vector
        size_t data_size = 0;
        in.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        
        tensor.data = std::make_shared<AennTensor::Data>(data_size);
        if (data_size > 0) {
            in.read(reinterpret_cast<char*>(tensor.data->data.data()), data_size * sizeof(AennTensor::data_type));
        }
    };

    read_tensor(state.train_images);
    read_tensor(state.train_labels);
    read_tensor(state.test_images);
    read_tensor(state.test_labels);

    state.is_ready.store(true);
    return true;
}

void training_loop(TrainingState& state) {
    while(state.is_training.load()) {
        //main predict -> backward -> step loop
    }
}

void start_training(TrainingState& state) {
    state.is_training = true;
    std::thread training_thread([&state] {
        training_loop(state);
    });
    training_thread.detach();
}

void stop_training(TrainingState& state) {
    state.is_training = false;
}

void show_file_dialog(TrainingState& state, bool& open_dialog, bool& is_loading) {
    if (open_dialog) {
        IGFD::FileDialogConfig config;
        config.path = ".";
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".train.data", config);
        open_dialog = false; // Reset trigger so it only opens once
    }

    // Display must be called EVERY frame to keep the popup alive and interactive
    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
            std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();

            printf("Chosen file: %s\n", filePathName.c_str());

            //HANDLE SELECTION
            if(is_loading) {

            } else {

            }
            
        }
        ImGuiFileDialog::Instance()->Close();
    }
}

void show_training_config(TrainingState& state) {
    const char* button_text = state.is_training? "Training" : "Train";
    const char* progress_text = "Training Progress";

    ImGui::DragInt("Epochs", &state.epochs, .2, 1, 50);
    ImGui::DragFloat("Learn Rate", &state.learning_rate, .001, 0.001, .5);
    ImGui::DragInt("Batch Size", &state.batch_size, 1, 1, state.num_train);
    ImGui::DragInt("Train Size", &state.num_train, 100, 1, 60000);
    ImGui::DragInt("Test Size", &state.num_test, 20, 1, 10000);
    ImGui::ProgressBar(state.progress_fraction, ImVec2(-1.0f, 0.0f), progress_text);

    float buttonWidth = ImGui::CalcTextSize(button_text).x; // Specify your desired button width
    set_center_ofs(buttonWidth);
    if (ImGui::Button(button_text, ImVec2(buttonWidth + 10, 0.0f))) {
        if(state.is_ready) {
            if(!state.is_training) {
                start_training(state);
            } else 
                stop_training(state);
        }
    }
}

void show_training_graph(const TrainingState& state) {
    static float xs[1001], ys1[1001], ys2[1001];
    for (int i = 0; i < 1001; ++i) {
        xs[i]  = (i*0.1f);
        ys1[i] = sinf(xs[i]) * 3 + 1;
        ys2[i] = cosf(xs[i]) * 0.2f + 0.5f;
    }

    if (ImPlot::BeginPlot("Line Plots")) {
        ImPlot::SetupAxes("Epochs","Loss");
        ImPlot::SetupAxesLimits(0, 100, 0, 10);
        ImPlot::SetupAxis(ImAxis_Y2, "Accuracy (%)", ImPlotAxisFlags_AuxDefault);
        ImPlot::SetupAxisLimits(ImAxis_Y2, 0, 100);

        //plot loss
        ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
        ImPlot::PlotLine("f(x) = sin(x)*3+1", xs, ys1, 1001);

        //plot accuracy
        ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
        ImPlot::PlotLine("f(x) = cos(x)*.2+.5", xs, ys2, 1001);

        ImPlot::EndPlot();
    }
}

void show_prediciton_image(const TrainingState& state) {
    if (!state.is_ready) return;
    
    static int image = 0;
    ImGui::DragInt("View Guess", &image, 1, 0, state.num_test-1);

    Tensor img = state.test_images.slice({{image, image}, {-1,-1}});
    auto raw = img.data->data.data() + img.offset;
    // printf("Showing image: %d, ofs into test_images: %d", image, img.offset);
    auto image_id = GetMNISTTexture(raw);
    ImGui::Image(image_id, {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().x});
}
