#pragma once

#include <smartassert.h>
#include <tensor.h>
#include <thread>
#include <cmath>

namespace SimpleModels {
    using namespace AennTensor;

    //[index, prediction, Expected]
    using Guesses = std::vector<std::tuple<int,int,int>>;
    
    
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

        return true;
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

    std::tuple<float, Guesses> evaluate_model(SimpleModel& model, const Tensor& inp, const Tensor& exp) {
        Tensor test_logits = model.forward(inp);
        Tensor test_probs = softmax(test_logits);

        size_t num_test = inp.shape[0];
        size_t correct = 0;
        Guesses guesses;
        for (size_t i = 0; i < num_test; ++i) {
            size_t pred = 0;
            float max_p = test_probs[i * 10];
            for (size_t c = 1; c < 10; ++c) {
                if (test_probs[i * 10 + c] > max_p) {
                    max_p = test_probs[i * 10 + c];
                    pred = c;
                }
            }

            size_t actual = 0;
            for (size_t c = 0; c < 10; ++c) {
                if (exp[i * 10 + c] > 0.5f) {
                    actual = c;
                    break;
                }
            }

            if (pred == actual) correct++;
            else guesses.push_back({1,pred,actual});
        }

        float accuracy = (100.0f * correct / num_test);
        return {accuracy, guesses};
    }
}
