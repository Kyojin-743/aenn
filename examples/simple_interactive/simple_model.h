#pragma once

#include <fstream>
#include <smartassert.h>
#include <tensor.h>
#include <thread>
#include <cmath>

namespace SimpleModels {
    using namespace AennTensor;

    //[index, prediction, Expected]
    using Guess = std::tuple<int,int,int>;
    using Guesses = std::vector<Guess>;

    class Activation {
    public:
        virtual ~Activation() = default;
        virtual void forward_gpu(const float* d_in, float* d_out, int n) = 0;
        virtual void backward_gpu(const float* d_grad_out, float* d_grad_in, int n) = 0;
    };

      class Linear : public Activation {
    public:
        void forward_gpu(const float* d_in, float* d_out, int n) override;
        void backward_gpu(const float* d_grad_out, float* d_grad_in, int n) override;
    };

    class ReLU : public Activation {
    public:
        const float* d_cached_input{nullptr};
        int cached_n{0};

        void forward_gpu(const float* d_in, float* d_out, int n) override;
        void backward_gpu(const float* d_grad_out, float* d_grad_in, int n) override;
    };

    class Sigmoid : public Activation {
    public:
        const float* d_cached_output{nullptr};
        int cached_n{0};

        void forward_gpu(const float* d_in, float* d_out, int n) override;
        void backward_gpu(const float* d_grad_out, float* d_grad_in, int n) override;
    };

    class SimpleModel {
    public:
        using LayerSpec = std::tuple<size_t, std::shared_ptr<Activation>>;
        using Description = std::vector<LayerSpec>;

        std::vector<std::shared_ptr<Activation>> activations;
        std::vector<Tensor> weights;
        std::vector<Tensor> grads;

        std::vector<float*> d_weights;
        std::vector<float*> d_grads;
        std::vector<float*> d_activations;
        std::vector<float*> d_pre_activated;
        float* d_aug_input{nullptr};
        float* d_aug_transpose{nullptr};

        SimpleModel(Description desc);
        ~SimpleModel();

        void sync_weights_to_host();
        void update_weights(float lr, size_t batch_size);
    };


    bool load_mnist_images(const std::string& path, Tensor& images, size_t max_samples = 0, size_t max_threads = 4);
    bool load_mnist_labels(const std::string& path, Tensor& labels, size_t max_samples = 0);
    
    Tensor cross_entropy_loss_from_gpu(const float* d_probs, const float* d_labels, size_t batch_size, size_t num_classes);
    float train_one_epoch(SimpleModel& model, const Tensor& train_images, const Tensor& train_labels, size_t batch_size, float learn_rate);
    std::tuple<float, Guesses> evaluate_model(SimpleModel& model, const Tensor& inp, const Tensor& exp);

}