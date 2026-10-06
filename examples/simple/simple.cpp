#include <iostream>

#include <tensor.h>

using namespace AennTensor;

enum Activations {
    Linear,
    ReLu,
    Sigmoid
};

class SimpleModel {
    using Layer = std::vector<Tensor>;
public:
    SimpleModel(std::vector<std::array<size_t, 2>> description) {

    }

    Tensor forward(const Tensor& input) {

    }
    std::vector<Layer> layers;
    std::vector<Layer> grads;
    std::vector<Activations> activations;
};

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& labels);
void backward_step(SimpleModel& model, const Tensor& output);


int main(void) {
    
    SimpleModel model({
        {28*28, Linear},
        {128, ReLu},
        {64, ReLu},
        {10, Sigmoid}
    });

}
