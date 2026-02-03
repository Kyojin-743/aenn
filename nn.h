#ifndef __NN_H_
#define __NN_H_

#include <vector>
#include <cassert>
#include <memory>

#include "tensor.h"

template <typename T>
class Model {
using vector = std::vector;
using shared_ptr = std::shared_ptr;

public:
    Model(size_t nFanIn, vector<size_t>& vLayerDescription, vector<Activations>& vActivationDescription) {
        assert(vLayerDescription.size() == vActivationDescription.size());
        size_t nLayerCount = vLayerDescription.size();
        for(size_t i{0}; i < nLayerCount; i++) {
            vWeights.emplace_back(/*random tensor*/)
        }
    }
    ~Model() {}

    vector<shared_ptr<Tensor<T>>> vWeights;
};

#endif//__NN_H_


#ifdef NN_IMPLEMENTATION

#endif