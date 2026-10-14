#pragma once

#include <memory>
#include <functional>
#include <any>

#include <smartassert.h>
#include <tensor.h>

namespace AennAutoGrad
{
    class Node;
    class Tensor;
    class Function;
    class Ctx;

    using Data = AennTensor::Tensor;

    class Ctx {
public:
        Ctx() = default;

        template <typename... Args>
        void saved_tensors(Args&&... tensors) {
            (tensor_data.push_back(std::forward<Args>(tensors)), ...);
        }

        const std::vector<Data>& get_tensors() const {
            return tensor_data;
        }

        const Data& get_tensor(size_t index) const {
            ASSERT(index < tensor_data.size()) << "Index out of range (" << index << " > " << tensor_data.size() <<")";
            return tensor_data[index];
        }

        template <typename T>
        void save_any(T&& value) {
            any_data.push_back(std::make_any<T>(std::forward<T>(value)));
        }

        template <typename T>
        T get_any(size_t index) const {
            ASSERT(index < any_data.size()) << "Index out of range (" << index << " > " << any_data.size() <<")";
            return std::any_cast<T>(any_data[index]);
        }

        void clear() {
            tensor_data.clear();
            any_data.clear();
        }

        bool has_data() const {
            return !tensor_data.empty() || !any_data.empty();
        }

        std::vector<Data> tensor_data;
        std::vector<std::any> any_data;
    };

    class Function {

    };

    class Node {
public:

        Data value;
        Data grad;

        bool required_grad;
        bool is_leaf;

        Function _backward;
        std::vector<std::shared_ptr<Node>> parents;
    };

    //user-handle, to allow it to go out of scope but keep the nodes persistent
    class Tensor {
public:
        std::shared_ptr<Node> node;
    };

}