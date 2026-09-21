#include <tensor.h>
#include <numeric>
#include <functional>
#include <cassert>

namespace AennTensor {

     size_t calculate_numel(const Shape& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    }

    void Tensor::get_strides_() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;

        size_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    Tensor::Tensor(const Shape& shape, data_type fill_value)
        : shape_(shape), numel_(calculate_numel(shape)) {
        get_strides_();
        data_ = std::make_shared<Data>(numel_);
        std::fill_n(data_->data(), numel_, fill_value);
    }

    Tensor::Tensor(const Shape& shape, const std::vector<data_type>& data)
        : shape_(shape), numel_(calculate_numel(shape)) {
        assert(data.size() == numel_);
        get_strides_();
        data_ = std::make_shared<Data>(data);
    }

    data_type& Tensor::at(const Shape& indices) {
        assert(indices.size() == shape_.size());
        size_t flat_idx = offset_;
        for (size_t i = 0; i < indices.size(); ++i) {
            flat_idx += indices[i] * strides_[i];
        }
        return data_->data()[flat_idx];
    }


    Tensor Tensor::add(const Tensor& other) const {
        assert(this->shape() == other.shape());
        Tensor sum(shape_);

        const data_type* p_a = data();
        const data_type* p_b = other.data();
        data_type* p_s = sum.data();
        for(size_t i(0); i < numel_; ++i) {
            p_s[i] = p_a[i] + p_b[i];
        }
        return sum;
    }

    Tensor Tensor::mul(const Tensor& other) const {
        assert(this->shape() == other.shape());
        Tensor product(shape_);

        const data_type* p_a = data();
        const data_type* p_b = other.data();
        data_type* p_p = product.data();
        for(size_t i(0); i < numel_; ++i) {
            p_p[i] = p_a[i] * p_b[i];
        }
        return product;
    }

    Tensor Tensor::matmul(const Tensor& other) const {

    }
}