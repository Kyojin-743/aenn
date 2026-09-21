#pragma once

#include <vector>
#include <memory>

namespace AennTensor {
    //For now, no templates there will be 1 type for everything, per compile
    using data_type = float;
    using Shape = std::vector<size_t>;
    using Strides = std::vector<size_t>;

    class Data {
    public:
        explicit Data(size_t size) : data_(size, static_cast<data_type>(0)) {}
        Data(const std::vector<data_type>& v_from) :  data_(v_from) {}

        data_type* data() { return data_.data(); }
        const data_type* data() const {return data_.data(); }
        size_t size() { return data_.size(); }

    private:
        std::vector<data_type> data_;
    };

    class Tensor {
    public:
        Tensor(const Shape& shape, data_type fill = static_cast<data_type>(0));
        Tensor(const Shape& shape, const std::vector<data_type>& v_from);

        const Shape& shape() const { return shape_; }
        const Strides& strides() const { return strides_; }
        size_t size() const { return numel_; }
        size_t ndim() const { return shape_.size(); }
        bool is_contiguous() const;

        data_type* data() { return data_->data() + offset_; }
        const data_type* data() const { return data_->data() + offset_; }
        
        data_type& operator[](size_t flat_idx);
        const data_type& operator[](size_t flat_idx) const;
        data_type& at(const Shape& indices);

        Tensor reshape(const Shape& new_shape) const;
        Tensor transpose(size_t dim0, size_t dim1) const;

        Tensor add(const Tensor& other) const;
        Tensor mul(const Tensor& other) const;
        Tensor matmul(const Tensor& other) const;

        Tensor operator+(const Tensor& other) const { return add(other); }
        Tensor operator*(const Tensor& other) const { return mul(other); }
    private:
        std::shared_ptr<Data> data_;
        Shape shape_;
        Strides strides_;
        size_t offset_{0};
        size_t numel_{0};

        void get_strides_();    
    };

     size_t calculate_numel(const Shape& shape);
}