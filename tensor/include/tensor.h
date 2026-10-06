//AENN/tensor/tensor.h
#pragma once

#include <vector>
#include <array>
#include <memory>   //shared_ptr
#include <string>  
#include <iostream>
#include <functional>
#include <algorithm> //std::all_of

#include <cstddef> //ptrdiff_t

namespace AennTensor {
    //Forward declare for use in 'using' directives
    class Tensor;
    class Data;
    enum Distribution;

    //Typenames
    using data_type = float;
    using Vector = std::vector<data_type>;
    using Shape = std::vector<size_t>;
    using Strides = std::vector<size_t>;
    using Index = std::vector<size_t>;
    using Range = std::array<ptrdiff_t, 2>;
    using Ranges = std::vector<Range>;
    using Axes = std::vector<size_t>;
    using ReduceFn = std::function<data_type(const Tensor&)>;

    enum Distribution {
        Uniform,
        Normal
    };

    class Data {
    public:
        explicit Data(size_t size) : data(size, static_cast<data_type>(0)) {};
        Data(const Vector& vec) : data(vec) {};
        Vector data;
    };

    class Tensor {
    public:
        //Constructors
        Tensor() {};    //default
        explicit Tensor(const Shape& shape, data_type fill = static_cast<data_type>(0));
        explicit Tensor(const Shape& shape, const Vector& vec);
        Tensor(const Tensor& other);

        //Static Functions 
        static Tensor Zeroes(const Shape& shape);
        static Tensor Ones(const Shape& shape);
        static Tensor Random(const Shape& shape, 
            data_type min = static_cast<data_type>(0), 
            data_type max = static_cast<data_type>(1),
            Distribution dist = Distribution::Uniform);
        static Tensor Identity(const Shape& shape);

        //Overloads
        Tensor operator+(data_type scalar);
        Tensor operator-(data_type scalar);
        Tensor operator*(data_type scalar);
        Tensor operator/(data_type scalar);
        
        Tensor operator+(const Tensor& other);
        Tensor operator-(const Tensor& other);
        Tensor operator*(const Tensor& other);
        Tensor operator/(const Tensor& other);

        data_type& operator[](const Index& index);
        const data_type& operator[](const Index& index) const;

        data_type& operator[](size_t flat_idx);
        const data_type& operator[](size_t flat_idx) const;

        //short-hand for slicing
        Tensor operator[](const Ranges& nd_range);
        const Tensor operator[](const Ranges& nd_range) const;

        //Friends
        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

        //misc functions
        std::string tostr(bool pretty=true) const;
        const char* c_str(bool pretty=true) const;
        size_t flat_index(const Index& index) const;
        Index from_flat(size_t flat_idx) const;
        size_t size() const;

        //Tensor Manipulation Functions (always returns new)
        Tensor reshape(const Shape& new_shape) const;
        Tensor transpose(void) const;
        Tensor slice(const Ranges& nd_range) const;
        Tensor sum(const Axes& axes, bool keep_dims=false) const;
        Tensor mean(const Axes& axes, bool keep_dims=false) const;
        Tensor min(const Axes& axes, bool keep_dims=false) const;
        Tensor max(const Axes& axes, bool keep_dims=false) const;

        //fields
        std::shared_ptr<Data> data;
        Shape shape;
        Strides strides;
        size_t offset;
    };


    size_t numel(const Shape& shape);
    size_t get_flat_index(const Shape& shape, const Strides& strides, const Index& index);
    Index from_flat_idx(const Shape& shape, const Strides& strides, size_t flat_idx);
    Tensor matmul(const Tensor& a, const Tensor& b);
}