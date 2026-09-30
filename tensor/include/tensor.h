//AENN/tensor/tensor.h
#pragma once

#include <vector>
#include <array>
#include <memory>   //shared_ptr
#include <string>  
#include <iostream>

#include <cstddef> //ptrdiff_t

namespace AennTensor {
    using data_type = float;
    using Vector = std::vector<data_type>;
    using Shape = std::vector<size_t>;
    using Strides = std::vector<size_t>;
    using Index = std::vector<size_t>;
    using Range = std::array<ptrdiff_t, 2>;
    using Ranges = std::vector<Range>;

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

        Tensor operator[](const Ranges& nd_range);
        const Tensor operator[](const Ranges& nd_range) const;


        //Friends
        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

        //misc functions
        std::string tostr(bool pretty=true) const;
        const char* c_str(bool pretty=true) const;
        Tensor reshape(const Shape& new_shape) const;
        Tensor transpose(void) const;
        Tensor slice(const Ranges& nd_range) const;

        //fields
        std::shared_ptr<Data> data;
        Shape shape;
        Strides strides;
        size_t offset;
    };

    size_t numel(const Shape& shape);
    size_t get_flat_index(const Shape& shape, const Strides&, const Index& index);
    Tensor matmul(const Tensor& a, const Tensor& b);
}