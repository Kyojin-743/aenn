//AENN/tensor/tensor.cpp
#include <tensor.h>
#include <smartassert.h>

#include <random>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <cassert>

namespace AennTensor {
        // std::random_device rd;
        std::mt19937 gen(42);//rd());

         size_t numel(const Shape& shape) {
            size_t prod = 1;
            for(auto s : shape) prod *= s; 
            return prod;
        }
        
        size_t get_flat_index(const Shape& shape, const Index& index) {
            assert(shape.size() == index.size());
            size_t flat_idx = 0;
            size_t current_stride = 1;

            // Traverse from rightmost (innermost) dimension to leftmost
            for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                if (index[d] >= shape[d]) {
                    std::cerr << "index[" << d << "]=" << index[d] 
                            << ", shape[" << d << "]=" << shape[d] << "\n";
                    exit(1);
                }
                flat_idx += index[d] * current_stride;
                current_stride *= shape[d];
            }

            return flat_idx;
        }

        Tensor::Tensor(const Shape& shape, data_type fill) {
            this->shape = shape;
            this->strides = shape;
            data = std::make_shared<Data>(numel(shape));
            if(fill != static_cast<data_type>(0)) {
                for(auto& elem : data->data)
                    elem = fill;
            }
        }

        Tensor::Tensor(const Shape& shape, const Vector& vec) {
            this->shape = shape;
            this->strides= shape;
            data = std::make_shared<Data>(vec);
        }

        Tensor Tensor::Zeroes(const Shape& shape) {
            Tensor t(shape);
            return t;
        }

        Tensor Tensor::Ones(const Shape& shape) {
            Tensor t(shape, 1);
            return t;
        }

        Tensor Tensor::Random(const Shape& shape, data_type min, data_type max, Distribution dist) {
            Tensor t(shape);
            switch(dist) {
                case Distribution::Uniform: {
                    std::uniform_real_distribution<data_type> rand_uniform(min, max);
                    for(auto& elem : t.data->data) {
                        data_type r = rand_uniform(gen);
                        elem = r;
                    }
                }break;
                case Distribution::Normal: {
                    std::normal_distribution<data_type> rand_normal(min, max);
                    for(auto& elem : t.data->data) {
                        auto t = rand_normal(gen);
                        elem = t;
                    }
                }break;
            }
            return t;
        }

        Tensor Tensor::Identity(const Shape& shape) {//enforce square
            assert(shape.size() == 2 && "TODO: generalized contraction-neutral identity tensor creation");
            Tensor t(shape, static_cast<data_type>(0));
            size_t n = shape[0];
            for(size_t i(0ULL); i < n; ++i) {
                t.data->data[i * n + i] = static_cast<data_type>(1);
            }
            return t;
        }
        
    void dim_str_recurse(std::string& str, const Tensor& t, 
                        size_t dim, size_t flat_offset, const Strides& strides,
                        size_t indent) {
        size_t last_dim = t.shape.size() - 1;

        // Base Case: Innermost dimension (e.g., columns)
        if (dim == last_dim) {
            str += "[ ";
            for (size_t i = 0; i < t.shape[dim]; ++i) {
                size_t idx = flat_offset + i * strides[dim];
                str += std::to_string(t[idx]) + (i + 1 < t.shape[dim] ? " " : "");
            }
            str += " ]";
            return;
        }

        // Recursive Case: Outer dimensions
        str += "[";
        for (size_t i = 0; i < t.shape[dim]; ++i) {
            if (i > 0) {
                str += ",\n" + std::string(indent + 1, ' ');
            }
            size_t next_offset = flat_offset + i * strides[dim];
            dim_str_recurse(str, t, dim + 1, next_offset, strides, indent + 1);
        }
        str += "]";
    }

    std::string Tensor::tostr(bool pretty) const {
        if (shape.empty()) return "[]";

        // Row-major strides: strides[last_dim] = 1
        // strides[i] = strides[i + 1] * shape[i + 1]
        Strides strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        if (pretty) {
            std::string result;
            dim_str_recurse(result, *this, 0, 0, strides, 0);
            return result;
        } else {
            std::string result;
            size_t num_elements = numel(this->shape);

            for (size_t i = 0; i < num_elements; ++i) {
                if (i % strides[shape.size() - 1] == 0 && i != 0) {
                    result += "\n";
                }
                result += std::string(4, ' ') + std::to_string(data->data[i]);
            }

            return result;
        }
    }
        
    const char* Tensor::c_str(bool pretty) const {
        return strdup(tostr(pretty).c_str());
    }

    Tensor Tensor::operator+(data_type scalar) {
        Tensor t(this->shape);
        for(auto& elem : t.data->data) {
            elem += scalar;
        }
        return t;
    }

    Tensor Tensor::operator-(data_type scalar) {
        Tensor t(this->shape);
        for(auto& elem : t.data->data) {
            elem -= scalar;
        }
        return t;
    }

    Tensor Tensor::operator*(data_type scalar) {
        Tensor t(this->shape);
        for(auto& elem : t.data->data) {
            elem *= scalar;
        }
        return t;
    }

    Tensor Tensor::operator/(data_type scalar) {
        Tensor t(this->shape);
        for(auto& elem : t.data->data) {
            elem /= scalar;
        }
        return t;
    }

    Tensor Tensor::operator+(const Tensor& other) {
        assert(this->shape == other.shape);
        Tensor t(this->shape);
        size_t num_elements = numel(this->shape);   
        for (size_t i(0ULL); i < num_elements; ++i) {
            t[i] = (*this)[i] + other[i];
        }
        return t;
    }

    Tensor Tensor::operator-(const Tensor& other) {
        assert(this->shape == other.shape);
        Tensor t(this->shape);
        size_t num_elements = numel(this->shape);   
        for (size_t i(0ULL); i < num_elements; ++i) {
            t[i] = (*this)[i] - other[i];
        }
        return t;
    }

    Tensor Tensor::operator*(const Tensor& other) {
        assert(this->shape == other.shape);
        Tensor t(this->shape);
        size_t num_elements = numel(this->shape);   
        for (size_t i(0ULL); i < num_elements; ++i) {
            t[i] = (*this)[i] * other[i];
        }
        return t;
    }

    Tensor Tensor::operator/(const Tensor& other) {
        assert(this->shape == other.shape);
        Tensor t(this->shape);
        size_t num_elements = numel(this->shape);   
        for (size_t i(0ULL); i < num_elements; ++i) {
            t[i] = (*this)[i] / other[i];
        }
        return t;
    }

    data_type& Tensor::operator[](const Index& index) {
        return (this->data->data)[get_flat_index(this->shape, index)];
    }
    
    const data_type& Tensor::operator[](const Index& index) const {
        return (this->data->data)[get_flat_index(this->shape, index)];
    }

    data_type& Tensor::operator[](size_t flat_idx) {
        return (this->data->data)[flat_idx];
    }

    const data_type& Tensor::operator[](size_t flat_idx) const {
        return (this->data->data)[flat_idx];
    }

    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << tensor.tostr();
        return os;
    }

    Tensor matmul(const Tensor& a, const Tensor& b) {
        ASSERT(a.shape.size() == 2) << "Expected Degree 2. Got: " << a.shape.size();
        ASSERT(b.shape.size() == 2) << "Expected Degree 2. Got: " << b.shape.size();
        ASSERT(a.shape[1] == b.shape[0]) << "Inner dimensions must match. Got: " << a.shape[1] << " and " << b.shape[0];

        size_t M = a.shape[0];
        size_t K = a.shape[1];
        size_t N = b.shape[1];

        Tensor t({M, N}, static_cast<data_type>(0));

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                data_type val_a = a[{i, k}];
                for (size_t j = 0; j < N; ++j) {
                    t[{i, j}] += val_a * b[{k, j}];
                }
            }
        }

        return t;
    }
}