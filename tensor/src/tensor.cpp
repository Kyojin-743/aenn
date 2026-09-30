#include <tensor.h>

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

            for (size_t d = 0; d < shape.size(); ++d) {
                std::stringstream msg;
                msg << "" << "index[" << d << "]=" << index[d] << ", shape[" << d << "]=" << shape[d] << "";
                if (index[d] >= shape[d]) {
                    std::cerr << msg.str();
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
            size_t dim_idx, size_t flat_offset, const Strides strides,
            size_t indent) {
            // Base Case: Innermost dimension (shape[0])
            if (dim_idx == 0) {
                std::cout << "[ ";
                for (size_t i = 0; i < t.shape[0]; ++i) {
                    size_t idx = flat_offset + i * strides[0];
                    std::cout << t[idx] << (i + 1 < t.shape[0] ? " " : "");
                }
                std::cout << " ]";
                return;
            }

            // Recursive Case: Outer dimensions (shape[dim_idx])
            std::cout << "[";
            for (size_t i = 0; i < t.shape[dim_idx]; ++i) {
                if (i > 0) {
                    std::cout << ",\n" << std::string(indent + 1, ' ');
                }
                size_t next_offset = flat_offset + i * strides[dim_idx];
                dim_str_recurse(str, t, dim_idx - 1, next_offset, strides, indent + 1);
            }
            std::cout << "]";
        }

        std::string Tensor::tostr(bool pretty) const {
            if (pretty) {
                std::string result;
                Strides strides(shape.size(), 1);
                for (size_t i(1ULL); i < shape.size(); ++i) {
                    strides[i] = strides[i - 1] * shape[i - 1];
                }
                size_t outer_dim = shape.size() - 1;
                dim_str_recurse(result, *this, outer_dim, 0, strides, 0);
                return result;
            } else {
                std::string result;
                size_t num_elements = numel(this->shape);

                for(size_t i(0ULL); i < num_elements; ++i) {
                    if (i % strides[0] == 0 && i != 0) {
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
                
            }
            return t;
        }

        Tensor Tensor::operator-(const Tensor& other) {
            assert(this->shape == other.shape);
            Tensor t(this->shape);
            return t;
        }

        Tensor Tensor::operator*(const Tensor& other) {
            assert(this->shape == other.shape);
            Tensor t(this->shape);
            return t;
        }

        Tensor Tensor::operator/(const Tensor& other) {
            assert(this->shape == other.shape);
            Tensor t(this->shape);
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

}