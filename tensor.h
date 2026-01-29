#pragma once

#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <array>

inline bool getDebugFlag(void) {
#ifdef DEBUG
    return true;
#else
    return false;
#endif
}

template <typename T>
class Tensor {
public:
    //===================================================================
    //CONSTRUCTORS
    //===================================================================

    Tensor(void) {
        rows = 0;
        cols = 0;
    }

    Tensor(size_t r, size_t c) 
        : rows(r), cols(c){
        if(getDebugFlag()) printf("[Tensor Constructor]\n");
        m_data.resize(rows * cols);
    }

    Tensor(const std::array<size_t, 2>& index) 
        : rows(index[0]), cols(index[1]){
        m_data.resize(rows * cols);
    }

    Tensor(const Tensor& other) 
        : rows(other.rows), cols(other.cols) {
        m_data.resize(rows * cols);
        m_data = other.m_data;
    }

    Tensor(Tensor&& other) noexcept 
        : rows(other.rows), cols(other.cols), m_data(other.m_data) {
        if(getDebugFlag()) printf("[Tensor Move Constructor]\n");
    }

    Tensor(size_t r, size_t c, const T* arr) 
        : rows(r), cols(c) {
        m_data.resize(rows * cols);
        memcpy(m_data.data(), arr, sizeof(float) * rows * cols);
    }

    ~Tensor() { }

    //===================================================================
    //METHODS
    //===================================================================

    T& at(size_t i) {
        return m_data[i];
    }

    T at(size_t i) const {
        return m_data[i];
    }

    T& at(size_t r, size_t c) {
        return m_data[r * cols + c];
    }

    T at(size_t r, size_t c) const {
        return m_data[r * cols + c];
    }
    
    T at(const std::array<size_t, 2>& index) const {
        return m_data[index[0] * cols + index[1]];
    }

    std::array<size_t, 2> dim(void) const {
        return {rows, cols};
    }
    
    size_t size(void) const{
        return rows * cols;
    }

    void print(const char name[] = "Tensor") const {
        std::cout << name << " = [\n";
        for(size_t r = 0; r < rows; ++r) {
            std::cout << '\t';
            for(size_t c = 0; c < cols; ++c) {
                std::cout << std::setw(10) 
                          << std::fixed
                          << std::setprecision(5)
                          << at(r, c);
            }
            std::cout << '\n';
        }
        std::cout << "]" << std::endl;
    }
    
    const void* data() const {
        return m_data.data();
    }

    //=============================
    // MODIFY THE TENSOR
    //=============================
    Tensor& clear() {
        memset(m_data.data(), 0, sizeof(T) * m_data.size());
        return *this;
    }

    Tensor& fill(T fValue) {
        for(size_t i = 0; i < rows * cols; ++i)
            m_data[i] = fValue;
        return *this;
    }
    
    Tensor& random(T min = 0.0f, T max = 1.0f) {
        for(size_t i = 0; i < rows * cols; ++i) {
            T r = (rand() / (T)RAND_MAX);
            m_data[i] = r * (max - min) + min;
        }
        return *this;
    }
        
    Tensor& squared(void) {
        for(size_t r = 0; r < rows; ++r)
            for(size_t c = 0; c < cols; ++c) 
                this->at(r, c) *= this->at(r, c);
        return *this;
    }

    T average(void) const {
        T sum = (T)0;
        for(size_t r = 0; r < rows; ++r)
            for(size_t c = 0; c < cols; ++c) 
                sum += this->at(r, c);
        return ( sum / ((T)rows*cols) );
    }

    Tensor& copy(Tensor& src){
        assert( (src.rows * src.cols) == (rows * cols) );
        for(size_t i = 0; i < rows * cols; ++i) 
            m_data.at(i) = src.data->at(i);
        return *this;
    }

    Tensor& copy(std::vector<T>& data){
        assert( (data.size()) == (rows * cols) );
        for(size_t i = 0; i < rows * cols; ++i) 
            m_data.at(i) = data.at(i);
        return *this;
    }

    Tensor& resize(size_t r, size_t c){
        assert( (r * c) == (rows * cols) );
        rows = r;
        cols = c;
        return *this;
    }
    
    //=============================
    // RETURN NEW TENSOR 
    //=============================
    
    Tensor row(size_t startRow, size_t rowCount = 1) const {
        Tensor sub(rowCount, cols);
        
        for(size_t r = startRow; r < startRow + rowCount; ++r) {
            for(size_t c = 0; c < cols; ++c) {
                sub.at(r - startRow, c) = this->at(r, c);
            }
        }
        return sub;
    }

    Tensor col(size_t startCol, size_t colCount = 1) const {
        Tensor sub(rows, colCount);
        
        for(size_t r = 0; r < rows; ++r) {
            for(size_t c = startCol; c < startCol + colCount; ++c) {
                sub.at(r, c - startCol) = this->at(r, c);
            }
        }
        return sub;
    }

    Tensor transpose(void) const {
        Tensor transposed(cols, rows);
        for(size_t r = 0; r < rows; ++r)
            for(size_t c = 0; c < cols; ++c) 
                transposed.at(c, r) = this->at(r,c);
        return transposed;
    }

    Tensor apply(T func(T)) const {
        Tensor activated(rows, cols);
        for(size_t r = 0; r < rows; ++r)
            for(size_t c = 0; c < cols; ++c) 
                activated.at(r, c) = func(this->at(r,c));
        return activated;
    }

    //===================================================================
    // OPERATORS
    //===================================================================
    
    Tensor& operator=(Tensor&& other) noexcept {
        if(getDebugFlag()) printf("[Tensor Move Assignment]\n");
        if (this == &other) 
            return *this;
        rows = other.rows;
        cols = other.cols;
        m_data = other.m_data;
        return *this;
    }    
    
    Tensor& operator=(const Tensor& other) {
        if(getDebugFlag()) printf("[Tensor Copy Assignment]\n");
        if (this == &other) 
            return *this;
        rows = other.rows;
        cols = other.cols;
        m_data.resize(rows * cols);
        m_data = other.m_data;
        return *this;
    }

    Tensor& operator*=(T scalar) {
        for(size_t r = 0; r < rows; ++r)
            for(size_t c = 0; c < cols; ++c)
                this->at(r, c) *= scalar;
        return *this;
    }
    
    Tensor& operator/=(T scalar) {
        for(size_t r = 0; r < rows; ++r)
            for(size_t c = 0; c < cols; ++c)
                this->at(r, c) /= scalar;
        return *this;
    }

    Tensor& operator*=(const Tensor& other) {
    assert(this->dim() == other.dim());
        for(size_t r = 0; r < rows; ++r)
            for(size_t c = 0; c < cols; ++c)
                this->at(r, c) *= other.at(r, c);
        return *this;
    }

    Tensor& operator+=(const Tensor& other) {
    assert(this->dim() == other.dim());
        for(size_t r = 0; r < rows; ++r)
            for(size_t c = 0; c < cols; ++c)
                this->at(r, c) += other.at(r, c);
        return *this;
    }

    Tensor& operator-=(const Tensor& other) {
    assert(this->dim() == other.dim());
        for(size_t r = 0; r < rows; ++r)
            for(size_t c = 0; c < cols; ++c)
                this->at(r, c) -= other.at(r, c);
        return *this;
    }

    Tensor& operator=(const std::vector<T>& vecFloats) {
        assert(vecFloats.size() == rows * cols);
        m_data.resize(rows * cols);
        m_data = vecFloats;
        return *this;
    }

    template <typename U>
    friend Tensor<U> operator+(const Tensor<U>& lhs, U scalar);
    template <typename U>
    friend Tensor<U> operator-(const Tensor<U>& lhs, U scalar);
    template <typename U>
    friend Tensor<U> operator*(const Tensor<U>& lhs, U scalar);
    template <typename U>
    friend Tensor<U> operator/(const Tensor<U>& lhs, U scalar);
    template <typename U>
    friend Tensor<U> operator+(const Tensor<U>& lhs, const Tensor<U>& rhs);
    template <typename U>
    friend Tensor<U> operator-(const Tensor<U>& lhs, const Tensor<U>& rhs);
    template <typename U>
    friend Tensor<U> operator*(const Tensor<U>& lhs, const Tensor<U>& rhs);
    template <typename U>
    friend Tensor<U> hadamard(const Tensor<U>& lhs, const Tensor<U>& rhs);
    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& obj);
    //===================================================================
    //STATICS
    //===================================================================
    
    static Tensor identity(size_t r, size_t c) {
        Tensor<T> identity(r, c);
        for(size_t i = 0; i < r; ++i)
            for(size_t j = 0; j < c; ++j)
                if(i == j) 
                    identity.at(i, j) = 1;
        return identity;
    }

private:
    size_t rows, cols;
    std::vector <T> m_data;
};

//=============================
// RETURN NEW TENSOR 
//=============================
template <typename T>
Tensor<T> hadamard(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    assert(lhs.dim() == rhs.dim());
    Tensor<T> result(lhs.rows, lhs.cols);

    for(size_t r = 0; r < lhs.rows; ++r)
        for(size_t c = 0; c < lhs.cols; ++c)
            result.at(r, c) = lhs.at(r, c) * rhs.at(r, c);

    return result;
}
    
template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& obj) {
    size_t nRows = obj.dim()[0];
    size_t nCols = obj.dim()[1];

    os << '[';
    for(size_t r = 0; r < nRows; ++r) {
        for(size_t c = 0; c < nCols; ++c) {
            os << obj.at(r, c);// << (((r == (nRows - 1)) || (c == (nCols - 1)))?"":" ");
        }
        
        os << ((obj.dim()[0] == 1)? "": "\n");
    }
    os << ']';
    return os;
}

//Scalar Addition
template <typename T>
Tensor<T> operator+(const Tensor<T>& lhs, T scalar) {
    Tensor<T> result(lhs.rows, lhs.cols);
    
    for(size_t r = 0; r < lhs.rows; ++r)
        for(size_t c = 0; c < lhs.cols; ++c)
            result.at(r, c) = lhs.at(r, c) + scalar;
            
    return result;
}
    
//Scalar Subtraction
template <typename T>
Tensor<T> operator-(const Tensor<T>& lhs, T scalar) {
    Tensor<T> result(lhs.rows, lhs.cols);
    
    for(size_t r = 0; r < lhs.rows; ++r)
        for(size_t c = 0; c < lhs.cols; ++c)
            result.at(r, c) = lhs.at(r, c) - scalar;
            
    return result;
}

//Scalar Multiplication
template <typename T>
Tensor<T> operator*(const Tensor<T>& lhs, T scalar) {
    Tensor<T> result(lhs.rows, lhs.cols);
    
    for(size_t r = 0; r < lhs.rows; ++r)
        for(size_t c = 0; c < lhs.cols; ++c)
            result.at(r, c) = lhs.at(r, c) * scalar;
    return result;
}

//Scalar Division
template <typename T>
Tensor<T> operator/(const Tensor<T>& lhs, T scalar) {
    Tensor<T> result(lhs.rows, lhs.cols);
    
    for(size_t r = 0; r < lhs.rows; ++r)
        for(size_t c = 0; c < lhs.cols; ++c)
            result.at(r, c) = lhs.at(r, c) / scalar;
    return result;
}

//Matrix Element-wise Addition
template <typename T>
Tensor<T> operator+(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    assert(lhs.dim() == rhs.dim());
    Tensor<T> result(lhs.rows, lhs.cols);
    
    for(size_t r = 0; r < lhs.rows; ++r)
        for(size_t c = 0; c < lhs.cols; ++c)
            result.at(r, c) = lhs.at(r, c) + rhs.at(r, c);
            
    return result;
}
    
//Matrix Element-wise Subtraction
template <typename T>
Tensor<T> operator-(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    assert(lhs.dim() == rhs.dim());
    Tensor<T> result(lhs.rows, lhs.cols);
    
    for(size_t r = 0; r < lhs.rows; ++r)
        for(size_t c = 0; c < lhs.cols; ++c)
            result.at(r, c) = lhs.at(r, c) - rhs.at(r, c);
            
    return result;
}

//Cross Product
template <typename T>
Tensor<T> operator*(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    /*
        lhs.c == rhs.r
        RxC   x   RxC
    */
    assert(lhs.cols == rhs.rows);
    
    size_t outRows  = lhs.rows;
    size_t outCols  = rhs.cols;
    size_t inner    = lhs.cols;
    Tensor<T> resultant(outRows, outCols);
    
    for(size_t r = 0; r < outRows; ++r)
        for(size_t c = 0; c < outCols; ++c)
            for(size_t i = 0; i < inner; i++)
                resultant.at(r, c) += lhs.at(r, i) * rhs.at(i, c);
    
    return resultant;
}



