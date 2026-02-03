#ifndef __TENSOR_H_
#define __TENSOR_H_

#include <cassert>
#include <array>
#include <vector>
#include <memory>
#include <iostream>

#include <random>
#include <algorithm>

template <typename T>
class Tensor {

    template <typename U>
    using vector = std::vector<U>;

    template <typename U>
    using shared_ptr = std::shared_ptr<U>;
    
    template <typename U, typename... Args>
    static auto make_shared(Args&&... args) {
        return std::make_shared<U>(std::forward<Args>(args)...);
    }

public:
    /*Default Shape will be 1xN where N is the length of data*/
    Tensor(const vector<T>& vData = vector<size_t>(), const vector<size_t>& vShape = vector<size_t>())
        : vData(vData), vShape(vShape) {
            assert(0 != vShape.size());  //Must have a shape
            if(0 != vData.size()) { //non-empty tensor init, validate size
                size_t nCount{1};
                for(size_t s : vShape) nCount *= s;
                assert(vData.size() == nCount);
            }
        }
    ~Tensor() {}

    static shared_ptr<Tensor<T>> zeros(const vector<size_t>& vShape) {
        size_t nCount{1};
        for(size_t s : vShape) nCount *= s;
        if(0 == vShape.size()) nCount = 0;
        assert(0 != nCount);
        return make_shared<Tensor<T>>(vector<T>(nCount, static_cast<T>(0)), vShape);
    }

    template <typename Dist = std::uniform_real_distribution<T>>
    static shared_ptr<Tensor<T>> random(const vector<size_t>& vShape, Dist dist = Dist(0, 1)){
        auto tRandom = make_shared<Tensor<T>>(vector<T>(), vShape);

        static std::random_device rd;
        static std::mt19937 gen(rd());

        size_t nCount{1};
        for(size_t dim : vShape) nCount *= dim;

        for(size_t i{0}; i < nCount; i++)
            tRandom->vData.emplace_back(dist(gen));
        
        return tRandom;
    }

    void print() {
        size_t i{0};
        for( auto t : vData){
            std::cout << t << ", ";
        }
    }

    vector<size_t> vShape;
    vector<T> vData;
};

#endif//_s_TENSOR_H_

#ifdef TENSOR_IMPLEMENTATION

#endif