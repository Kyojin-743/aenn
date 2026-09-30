#include <cstdio>
#include <iostream>
// #include <autograd.h>
#include <tensor.h>


int main(void) {


    auto t1 = AennTensor::Tensor::Random({3,4,2});

    std::cout << t1 << "\n\n";
    std::cout << t1[{1,0,0}] << "\n\n";

    auto t2 = AennTensor::Tensor(t1.shape, 2.0f);

    std::cout << t2 << "\n\n";

    std::cout << "Addition:\n" << t1 + t2 << "\n\n";
    std::cout << "Subtraction:\n" << t1 - t2 << "\n\n";
    std::cout << "Multiplication:\n" << t1 * t2 << "\n\n";
    std::cout << "Division:\n" << t1 / t2 << "\n\n";

    auto t3 = AennTensor::Tensor({3,3}, {1,2,3,4,5,6,7,8,9});

    std::cout << t3 << "\n\n";

    auto t4 = AennTensor::Tensor::Random({5,8});
    auto t5 = AennTensor::Tensor::Random({8,3});

    std::cout << t4 << "\n\n";
    std::cout << t5 << "\n\n";

    std::cout << AennTensor::matmul(t4, t5) << "\n\n";


    std::cout << std::endl;
}