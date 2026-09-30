#include <cstdio>
#include <iostream>
// #include <autograd.h>
#include <tensor.h>


int main(void) {

    auto t1 = AennTensor::Tensor::Random({3,4,2});

    // printf("%s\n", t1.c_str());
    std::cout << t1 << std::endl;
    std::cout << t1[{1,0,0}] << std::endl;
}