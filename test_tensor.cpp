#include <ctime>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <chrono>
#include <ctime>

#include "tensor.h"


int main(void) {

    auto z = Tensor<float>::random({3, 3});

    z->print();

    return 0;
}