#include <cstdio>
#include <iostream>
// #include <autograd.h>
#include <tensor.h>

using namespace AennTensor;
using namespace std;
int main(void) {

    Tensor t({2, 3}, Vector{1, 2, 3, 
                            4, 5, 6});
    Tensor transposed = t.transpose();

    cout << "Original tensor:\n" << t << "\n\n";

    cout << "Transpose tensor:\n" << transposed << "\n\n";

    cout << "transposed[0,0]: " << transposed[{0, 0}] << "\n";
    cout << "transposed[0,1]: " << transposed[{0, 1}] << "\n";
    cout << "transposed[2,0]: " << transposed[{2, 0}] << "\n";
    cout << "transposed[2,1]: " << transposed[{2, 1}] << "\n";

}