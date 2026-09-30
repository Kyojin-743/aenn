// #include <gtest/gtest.h>
// #include <tensor.h>

// #include <cstdio>

// using namespace AennTensor;

// TEST(TensorTest, ZeroInitialize) {
//     auto t = Tensor(Shape{2,2});
//     ASSERT_EQ(t.size(), 4) << "Size should be 4!";
//     ASSERT_EQ(t.ndim(), 2) << "Dimension should be 2!";

//     auto data = t.data();
//     for(auto i{0}; i < t.size(); i++) {
//         ASSERT_EQ(data[i], 0) << "Should be 0!";
//     }
// }

// TEST(TensorTest, InitializeFromVector) {
//     auto v = std::vector<data_type>({1,2,3,4,5,6,7,8,9,10});
//     auto t = Tensor(Shape{2,5}, v);

//     auto data = t.data();
//     for(auto i{0}; i < t.size(); i++) {
//         ASSERT_EQ(data[i], i+1);
//     }
// }

// TEST(TensorTest, Indexing) {
//     auto v = std::vector<data_type>({1,2,3,4,5,6,7,8});
//     auto t = Tensor(Shape{2,2,2}, v);

//     auto counter{1};
//     for(size_t i{0}; i < t.shape()[0]; i++) {
//         for(size_t j{0}; j < t.shape()[1]; j++) {
//             for(size_t k{0}; k < t.shape()[2]; k++) {
//                 ASSERT_EQ(t.at(Shape{i,j,k}), counter);
//                 counter++;
//             }
//         }
//     }
// }

// TEST(TensorTest, Operations) {
//     auto v1 = std::vector<data_type>({1.0,1.0,1.0,1.0});
//     auto v2 = std::vector<data_type>({2.0,2.0,2.0,2.0});
    
//     auto t1 = Tensor(Shape{2,2}, v1);
//     auto t2 = Tensor(Shape{2,2}, v2);

//     auto add_tensor   = t1.add(t2);
//     auto piece_tensor = t1.mul(t2);
//     auto mul_tensor   = t1.matmul(t2);

//     auto add   = add_tensor.data();
//     auto piece = piece_tensor.data();
//     // auto mul   = mul_tensor.data(); 

//     float add_exp[] = {3.0, 3.0, 3.0, 3.0};
//     float piece_exp[] = {2.0, 2.0, 2.0, 2.0};
//     // float mul_exp[] = {4.0, 4.0, 4.0, 4.0};

//     for(size_t i(0); i < 4; ++i) {
//         ASSERT_EQ(add_exp[i], add[i]) << "exp: " << add_exp[i] << " | got: " << add[i];
//         ASSERT_EQ(piece_exp[i], piece[i]) << "exp: " << piece_exp[i] << " | got: " << piece[i];;
//         // ASSERT_EQ(mul_exp[i], mul[i]) << "exp: " << mul_exp[i] << " | got: " << mul[i];;
//     }
// }

// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }