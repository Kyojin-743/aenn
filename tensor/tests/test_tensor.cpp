#include <gtest/gtest.h>
#include <tensor.h>

#include <cstdio>

using namespace AennTensor;

//This is a change

TEST(TensorTest, CalculateNumel) {
    auto numel = calculate_numel(Shape{2,2,2});
    ASSERT_EQ(numel, 8);
}

TEST(TensorTest, ZeroInitialize) {
    auto t = Tensor(Shape{2,2});
    ASSERT_EQ(t.size(), 4) << "Size should be 4!";
    ASSERT_EQ(t.ndim(), 2) << "Dimension should be 2!";

    auto data = t.data();
    for(auto i{0}; i < t.size(); i++) {
        ASSERT_EQ(data[i], 0) << "Should be 0!";
    }
}

TEST(TensorTest, InitializeFromVector) {
    auto v = std::vector<data_type>({1,2,3,4,5,6,7,8,9,10});
    auto t = Tensor(Shape{2,5}, v);

    auto data = t.data();
    for(auto i{0}; i < t.size(); i++) {
        ASSERT_EQ(data[i], i+1);
    }
}

TEST(TensorTest, Indexing) {
    auto v = std::vector<data_type>({1,2,3,4,5,6,7,8});
    auto t = Tensor(Shape{2,2,2}, v);

    auto counter{1};
    for(size_t i{0}; i < t.shape()[0]; i++) {
        for(size_t j{0}; j < t.shape()[1]; j++) {
            for(size_t k{0}; k < t.shape()[2]; k++) {
                ASSERT_EQ(t.at(Shape{i,j,k}), counter);
                counter++;
            }
        }
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}