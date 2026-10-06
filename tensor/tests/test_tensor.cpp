#include <gtest/gtest.h>
#include <tensor.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace AennTensor;

// Utility helper for floating point comparisons
constexpr float EPSILON = 1e-4f;

// --- CONSTRUCTORS & FACTORIES ---

TEST(Constructors, Empty) {
    // Edge case: Default-constructed tensor (empty data pointer/shape)
    Tensor t;
    EXPECT_EQ(t.data, nullptr);
    EXPECT_TRUE(t.shape.empty());
    EXPECT_EQ(t.size(), 0u);
}

TEST(Constructors, Fill) {
    // Normal case: 2D Tensor filled with default 0
    Tensor t1({2, 3});
    EXPECT_EQ(t1.shape, (Shape{2, 3}));
    EXPECT_EQ(t1.size(), 6u);
    for (size_t i = 0; i < t1.size(); ++i) {
        EXPECT_FLOAT_EQ(((float)t1[i]), 0.0f);
    }

    // Normal case: 3D Tensor filled with a specific value
    Tensor t2({2, 2, 2}, 3.14f);
    EXPECT_EQ(t2.shape, (Shape{2, 2, 2}));
    EXPECT_EQ(t2.size(), 8u);
    for (size_t i = 0; i < t2.size(); ++i) {
        EXPECT_FLOAT_EQ(((float)t2[i]), 3.14f);
    }

    // Edge case: 1D Tensor with zero size element
    Tensor t_zero({0});
    EXPECT_EQ(t_zero.size(), 0u);

    // Edge case: Factory functions (Zeroes and Ones)
    Tensor zeros = Tensor::Zeroes({3, 3});
    EXPECT_FLOAT_EQ(((float)zeros[{0, 0}]), 0.0f);

    Tensor ones = Tensor::Ones({3, 3});
    EXPECT_FLOAT_EQ(((float)ones[{0, 0}]), 1.0f);
}

TEST(Constructors, Vector) {
    // Normal case: Initialized with a vector matching shape total size
    Vector data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t({2, 2}, data);
    EXPECT_EQ(t.size(), 4u);
    EXPECT_FLOAT_EQ(((float)t[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ(((float)t[{1, 1}]), 4.0f);

    // Edge case: Copy Constructor
    Tensor copy_t(t);
    EXPECT_EQ(copy_t.shape, t.shape);
    EXPECT_FLOAT_EQ(((float)copy_t[{0, 0}]), 1.0f);
}

TEST(Constructors, Random) {
    // Normal case: Uniform distribution bounded check
    Shape shape = {10, 10};
    float min_val = -2.0f;
    float max_val = 5.0f;
    Tensor rand_u = Tensor::Random(shape, min_val, max_val, Distribution::Uniform);

    EXPECT_EQ(rand_u.shape, shape);
    for (size_t i = 0; i < rand_u.size(); ++i) {
        EXPECT_GE(rand_u[i], min_val);
        EXPECT_LE(rand_u[i], max_val);
    }

    // Normal case: Normal distribution check
    Tensor rand_n = Tensor::Random(shape, 0.0f, 1.0f, Distribution::Normal);
    EXPECT_EQ(rand_n.shape, shape);
}


// --- SCALAR OPERATORS ---

TEST(Operators, AddScalar) {
    Tensor t({2, 2}, 2.0f);
    Tensor res = t + 3.0f;

    EXPECT_FLOAT_EQ(((float)res[{0, 0}]), 5.0f);
    // Zero add edge case
    Tensor res_zero = t + 0.0f;
    EXPECT_FLOAT_EQ(((float)res_zero[{0, 0}]), 2.0f);
}

TEST(Operators, SubScalar) {
    Tensor t({2, 2}, 5.0f);
    Tensor res = t - 3.0f;

    EXPECT_FLOAT_EQ(((float)res[{0, 0}]), 2.0f);
    // Negative result check
    Tensor res_neg = t - 10.0f;
    EXPECT_FLOAT_EQ(((float)res_neg[{0, 0}]), -5.0f);
}

TEST(Operators, MulScalar) {
    Tensor t({2, 2}, 3.0f);
    Tensor res = t * 2.5f;

    EXPECT_FLOAT_EQ(((float)res[{0, 0}]), 7.5f);
    // Zero multiplication edge case
    Tensor res_zero = t * 0.0f;
    EXPECT_FLOAT_EQ(((float)res_zero[{0, 0}]), 0.0f);
}

TEST(Operators, DivScalar) {
    Tensor t({2, 2}, 10.0f);
    Tensor res = t / 2.0f;

    EXPECT_FLOAT_EQ(((float)res[{0, 0}]), 5.0f);
    // Edge case: Division by infinity / zero boundary check
    Tensor res_inf = t / 0.0f;
    EXPECT_TRUE(std::isinf(res_inf[{0, 0}]));
}


// --- TENSOR-TENSOR OPERATORS ---

TEST(Operators, AddTensor) {
    Tensor a({2, 2}, Vector{1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b({2, 2}, Vector{10.0f, 20.0f, 30.0f, 40.0f});
    Tensor res = a + b;

    EXPECT_FLOAT_EQ(((float)res[{0, 0}]), 11.0f);
    EXPECT_FLOAT_EQ(((float)res[{1, 1}]), 44.0f);
}

TEST(Operators, SubTensor) {
    Tensor a({2, 2}, Vector{10.0f, 20.0f, 30.0f, 40.0f});
    Tensor b({2, 2}, Vector{1.0f, 2.0f, 3.0f, 4.0f});
    Tensor res = a - b;

    EXPECT_FLOAT_EQ(((float)res[{0, 0}]), 9.0f);
    EXPECT_FLOAT_EQ(((float)res[{1, 1}]), 36.0f);
}

TEST(Operators, MulTensor) {
    // Element-wise multiplication
    Tensor a({2, 2}, Vector{2.0f, 3.0f, 4.0f, 5.0f});
    Tensor b({2, 2}, Vector{3.0f, 4.0f, 5.0f, 6.0f});
    Tensor res = a * b;

    EXPECT_FLOAT_EQ(((float)res[{0, 0}]), 6.0f);
    EXPECT_FLOAT_EQ(((float)res[{1, 1}]), 30.0f);
}

TEST(Operators, DivTensor) {
    // Element-wise division
    Tensor a({2, 2}, Vector{10.0f, 12.0f, 14.0f, 16.0f});
    Tensor b({2, 2}, Vector{2.0f, 3.0f, 2.0f, 4.0f});
    Tensor res = a / b;

    EXPECT_FLOAT_EQ(((float)res[{0, 0}]), 5.0f);
    EXPECT_FLOAT_EQ(((float)res[{1, 1}]), 4.0f);
}


// --- INDEXING OPERATORS ---

TEST(Operators, IndexCoordinate) {
    Tensor t({2, 3}, Vector{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Read access
    EXPECT_FLOAT_EQ(((float)t[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ(((float)t[{0, 2}]), 3.0f);
    EXPECT_FLOAT_EQ(((float)t[{1, 0}]), 4.0f);

    // Write access
    t[{1, 2}] = 99.0f;
    EXPECT_FLOAT_EQ(((float)t[{1, 2}]), 99.0f);
}

TEST(Operators, IndexFlat) {
    Tensor t({2, 3}, Vector{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});

    // 1D flat access
    EXPECT_FLOAT_EQ(((float)t[0u]), 10.0f);
    EXPECT_FLOAT_EQ(((float)t[5u]), 60.0f);

    // Write modification via flat index
    t[2u] = 300.0f;
    EXPECT_FLOAT_EQ(((float)t[{0, 2}]), 300.0f);
}

TEST(Operators, IndexSlice) {
    Tensor t({4, 4}, 1.0f);
    
    // Slice operator shorthand test
    Ranges ranges = {{0, 2}, {1, 3}}; // slice rows 0..2, cols 1..3
    Tensor sliced = t[ranges];

    EXPECT_EQ(sliced.shape, (Shape{2, 2}));
}


// --- HELPER & STRING FUNCTIONS ---

TEST(Functions, toString) {
    Tensor t({2, 2}, Vector{1.0f, 2.0f, 3.0f, 4.0f});

    std::string str_pretty = t.tostr(true);
    std::string str_raw = t.tostr(false);

    EXPECT_FALSE(str_pretty.empty());
    EXPECT_FALSE(str_raw.empty());
    EXPECT_NE(t.c_str(), nullptr);
}

TEST(Functions, FlatIdx) {
    Tensor t({2, 3, 4}); // 3D Tensor
    Index idx = {1, 2, 3};

    size_t flat = t.flat_index(idx);
    EXPECT_LT(flat, t.size());

    // Origin index
    EXPECT_EQ(t.flat_index({0, 0, 0}), 0u);
}

TEST(Functions, FromFlat) {
    Tensor t({2, 3, 4});
    size_t target_flat = 15;

    Index idx = t.from_flat(target_flat);
    size_t recomputed_flat = t.flat_index(idx);

    EXPECT_EQ(recomputed_flat, target_flat);

    // Origin boundary test
    Index origin = t.from_flat(0);
    EXPECT_EQ(origin, (Index{0, 0, 0}));
}


// --- TENSOR MANIPULATION ---

TEST(Manipulation, Reshape) {
    Tensor t({2, 6}, Vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    // Reshape 2D -> 3D preserving element count
    Tensor reshaped = t.reshape({2, 3, 2});
    EXPECT_EQ(reshaped.shape, (Shape{2, 3, 2}));
    EXPECT_EQ(reshaped.size(), 12u);
    EXPECT_FLOAT_EQ(((float)reshaped[0u]), 1.0f);
    EXPECT_FLOAT_EQ(((float)reshaped[11u]), 12.0f);

    // Reshape 2D -> 1D
    Tensor flat = t.reshape({12});
    EXPECT_EQ(flat.shape, (Shape{12}));
}

TEST(Manipulation, Transpose) {
    // 2D matrix transpose
    Tensor t({2, 3}, Vector{1, 2, 3, 
                            4, 5, 6});
    Tensor transposed = t.transpose();

    EXPECT_EQ(transposed.shape, (Shape{3, 2}));
    EXPECT_FLOAT_EQ(((float)transposed[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ(((float)transposed[{0, 1}]), 4.0f);
    EXPECT_FLOAT_EQ(((float)transposed[{2, 0}]), 3.0f);
    EXPECT_FLOAT_EQ(((float)transposed[{2, 1}]), 6.0f);
}

// --- TENSOR Reductions ---

TEST(Reduction, Sum) {
    Tensor t({2, 3}, Vector{1, 2, 3, 
                            4, 5, 6});

    // Sum along axis 0 (columns sum)
    Tensor sum_axis0 = t.sum({0}, false);
    EXPECT_EQ(sum_axis0.size(), 3u);
    EXPECT_FLOAT_EQ(((float)sum_axis0[0u]), 5.0f);
    EXPECT_FLOAT_EQ(((float)sum_axis0[1u]), 7.0f);
    EXPECT_FLOAT_EQ(((float)sum_axis0[2u]), 9.0f);

    // Sum along axis 1 keeping dimensions
    Tensor sum_axis1_keep = t.sum({1}, true);
    EXPECT_EQ(sum_axis1_keep.shape, (Shape{2, 1}));
    EXPECT_FLOAT_EQ(((float)sum_axis1_keep[{0, 0}]), 6.0f);
    EXPECT_FLOAT_EQ(((float)sum_axis1_keep[{1, 0}]), 15.0f);
}

TEST(Reduction, Mean) {
    Tensor t({2, 2}, Vector{2.0f, 4.0f, 
                            6.0f, 8.0f});

    // Mean across all dimensions / single axis
    Tensor mean_axis = t.mean({0}, false);
    EXPECT_FLOAT_EQ(((float)mean_axis[0u]), 4.0f);
    EXPECT_FLOAT_EQ(((float)mean_axis[1u]), 6.0f);
}

TEST(Reduction, Min) {
    Tensor t({2, 3}, Vector{10, 2, 30, 
                            4,  5, 1});

    // Min along axis 1
    Tensor min_axis1 = t.min({1}, false);
    EXPECT_FLOAT_EQ(((float)min_axis1[0u]), 2.0f);
    EXPECT_FLOAT_EQ(((float)min_axis1[1u]), 1.0f);
}

TEST(Reduction, Max) {
    Tensor t({2, 3}, Vector{10, 2, 30, 
                            4,  5, 1});

    // Max along axis 0
    Tensor max_axis0 = t.max({0}, false);
    EXPECT_FLOAT_EQ(((float)max_axis0[0u]), 10.0f);
    EXPECT_FLOAT_EQ(((float)max_axis0[1u]), 5.0f);
    EXPECT_FLOAT_EQ(((float)max_axis0[2u]), 30.0f);
}

// --- GLOBAL UTILITIES ---

TEST(GlobalFunctions, Matmul) {
    // Matrix multiplication: (2x3) * (3x2) -> (2x2)
    Tensor a({2, 3}, Vector{1, 2, 3, 
                            4, 5, 6});
    Tensor b({3, 2}, Vector{7, 8, 
                            9, 1, 
                            2, 3});

    Tensor res = matmul(a, b);
    EXPECT_EQ(res.shape, (Shape{2, 2}));
    EXPECT_FLOAT_EQ(((float)res[{0, 0}]), 31.0f); // 1*7 + 2*9 + 3*2
    EXPECT_FLOAT_EQ(((float)res[{0, 1}]), 19.0f); // 1*8 + 2*1 + 3*3
    EXPECT_FLOAT_EQ(((float)res[{1, 0}]), 85.0f); // 4*7 + 5*9 + 6*2
    EXPECT_FLOAT_EQ(((float)res[{1, 1}]), 55.0f); // 4*8 + 5*1 + 6*3
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}