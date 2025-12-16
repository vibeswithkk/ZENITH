// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#include "zenith/kernels.hpp"

namespace zenith {
namespace kernels {
namespace testing {

// ============================================================================
// Utility Functions
// ============================================================================

// Compare two float arrays with tolerance
bool arrays_almost_equal(const float *a, const float *b, size_t size,
                         float rtol = 1e-5f, float atol = 1e-6f) {
  for (size_t i = 0; i < size; ++i) {
    float diff = std::abs(a[i] - b[i]);
    float max_val = std::max(std::abs(a[i]), std::abs(b[i]));
    if (diff > atol + rtol * max_val) {
      return false;
    }
  }
  return true;
}

// ============================================================================
// CPU Feature Detection Tests
// ============================================================================

TEST(CpuFeaturesTest, DetectFeatures) {
  auto features = get_cpu_features();
  // These should at least not crash
  (void)features.has_sse;
  (void)features.has_sse2;
  (void)features.has_avx;
  (void)features.has_avx2;
  (void)features.has_fma;
}

// ============================================================================
// Memory Utilities Tests
// ============================================================================

TEST(MemoryTest, AlignedAlloc) {
  void *ptr = aligned_alloc(1024, 32);
  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(is_aligned(ptr, 32));
  aligned_free(ptr);
}

TEST(MemoryTest, AlignedAlloc64) {
  void *ptr = aligned_alloc(1024, 64);
  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(is_aligned(ptr, 64));
  aligned_free(ptr);
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

TEST(MatMulTest, SmallMatrix) {
  // A: 2x3, B: 3x2, C: 2x2
  float A[] = {1, 2, 3, 4, 5, 6};
  float B[] = {1, 2, 3, 4, 5, 6};
  float C[4] = {0};

  matmul_f32(A, B, C, 2, 2, 3);

  // Expected: C = A @ B
  // [1,2,3] @ [1,2]   = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
  //          [3,4]
  //          [5,6]
  // [4,5,6] @ same    = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]

  EXPECT_NEAR(C[0], 22.0f, 1e-5f);
  EXPECT_NEAR(C[1], 28.0f, 1e-5f);
  EXPECT_NEAR(C[2], 49.0f, 1e-5f);
  EXPECT_NEAR(C[3], 64.0f, 1e-5f);
}

TEST(MatMulTest, LargerMatrix) {
  const int M = 16, N = 16, K = 16;
  std::vector<float> A(M * K), B(K * N), C(M * N), C_ref(M * N);

  // Initialize with simple values
  for (int i = 0; i < M * K; ++i)
    A[i] = static_cast<float>(i % 10) * 0.1f;
  for (int i = 0; i < K * N; ++i)
    B[i] = static_cast<float>(i % 10) * 0.1f;

  // Compute using kernel
  matmul_f32(A.data(), B.data(), C.data(), M, N, K);

  // Compute reference
  matmul_f32_naive(A.data(), B.data(), C_ref.data(), M, N, K);

  // Compare
  EXPECT_TRUE(arrays_almost_equal(C.data(), C_ref.data(), M * N));
}

TEST(MatMulTest, AVX2VsNaive) {
  if (!get_cpu_features().has_avx2) {
    GTEST_SKIP() << "AVX2 not available";
  }

  const int M = 64, N = 64, K = 64;
  std::vector<float> A(M * K), B(K * N), C_avx2(M * N), C_naive(M * N);

  // Random-ish initialization
  for (int i = 0; i < M * K; ++i)
    A[i] = static_cast<float>((i * 7) % 100) * 0.01f;
  for (int i = 0; i < K * N; ++i)
    B[i] = static_cast<float>((i * 11) % 100) * 0.01f;

  matmul_f32_avx2(A.data(), B.data(), C_avx2.data(), M, N, K);
  matmul_f32_naive(A.data(), B.data(), C_naive.data(), M, N, K);

  EXPECT_TRUE(arrays_almost_equal(C_avx2.data(), C_naive.data(), M * N));
}

// ============================================================================
// ReLU Tests
// ============================================================================

TEST(ReluTest, Basic) {
  float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  relu_f32(data, 5);

  EXPECT_EQ(data[0], 0.0f);
  EXPECT_EQ(data[1], 0.0f);
  EXPECT_EQ(data[2], 0.0f);
  EXPECT_EQ(data[3], 1.0f);
  EXPECT_EQ(data[4], 2.0f);
}

TEST(ReluTest, AVX2) {
  if (!get_cpu_features().has_avx2) {
    GTEST_SKIP() << "AVX2 not available";
  }

  std::vector<float> data(32);
  for (int i = 0; i < 32; ++i) {
    data[i] = static_cast<float>(i - 16);
  }

  relu_f32_avx2(data.data(), data.size());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(data[i], 0.0f);
  }
  for (int i = 16; i < 32; ++i) {
    EXPECT_EQ(data[i], static_cast<float>(i - 16));
  }
}

// ============================================================================
// Element-wise Operations Tests
// ============================================================================

TEST(AddTest, Basic) {
  float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float B[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float C[4];

  add_f32(A, B, C, 4);

  EXPECT_EQ(C[0], 6.0f);
  EXPECT_EQ(C[1], 8.0f);
  EXPECT_EQ(C[2], 10.0f);
  EXPECT_EQ(C[3], 12.0f);
}

TEST(SubTest, Basic) {
  float A[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float B[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float C[4];

  sub_f32(A, B, C, 4);

  EXPECT_EQ(C[0], 4.0f);
  EXPECT_EQ(C[1], 4.0f);
  EXPECT_EQ(C[2], 4.0f);
  EXPECT_EQ(C[3], 4.0f);
}

TEST(MulTest, Basic) {
  float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float B[] = {2.0f, 3.0f, 4.0f, 5.0f};
  float C[4];

  mul_f32(A, B, C, 4);

  EXPECT_EQ(C[0], 2.0f);
  EXPECT_EQ(C[1], 6.0f);
  EXPECT_EQ(C[2], 12.0f);
  EXPECT_EQ(C[3], 20.0f);
}

// ============================================================================
// Reduction Tests
// ============================================================================

TEST(SumTest, Basic) {
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float result = sum_f32(data, 5);
  EXPECT_NEAR(result, 15.0f, 1e-5f);
}

TEST(MeanTest, Basic) {
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float result = mean_f32(data, 5);
  EXPECT_NEAR(result, 3.0f, 1e-5f);
}

TEST(MaxTest, Basic) {
  float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
  float result = max_f32(data, 5);
  EXPECT_EQ(result, 5.0f);
}

TEST(MinTest, Basic) {
  float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
  float result = min_f32(data, 5);
  EXPECT_EQ(result, 1.0f);
}

// ============================================================================
// Activation Function Tests
// ============================================================================

TEST(SigmoidTest, Basic) {
  float data[] = {0.0f, 1.0f, -1.0f};
  sigmoid_f32(data, 3);

  EXPECT_NEAR(data[0], 0.5f, 1e-5f);
  EXPECT_NEAR(data[1], 0.7310586f, 1e-5f);
  EXPECT_NEAR(data[2], 0.2689414f, 1e-5f);
}

TEST(TanhTest, Basic) {
  float data[] = {0.0f, 1.0f, -1.0f};
  tanh_f32(data, 3);

  EXPECT_NEAR(data[0], 0.0f, 1e-5f);
  EXPECT_NEAR(data[1], 0.7615942f, 1e-5f);
  EXPECT_NEAR(data[2], -0.7615942f, 1e-5f);
}

// ============================================================================
// Softmax Tests
// ============================================================================

TEST(SoftmaxTest, Basic) {
  float input[] = {1.0f, 2.0f, 3.0f};
  float output[3];

  softmax_f32(input, output, 3, 3);

  // Softmax should sum to 1
  float sum = output[0] + output[1] + output[2];
  EXPECT_NEAR(sum, 1.0f, 1e-5f);

  // Larger inputs should have larger outputs
  EXPECT_LT(output[0], output[1]);
  EXPECT_LT(output[1], output[2]);
}

// ============================================================================
// Convolution Tests
// ============================================================================

TEST(Conv2DTest, SimpleConv) {
  // Input: 1x1x4x4 (N, C, H, W)
  float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  // Weight: 1x1x2x2 (C_out, C_in, K_h, K_w)
  float weight[4] = {1, 0, 0, 1}; // Diagonal filter
  float output[9] = {0};

  conv2d_f32(input, weight, nullptr, output, 1, 1, 4, 4, 1, 2, 2, 1, 1, 0, 0);

  // Output should be 3x3
  // Expected: top-left + bottom-right of each 2x2 window
  EXPECT_NEAR(output[0], 1 + 6, 1e-5f);   // 7
  EXPECT_NEAR(output[1], 2 + 7, 1e-5f);   // 9
  EXPECT_NEAR(output[2], 3 + 8, 1e-5f);   // 11
  EXPECT_NEAR(output[3], 5 + 10, 1e-5f);  // 15
  EXPECT_NEAR(output[4], 6 + 11, 1e-5f);  // 17
  EXPECT_NEAR(output[5], 7 + 12, 1e-5f);  // 19
  EXPECT_NEAR(output[6], 9 + 14, 1e-5f);  // 23
  EXPECT_NEAR(output[7], 10 + 15, 1e-5f); // 25
  EXPECT_NEAR(output[8], 11 + 16, 1e-5f); // 27
}

// ============================================================================
// MaxPool Tests
// ============================================================================

TEST(MaxPoolTest, Simple2x2) {
  // Input: 1x1x4x4
  float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float output[4] = {0};

  maxpool2d_f32(input, output, 1, 1, 4, 4, 2, 2, 2, 2);

  // Output: 1x1x2x2, each is max of 2x2 block
  EXPECT_EQ(output[0], 6.0f);  // max(1,2,5,6)
  EXPECT_EQ(output[1], 8.0f);  // max(3,4,7,8)
  EXPECT_EQ(output[2], 14.0f); // max(9,10,13,14)
  EXPECT_EQ(output[3], 16.0f); // max(11,12,15,16)
}

// ============================================================================
// AvgPool Tests
// ============================================================================

TEST(AvgPoolTest, Simple2x2) {
  // Input: 1x1x4x4
  float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float output[4] = {0};

  avgpool2d_f32(input, output, 1, 1, 4, 4, 2, 2, 2, 2);

  // Output: 1x1x2x2, each is avg of 2x2 block
  EXPECT_NEAR(output[0], (1 + 2 + 5 + 6) / 4.0f, 1e-5f);
  EXPECT_NEAR(output[1], (3 + 4 + 7 + 8) / 4.0f, 1e-5f);
  EXPECT_NEAR(output[2], (9 + 10 + 13 + 14) / 4.0f, 1e-5f);
  EXPECT_NEAR(output[3], (11 + 12 + 15 + 16) / 4.0f, 1e-5f);
}

} // namespace testing
} // namespace kernels
} // namespace zenith
