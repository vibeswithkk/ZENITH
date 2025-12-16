// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#include <gtest/gtest.h>
#include <zenith/zenith.hpp>

// ============================================================================
// TensorDescriptor Tests
// ============================================================================

TEST(TensorTest, DefaultConstructor) {
  zenith::TensorDescriptor tensor;
  EXPECT_EQ(tensor.name(), "");
  EXPECT_EQ(tensor.dtype(), zenith::DataType::Float32);
  EXPECT_EQ(tensor.layout(), zenith::Layout::NCHW);
}

TEST(TensorTest, FullConstructor) {
  zenith::TensorDescriptor tensor("my_tensor", zenith::Shape{1, 3, 224, 224},
                                  zenith::DataType::Float16,
                                  zenith::Layout::NCHW);

  EXPECT_EQ(tensor.name(), "my_tensor");
  EXPECT_EQ(tensor.shape().rank(), 4);
  EXPECT_EQ(tensor.dtype(), zenith::DataType::Float16);
  EXPECT_EQ(tensor.layout(), zenith::Layout::NCHW);
}

TEST(TensorTest, SizeBytesCalculation) {
  zenith::TensorDescriptor tensor("test", zenith::Shape{2, 3, 4}, // 24 elements
                                  zenith::DataType::Float32, // 4 bytes each
                                  zenith::Layout::NCHW);

  EXPECT_EQ(tensor.size_bytes(), 96); // 24 * 4 = 96
}

TEST(TensorTest, IsValidCheck) {
  zenith::TensorDescriptor empty;
  EXPECT_FALSE(empty.is_valid());

  zenith::TensorDescriptor valid("valid", zenith::Shape{1, 2, 3},
                                 zenith::DataType::Float32);
  EXPECT_TRUE(valid.is_valid());
}

TEST(TensorTest, Mutators) {
  zenith::TensorDescriptor tensor;

  tensor.set_name("updated");
  EXPECT_EQ(tensor.name(), "updated");

  tensor.set_dtype(zenith::DataType::Int8);
  EXPECT_EQ(tensor.dtype(), zenith::DataType::Int8);

  tensor.set_layout(zenith::Layout::NHWC);
  EXPECT_EQ(tensor.layout(), zenith::Layout::NHWC);
}
