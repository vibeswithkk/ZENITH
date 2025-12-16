// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#include <gtest/gtest.h>
#include <zenith/zenith.hpp>

// ============================================================================
// DataType Tests
// ============================================================================

TEST(TypesTest, DTypeSizeReturnsCorrectBytes) {
  EXPECT_EQ(zenith::dtype_size(zenith::DataType::Float32), 4);
  EXPECT_EQ(zenith::dtype_size(zenith::DataType::Float16), 2);
  EXPECT_EQ(zenith::dtype_size(zenith::DataType::Float64), 8);
  EXPECT_EQ(zenith::dtype_size(zenith::DataType::Int8), 1);
  EXPECT_EQ(zenith::dtype_size(zenith::DataType::Int32), 4);
  EXPECT_EQ(zenith::dtype_size(zenith::DataType::Int64), 8);
  EXPECT_EQ(zenith::dtype_size(zenith::DataType::Bool), 1);
}

TEST(TypesTest, DTypeToStringReturnsCorrectName) {
  EXPECT_EQ(zenith::dtype_to_string(zenith::DataType::Float32), "float32");
  EXPECT_EQ(zenith::dtype_to_string(zenith::DataType::Float16), "float16");
  EXPECT_EQ(zenith::dtype_to_string(zenith::DataType::Int8), "int8");
  EXPECT_EQ(zenith::dtype_to_string(zenith::DataType::Bool), "bool");
}

// ============================================================================
// Shape Tests
// ============================================================================

TEST(ShapeTest, DefaultConstructorCreatesEmptyShape) {
  zenith::Shape shape;
  EXPECT_EQ(shape.rank(), 0);
  EXPECT_EQ(shape.numel(), 0);
}

TEST(ShapeTest, InitializerListConstructorWorks) {
  zenith::Shape shape{2, 3, 4};
  EXPECT_EQ(shape.rank(), 3);
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);
}

TEST(ShapeTest, NumelCalculatesCorrectly) {
  zenith::Shape shape{2, 3, 4};
  EXPECT_EQ(shape.numel(), 24);
}

TEST(ShapeTest, DynamicShapeDetected) {
  zenith::Shape static_shape{2, 3, 4};
  EXPECT_FALSE(static_shape.is_dynamic());

  zenith::Shape dynamic_shape{2, -1, 4}; // -1 indicates dynamic
  EXPECT_TRUE(dynamic_shape.is_dynamic());
}

TEST(ShapeTest, EqualityComparison) {
  zenith::Shape a{2, 3, 4};
  zenith::Shape b{2, 3, 4};
  zenith::Shape c{2, 3, 5};

  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

// ============================================================================
// Status Tests
// ============================================================================

TEST(StatusTest, DefaultStatusIsOk) {
  zenith::Status status;
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(status.code(), zenith::StatusCode::Ok);
}

TEST(StatusTest, ErrorStatusWorks) {
  auto status = zenith::Status::Error(zenith::StatusCode::InvalidArgument,
                                      "Test error message");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), zenith::StatusCode::InvalidArgument);
  EXPECT_EQ(status.message(), "Test error message");
}
