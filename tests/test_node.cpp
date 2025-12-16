// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#include <gtest/gtest.h>
#include <zenith/zenith.hpp>

// ============================================================================
// Node Tests
// ============================================================================

TEST(NodeTest, DefaultConstructor) {
  zenith::Node node;
  EXPECT_EQ(node.op_type(), "");
  EXPECT_EQ(node.name(), "");
  EXPECT_EQ(node.num_inputs(), 0);
  EXPECT_EQ(node.num_outputs(), 0);
}

TEST(NodeTest, FullConstructor) {
  std::vector<zenith::TensorDescriptor> inputs = {zenith::TensorDescriptor(
      "input", zenith::Shape{1, 3, 224, 224}, zenith::DataType::Float32)};
  std::vector<zenith::TensorDescriptor> outputs = {zenith::TensorDescriptor(
      "output", zenith::Shape{1, 64, 112, 112}, zenith::DataType::Float32)};
  zenith::AttributeMap attrs = {
      {"kernel_size", int64_t(3)},
      {"stride", int64_t(2)},
  };

  zenith::Node node(zenith::ops::CONV, "conv1", inputs, outputs, attrs);

  EXPECT_EQ(node.op_type(), "Conv");
  EXPECT_EQ(node.name(), "conv1");
  EXPECT_EQ(node.num_inputs(), 1);
  EXPECT_EQ(node.num_outputs(), 1);
}

TEST(NodeTest, IsOpCheck) {
  zenith::Node node;
  node.set_op_type(zenith::ops::RELU);

  EXPECT_TRUE(node.is_op(zenith::ops::RELU));
  EXPECT_FALSE(node.is_op(zenith::ops::CONV));
}

TEST(NodeTest, AddInputOutput) {
  zenith::Node node;

  zenith::TensorDescriptor input("in", zenith::Shape{1, 10},
                                 zenith::DataType::Float32);
  zenith::TensorDescriptor output("out", zenith::Shape{1, 5},
                                  zenith::DataType::Float32);

  node.add_input(input);
  node.add_output(output);

  EXPECT_EQ(node.num_inputs(), 1);
  EXPECT_EQ(node.num_outputs(), 1);
}

TEST(NodeTest, AttributeAccess) {
  zenith::Node node;

  node.set_attr("learning_rate", 0.001);
  node.set_attr("num_layers", int64_t(5));
  node.set_attr("activation", std::string("relu"));

  EXPECT_TRUE(node.has_attr("learning_rate"));
  EXPECT_TRUE(node.has_attr("num_layers"));
  EXPECT_FALSE(node.has_attr("nonexistent"));

  auto lr = node.get_attr<double>("learning_rate");
  EXPECT_TRUE(lr.has_value());
  EXPECT_DOUBLE_EQ(lr.value(), 0.001);
}

TEST(NodeTest, Clone) {
  zenith::Node original(zenith::ops::MATMUL, "matmul1", {}, {});
  original.set_attr("transpose_a", false);

  auto cloned = original.clone();

  EXPECT_EQ(cloned->op_type(), original.op_type());
  EXPECT_EQ(cloned->name(), original.name());
  // ID should be different
  EXPECT_NE(cloned->id(), original.id());
}

TEST(NodeTest, OpTypeConstants) {
  EXPECT_EQ(std::string(zenith::ops::RELU), "Relu");
  EXPECT_EQ(std::string(zenith::ops::CONV), "Conv");
  EXPECT_EQ(std::string(zenith::ops::MATMUL), "MatMul");
  EXPECT_EQ(std::string(zenith::ops::BATCH_NORM), "BatchNormalization");
}
