// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#include <gtest/gtest.h>
#include <zenith/zenith.hpp>

// ============================================================================
// GraphIR Tests
// ============================================================================

TEST(GraphIRTest, DefaultConstructor) {
  zenith::GraphIR graph;
  EXPECT_EQ(graph.name(), "");
  EXPECT_EQ(graph.num_nodes(), 0);
}

TEST(GraphIRTest, NamedConstructor) {
  zenith::GraphIR graph("my_model");
  EXPECT_EQ(graph.name(), "my_model");
}

TEST(GraphIRTest, AddNode) {
  zenith::GraphIR graph("test_graph");

  zenith::TensorDescriptor input("x", zenith::Shape{1, 10},
                                 zenith::DataType::Float32);
  zenith::TensorDescriptor output("y", zenith::Shape{1, 10},
                                  zenith::DataType::Float32);

  auto *node = graph.add_node(zenith::ops::RELU, "relu1", {input}, {output});

  EXPECT_EQ(graph.num_nodes(), 1);
  EXPECT_NE(node, nullptr);
  EXPECT_EQ(node->name(), "relu1");
}

TEST(GraphIRTest, GetNodeByName) {
  zenith::GraphIR graph("test");

  graph.add_node(zenith::ops::RELU, "relu1", {}, {});
  graph.add_node(zenith::ops::SIGMOID, "sigmoid1", {}, {});

  auto *found = graph.get_node("relu1");
  EXPECT_NE(found, nullptr);
  EXPECT_EQ(found->op_type(), zenith::ops::RELU);

  auto *not_found = graph.get_node("nonexistent");
  EXPECT_EQ(not_found, nullptr);
}

TEST(GraphIRTest, RemoveNode) {
  zenith::GraphIR graph;
  graph.add_node(zenith::ops::RELU, "relu1", {}, {});
  graph.add_node(zenith::ops::SIGMOID, "sigmoid1", {}, {});

  EXPECT_EQ(graph.num_nodes(), 2);

  bool removed = graph.remove_node("relu1");
  EXPECT_TRUE(removed);
  EXPECT_EQ(graph.num_nodes(), 1);
  EXPECT_EQ(graph.get_node("relu1"), nullptr);

  bool not_removed = graph.remove_node("nonexistent");
  EXPECT_FALSE(not_removed);
}

TEST(GraphIRTest, InputsOutputs) {
  zenith::GraphIR graph;

  zenith::TensorDescriptor input("input", zenith::Shape{1, 3, 224, 224},
                                 zenith::DataType::Float32);
  zenith::TensorDescriptor output("output", zenith::Shape{1, 1000},
                                  zenith::DataType::Float32);

  graph.add_input(input);
  graph.add_output(output);

  EXPECT_EQ(graph.inputs().size(), 1);
  EXPECT_EQ(graph.outputs().size(), 1);
  EXPECT_EQ(graph.inputs()[0].name(), "input");
  EXPECT_EQ(graph.outputs()[0].name(), "output");
}

TEST(GraphIRTest, FindNodesByOp) {
  zenith::GraphIR graph;

  graph.add_node(zenith::ops::CONV, "conv1", {}, {});
  graph.add_node(zenith::ops::RELU, "relu1", {}, {});
  graph.add_node(zenith::ops::CONV, "conv2", {}, {});
  graph.add_node(zenith::ops::RELU, "relu2", {}, {});

  auto convs = graph.find_nodes_by_op(zenith::ops::CONV);
  EXPECT_EQ(convs.size(), 2);

  auto relus = graph.find_nodes_by_op(zenith::ops::RELU);
  EXPECT_EQ(relus.size(), 2);

  auto pools = graph.find_nodes_by_op(zenith::ops::MAX_POOL);
  EXPECT_EQ(pools.size(), 0);
}

TEST(GraphIRTest, CountOps) {
  zenith::GraphIR graph;

  graph.add_node(zenith::ops::CONV, "conv1", {}, {});
  graph.add_node(zenith::ops::CONV, "conv2", {}, {});
  graph.add_node(zenith::ops::RELU, "relu1", {}, {});

  auto counts = graph.count_ops();

  EXPECT_EQ(counts[zenith::ops::CONV], 2);
  EXPECT_EQ(counts[zenith::ops::RELU], 1);
}

TEST(GraphIRTest, ValidationFailsOnEmpty) {
  zenith::GraphIR graph;

  auto status = graph.validate();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), zenith::StatusCode::InvalidGraph);
}

TEST(GraphIRTest, ValidationFailsOnNoInputs) {
  zenith::GraphIR graph;
  graph.add_node(zenith::ops::RELU, "relu1", {}, {});

  auto status = graph.validate();
  EXPECT_FALSE(status.ok());
}

TEST(GraphIRTest, ValidationPassesOnValidGraph) {
  zenith::GraphIR graph;

  zenith::TensorDescriptor input("input", zenith::Shape{1, 10},
                                 zenith::DataType::Float32);
  zenith::TensorDescriptor output("output", zenith::Shape{1, 10},
                                  zenith::DataType::Float32);

  graph.add_input(input);
  graph.add_output(output);
  graph.add_node(zenith::ops::RELU, "relu1", {input}, {output});

  auto status = graph.validate();
  EXPECT_TRUE(status.ok());
}

TEST(GraphIRTest, Clone) {
  zenith::GraphIR original("original");
  original.add_input(zenith::TensorDescriptor("in", zenith::Shape{1},
                                              zenith::DataType::Float32));
  original.add_output(zenith::TensorDescriptor("out", zenith::Shape{1},
                                               zenith::DataType::Float32));
  original.add_node(zenith::ops::IDENTITY, "id1", {}, {});

  auto cloned = original.clone();

  EXPECT_EQ(cloned->name(), original.name());
  EXPECT_EQ(cloned->num_nodes(), original.num_nodes());
  EXPECT_EQ(cloned->inputs().size(), original.inputs().size());
  EXPECT_EQ(cloned->outputs().size(), original.outputs().size());
}

TEST(GraphIRTest, Summary) {
  zenith::GraphIR graph("test_model");
  graph.add_input(zenith::TensorDescriptor("in", zenith::Shape{1},
                                           zenith::DataType::Float32));
  graph.add_output(zenith::TensorDescriptor("out", zenith::Shape{1},
                                            zenith::DataType::Float32));
  graph.add_node(zenith::ops::CONV, "conv1", {}, {});

  std::string summary = graph.summary();

  EXPECT_TRUE(summary.find("test_model") != std::string::npos);
  EXPECT_TRUE(summary.find("Nodes: 1") != std::string::npos);
  EXPECT_TRUE(summary.find("Conv") != std::string::npos);
}
