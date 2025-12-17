// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zenith/backend.hpp>
#include <zenith/cpu_backend.hpp>
#include <zenith/dispatcher.hpp>
#include <zenith/kernels.hpp>
#include <zenith/zenith.hpp>

#ifdef ZENITH_HAS_CUDA
#include <zenith/cublas_ops.hpp>
#include <zenith/cuda_backend.hpp>
#include <zenith/gpu_tensor.hpp>
#ifdef ZENITH_HAS_CUDNN
#include <zenith/cudnn_ops.hpp>
#endif
#endif

namespace py = pybind11;

// ============================================================================
// Helper Functions for NumPy Array Conversion
// ============================================================================

// Get contiguous array from arbitrary NumPy input
py::array_t<float> ensure_contiguous_float32(py::array_t<float> arr) {
  auto buf = arr.request();
  if (!(arr.flags() & py::array::c_style)) {
    // Need to make a contiguous copy
    return py::array_t<float>::ensure(arr);
  }
  return arr;
}

// ============================================================================
// Kernel Wrapper Functions
// ============================================================================

namespace kernels_py {

/// Matrix multiplication: C = A @ B
/// A: [M, K], B: [K, N], C: [M, N]
py::array_t<float> matmul(py::array_t<float> A, py::array_t<float> B) {
  // Get buffer info for input arrays
  auto buf_a = A.request();
  auto buf_b = B.request();

  // Validate dimensions
  if (buf_a.ndim != 2 || buf_b.ndim != 2) {
    throw std::runtime_error("matmul requires 2D arrays");
  }

  int M = buf_a.shape[0];
  int K = buf_a.shape[1];
  int K2 = buf_b.shape[0];
  int N = buf_b.shape[1];

  if (K != K2) {
    throw std::runtime_error("Inner dimensions do not match: " +
                             std::to_string(K) + " vs " + std::to_string(K2));
  }

  // Allocate output array
  auto result = py::array_t<float>({M, N});
  auto buf_c = result.request();

  // Get raw pointers
  const float *ptr_a = static_cast<const float *>(buf_a.ptr);
  const float *ptr_b = static_cast<const float *>(buf_b.ptr);
  float *ptr_c = static_cast<float *>(buf_c.ptr);

  // Call the kernel
  zenith::kernels::matmul_f32(ptr_a, ptr_b, ptr_c, M, N, K);

  return result;
}

/// ReLU activation (in-place or return new array)
py::array_t<float> relu(py::array_t<float> input, bool inplace = false) {
  auto buf = input.request();

  py::array_t<float> output;
  float *ptr;

  if (inplace) {
    // Modify in-place
    ptr = static_cast<float *>(buf.ptr);
    output = input;
  } else {
    // Create copy
    output = py::array_t<float>(buf.shape);
    auto out_buf = output.request();
    ptr = static_cast<float *>(out_buf.ptr);

    // Copy input to output
    std::memcpy(ptr, buf.ptr, buf.size * sizeof(float));
  }

  zenith::kernels::relu_f32(ptr, buf.size);

  return output;
}

/// Sigmoid activation
py::array_t<float> sigmoid(py::array_t<float> input) {
  auto buf = input.request();

  auto output = py::array_t<float>(buf.shape);
  auto out_buf = output.request();
  float *ptr = static_cast<float *>(out_buf.ptr);

  // Copy input to output
  std::memcpy(ptr, buf.ptr, buf.size * sizeof(float));

  zenith::kernels::sigmoid_f32(ptr, buf.size);

  return output;
}

/// Tanh activation
py::array_t<float> tanh_activation(py::array_t<float> input) {
  auto buf = input.request();

  auto output = py::array_t<float>(buf.shape);
  auto out_buf = output.request();
  float *ptr = static_cast<float *>(out_buf.ptr);

  std::memcpy(ptr, buf.ptr, buf.size * sizeof(float));

  zenith::kernels::tanh_f32(ptr, buf.size);

  return output;
}

/// Element-wise addition: C = A + B
py::array_t<float> add(py::array_t<float> A, py::array_t<float> B) {
  auto buf_a = A.request();
  auto buf_b = B.request();

  if (buf_a.size != buf_b.size) {
    throw std::runtime_error("Array sizes must match for addition");
  }

  auto output = py::array_t<float>(buf_a.shape);
  auto out_buf = output.request();

  zenith::kernels::add_f32(static_cast<const float *>(buf_a.ptr),
                           static_cast<const float *>(buf_b.ptr),
                           static_cast<float *>(out_buf.ptr), buf_a.size);

  return output;
}

/// Sum reduction
float sum(py::array_t<float> input) {
  auto buf = input.request();
  return zenith::kernels::sum_f32(static_cast<const float *>(buf.ptr),
                                  buf.size);
}

/// Mean reduction
float mean(py::array_t<float> input) {
  auto buf = input.request();
  return zenith::kernels::mean_f32(static_cast<const float *>(buf.ptr),
                                   buf.size);
}

/// Max reduction
float max(py::array_t<float> input) {
  auto buf = input.request();
  return zenith::kernels::max_f32(static_cast<const float *>(buf.ptr),
                                  buf.size);
}

/// Min reduction
float min(py::array_t<float> input) {
  auto buf = input.request();
  return zenith::kernels::min_f32(static_cast<const float *>(buf.ptr),
                                  buf.size);
}

/// Softmax activation
py::array_t<float> softmax(py::array_t<float> input, int axis = -1) {
  auto buf = input.request();

  // Handle negative axis
  if (axis < 0)
    axis = buf.ndim + axis;

  if (axis < 0 || axis >= buf.ndim) {
    throw std::runtime_error("Invalid axis for softmax");
  }

  auto output = py::array_t<float>(buf.shape);
  auto out_buf = output.request();

  size_t axis_size = buf.shape[axis];

  zenith::kernels::softmax_f32(static_cast<const float *>(buf.ptr),
                               static_cast<float *>(out_buf.ptr), buf.size,
                               axis_size);

  return output;
}

/// Conv2D operation
/// input: [N, C_in, H, W], weight: [C_out, C_in, K_h, K_w]
py::array_t<float> conv2d(py::array_t<float> input, py::array_t<float> weight,
                          py::object bias = py::none(), int stride = 1,
                          int padding = 0) {
  auto buf_in = input.request();
  auto buf_w = weight.request();

  if (buf_in.ndim != 4 || buf_w.ndim != 4) {
    throw std::runtime_error("conv2d requires 4D input and weight tensors");
  }

  int N = buf_in.shape[0];
  int C_in = buf_in.shape[1];
  int H = buf_in.shape[2];
  int W = buf_in.shape[3];
  int C_out = buf_w.shape[0];
  int K_h = buf_w.shape[2];
  int K_w = buf_w.shape[3];

  // Calculate output dimensions
  int H_out = (H + 2 * padding - K_h) / stride + 1;
  int W_out = (W + 2 * padding - K_w) / stride + 1;

  // Allocate output
  auto output = py::array_t<float>({N, C_out, H_out, W_out});
  auto out_buf = output.request();

  // Get bias pointer if provided
  const float *bias_ptr = nullptr;
  if (!bias.is_none()) {
    auto bias_arr = bias.cast<py::array_t<float>>();
    bias_ptr = static_cast<const float *>(bias_arr.request().ptr);
  }

  zenith::kernels::conv2d_f32(static_cast<const float *>(buf_in.ptr),
                              static_cast<const float *>(buf_w.ptr), bias_ptr,
                              static_cast<float *>(out_buf.ptr), N, C_in, H, W,
                              C_out, K_h, K_w, stride, stride, padding,
                              padding);

  return output;
}

/// MaxPool2D operation
py::array_t<float> maxpool2d(py::array_t<float> input, int kernel_size = 2,
                             int stride = 2) {
  auto buf = input.request();

  if (buf.ndim != 4) {
    throw std::runtime_error("maxpool2d requires 4D input tensor");
  }

  int N = buf.shape[0];
  int C = buf.shape[1];
  int H = buf.shape[2];
  int W = buf.shape[3];

  int H_out = (H - kernel_size) / stride + 1;
  int W_out = (W - kernel_size) / stride + 1;

  auto output = py::array_t<float>({N, C, H_out, W_out});
  auto out_buf = output.request();

  zenith::kernels::maxpool2d_f32(static_cast<const float *>(buf.ptr),
                                 static_cast<float *>(out_buf.ptr), N, C, H, W,
                                 kernel_size, kernel_size, stride, stride);

  return output;
}

/// Get CPU features info
py::dict get_cpu_info() {
  auto features = zenith::kernels::get_cpu_features();
  py::dict info;
  info["has_sse"] = features.has_sse;
  info["has_sse2"] = features.has_sse2;
  info["has_sse3"] = features.has_sse3;
  info["has_sse4_1"] = features.has_sse4_1;
  info["has_sse4_2"] = features.has_sse4_2;
  info["has_avx"] = features.has_avx;
  info["has_avx2"] = features.has_avx2;
  info["has_avx512f"] = features.has_avx512f;
  info["has_fma"] = features.has_fma;
  info["has_neon"] = features.has_neon;
  return info;
}

} // namespace kernels_py

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(_zenith_core, m) {
  m.doc() = "Zenith Core C++ Library Python Bindings";

  // Version info
  m.attr("__version__") = zenith::VERSION;

  // ========================================================================
  // DataType Enum
  // ========================================================================
  py::enum_<zenith::DataType>(m, "DataType")
      .value("Float32", zenith::DataType::Float32)
      .value("Float16", zenith::DataType::Float16)
      .value("BFloat16", zenith::DataType::BFloat16)
      .value("Float64", zenith::DataType::Float64)
      .value("Int8", zenith::DataType::Int8)
      .value("Int16", zenith::DataType::Int16)
      .value("Int32", zenith::DataType::Int32)
      .value("Int64", zenith::DataType::Int64)
      .value("UInt8", zenith::DataType::UInt8)
      .value("Bool", zenith::DataType::Bool)
      .export_values();

  // ========================================================================
  // Layout Enum
  // ========================================================================
  py::enum_<zenith::Layout>(m, "Layout")
      .value("NCHW", zenith::Layout::NCHW)
      .value("NHWC", zenith::Layout::NHWC)
      .value("NC", zenith::Layout::NC)
      .export_values();

  // ========================================================================
  // StatusCode Enum
  // ========================================================================
  py::enum_<zenith::StatusCode>(m, "StatusCode")
      .value("Ok", zenith::StatusCode::Ok)
      .value("InvalidArgument", zenith::StatusCode::InvalidArgument)
      .value("InvalidInput", zenith::StatusCode::InvalidInput)
      .value("InvalidOutput", zenith::StatusCode::InvalidOutput)
      .value("NotFound", zenith::StatusCode::NotFound)
      .value("AlreadyExists", zenith::StatusCode::AlreadyExists)
      .value("OutOfMemory", zenith::StatusCode::OutOfMemory)
      .value("NotImplemented", zenith::StatusCode::NotImplemented)
      .value("UnsupportedOp", zenith::StatusCode::UnsupportedOp)
      .value("InternalError", zenith::StatusCode::InternalError)
      .value("InvalidGraph", zenith::StatusCode::InvalidGraph)
      .value("OptimizationFailed", zenith::StatusCode::OptimizationFailed)
      .export_values();

  // ========================================================================
  // Shape Class
  // ========================================================================
  py::class_<zenith::Shape>(m, "Shape")
      .def(py::init<>())
      .def(py::init<std::vector<int64_t>>())
      .def("rank", &zenith::Shape::rank)
      .def("numel", &zenith::Shape::numel)
      .def("dims", &zenith::Shape::dims)
      .def("is_dynamic", &zenith::Shape::is_dynamic)
      .def("__getitem__", [](const zenith::Shape &s, size_t i) { return s[i]; })
      .def("__len__", &zenith::Shape::rank)
      .def("__repr__", [](const zenith::Shape &s) {
        std::string result = "Shape([";
        const auto &dims = s.dims();
        for (size_t i = 0; i < dims.size(); ++i) {
          if (i > 0)
            result += ", ";
          result += std::to_string(dims[i]);
        }
        result += "])";
        return result;
      });

  // ========================================================================
  // Status Class
  // ========================================================================
  py::class_<zenith::Status>(m, "Status")
      .def(py::init<>())
      .def("ok", &zenith::Status::ok)
      .def("code", &zenith::Status::code)
      .def("message", &zenith::Status::message)
      .def("__bool__", &zenith::Status::ok)
      .def("__repr__", [](const zenith::Status &s) {
        return s.ok() ? "Status(Ok)" : "Status(Error: " + s.message() + ")";
      });

  // ========================================================================
  // TensorDescriptor Class
  // ========================================================================
  py::class_<zenith::TensorDescriptor>(m, "TensorDescriptor")
      .def(py::init<>())
      .def(py::init<std::string, zenith::Shape, zenith::DataType,
                    zenith::Layout>(),
           py::arg("name"), py::arg("shape"),
           py::arg("dtype") = zenith::DataType::Float32,
           py::arg("layout") = zenith::Layout::NCHW)
      .def_property("name", &zenith::TensorDescriptor::name,
                    &zenith::TensorDescriptor::set_name)
      .def_property("shape", &zenith::TensorDescriptor::shape,
                    &zenith::TensorDescriptor::set_shape)
      .def_property("dtype", &zenith::TensorDescriptor::dtype,
                    &zenith::TensorDescriptor::set_dtype)
      .def_property("layout", &zenith::TensorDescriptor::layout,
                    &zenith::TensorDescriptor::set_layout)
      .def("size_bytes", &zenith::TensorDescriptor::size_bytes)
      .def("is_valid", &zenith::TensorDescriptor::is_valid)
      .def("__repr__", [](const zenith::TensorDescriptor &t) {
        return "TensorDescriptor(name='" + t.name() +
               "', dtype=" + zenith::dtype_to_string(t.dtype()) + ")";
      });

  // ========================================================================
  // Node Class
  // ========================================================================
  py::class_<zenith::Node>(m, "Node")
      .def(py::init<>())
      .def(py::init<
               std::string, std::string, std::vector<zenith::TensorDescriptor>,
               std::vector<zenith::TensorDescriptor>, zenith::AttributeMap>(),
           py::arg("op_type"), py::arg("name"), py::arg("inputs"),
           py::arg("outputs"), py::arg("attrs") = zenith::AttributeMap{})
      .def_property_readonly("id", &zenith::Node::id)
      .def_property_readonly("op_type", &zenith::Node::op_type)
      .def_property_readonly("name", &zenith::Node::name)
      .def_property_readonly("inputs", &zenith::Node::inputs)
      .def_property_readonly("outputs", &zenith::Node::outputs)
      .def("num_inputs", &zenith::Node::num_inputs)
      .def("num_outputs", &zenith::Node::num_outputs)
      .def("is_op", &zenith::Node::is_op)
      .def("add_input", &zenith::Node::add_input)
      .def("add_output", &zenith::Node::add_output)
      .def("__repr__", [](const zenith::Node &n) {
        return "Node(op='" + n.op_type() + "', name='" + n.name() + "')";
      });

  // ========================================================================
  // GraphIR Class
  // ========================================================================
  py::class_<zenith::GraphIR>(m, "GraphIR")
      .def(py::init<>())
      .def(py::init<std::string>(), py::arg("name"))
      .def_property("name", &zenith::GraphIR::name, &zenith::GraphIR::set_name)
      .def(
          "add_node",
          [](zenith::GraphIR &g, const std::string &op_type,
             const std::string &name,
             const std::vector<zenith::TensorDescriptor> &inputs,
             const std::vector<zenith::TensorDescriptor> &outputs) {
            return g.add_node(op_type, name, inputs, outputs);
          },
          py::return_value_policy::reference)
      .def("get_node", &zenith::GraphIR::get_node,
           py::return_value_policy::reference)
      .def("remove_node", &zenith::GraphIR::remove_node)
      .def("num_nodes", &zenith::GraphIR::num_nodes)
      .def("set_inputs", &zenith::GraphIR::set_inputs)
      .def("set_outputs", &zenith::GraphIR::set_outputs)
      .def("add_input", &zenith::GraphIR::add_input)
      .def("add_output", &zenith::GraphIR::add_output)
      .def_property_readonly("inputs", &zenith::GraphIR::inputs)
      .def_property_readonly("outputs", &zenith::GraphIR::outputs)
      .def("find_nodes_by_op", &zenith::GraphIR::find_nodes_by_op,
           py::return_value_policy::reference)
      .def("validate", &zenith::GraphIR::validate)
      .def("count_ops", &zenith::GraphIR::count_ops)
      .def("summary", &zenith::GraphIR::summary)
      .def("__len__", &zenith::GraphIR::num_nodes)
      .def("__repr__", [](const zenith::GraphIR &g) {
        return "GraphIR(name='" + g.name() +
               "', nodes=" + std::to_string(g.num_nodes()) + ")";
      });

  // ========================================================================
  // Utility Functions
  // ========================================================================
  m.def("get_version", &zenith::get_version, "Get Zenith version string");
  m.def("dtype_size", &zenith::dtype_size, "Get size in bytes for a data type");
  m.def("dtype_to_string", &zenith::dtype_to_string,
        "Get string representation of data type");

  // ========================================================================
  // Kernel Submodule
  // ========================================================================
  auto kernels = m.def_submodule("kernels", "Zenith kernel operations");

  kernels.def("matmul", &kernels_py::matmul, py::arg("A"), py::arg("B"),
              "Matrix multiplication: C = A @ B");

  kernels.def("relu", &kernels_py::relu, py::arg("input"),
              py::arg("inplace") = false, "ReLU activation: max(0, x)");

  kernels.def("sigmoid", &kernels_py::sigmoid, py::arg("input"),
              "Sigmoid activation: 1 / (1 + exp(-x))");

  kernels.def("tanh", &kernels_py::tanh_activation, py::arg("input"),
              "Tanh activation");

  kernels.def("add", &kernels_py::add, py::arg("A"), py::arg("B"),
              "Element-wise addition: C = A + B");

  kernels.def("sum", &kernels_py::sum, py::arg("input"), "Sum of all elements");

  kernels.def("mean", &kernels_py::mean, py::arg("input"),
              "Mean of all elements");

  kernels.def("max", &kernels_py::max, py::arg("input"),
              "Maximum element value");

  kernels.def("min", &kernels_py::min, py::arg("input"),
              "Minimum element value");

  kernels.def("softmax", &kernels_py::softmax, py::arg("input"),
              py::arg("axis") = -1, "Softmax activation");

  kernels.def("conv2d", &kernels_py::conv2d, py::arg("input"),
              py::arg("weight"), py::arg("bias") = py::none(),
              py::arg("stride") = 1, py::arg("padding") = 0,
              "2D Convolution (NCHW layout)");

  kernels.def("maxpool2d", &kernels_py::maxpool2d, py::arg("input"),
              py::arg("kernel_size") = 2, py::arg("stride") = 2,
              "2D Max Pooling");

  kernels.def("get_cpu_info", &kernels_py::get_cpu_info,
              "Get CPU feature information (SSE, AVX, etc.)");

  // ========================================================================
  // Backend Submodule
  // ========================================================================
  auto backends = m.def_submodule("backends", "Hardware backend management");

  backends.def(
      "list_available",
      []() {
        std::vector<std::string> available;
        available.push_back("cpu"); // Always available
#ifdef ZENITH_HAS_CUDA
        if (zenith::cublas::is_cublas_available()) {
          available.push_back("cuda");
        }
#endif
        return available;
      },
      "List all available backends");

  backends.def(
      "is_cuda_available",
      []() {
#ifdef ZENITH_HAS_CUDA
        return zenith::cublas::is_cublas_available();
#else
        return false;
#endif
      },
      "Check if CUDA backend is available");

  backends.def(
      "is_cudnn_available",
      []() {
#ifdef ZENITH_HAS_CUDNN
        return zenith::cudnn::is_cudnn_available();
#else
        return false;
#endif
      },
      "Check if cuDNN is available");

  backends.def(
      "get_cudnn_version",
      []() {
#ifdef ZENITH_HAS_CUDNN
        return zenith::cudnn::get_cudnn_version();
#else
        return 0;
#endif
      },
      "Get cuDNN version number");

  // ========================================================================
  // CUDA Kernel Submodule (only if compiled with CUDA)
  // ========================================================================
#ifdef ZENITH_HAS_CUDA
  auto cuda = m.def_submodule("cuda", "CUDA accelerated operations");

  cuda.def(
      "matmul",
      [](py::array_t<float> A, py::array_t<float> B) {
        auto buf_a = A.request();
        auto buf_b = B.request();

        if (buf_a.ndim != 2 || buf_b.ndim != 2) {
          throw std::runtime_error("matmul requires 2D arrays");
        }

        int M = buf_a.shape[0];
        int K = buf_a.shape[1];
        int K2 = buf_b.shape[0];
        int N = buf_b.shape[1];

        if (K != K2) {
          throw std::runtime_error("Inner dimensions do not match");
        }

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));

        // Copy to device
        cudaMemcpy(d_A, buf_a.ptr, M * K * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, buf_b.ptr, K * N * sizeof(float),
                   cudaMemcpyHostToDevice);

        // Call cuBLAS
        auto status = zenith::cublas::gemm_f32(d_A, d_B, d_C, M, N, K);

        // Allocate output and copy back
        auto result = py::array_t<float>({M, N});
        auto buf_c = result.request();
        cudaMemcpy(buf_c.ptr, d_C, M * N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        if (!status.ok()) {
          throw std::runtime_error("cuBLAS matmul failed: " + status.message());
        }

        return result;
      },
      py::arg("A"), py::arg("B"), "Matrix multiplication using cuBLAS");

  cuda.def(
      "is_available", []() { return zenith::cublas::is_cublas_available(); },
      "Check if CUDA is available");

  // ========================================================================
  // GpuTensor Class Bindings
  // ========================================================================
  py::class_<zenith::GpuTensor>(cuda, "GpuTensor")
      .def(py::init<>())
      .def("is_valid", &zenith::GpuTensor::is_valid)
      .def("numel", &zenith::GpuTensor::numel)
      .def("size_bytes", &zenith::GpuTensor::size_bytes)
      .def("ndim", &zenith::GpuTensor::ndim)
      .def(
          "to_numpy",
          [](zenith::GpuTensor &self) {
            if (!self.is_valid()) {
              throw std::runtime_error("Invalid GpuTensor");
            }
            std::vector<ssize_t> py_shape;
            for (size_t i = 0; i < self.shape().rank(); ++i) {
              py_shape.push_back(self.shape()[i]);
            }
            auto result = py::array_t<float>(py_shape);
            auto buf = result.request();
            self.to_host(buf.ptr);
            return result;
          },
          "Copy tensor data from GPU to NumPy array")
      .def("__repr__", [](const zenith::GpuTensor &self) {
        if (!self.is_valid())
          return std::string("GpuTensor(invalid)");
        std::string s = "GpuTensor(shape=[";
        for (size_t i = 0; i < self.shape().rank(); ++i) {
          if (i > 0)
            s += ", ";
          s += std::to_string(self.shape()[i]);
        }
        s += "], device=cuda)";
        return s;
      });

  // ========================================================================
  // GPU Tensor Factory Functions
  // ========================================================================
  cuda.def(
      "to_gpu",
      [](py::array_t<float> arr) {
        auto buf = arr.request();
        std::vector<int64_t> dims;
        for (ssize_t i = 0; i < buf.ndim; ++i) {
          dims.push_back(buf.shape[i]);
        }
        return zenith::GpuTensor::from_host(buf.ptr, zenith::Shape(dims),
                                            zenith::DataType::Float32);
      },
      py::arg("array"), "Copy NumPy array to GPU tensor");

  cuda.def(
      "empty",
      [](py::tuple shape) {
        std::vector<int64_t> dims;
        for (auto d : shape) {
          dims.push_back(py::cast<int64_t>(d));
        }
        return zenith::GpuTensor::empty(zenith::Shape(dims));
      },
      py::arg("shape"), "Create empty GPU tensor");

  // ========================================================================
  // Optimized Operations on GpuTensor (no copy!)
  // ========================================================================
  cuda.def(
      "matmul_gpu",
      [](zenith::GpuTensor &A, zenith::GpuTensor &B) {
        if (!A.is_valid() || !B.is_valid()) {
          throw std::runtime_error("Invalid input tensors");
        }
        if (A.ndim() != 2 || B.ndim() != 2) {
          throw std::runtime_error("matmul_gpu requires 2D tensors");
        }

        int M = A.dim(0);
        int K = A.dim(1);
        int K2 = B.dim(0);
        int N = B.dim(1);

        if (K != K2) {
          throw std::runtime_error("Inner dimensions do not match");
        }

        // Create output tensor on GPU
        zenith::GpuTensor C(zenith::Shape({M, N}), zenith::DataType::Float32);

        // Call cuBLAS - NO MEMORY COPIES!
        auto status =
            zenith::cublas::gemm_f32(A.data_ptr<float>(), B.data_ptr<float>(),
                                     C.data_ptr<float>(), M, N, K);

        if (!status.ok()) {
          throw std::runtime_error("cuBLAS matmul failed: " + status.message());
        }

        return C;
      },
      py::arg("A"), py::arg("B"),
      "Matrix multiplication on GPU tensors (zero copy)");

  // ========================================================================
  // Memory Pool Management
  // ========================================================================
  cuda.def(
      "memory_stats",
      []() {
        auto stats = zenith::get_gpu_memory_stats();
        py::dict result;
        result["allocations"] = stats.allocations;
        result["cache_hits"] = stats.cache_hits;
        result["cache_returns"] = stats.cache_returns;
        result["total_allocated"] = stats.total_allocated;
        return result;
      },
      "Get GPU memory pool statistics");

  cuda.def("clear_memory_pool", &zenith::clear_gpu_memory_pool,
           "Clear GPU memory pool");

#ifdef ZENITH_HAS_CUDNN
  cuda.def(
      "relu",
      [](py::array_t<float> input) {
        auto buf = input.request();

        if (buf.ndim != 4) {
          throw std::runtime_error("cuDNN relu requires 4D tensor [N,C,H,W]");
        }

        int N = buf.shape[0];
        int C = buf.shape[1];
        int H = buf.shape[2];
        int W = buf.shape[3];

        // Allocate device memory
        float *d_in, *d_out;
        size_t size = N * C * H * W * sizeof(float);
        cudaMalloc(&d_in, size);
        cudaMalloc(&d_out, size);

        cudaMemcpy(d_in, buf.ptr, size, cudaMemcpyHostToDevice);

        auto status = zenith::cudnn::relu_forward(d_in, d_out, N, C, H, W);

        auto result = py::array_t<float>(buf.shape);
        auto out_buf = result.request();
        cudaMemcpy(out_buf.ptr, d_out, size, cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN relu failed: " + status.message());
        }

        return result;
      },
      py::arg("input"), "ReLU activation using cuDNN (4D tensor)");

  cuda.def("has_cudnn", []() { return true; });
#else
  cuda.def("has_cudnn", []() { return false; });
#endif

#endif // ZENITH_HAS_CUDA
}
