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
#include <zenith/cublas_attention.hpp> // cuBLAS-based attention
#include <zenith/cublas_ops.hpp>
#include <zenith/cuda_backend.hpp>
#include <zenith/cuda_kernels.hpp>    // For GELU, LayerNorm, Softmax
#include <zenith/flash_attention.hpp> // FlashAttention for Transformer
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
          "dim",
          [](zenith::GpuTensor &self, int axis) {
            if (!self.is_valid()) {
              throw std::runtime_error("Invalid GpuTensor");
            }
            if (axis < 0 || axis >= static_cast<int>(self.shape().rank())) {
              throw std::runtime_error("Axis out of range");
            }
            return self.shape()[axis];
          },
          py::arg("axis"), "Get dimension at axis")
      .def_property_readonly(
          "shape",
          [](zenith::GpuTensor &self) {
            if (!self.is_valid()) {
              return py::tuple();
            }
            py::list dims;
            for (size_t i = 0; i < self.shape().rank(); ++i) {
              dims.append(self.shape()[i]);
            }
            return py::tuple(dims);
          },
          "Get tensor shape as tuple")
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

  // ========================================================================
  // Conv2D using cuDNN
  // ========================================================================
  cuda.def(
      "conv2d",
      [](py::array_t<float> input, py::array_t<float> weight,
         py::object bias_obj, int stride, int padding) {
        auto buf_in = input.request();
        auto buf_w = weight.request();

        if (buf_in.ndim != 4 || buf_w.ndim != 4) {
          throw std::runtime_error("conv2d requires 4D tensors: input "
                                   "[N,C,H,W], weight [C_out,C_in,K,K]");
        }

        int N = buf_in.shape[0];
        int C_in = buf_in.shape[1];
        int H = buf_in.shape[2];
        int W = buf_in.shape[3];
        int C_out = buf_w.shape[0];
        int K_h = buf_w.shape[2];
        int K_w = buf_w.shape[3];

        int H_out = (H + 2 * padding - K_h) / stride + 1;
        int W_out = (W + 2 * padding - K_w) / stride + 1;

        // Allocate device memory
        float *d_in, *d_w, *d_out, *d_bias = nullptr;
        size_t in_size = N * C_in * H * W * sizeof(float);
        size_t w_size = C_out * C_in * K_h * K_w * sizeof(float);
        size_t out_size = N * C_out * H_out * W_out * sizeof(float);

        cudaMalloc(&d_in, in_size);
        cudaMalloc(&d_w, w_size);
        cudaMalloc(&d_out, out_size);
        cudaMemcpy(d_in, buf_in.ptr, in_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, buf_w.ptr, w_size, cudaMemcpyHostToDevice);

        // Handle optional bias
        if (!bias_obj.is_none()) {
          auto bias = bias_obj.cast<py::array_t<float>>();
          auto buf_b = bias.request();
          cudaMalloc(&d_bias, C_out * sizeof(float));
          cudaMemcpy(d_bias, buf_b.ptr, C_out * sizeof(float),
                     cudaMemcpyHostToDevice);
        }

        // Workspace for cuDNN
        size_t workspace_size = 0;
        zenith::cudnn::conv2d_get_workspace_size(N, C_in, H, W, C_out, K_h, K_w,
                                                 stride, stride, padding,
                                                 padding, &workspace_size);
        void *workspace = nullptr;
        if (workspace_size > 0) {
          cudaMalloc(&workspace, workspace_size);
        }

        auto status = zenith::cudnn::conv2d_forward(
            d_in, d_w, d_bias, d_out, N, C_in, H, W, C_out, K_h, K_w, stride,
            stride, padding, padding, workspace, workspace_size);

        auto result = py::array_t<float>({N, C_out, H_out, W_out});
        auto buf_out = result.request();
        cudaMemcpy(buf_out.ptr, d_out, out_size, cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_w);
        cudaFree(d_out);
        if (d_bias)
          cudaFree(d_bias);
        if (workspace)
          cudaFree(workspace);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN conv2d failed: " + status.message());
        }
        return result;
      },
      py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
      py::arg("stride") = 1, py::arg("padding") = 0,
      "2D Convolution using cuDNN");

  // ========================================================================
  // BatchNorm using cuDNN
  // ========================================================================
  cuda.def(
      "batch_norm",
      [](py::array_t<float> input, py::array_t<float> gamma,
         py::array_t<float> beta, py::array_t<float> mean,
         py::array_t<float> var, double epsilon) {
        auto buf_in = input.request();

        if (buf_in.ndim != 4) {
          throw std::runtime_error("batch_norm requires 4D tensor [N,C,H,W]");
        }

        int N = buf_in.shape[0];
        int C = buf_in.shape[1];
        int H = buf_in.shape[2];
        int W = buf_in.shape[3];
        size_t size = N * C * H * W * sizeof(float);

        float *d_in, *d_out, *d_gamma, *d_beta, *d_mean, *d_var;
        cudaMalloc(&d_in, size);
        cudaMalloc(&d_out, size);
        cudaMalloc(&d_gamma, C * sizeof(float));
        cudaMalloc(&d_beta, C * sizeof(float));
        cudaMalloc(&d_mean, C * sizeof(float));
        cudaMalloc(&d_var, C * sizeof(float));

        cudaMemcpy(d_in, buf_in.ptr, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gamma, gamma.request().ptr, C * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, beta.request().ptr, C * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_mean, mean.request().ptr, C * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_var, var.request().ptr, C * sizeof(float),
                   cudaMemcpyHostToDevice);

        auto status = zenith::cudnn::batchnorm_forward_inference(
            d_in, d_out, d_gamma, d_beta, d_mean, d_var, N, C, H, W, epsilon);

        auto result = py::array_t<float>(buf_in.shape);
        auto buf_out = result.request();
        cudaMemcpy(buf_out.ptr, d_out, size, cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_gamma);
        cudaFree(d_beta);
        cudaFree(d_mean);
        cudaFree(d_var);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN batch_norm failed: " +
                                   status.message());
        }
        return result;
      },
      py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("mean"),
      py::arg("var"), py::arg("epsilon") = 1e-5,
      "Batch Normalization using cuDNN (inference mode)");

  // ========================================================================
  // MaxPool2D using cuDNN
  // ========================================================================
  cuda.def(
      "maxpool2d",
      [](py::array_t<float> input, int kernel_size, int stride, int padding) {
        auto buf = input.request();

        if (buf.ndim != 4) {
          throw std::runtime_error("maxpool2d requires 4D tensor [N,C,H,W]");
        }

        int N = buf.shape[0];
        int C = buf.shape[1];
        int H = buf.shape[2];
        int W = buf.shape[3];
        int H_out = (H + 2 * padding - kernel_size) / stride + 1;
        int W_out = (W + 2 * padding - kernel_size) / stride + 1;

        float *d_in, *d_out;
        size_t in_size = N * C * H * W * sizeof(float);
        size_t out_size = N * C * H_out * W_out * sizeof(float);
        cudaMalloc(&d_in, in_size);
        cudaMalloc(&d_out, out_size);
        cudaMemcpy(d_in, buf.ptr, in_size, cudaMemcpyHostToDevice);

        auto status = zenith::cudnn::maxpool2d_forward(
            d_in, d_out, N, C, H, W, kernel_size, kernel_size, stride, stride,
            padding, padding);

        auto result = py::array_t<float>({N, C, H_out, W_out});
        auto buf_out = result.request();
        cudaMemcpy(buf_out.ptr, d_out, out_size, cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN maxpool2d failed: " +
                                   status.message());
        }
        return result;
      },
      py::arg("input"), py::arg("kernel_size") = 2, py::arg("stride") = 2,
      py::arg("padding") = 0, "Max Pooling 2D using cuDNN");

  // ========================================================================
  // GlobalAvgPool using cuDNN
  // ========================================================================
  cuda.def(
      "global_avgpool",
      [](py::array_t<float> input) {
        auto buf = input.request();

        if (buf.ndim != 4) {
          throw std::runtime_error(
              "global_avgpool requires 4D tensor [N,C,H,W]");
        }

        int N = buf.shape[0];
        int C = buf.shape[1];
        int H = buf.shape[2];
        int W = buf.shape[3];

        float *d_in, *d_out;
        size_t in_size = N * C * H * W * sizeof(float);
        size_t out_size = N * C * sizeof(float); // H_out=1, W_out=1
        cudaMalloc(&d_in, in_size);
        cudaMalloc(&d_out, out_size);
        cudaMemcpy(d_in, buf.ptr, in_size, cudaMemcpyHostToDevice);

        auto status =
            zenith::cudnn::global_avgpool_forward(d_in, d_out, N, C, H, W);

        auto result = py::array_t<float>({N, C, 1, 1});
        auto buf_out = result.request();
        cudaMemcpy(buf_out.ptr, d_out, out_size, cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN global_avgpool failed: " +
                                   status.message());
        }
        return result;
      },
      py::arg("input"), "Global Average Pooling using cuDNN");

  // ========================================================================
  // Element-wise Add using cuDNN (for residual connections)
  // ========================================================================
  cuda.def(
      "add",
      [](py::array_t<float> a, py::array_t<float> b) {
        auto buf_a = a.request();
        auto buf_b = b.request();

        if (buf_a.ndim != 4 || buf_b.ndim != 4) {
          throw std::runtime_error("add requires 4D tensors [N,C,H,W]");
        }

        int N = buf_a.shape[0];
        int C = buf_a.shape[1];
        int H = buf_a.shape[2];
        int W = buf_a.shape[3];
        size_t size = N * C * H * W * sizeof(float);

        float *d_a, *d_b, *d_out;
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_out, size);
        cudaMemcpy(d_a, buf_a.ptr, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, buf_b.ptr, size, cudaMemcpyHostToDevice);

        auto status = zenith::cudnn::add_tensors(d_a, d_b, d_out, N, C, H, W);

        auto result = py::array_t<float>(buf_a.shape);
        auto buf_out = result.request();
        cudaMemcpy(buf_out.ptr, d_out, size, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN add failed: " + status.message());
        }
        return result;
      },
      py::arg("a"), py::arg("b"), "Element-wise Add using cuDNN");

  // ========================================================================
  // GPU-RESIDENT CNN OPERATIONS (Zero-Copy Pipeline)
  // These operate directly on GpuTensor without CPU transfers
  // ========================================================================

  // Conv2D on GPU tensors
  cuda.def(
      "conv2d_gpu",
      [](zenith::GpuTensor &input, zenith::GpuTensor &weight,
         py::object bias_tensor, int stride, int padding) {
        if (!input.is_valid() || !weight.is_valid()) {
          throw std::runtime_error("Invalid input tensors");
        }
        if (input.ndim() != 4 || weight.ndim() != 4) {
          throw std::runtime_error("conv2d_gpu requires 4D tensors");
        }

        int N = input.dim(0);
        int C_in = input.dim(1);
        int H = input.dim(2);
        int W = input.dim(3);
        int C_out = weight.dim(0);
        int K_h = weight.dim(2);
        int K_w = weight.dim(3);

        int H_out = (H + 2 * padding - K_h) / stride + 1;
        int W_out = (W + 2 * padding - K_w) / stride + 1;

        // Create output tensor on GPU
        zenith::GpuTensor output(zenith::Shape({N, C_out, H_out, W_out}));

        // Get workspace
        size_t workspace_size = 0;
        zenith::cudnn::conv2d_get_workspace_size(N, C_in, H, W, C_out, K_h, K_w,
                                                 stride, stride, padding,
                                                 padding, &workspace_size);
        void *workspace = nullptr;
        if (workspace_size > 0) {
          cudaMalloc(&workspace, workspace_size);
        }

        // Handle optional bias
        float *d_bias = nullptr;
        if (!bias_tensor.is_none()) {
          auto &bias_gpu = bias_tensor.cast<zenith::GpuTensor &>();
          d_bias = bias_gpu.data_ptr<float>();
        }

        auto status = zenith::cudnn::conv2d_forward(
            input.data_ptr<float>(), weight.data_ptr<float>(), d_bias,
            output.data_ptr<float>(), N, C_in, H, W, C_out, K_h, K_w, stride,
            stride, padding, padding, workspace, workspace_size);

        if (workspace)
          cudaFree(workspace);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN conv2d_gpu failed: " +
                                   status.message());
        }
        return output;
      },
      py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
      py::arg("stride") = 1, py::arg("padding") = 0,
      "Conv2D on GPU tensors (zero copy)");

  // ReLU on GPU tensor
  cuda.def(
      "relu_gpu",
      [](zenith::GpuTensor &input) {
        if (!input.is_valid() || input.ndim() != 4) {
          throw std::runtime_error("relu_gpu requires valid 4D GpuTensor");
        }

        int N = input.dim(0);
        int C = input.dim(1);
        int H = input.dim(2);
        int W = input.dim(3);

        zenith::GpuTensor output(input.shape());

        auto status = zenith::cudnn::relu_forward(
            input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN relu_gpu failed: " +
                                   status.message());
        }
        return output;
      },
      py::arg("input"), "ReLU on GPU tensor (zero copy)");

  // BatchNorm on GPU tensor
  cuda.def(
      "batch_norm_gpu",
      [](zenith::GpuTensor &input, zenith::GpuTensor &gamma,
         zenith::GpuTensor &beta, zenith::GpuTensor &mean,
         zenith::GpuTensor &var, double epsilon) {
        if (!input.is_valid() || input.ndim() != 4) {
          throw std::runtime_error(
              "batch_norm_gpu requires valid 4D GpuTensor");
        }

        int N = input.dim(0);
        int C = input.dim(1);
        int H = input.dim(2);
        int W = input.dim(3);

        zenith::GpuTensor output(input.shape());

        auto status = zenith::cudnn::batchnorm_forward_inference(
            input.data_ptr<float>(), output.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(),
            mean.data_ptr<float>(), var.data_ptr<float>(), N, C, H, W, epsilon);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN batch_norm_gpu failed: " +
                                   status.message());
        }
        return output;
      },
      py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("mean"),
      py::arg("var"), py::arg("epsilon") = 1e-5,
      "BatchNorm on GPU tensor (zero copy)");

  // MaxPool2D on GPU tensor
  cuda.def(
      "maxpool2d_gpu",
      [](zenith::GpuTensor &input, int kernel_size, int stride, int padding) {
        if (!input.is_valid() || input.ndim() != 4) {
          throw std::runtime_error("maxpool2d_gpu requires valid 4D GpuTensor");
        }

        int N = input.dim(0);
        int C = input.dim(1);
        int H = input.dim(2);
        int W = input.dim(3);
        int H_out = (H + 2 * padding - kernel_size) / stride + 1;
        int W_out = (W + 2 * padding - kernel_size) / stride + 1;

        zenith::GpuTensor output(zenith::Shape({N, C, H_out, W_out}));

        auto status = zenith::cudnn::maxpool2d_forward(
            input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W,
            kernel_size, kernel_size, stride, stride, padding, padding);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN maxpool2d_gpu failed: " +
                                   status.message());
        }
        return output;
      },
      py::arg("input"), py::arg("kernel_size") = 2, py::arg("stride") = 2,
      py::arg("padding") = 0, "MaxPool2D on GPU tensor (zero copy)");

  // Element-wise Add on GPU tensors
  cuda.def(
      "add_gpu",
      [](zenith::GpuTensor &a, zenith::GpuTensor &b) {
        if (!a.is_valid() || !b.is_valid() || a.ndim() != 4 || b.ndim() != 4) {
          throw std::runtime_error("add_gpu requires valid 4D GpuTensors");
        }

        int N = a.dim(0);
        int C = a.dim(1);
        int H = a.dim(2);
        int W = a.dim(3);

        zenith::GpuTensor output(a.shape());

        auto status =
            zenith::cudnn::add_tensors(a.data_ptr<float>(), b.data_ptr<float>(),
                                       output.data_ptr<float>(), N, C, H, W);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN add_gpu failed: " + status.message());
        }
        return output;
      },
      py::arg("a"), py::arg("b"),
      "Element-wise Add on GPU tensors (zero copy)");

  // Global Average Pool on GPU tensor
  cuda.def(
      "global_avgpool_gpu",
      [](zenith::GpuTensor &input) {
        if (!input.is_valid() || input.ndim() != 4) {
          throw std::runtime_error(
              "global_avgpool_gpu requires valid 4D GpuTensor");
        }

        int N = input.dim(0);
        int C = input.dim(1);
        int H = input.dim(2);
        int W = input.dim(3);

        // Output is [N, C, 1, 1]
        zenith::GpuTensor output(zenith::Shape({N, C, 1, 1}));

        auto status = zenith::cudnn::global_avgpool_forward(
            input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W);

        if (!status.ok()) {
          throw std::runtime_error("cuDNN global_avgpool_gpu failed: " +
                                   status.message());
        }
        return output;
      },
      py::arg("input"), "Global Average Pool on GPU tensor (zero copy)");

  // ========================================================================
  // TRANSFORMER/BERT GPU OPERATIONS
  // These use custom CUDA kernels for Transformer model support
  // ========================================================================

  // GELU activation on GPU tensor
  cuda.def(
      "gelu_gpu",
      [](zenith::GpuTensor &input) {
        if (!input.is_valid()) {
          throw std::runtime_error("gelu_gpu requires valid GpuTensor");
        }

        zenith::GpuTensor output(input.shape());
        size_t size = input.numel();

        zenith::cuda_kernels::gelu_f32(input.data_ptr<float>(),
                                       output.data_ptr<float>(), size);
        cudaDeviceSynchronize();

        return output;
      },
      py::arg("input"), "GELU activation on GPU tensor (zero copy)");

  // LayerNorm on GPU tensor
  cuda.def(
      "layernorm_gpu",
      [](zenith::GpuTensor &input, zenith::GpuTensor &gamma,
         zenith::GpuTensor &beta, double eps) {
        if (!input.is_valid() || input.ndim() != 2) {
          throw std::runtime_error(
              "layernorm_gpu requires valid 2D GpuTensor [batch, hidden]");
        }

        int batch = input.dim(0);
        int hidden = input.dim(1);

        zenith::GpuTensor output(input.shape());

        zenith::cuda_kernels::layernorm_f32(
            input.data_ptr<float>(), output.data_ptr<float>(),
            gamma.data_ptr<float>(), beta.data_ptr<float>(), batch, hidden,
            static_cast<float>(eps));
        cudaDeviceSynchronize();

        return output;
      },
      py::arg("input"), py::arg("gamma"), py::arg("beta"),
      py::arg("eps") = 1e-5, "LayerNorm on GPU tensor (zero copy)");

  // GPU Matrix Multiplication using cuBLAS
  // C = A @ B, A: [M, K], B: [K, N] -> C: [M, N]
  cuda.def(
      "matmul_gpu",
      [](zenith::GpuTensor &A, zenith::GpuTensor &B) {
        if (!A.is_valid() || !B.is_valid()) {
          throw std::runtime_error("matmul_gpu requires valid GpuTensors");
        }
        if (A.ndim() != 2 || B.ndim() != 2) {
          throw std::runtime_error("matmul_gpu requires 2D tensors");
        }
        if (A.dim(1) != B.dim(0)) {
          throw std::runtime_error("matmul_gpu dimension mismatch");
        }

        int M = A.dim(0);
        int K = A.dim(1);
        int N = B.dim(1);

        zenith::Shape out_shape({M, N});
        zenith::GpuTensor output(out_shape);

        auto status =
            zenith::cublas::gemm_f32(A.data_ptr<float>(), B.data_ptr<float>(),
                                     output.data_ptr<float>(), M, N, K);

        if (!status.ok()) {
          throw std::runtime_error("cuBLAS matmul_gpu failed");
        }

        return output;
      },
      py::arg("A"), py::arg("B"),
      "GPU matmul using cuBLAS: C[M,N] = A[M,K] @ B[K,N]");

  // GPU Linear layer: Y = X @ W^T + bias
  // X: [M, K], W: [N, K], bias: [N] -> Y: [M, N]
  cuda.def(
      "linear_gpu",
      [](zenith::GpuTensor &X, zenith::GpuTensor &W, zenith::GpuTensor &bias) {
        if (!X.is_valid() || !W.is_valid()) {
          throw std::runtime_error("linear_gpu requires valid GpuTensors");
        }
        if (X.ndim() != 2 || W.ndim() != 2) {
          throw std::runtime_error("linear_gpu requires 2D tensors");
        }

        int M = X.dim(0); // batch*seq
        int K = X.dim(1); // input features
        int N = W.dim(0); // output features

        if (W.dim(1) != K) {
          throw std::runtime_error("linear_gpu: W.dim(1) must match X.dim(1)");
        }

        zenith::Shape out_shape({M, N});
        zenith::GpuTensor output(out_shape);

        // Y = X @ W^T: [M,K] @ [K,N] = [M,N] (W is [N,K], need transpose)
        auto status = zenith::cublas::gemm_f32(
            X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
            M, N, K, 1.0f, 0.0f, false, true); // trans_b = true for W^T

        if (!status.ok()) {
          throw std::runtime_error("cuBLAS linear_gpu matmul failed");
        }

        // Add bias if valid
        if (bias.is_valid() && bias.dim(0) == N) {
          // Broadcast add bias to each row
          int blocks = (M * N + 255) / 256;
          float *out_ptr = output.data_ptr<float>();
          const float *bias_ptr = bias.data_ptr<float>();
          // Simple bias add with kernel
          zenith::cuda_kernels::add_bias_f32(out_ptr, bias_ptr, M, N);
          cudaDeviceSynchronize();
        }

        return output;
      },
      py::arg("X"), py::arg("W"), py::arg("bias"),
      "GPU linear: Y[M,N] = X[M,K] @ W[N,K]^T + bias[N]");

  // Softmax on GPU tensor (2D)
  cuda.def(
      "softmax_gpu",
      [](zenith::GpuTensor &input) {
        if (!input.is_valid() || input.ndim() != 2) {
          throw std::runtime_error(
              "softmax_gpu requires valid 2D GpuTensor [batch, seq_len]");
        }

        int batch = input.dim(0);
        int len = input.dim(1);

        zenith::GpuTensor output(input.shape());

        zenith::cuda_kernels::softmax_2d_f32(
            input.data_ptr<float>(), output.data_ptr<float>(), batch, len);
        cudaDeviceSynchronize();

        return output;
      },
      py::arg("input"), "Softmax on GPU tensor (zero copy)");

  // FlashAttention for Multi-Head Attention
  // Q, K, V, O: [batch, num_heads, seq_len, head_dim]
  cuda.def(
      "flash_attention_gpu",
      [](zenith::GpuTensor &Q, zenith::GpuTensor &K, zenith::GpuTensor &V) {
        if (!Q.is_valid() || !K.is_valid() || !V.is_valid()) {
          throw std::runtime_error(
              "flash_attention_gpu requires valid GpuTensors");
        }
        if (Q.ndim() != 4 || K.ndim() != 4 || V.ndim() != 4) {
          throw std::runtime_error("flash_attention_gpu requires 4D tensors "
                                   "[batch, heads, seq, dim]");
        }

        int batch_size = Q.dim(0);
        int num_heads = Q.dim(1);
        int seq_len = Q.dim(2);
        int head_dim = Q.dim(3);

        zenith::GpuTensor output(Q.shape());

        zenith::flash_attention::flash_attention_forward(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            output.data_ptr<float>(), batch_size, num_heads, seq_len, head_dim);
        cudaDeviceSynchronize();

        return output;
      },
      py::arg("Q"), py::arg("K"), py::arg("V"),
      "FlashAttention on GPU tensors [batch, heads, seq, dim] - memory "
      "efficient");

  // cuBLAS-based attention (high performance)
  cuda.def(
      "cublas_attention_gpu",
      [](zenith::GpuTensor &Q, zenith::GpuTensor &K, zenith::GpuTensor &V) {
        if (!Q.is_valid() || !K.is_valid() || !V.is_valid()) {
          throw std::runtime_error(
              "cublas_attention_gpu requires valid GpuTensors");
        }
        if (Q.ndim() != 4) {
          throw std::runtime_error("cublas_attention_gpu requires 4D tensors");
        }

        int batch_size = Q.dim(0);
        int num_heads = Q.dim(1);
        int seq_len = Q.dim(2);
        int head_dim = Q.dim(3);

        zenith::GpuTensor output(Q.shape());

        zenith::cublas_attention::cublas_attention_forward_alloc(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            output.data_ptr<float>(), batch_size, num_heads, seq_len, head_dim);

        return output;
      },
      py::arg("Q"), py::arg("K"), py::arg("V"),
      "cuBLAS attention [batch, heads, seq, dim] - use Tensor Cores");

  // Element-wise add for 2D tensors (for residual connections)
  cuda.def(
      "add_2d_gpu",
      [](zenith::GpuTensor &A, zenith::GpuTensor &B) {
        if (!A.is_valid() || !B.is_valid()) {
          throw std::runtime_error("add_2d_gpu requires valid GpuTensors");
        }
        if (A.ndim() != 2 || B.ndim() != 2) {
          throw std::runtime_error("add_2d_gpu requires 2D tensors");
        }

        int M = A.dim(0);
        int N = A.dim(1);

        zenith::GpuTensor output(A.shape());
        zenith::cuda_kernels::add_2d_f32(A.data_ptr<float>(),
                                         B.data_ptr<float>(),
                                         output.data_ptr<float>(), M, N);
        return output;
      },
      py::arg("A"), py::arg("B"), "Element-wise add: C = A + B (2D tensors)");

  // Transpose for attention: [batch, seq, heads, dim] -> [batch, heads, seq,
  // dim]
  cuda.def(
      "transpose_for_attention",
      [](zenith::GpuTensor &input, int batch, int seq, int heads, int dim) {
        if (!input.is_valid()) {
          throw std::runtime_error(
              "transpose_for_attention needs valid tensor");
        }

        zenith::Shape out_shape({batch, heads, seq, dim});
        zenith::GpuTensor output(out_shape);

        zenith::cuda_kernels::transpose_0213_f32(input.data_ptr<float>(),
                                                 output.data_ptr<float>(),
                                                 batch, seq, heads, dim);
        return output;
      },
      py::arg("input"), py::arg("batch"), py::arg("seq"), py::arg("heads"),
      py::arg("dim"), "Transpose [B,S,H,D] -> [B,H,S,D] for attention");

  // Inverse transpose: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
  cuda.def(
      "transpose_from_attention",
      [](zenith::GpuTensor &input, int batch, int heads, int seq, int dim) {
        if (!input.is_valid()) {
          throw std::runtime_error(
              "transpose_from_attention needs valid tensor");
        }

        zenith::Shape out_shape({batch, seq, heads, dim});
        zenith::GpuTensor output(out_shape);

        zenith::cuda_kernels::transpose_0213_inv_f32(input.data_ptr<float>(),
                                                     output.data_ptr<float>(),
                                                     batch, heads, seq, dim);
        return output;
      },
      py::arg("input"), py::arg("batch"), py::arg("heads"), py::arg("seq"),
      py::arg("dim"), "Transpose [B,H,S,D] -> [B,S,H,D] from attention");

  // Reshape 4D to 2D: [batch, seq, heads, dim] -> [batch*seq, heads*dim]
  cuda.def(
      "reshape_4d_to_2d",
      [](zenith::GpuTensor &input, int batch, int seq, int hidden) {
        if (!input.is_valid()) {
          throw std::runtime_error("reshape_4d_to_2d needs valid tensor");
        }
        zenith::Shape out_shape({batch * seq, hidden});
        zenith::GpuTensor output(out_shape);
        cudaMemcpy(output.data_ptr<float>(), input.data_ptr<float>(),
                   batch * seq * hidden * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        return output;
      },
      py::arg("input"), py::arg("batch"), py::arg("seq"), py::arg("hidden"),
      "Reshape [B,S,H,D] -> [B*S, H*D]");

  cuda.def("has_cudnn", []() { return true; });
#else
  cuda.def("has_cudnn", []() { return false; });
#endif

#endif // ZENITH_HAS_CUDA
}
