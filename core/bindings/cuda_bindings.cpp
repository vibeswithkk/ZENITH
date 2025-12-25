// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Zenith Native CUDA Bindings - pybind11/torch extension
// Exposes CUDA kernels to Python via PyTorch extension mechanism

#ifdef ZENITH_HAS_CUDA

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Forward declarations from our CUDA kernels
namespace zenith {
namespace cuda_kernels {

void relu_f32(float *data, size_t size);
void gelu_f32(const float *input, float *output, size_t size);
void layernorm_f32(const float *input, float *output, const float *gamma,
                   const float *beta, int batch, int hidden, float eps);
void matmul_f32(const float *A, const float *B, float *C, int M, int N, int K);
void softmax_2d_f32(const float *input, float *output, int batch, int len);
void add_f32(const float *A, const float *B, float *C, size_t size);
void wmma_matmul_f16(const half *A, const half *B, float *C, int M, int N,
                     int K);

} // namespace cuda_kernels

namespace flash_attention {

void flash_attention_forward(const float *Q, const float *K, const float *V,
                             float *O, int batch_size, int num_heads,
                             int seq_len, int head_dim);

} // namespace flash_attention
} // namespace zenith

// Wrapper functions that work with PyTorch tensors
torch::Tensor relu_cuda(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

  auto output = input.clone();
  zenith::cuda_kernels::relu_f32(output.data_ptr<float>(), output.numel());
  return output;
}

torch::Tensor gelu_cuda(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

  auto output = torch::empty_like(input);
  zenith::cuda_kernels::gelu_f32(input.data_ptr<float>(),
                                 output.data_ptr<float>(), input.numel());
  return output;
}

torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor gamma,
                             torch::Tensor beta, float eps = 1e-5) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "Input must be 2D [batch, hidden]");

  auto output = torch::empty_like(input);
  int batch = input.size(0);
  int hidden = input.size(1);

  zenith::cuda_kernels::layernorm_f32(
      input.data_ptr<float>(), output.data_ptr<float>(),
      gamma.data_ptr<float>(), beta.data_ptr<float>(), batch, hidden, eps);
  return output;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  TORCH_CHECK(B.size(0) == K, "Matrix dimensions must match");

  auto C = torch::empty({M, N}, A.options());

  zenith::cuda_kernels::matmul_f32(A.data_ptr<float>(), B.data_ptr<float>(),
                                   C.data_ptr<float>(), M, N, K);
  return C;
}

torch::Tensor flash_attention_cuda(torch::Tensor Q, torch::Tensor K,
                                   torch::Tensor V) {
  TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
  TORCH_CHECK(Q.dim() == 4, "Q must be 4D [batch, heads, seq, dim]");

  int batch = Q.size(0);
  int heads = Q.size(1);
  int seq = Q.size(2);
  int dim = Q.size(3);

  auto O = torch::empty_like(Q);

  zenith::flash_attention::flash_attention_forward(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      O.data_ptr<float>(), batch, heads, seq, dim);
  return O;
}

// WMMA Tensor Core MatMul wrapper
// Takes FP16 inputs, outputs FP32 for numerical stability
torch::Tensor wmma_matmul_cuda(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
  TORCH_CHECK(A.scalar_type() == torch::kFloat16,
              "A must be FP16 (use .half())");
  TORCH_CHECK(B.scalar_type() == torch::kFloat16,
              "B must be FP16 (use .half())");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  TORCH_CHECK(B.size(0) == K, "Matrix dimensions must match");

  // Output is FP32 for numerical stability
  auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));

  zenith::cuda_kernels::wmma_matmul_f16(
      reinterpret_cast<const half *>(A.data_ptr<at::Half>()),
      reinterpret_cast<const half *>(B.data_ptr<at::Half>()),
      C.data_ptr<float>(), M, N, K);
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Zenith Native CUDA Kernels";

  m.def("relu", &relu_cuda, "ReLU activation (CUDA)");
  m.def("gelu", &gelu_cuda, "GELU activation (CUDA)");
  m.def("layernorm", &layernorm_cuda, "LayerNorm (CUDA)", py::arg("input"),
        py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5);
  m.def("matmul", &matmul_cuda, "Matrix multiplication (CUDA)");
  m.def("flash_attention", &flash_attention_cuda, "Flash Attention (CUDA)");
  m.def("wmma_matmul", &wmma_matmul_cuda,
        "WMMA Tensor Core MatMul (FP16->FP32)");
}

#else
// Stub for non-CUDA builds
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Zenith CUDA Kernels (CUDA not available)";
}
#endif
