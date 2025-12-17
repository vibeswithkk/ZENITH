// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// FlashAttention Header - Forward declarations

#ifndef ZENITH_FLASH_ATTENTION_HPP
#define ZENITH_FLASH_ATTENTION_HPP

#ifdef ZENITH_HAS_CUDA

#include <cstddef>

namespace zenith {
namespace flash_attention {

/// FlashAttention forward pass
/// Q, K, V: [batch, num_heads, seq_len, head_dim]
/// O: output [batch, num_heads, seq_len, head_dim]
void flash_attention_forward(const float *Q, const float *K, const float *V,
                             float *O, int batch_size, int num_heads,
                             int seq_len, int head_dim);

} // namespace flash_attention
} // namespace zenith

#endif // ZENITH_HAS_CUDA
#endif // ZENITH_FLASH_ATTENTION_HPP
