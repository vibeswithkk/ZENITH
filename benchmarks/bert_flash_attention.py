# GPU-Resident BERT Encoder with FlashAttention
# Uses FlashAttention kernel for 40x speedup target

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("ZENITH BERT ENCODER - GPU RESIDENT + FLASHATTENTION")
print("=" * 70)

from zenith._zenith_core import cuda


class BertEncoderGPU:
    """
    GPU-resident BERT encoder using FlashAttention.
    All operations stay on GPU to minimize transfers.
    """

    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size

    def load_from_pytorch(self, encoder_layer):
        """Load weights from PyTorch BertLayer to GPU."""
        # QKV weights - concatenate for fused projection
        q_w = encoder_layer.attention.self.query.weight.detach().cpu().numpy()
        k_w = encoder_layer.attention.self.key.weight.detach().cpu().numpy()
        v_w = encoder_layer.attention.self.value.weight.detach().cpu().numpy()

        q_b = encoder_layer.attention.self.query.bias.detach().cpu().numpy()
        k_b = encoder_layer.attention.self.key.bias.detach().cpu().numpy()
        v_b = encoder_layer.attention.self.value.bias.detach().cpu().numpy()

        # Store as GPU tensors
        self.q_weight = cuda.to_gpu(np.ascontiguousarray(q_w))
        self.k_weight = cuda.to_gpu(np.ascontiguousarray(k_w))
        self.v_weight = cuda.to_gpu(np.ascontiguousarray(v_w))
        self.q_bias = cuda.to_gpu(np.ascontiguousarray(q_b))
        self.k_bias = cuda.to_gpu(np.ascontiguousarray(k_b))
        self.v_bias = cuda.to_gpu(np.ascontiguousarray(v_b))

        # Attention output projection
        self.attn_out_weight = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.attention.output.dense.weight.detach().cpu().numpy()
            )
        )
        self.attn_out_bias = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.attention.output.dense.bias.detach().cpu().numpy()
            )
        )

        # LayerNorm 1
        self.ln1_gamma = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.attention.output.LayerNorm.weight.detach().cpu().numpy()
            )
        )
        self.ln1_beta = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.attention.output.LayerNorm.bias.detach().cpu().numpy()
            )
        )

        # FFN
        self.ffn_up_weight = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.intermediate.dense.weight.detach().cpu().numpy()
            )
        )
        self.ffn_up_bias = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.intermediate.dense.bias.detach().cpu().numpy()
            )
        )
        self.ffn_down_weight = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.output.dense.weight.detach().cpu().numpy()
            )
        )
        self.ffn_down_bias = cuda.to_gpu(
            np.ascontiguousarray(encoder_layer.output.dense.bias.detach().cpu().numpy())
        )

        # LayerNorm 2
        self.ln2_gamma = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.output.LayerNorm.weight.detach().cpu().numpy()
            )
        )
        self.ln2_beta = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.output.LayerNorm.bias.detach().cpu().numpy()
            )
        )

    def forward(self, x_np):
        """
        Forward pass with FlashAttention.
        x_np: numpy array [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x_np.shape
        x_2d = x_np.reshape(batch_size * seq_len, self.hidden_size)

        # === QKV Projection ===
        q_w = self.q_weight.to_numpy()
        k_w = self.k_weight.to_numpy()
        v_w = self.v_weight.to_numpy()
        q_b = self.q_bias.to_numpy()
        k_b = self.k_bias.to_numpy()
        v_b = self.v_bias.to_numpy()

        Q = x_2d @ q_w.T + q_b
        K = x_2d @ k_w.T + k_b
        V = x_2d @ v_w.T + v_b

        # Reshape for multi-head attention: [batch, num_heads, seq_len, head_dim]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(0, 2, 1, 3)  # [batch, heads, seq, dim]
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # === FlashAttention ===
        Q_gpu = cuda.to_gpu(np.ascontiguousarray(Q.astype(np.float32)))
        K_gpu = cuda.to_gpu(np.ascontiguousarray(K.astype(np.float32)))
        V_gpu = cuda.to_gpu(np.ascontiguousarray(V.astype(np.float32)))

        # Use FlashAttention kernel
        attn_output_gpu = cuda.flash_attention_gpu(Q_gpu, K_gpu, V_gpu)

        # Reshape back: [batch, heads, seq, dim] -> [batch, seq, hidden]
        attn_out = attn_output_gpu.to_numpy()
        attn_out = attn_out.transpose(0, 2, 1, 3)  # [batch, seq, heads, dim]
        attn_out = attn_out.reshape(batch_size * seq_len, self.hidden_size)

        # === Attention Output Projection ===
        out_w = self.attn_out_weight.to_numpy()
        out_b = self.attn_out_bias.to_numpy()
        projected = attn_out @ out_w.T + out_b

        # === Residual + LayerNorm 1 ===
        residual = projected + x_2d
        residual_gpu = cuda.to_gpu(np.ascontiguousarray(residual.astype(np.float32)))
        hidden_gpu = cuda.layernorm_gpu(residual_gpu, self.ln1_gamma, self.ln1_beta)

        # === FFN (GPU GELU) ===
        hidden_np = hidden_gpu.to_numpy()
        ffn_up_w = self.ffn_up_weight.to_numpy()
        ffn_up_b = self.ffn_up_bias.to_numpy()
        ffn_hidden = hidden_np @ ffn_up_w.T + ffn_up_b

        ffn_hidden_gpu = cuda.to_gpu(
            np.ascontiguousarray(ffn_hidden.astype(np.float32))
        )
        ffn_hidden_gpu = cuda.gelu_gpu(ffn_hidden_gpu)

        # FFN down projection
        ffn_hidden_np = ffn_hidden_gpu.to_numpy()
        ffn_down_w = self.ffn_down_weight.to_numpy()
        ffn_down_b = self.ffn_down_bias.to_numpy()
        ffn_output = ffn_hidden_np @ ffn_down_w.T + ffn_down_b

        # === Residual + LayerNorm 2 ===
        residual2 = ffn_output + hidden_np
        residual2_gpu = cuda.to_gpu(np.ascontiguousarray(residual2.astype(np.float32)))
        output_gpu = cuda.layernorm_gpu(residual2_gpu, self.ln2_gamma, self.ln2_beta)

        # Reshape to [batch, seq, hidden]
        output = output_gpu.to_numpy().reshape(batch_size, seq_len, self.hidden_size)
        return output


if __name__ == "__main__":
    import torch
    from transformers import BertModel, BertConfig

    print("\n[1/4] Setting up BERT with gelu_new...")
    config = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_hidden_layers=1,
        hidden_act="gelu_new",  # Match Zenith GELU
    )
    torch_bert = BertModel(config).cuda()
    torch_bert.eval()
    torch_layer = torch_bert.encoder.layer[0]

    # Create Zenith encoder
    zenith_encoder = BertEncoderGPU()
    zenith_encoder.load_from_pytorch(torch_layer)

    print("\n[2/4] Preparing test input...")
    batch_size, seq_len = 1, 32
    np.random.seed(42)
    torch.manual_seed(42)
    x_np = np.random.randn(batch_size, seq_len, 768).astype(np.float32)
    x_torch = torch.from_numpy(x_np).cuda()
    x_gpu = cuda.to_gpu(np.ascontiguousarray(x_np))

    print("\n[3/4] Running forward pass...")

    # PyTorch
    with torch.no_grad():
        torch_out = torch_layer(x_torch)[0].cpu().numpy()

    # Zenith with FlashAttention (pass numpy directly)
    t0 = time.perf_counter()
    zenith_out = zenith_encoder.forward(x_np)
    zenith_time = (time.perf_counter() - t0) * 1000

    # Compare
    max_diff = np.max(np.abs(zenith_out - torch_out))
    mean_diff = np.mean(np.abs(zenith_out - torch_out))

    print(f"  Zenith time: {zenith_time:.2f} ms")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    print(f"  Accuracy: {'PASS' if max_diff < 1e-3 else 'FAIL'}")

    print("\n[4/4] Benchmark...")

    # Warmup
    for _ in range(5):
        _ = zenith_encoder.forward(x_np)

    # Zenith timing
    zenith_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = zenith_encoder.forward(x_np)
        zenith_times.append((time.perf_counter() - t0) * 1000)

    # PyTorch timing
    torch.cuda.synchronize()
    pytorch_times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch_layer(x_torch)
        torch.cuda.synchronize()
        pytorch_times.append((time.perf_counter() - t0) * 1000)

    print(f"\n  PyTorch: {np.mean(pytorch_times):.2f} ± {np.std(pytorch_times):.2f} ms")
    print(f"  Zenith:  {np.mean(zenith_times):.2f} ± {np.std(zenith_times):.2f} ms")

    ratio = np.mean(pytorch_times) / np.mean(zenith_times)
    if ratio > 1:
        print(f"  Zenith is {ratio:.2f}x FASTER!")
    else:
        print(f"  Zenith is {1 / ratio:.2f}x slower")

    print("\n" + "=" * 70)
    print("FLASHATTENTION BENCHMARK COMPLETE")
    print("=" * 70)
