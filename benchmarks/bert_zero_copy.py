# GPU-Resident BERT Encoder with Zero-Copy Operations
# All operations on GPU - NO to_numpy() calls!

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("ZENITH BERT ENCODER - ZERO-COPY GPU OPERATIONS")
print("=" * 70)

from zenith._zenith_core import cuda


class BertEncoderZeroCopy:
    """BERT Encoder Layer - 100% GPU, zero CPU copies."""

    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size

        # GPU tensors for weights
        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.q_bias = None
        self.k_bias = None
        self.v_bias = None
        self.attn_out_weight = None
        self.attn_out_bias = None
        self.ln1_gamma = None
        self.ln1_beta = None
        self.ffn_up_weight = None
        self.ffn_up_bias = None
        self.ffn_down_weight = None
        self.ffn_down_bias = None
        self.ln2_gamma = None
        self.ln2_beta = None

    def load_from_pytorch(self, torch_layer):
        """Load weights from PyTorch and keep on GPU."""
        self.q_weight = cuda.to_gpu(
            torch_layer.attention.self.query.weight.detach().cpu().numpy()
        )
        self.k_weight = cuda.to_gpu(
            torch_layer.attention.self.key.weight.detach().cpu().numpy()
        )
        self.v_weight = cuda.to_gpu(
            torch_layer.attention.self.value.weight.detach().cpu().numpy()
        )
        self.q_bias = cuda.to_gpu(
            torch_layer.attention.self.query.bias.detach().cpu().numpy()
        )
        self.k_bias = cuda.to_gpu(
            torch_layer.attention.self.key.bias.detach().cpu().numpy()
        )
        self.v_bias = cuda.to_gpu(
            torch_layer.attention.self.value.bias.detach().cpu().numpy()
        )
        self.attn_out_weight = cuda.to_gpu(
            torch_layer.attention.output.dense.weight.detach().cpu().numpy()
        )
        self.attn_out_bias = cuda.to_gpu(
            torch_layer.attention.output.dense.bias.detach().cpu().numpy()
        )
        self.ln1_gamma = cuda.to_gpu(
            torch_layer.attention.output.LayerNorm.weight.detach().cpu().numpy()
        )
        self.ln1_beta = cuda.to_gpu(
            torch_layer.attention.output.LayerNorm.bias.detach().cpu().numpy()
        )
        self.ffn_up_weight = cuda.to_gpu(
            torch_layer.intermediate.dense.weight.detach().cpu().numpy()
        )
        self.ffn_up_bias = cuda.to_gpu(
            torch_layer.intermediate.dense.bias.detach().cpu().numpy()
        )
        self.ffn_down_weight = cuda.to_gpu(
            torch_layer.output.dense.weight.detach().cpu().numpy()
        )
        self.ffn_down_bias = cuda.to_gpu(
            torch_layer.output.dense.bias.detach().cpu().numpy()
        )
        self.ln2_gamma = cuda.to_gpu(
            torch_layer.output.LayerNorm.weight.detach().cpu().numpy()
        )
        self.ln2_beta = cuda.to_gpu(
            torch_layer.output.LayerNorm.bias.detach().cpu().numpy()
        )

    def forward(self, x_gpu, batch_size=1, seq_len=32):
        """
        Full GPU forward pass - NO to_numpy() calls.
        x_gpu: GpuTensor [batch*seq, hidden_size]
        """
        # === QKV Projection (GPU cuBLAS) ===
        Q_gpu = cuda.linear_gpu(x_gpu, self.q_weight, self.q_bias)
        K_gpu = cuda.linear_gpu(x_gpu, self.k_weight, self.k_bias)
        V_gpu = cuda.linear_gpu(x_gpu, self.v_weight, self.v_bias)

        # === Transpose for attention (GPU kernel) ===
        # [batch*seq, hidden] treated as [batch, seq, heads, dim]
        Q_4d = cuda.transpose_for_attention(
            Q_gpu, batch_size, seq_len, self.num_heads, self.head_dim
        )
        K_4d = cuda.transpose_for_attention(
            K_gpu, batch_size, seq_len, self.num_heads, self.head_dim
        )
        V_4d = cuda.transpose_for_attention(
            V_gpu, batch_size, seq_len, self.num_heads, self.head_dim
        )

        # === cuBLAS Attention (GPU) ===
        attn_out_gpu = cuda.cublas_attention_gpu(Q_4d, K_4d, V_4d)

        # === Transpose back (GPU kernel) ===
        attn_2d = cuda.transpose_from_attention(
            attn_out_gpu, batch_size, self.num_heads, seq_len, self.head_dim
        )

        # === Output Projection (GPU cuBLAS) ===
        projected_gpu = cuda.linear_gpu(
            attn_2d, self.attn_out_weight, self.attn_out_bias
        )

        # === Residual + LayerNorm 1 (GPU) ===
        residual_gpu = cuda.add_2d_gpu(projected_gpu, x_gpu)
        hidden_gpu = cuda.layernorm_gpu(residual_gpu, self.ln1_gamma, self.ln1_beta)

        # === FFN Up + GELU (GPU) ===
        ffn_up_gpu = cuda.linear_gpu(hidden_gpu, self.ffn_up_weight, self.ffn_up_bias)
        ffn_up_gpu = cuda.gelu_gpu(ffn_up_gpu)

        # === FFN Down (GPU) ===
        ffn_down_gpu = cuda.linear_gpu(
            ffn_up_gpu, self.ffn_down_weight, self.ffn_down_bias
        )

        # === Residual + LayerNorm 2 (GPU) ===
        residual2_gpu = cuda.add_2d_gpu(ffn_down_gpu, hidden_gpu)
        output_gpu = cuda.layernorm_gpu(residual2_gpu, self.ln2_gamma, self.ln2_beta)

        return output_gpu


if __name__ == "__main__":
    import torch
    from transformers import BertModel, BertConfig

    print("\n[1/4] Setting up BERT...")
    config = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_hidden_layers=1,
        hidden_act="gelu_new",
    )
    torch_bert = BertModel(config).cuda()
    torch_bert.eval()
    torch_layer = torch_bert.encoder.layer[0]

    zenith_encoder = BertEncoderZeroCopy()
    zenith_encoder.load_from_pytorch(torch_layer)

    print("\n[2/4] Preparing test input...")
    batch_size, seq_len = 1, 32
    np.random.seed(42)
    torch.manual_seed(42)
    x_np = np.random.randn(batch_size, seq_len, 768).astype(np.float32)
    x_torch = torch.from_numpy(x_np).cuda()

    x_2d = x_np.reshape(batch_size * seq_len, 768)
    x_gpu = cuda.to_gpu(np.ascontiguousarray(x_2d))

    print("\n[3/4] Running forward pass...")

    with torch.no_grad():
        torch_out = torch_layer(x_torch)[0].cpu().numpy()

    t0 = time.perf_counter()
    zenith_out_gpu = zenith_encoder.forward(x_gpu, batch_size, seq_len)
    zenith_time = (time.perf_counter() - t0) * 1000

    zenith_out = zenith_out_gpu.to_numpy().reshape(batch_size, seq_len, 768)

    max_diff = np.max(np.abs(zenith_out - torch_out))
    mean_diff = np.mean(np.abs(zenith_out - torch_out))

    print(f"  Zenith time: {zenith_time:.2f} ms")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    print(f"  Accuracy: {'PASS' if max_diff < 1e-3 else 'FAIL'}")

    print("\n[4/4] Benchmark...")

    for _ in range(5):
        _ = zenith_encoder.forward(x_gpu, batch_size, seq_len)

    zenith_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = zenith_encoder.forward(x_gpu, batch_size, seq_len)
        zenith_times.append((time.perf_counter() - t0) * 1000)

    torch.cuda.synchronize()
    pytorch_times = []
    for _ in range(50):
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
    print("ZERO-COPY BERT ENCODER COMPLETE")
    print("=" * 70)
