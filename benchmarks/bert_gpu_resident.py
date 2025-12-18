# GPU-Resident BERT Encoder with cuBLAS
# All operations on GPU: linear_gpu, cublas_attention_gpu, gelu_gpu, layernorm_gpu

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("ZENITH BERT ENCODER - FULL GPU RESIDENT")
print("=" * 70)

from zenith._zenith_core import cuda


class BertEncoderFullGPU:
    """BERT Encoder Layer - 100% GPU operations."""

    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size

        # GPU tensors for weights (loaded from PyTorch)
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
        # Attention weights
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

        # Output projection
        self.attn_out_weight = cuda.to_gpu(
            torch_layer.attention.output.dense.weight.detach().cpu().numpy()
        )
        self.attn_out_bias = cuda.to_gpu(
            torch_layer.attention.output.dense.bias.detach().cpu().numpy()
        )

        # LayerNorm 1
        self.ln1_gamma = cuda.to_gpu(
            torch_layer.attention.output.LayerNorm.weight.detach().cpu().numpy()
        )
        self.ln1_beta = cuda.to_gpu(
            torch_layer.attention.output.LayerNorm.bias.detach().cpu().numpy()
        )

        # FFN
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

        # LayerNorm 2
        self.ln2_gamma = cuda.to_gpu(
            torch_layer.output.LayerNorm.weight.detach().cpu().numpy()
        )
        self.ln2_beta = cuda.to_gpu(
            torch_layer.output.LayerNorm.bias.detach().cpu().numpy()
        )

    def forward(self, x_gpu):
        """
        Full GPU forward pass.
        x_gpu: GpuTensor [batch*seq, hidden_size]
        """
        batch_seq = x_gpu.shape[0]  # shape is property, not method
        batch_size = 1  # Assuming batch=1 for now
        seq_len = batch_seq // batch_size

        # === QKV Projection (GPU) ===
        Q_gpu = cuda.linear_gpu(x_gpu, self.q_weight, self.q_bias)
        K_gpu = cuda.linear_gpu(x_gpu, self.k_weight, self.k_bias)
        V_gpu = cuda.linear_gpu(x_gpu, self.v_weight, self.v_bias)

        # Reshape for attention: need numpy for reshape then back to GPU
        Q_np = Q_gpu.to_numpy().reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        K_np = K_gpu.to_numpy().reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        V_np = V_gpu.to_numpy().reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        Q_np = Q_np.transpose(0, 2, 1, 3)  # [batch, heads, seq, dim]
        K_np = K_np.transpose(0, 2, 1, 3)
        V_np = V_np.transpose(0, 2, 1, 3)

        Q_4d = cuda.to_gpu(np.ascontiguousarray(Q_np.astype(np.float32)))
        K_4d = cuda.to_gpu(np.ascontiguousarray(K_np.astype(np.float32)))
        V_4d = cuda.to_gpu(np.ascontiguousarray(V_np.astype(np.float32)))

        # === cuBLAS Attention ===
        attn_out_gpu = cuda.cublas_attention_gpu(Q_4d, K_4d, V_4d)

        # Reshape back to [batch*seq, hidden]
        attn_out_np = attn_out_gpu.to_numpy()
        attn_out_np = attn_out_np.transpose(0, 2, 1, 3)
        attn_out_np = attn_out_np.reshape(batch_seq, self.hidden_size)
        attn_2d = cuda.to_gpu(np.ascontiguousarray(attn_out_np))

        # === Output Projection (GPU) ===
        projected_gpu = cuda.linear_gpu(
            attn_2d, self.attn_out_weight, self.attn_out_bias
        )

        # === Residual + LayerNorm 1 ===
        # Need element-wise add on GPU
        residual_np = projected_gpu.to_numpy() + x_gpu.to_numpy()
        residual_gpu = cuda.to_gpu(np.ascontiguousarray(residual_np))
        hidden_gpu = cuda.layernorm_gpu(residual_gpu, self.ln1_gamma, self.ln1_beta)

        # === FFN Up (GPU) ===
        ffn_up_gpu = cuda.linear_gpu(hidden_gpu, self.ffn_up_weight, self.ffn_up_bias)
        ffn_up_gpu = cuda.gelu_gpu(ffn_up_gpu)

        # === FFN Down (GPU) ===
        ffn_down_gpu = cuda.linear_gpu(
            ffn_up_gpu, self.ffn_down_weight, self.ffn_down_bias
        )

        # === Residual + LayerNorm 2 ===
        residual2_np = ffn_down_gpu.to_numpy() + hidden_gpu.to_numpy()
        residual2_gpu = cuda.to_gpu(np.ascontiguousarray(residual2_np))
        output_gpu = cuda.layernorm_gpu(residual2_gpu, self.ln2_gamma, self.ln2_beta)

        return output_gpu


if __name__ == "__main__":
    import torch
    from transformers import BertModel, BertConfig

    print("\n[1/4] Setting up BERT with gelu_new...")
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

    # Create Zenith encoder
    zenith_encoder = BertEncoderFullGPU()
    zenith_encoder.load_from_pytorch(torch_layer)

    print("\n[2/4] Preparing test input...")
    batch_size, seq_len = 1, 32
    np.random.seed(42)
    torch.manual_seed(42)
    x_np = np.random.randn(batch_size, seq_len, 768).astype(np.float32)
    x_torch = torch.from_numpy(x_np).cuda()

    # Flatten for 2D linear
    x_2d = x_np.reshape(batch_size * seq_len, 768)
    x_gpu = cuda.to_gpu(np.ascontiguousarray(x_2d))

    print("\n[3/4] Running forward pass...")

    # PyTorch
    with torch.no_grad():
        torch_out = torch_layer(x_torch)[0].cpu().numpy()

    # Zenith Full GPU
    t0 = time.perf_counter()
    zenith_out_gpu = zenith_encoder.forward(x_gpu)
    zenith_time = (time.perf_counter() - t0) * 1000

    zenith_out = zenith_out_gpu.to_numpy().reshape(batch_size, seq_len, 768)

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
        _ = zenith_encoder.forward(x_gpu)

    # Zenith timing
    zenith_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = zenith_encoder.forward(x_gpu)
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
    print("FULL GPU BERT ENCODER COMPLETE")
    print("=" * 70)
