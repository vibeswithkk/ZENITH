# Hybrid BERT Encoder: FP16 Attention + FP32 Others
# Best balance of speed (Tensor Cores) and accuracy

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("ZENITH BERT - HYBRID FP16/FP32 ENCODER (Optimized)")
print("=" * 70)

from zenith._zenith_core import cuda


class BertLayerHybrid:
    """Single BERT layer: FP16 attention, FP32 others."""

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
        """Load weights from PyTorch layer."""
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

    def forward(self, x_gpu, batch_size, seq_len):
        """Hybrid forward: FP16 attention, FP32 rest."""

        # === QKV Projection (FP32 cuBLAS) ===
        Q_gpu = cuda.linear_gpu(x_gpu, self.q_weight, self.q_bias)
        K_gpu = cuda.linear_gpu(x_gpu, self.k_weight, self.k_bias)
        V_gpu = cuda.linear_gpu(x_gpu, self.v_weight, self.v_bias)

        # === Transpose for attention (FP32 GPU kernel) ===
        Q_4d = cuda.transpose_for_attention(
            Q_gpu, batch_size, seq_len, self.num_heads, self.head_dim
        )
        K_4d = cuda.transpose_for_attention(
            K_gpu, batch_size, seq_len, self.num_heads, self.head_dim
        )
        V_4d = cuda.transpose_for_attention(
            V_gpu, batch_size, seq_len, self.num_heads, self.head_dim
        )

        # === FP16 Attention with Tensor Cores! ===
        # This is where we get the Tensor Core speedup
        attn_out = cuda.attention_fp16_gpu(Q_4d, K_4d, V_4d)

        # === Transpose back (FP32) ===
        attn_4d = cuda.transpose_from_attention(
            attn_out, batch_size, self.num_heads, seq_len, self.head_dim
        )
        attn_2d = cuda.reshape_4d_to_2d(attn_4d, batch_size, seq_len, self.hidden_size)

        # === Output Projection (FP32 cuBLAS) ===
        projected = cuda.linear_gpu(attn_2d, self.attn_out_weight, self.attn_out_bias)

        # === Residual + LayerNorm (FP32) ===
        residual = cuda.add_2d_gpu(projected, x_gpu)
        hidden = cuda.layernorm_gpu(residual, self.ln1_gamma, self.ln1_beta)

        # === FFN Up + GELU (FP32) ===
        ffn_up = cuda.linear_gpu(hidden, self.ffn_up_weight, self.ffn_up_bias)
        ffn_up = cuda.gelu_gpu(ffn_up)

        # === FFN Down (FP32) ===
        ffn_down = cuda.linear_gpu(ffn_up, self.ffn_down_weight, self.ffn_down_bias)

        # === Residual + LayerNorm (FP32) ===
        residual2 = cuda.add_2d_gpu(ffn_down, hidden)
        output = cuda.layernorm_gpu(residual2, self.ln2_gamma, self.ln2_beta)

        return output


class BertEncoderHybrid:
    """Full BERT Encoder with hybrid FP16/FP32."""

    def __init__(self, num_layers=12, hidden_size=768, num_heads=12):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = [
            BertLayerHybrid(hidden_size, num_heads) for _ in range(num_layers)
        ]

    def load_from_pytorch(self, torch_encoder):
        """Load all layer weights."""
        for i, layer in enumerate(self.layers):
            layer.load_from_pytorch(torch_encoder.layer[i])

    def forward(self, x_gpu, batch_size, seq_len):
        """Forward through all layers."""
        hidden = x_gpu
        for layer in self.layers:
            hidden = layer.forward(hidden, batch_size, seq_len)
        return hidden


if __name__ == "__main__":
    import torch
    from transformers import BertModel, BertConfig

    NUM_LAYERS = 12
    HIDDEN_SIZE = 768
    NUM_HEADS = 12
    BATCH_SIZE = 1
    SEQ_LEN = 128

    print(
        f"\n[Config] layers={NUM_LAYERS}, batch={BATCH_SIZE}, "
        f"seq={SEQ_LEN}, hidden={HIDDEN_SIZE}"
    )
    print("[Strategy] FP16 Attention + FP32 Others")

    print("\n[1/5] Loading PyTorch BERT-base...")
    config = BertConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        intermediate_size=3072,
        num_hidden_layers=NUM_LAYERS,
        hidden_act="gelu_new",
    )
    torch_bert = BertModel(config).cuda()
    torch_bert.eval()

    print("\n[2/5] Creating Zenith Hybrid encoder...")
    zenith_encoder = BertEncoderHybrid(NUM_LAYERS, HIDDEN_SIZE, NUM_HEADS)
    zenith_encoder.load_from_pytorch(torch_bert.encoder)
    print(f"  Loaded {NUM_LAYERS} layers")

    print("\n[3/5] Preparing input...")
    np.random.seed(42)
    torch.manual_seed(42)
    x_np = np.random.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).astype(np.float32)
    x_torch = torch.from_numpy(x_np).cuda()

    x_2d = x_np.reshape(BATCH_SIZE * SEQ_LEN, HIDDEN_SIZE)
    x_gpu = cuda.to_gpu(np.ascontiguousarray(x_2d))

    print("\n[4/5] Forward pass comparison...")

    # PyTorch reference
    with torch.no_grad():
        torch_out = torch_bert.encoder(x_torch)[0].cpu().numpy()

    # Zenith Hybrid
    t0 = time.perf_counter()
    zenith_out_gpu = zenith_encoder.forward(x_gpu, BATCH_SIZE, SEQ_LEN)
    cuda.sync()  # Single sync point (optimized - no per-kernel sync)
    zenith_time = (time.perf_counter() - t0) * 1000

    zenith_out = zenith_out_gpu.to_numpy().reshape(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    max_diff = np.max(np.abs(zenith_out - torch_out))
    mean_diff = np.mean(np.abs(zenith_out - torch_out))

    print(f"  Zenith first pass: {zenith_time:.2f} ms")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    print(f"  Accuracy: {'PASS' if max_diff < 0.1 else 'FAIL'}")

    print("\n[5/5] Benchmark (50 runs)...")

    # Warmup
    for _ in range(5):
        _ = zenith_encoder.forward(x_gpu, BATCH_SIZE, SEQ_LEN)

    # Zenith timing
    zenith_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = zenith_encoder.forward(x_gpu, BATCH_SIZE, SEQ_LEN)
        cuda.sync()  # Single sync per iteration
        zenith_times.append((time.perf_counter() - t0) * 1000)

    # PyTorch timing
    torch.cuda.synchronize()
    pytorch_times = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch_bert.encoder(x_torch)
        torch.cuda.synchronize()
        pytorch_times.append((time.perf_counter() - t0) * 1000)

    print("\n" + "=" * 70)
    print("RESULTS - 12-LAYER BERT-BASE HYBRID")
    print("=" * 70)
    print(f"\n  Strategy: FP16 Attention + FP32 Linear/LayerNorm/FFN")
    print(f"\n  PyTorch: {np.mean(pytorch_times):.2f} ± {np.std(pytorch_times):.2f} ms")
    print(f"  Zenith:  {np.mean(zenith_times):.2f} ± {np.std(zenith_times):.2f} ms")

    speedup = np.mean(pytorch_times) / np.mean(zenith_times)
    if speedup > 1:
        print(f"\n  Zenith is {speedup:.2f}x FASTER!")
    else:
        print(f"\n  Zenith is {1 / speedup:.2f}x slower")

    print(f"\n  Accuracy: max_diff = {max_diff:.2e}")
    print("=" * 70)
