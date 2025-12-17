# BERT Encoder Benchmark using Zenith Operators
# Tests a single BERT encoder layer with Zenith GPU-resident operations
# Compares accuracy against PyTorch reference

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("ZENITH BERT ENCODER BENCHMARK")
print("=" * 70)

from zenith._zenith_core import cuda


def zenith_linear(x_gpu, weight_gpu, bias_gpu):
    """Linear layer: y = xW^T + b (using matmul_gpu)"""
    # x: [batch, in_features], weight: [out_features, in_features]
    # We need to transpose weight for matmul
    # For now, use numpy matmul on CPU then upload
    x_np = x_gpu.to_numpy()
    w_np = weight_gpu.to_numpy()
    b_np = bias_gpu.to_numpy()

    out_np = x_np @ w_np.T + b_np
    return cuda.to_gpu(np.ascontiguousarray(out_np))


class ZenithBertEncoderLayer:
    """
    Single BERT encoder layer using Zenith GPU ops.

    Architecture:
    1. Multi-Head Self-Attention
    2. LayerNorm (residual connection)
    3. Feed-Forward Network (2 linear + GELU)
    4. LayerNorm (residual connection)
    """

    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size

    def load_from_pytorch(self, encoder_layer):
        """Load weights from PyTorch BertLayer."""
        # Attention weights
        self.q_weight = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.attention.self.query.weight.detach().cpu().numpy()
            )
        )
        self.q_bias = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.attention.self.query.bias.detach().cpu().numpy()
            )
        )

        self.k_weight = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.attention.self.key.weight.detach().cpu().numpy()
            )
        )
        self.k_bias = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.attention.self.key.bias.detach().cpu().numpy()
            )
        )

        self.v_weight = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.attention.self.value.weight.detach().cpu().numpy()
            )
        )
        self.v_bias = cuda.to_gpu(
            np.ascontiguousarray(
                encoder_layer.attention.self.value.bias.detach().cpu().numpy()
            )
        )

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

        # FFN weights
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
        Forward pass through encoder layer.
        x: [batch_size, seq_len, hidden_size] as numpy
        """
        batch_size, seq_len, _ = x_np.shape

        # Reshape for 2D operations: [batch*seq, hidden]
        x_2d = x_np.reshape(batch_size * seq_len, self.hidden_size)
        x_gpu = cuda.to_gpu(np.ascontiguousarray(x_2d))

        # === Multi-Head Self-Attention ===
        # Project Q, K, V
        Q = zenith_linear(x_gpu, self.q_weight, self.q_bias)  # [B*S, H]
        K = zenith_linear(x_gpu, self.k_weight, self.k_bias)
        V = zenith_linear(x_gpu, self.v_weight, self.v_bias)

        # Simple attention (not split into heads for MVP)
        # attn_scores = Q @ K^T / sqrt(d)
        Q_np = Q.to_numpy()
        K_np = K.to_numpy()
        V_np = V.to_numpy()

        # Reshape for attention: [batch, heads, seq, head_dim]
        Q_np = Q_np.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K_np = K_np.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V_np = V_np.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        Q_np = Q_np.transpose(0, 2, 1, 3)  # [B, H, S, D]
        K_np = K_np.transpose(0, 2, 1, 3)
        V_np = V_np.transpose(0, 2, 1, 3)

        # Attention scores
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = np.matmul(Q_np, K_np.transpose(0, 1, 3, 2)) * scale

        # Softmax (use GPU for each head)
        attn_probs = np.zeros_like(attn_scores)
        for b in range(batch_size):
            for h in range(self.num_heads):
                scores_2d = attn_scores[b, h].reshape(1, -1)
                scores_gpu = cuda.to_gpu(
                    np.ascontiguousarray(scores_2d.astype(np.float32))
                )
                probs_gpu = cuda.softmax_gpu(scores_gpu)
                attn_probs[b, h] = probs_gpu.to_numpy().reshape(seq_len, seq_len)

        # Apply attention to values
        attn_output = np.matmul(attn_probs, V_np)  # [B, H, S, D]

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3)  # [B, S, H, D]
        attn_output = attn_output.reshape(batch_size * seq_len, self.hidden_size)
        attn_output_gpu = cuda.to_gpu(
            np.ascontiguousarray(attn_output.astype(np.float32))
        )

        # Attention output projection
        attn_output_gpu = zenith_linear(
            attn_output_gpu, self.attn_out_weight, self.attn_out_bias
        )

        # Residual + LayerNorm 1
        residual_np = attn_output_gpu.to_numpy() + x_2d
        residual_gpu = cuda.to_gpu(np.ascontiguousarray(residual_np.astype(np.float32)))
        hidden = cuda.layernorm_gpu(residual_gpu, self.ln1_gamma, self.ln1_beta)

        # === Feed-Forward Network ===
        # Up projection + GELU
        ffn_hidden = zenith_linear(hidden, self.ffn_up_weight, self.ffn_up_bias)
        ffn_hidden = cuda.gelu_gpu(ffn_hidden)

        # Down projection
        ffn_output = zenith_linear(ffn_hidden, self.ffn_down_weight, self.ffn_down_bias)

        # Residual + LayerNorm 2
        residual2_np = ffn_output.to_numpy() + hidden.to_numpy()
        residual2_gpu = cuda.to_gpu(
            np.ascontiguousarray(residual2_np.astype(np.float32))
        )
        output = cuda.layernorm_gpu(residual2_gpu, self.ln2_gamma, self.ln2_beta)

        # Reshape back to [batch, seq, hidden]
        output_np = output.to_numpy()
        output_np = output_np.reshape(batch_size, seq_len, self.hidden_size)

        return output_np


if __name__ == "__main__":
    import torch

    try:
        from transformers import BertModel, BertConfig

        HAS_TRANSFORMERS = True
    except ImportError:
        HAS_TRANSFORMERS = False
        print("  transformers not installed, using PyTorch only")

    print("\n[1/4] Setting up BERT encoder layer...")

    if HAS_TRANSFORMERS:
        # Use HuggingFace BERT with gelu_new (tanh approximation) to match Zenith
        config = BertConfig(
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
            num_hidden_layers=1,
            hidden_act="gelu_new",  # Use tanh approximation like Zenith
        )
        torch_bert = BertModel(config).cuda()
        torch_bert.eval()
        torch_layer = torch_bert.encoder.layer[0]
        print(f"  Using HuggingFace BertModel")
    else:
        # Fallback: just test GELU and LayerNorm
        print("  Testing GELU and LayerNorm only...")

    # Test individual operators first
    print("\n[2/4] Testing individual Transformer operators...")

    # Test GELU
    test_input = np.random.randn(4, 768).astype(np.float32)
    test_gpu = cuda.to_gpu(np.ascontiguousarray(test_input))
    gelu_output = cuda.gelu_gpu(test_gpu)

    # Compare with PyTorch GELU (use approximate='tanh' to match Zenith implementation)
    # Note: PyTorch default GELU uses exact erf, Zenith uses tanh approximation from BERT paper
    torch_gelu = torch.nn.GELU(approximate="tanh")
    torch_output = torch_gelu(torch.from_numpy(test_input)).detach().numpy()
    gelu_diff = np.max(np.abs(gelu_output.to_numpy() - torch_output))
    print(
        f"  GELU: max_diff = {gelu_diff:.2e} {'[PASS]' if gelu_diff < 1e-4 else '[FAIL]'}"
    )

    # Test LayerNorm
    gamma = np.ones(768, dtype=np.float32)
    beta = np.zeros(768, dtype=np.float32)
    gamma_gpu = cuda.to_gpu(np.ascontiguousarray(gamma))
    beta_gpu = cuda.to_gpu(np.ascontiguousarray(beta))

    ln_output = cuda.layernorm_gpu(test_gpu, gamma_gpu, beta_gpu)

    # Compare with PyTorch LayerNorm
    torch_ln = torch.nn.LayerNorm(768).cuda()
    torch_ln.weight.data.fill_(1.0)
    torch_ln.bias.data.fill_(0.0)
    torch_ln_output = (
        torch_ln(torch.from_numpy(test_input).cuda()).detach().cpu().numpy()
    )
    ln_diff = np.max(np.abs(ln_output.to_numpy() - torch_ln_output))
    print(
        f"  LayerNorm: max_diff = {ln_diff:.2e} {'[PASS]' if ln_diff < 1e-4 else '[FAIL]'}"
    )

    # Test Softmax
    softmax_input = np.random.randn(4, 128).astype(np.float32)
    softmax_gpu = cuda.to_gpu(np.ascontiguousarray(softmax_input))
    softmax_output = cuda.softmax_gpu(softmax_gpu)

    torch_softmax = torch.nn.functional.softmax(
        torch.from_numpy(softmax_input), dim=-1
    ).numpy()
    softmax_diff = np.max(np.abs(softmax_output.to_numpy() - torch_softmax))
    print(
        f"  Softmax: max_diff = {softmax_diff:.2e} {'[PASS]' if softmax_diff < 1e-4 else '[FAIL]'}"
    )

    if HAS_TRANSFORMERS:
        print("\n[3/4] Testing full BERT encoder layer...")

        # Create Zenith encoder
        zenith_encoder = ZenithBertEncoderLayer()
        zenith_encoder.load_from_pytorch(torch_layer)

        # Test input
        batch_size, seq_len = 1, 32
        np.random.seed(42)
        torch.manual_seed(42)
        x_np = np.random.randn(batch_size, seq_len, 768).astype(np.float32)
        x_torch = torch.from_numpy(x_np).cuda()

        # PyTorch forward
        with torch.no_grad():
            torch_out = torch_layer(x_torch)[0].cpu().numpy()

        # Zenith forward
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

        # Benchmark
        print("\n[4/4] Performance benchmark...")

        # Warmup
        for _ in range(3):
            _ = zenith_encoder.forward(x_np)

        # Zenith
        zenith_times = []
        for i in range(10):
            t0 = time.perf_counter()
            _ = zenith_encoder.forward(x_np)
            zenith_times.append((time.perf_counter() - t0) * 1000)

        # PyTorch
        torch.cuda.synchronize()
        pytorch_times = []
        for i in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = torch_layer(x_torch)
            torch.cuda.synchronize()
            pytorch_times.append((time.perf_counter() - t0) * 1000)

        print(
            f"\n  PyTorch: {np.mean(pytorch_times):.2f} ± {np.std(pytorch_times):.2f} ms"
        )
        print(f"  Zenith:  {np.mean(zenith_times):.2f} ± {np.std(zenith_times):.2f} ms")
        print(f"  Ratio:   {np.mean(zenith_times) / np.mean(pytorch_times):.2f}x")

    print("\n" + "=" * 70)
    print("BERT ENCODER TEST COMPLETE")
    print("=" * 70)
