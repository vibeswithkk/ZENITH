# Debug BERT Encoder Layer-by-Layer
# Finds exact divergence point between Zenith and PyTorch

import sys

sys.path.insert(0, "build/python")

import numpy as np
import torch
from transformers import BertModel, BertConfig

print("=" * 70)
print("BERT ENCODER LAYER-BY-LAYER DEBUG")
print("=" * 70)

from zenith._zenith_core import cuda

# Setup
config = BertConfig(
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
    num_hidden_layers=1,
)
torch_bert = BertModel(config).cuda()
torch_bert.eval()
layer = torch_bert.encoder.layer[0]

# Test input
batch_size, seq_len = 1, 32
hidden_size = 768
np.random.seed(42)
torch.manual_seed(42)
x_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
x_torch = torch.from_numpy(x_np).cuda()

print(f"\nInput shape: {x_np.shape}")

# ============================================================================
# STEP 1: QKV Projections
# ============================================================================
print("\n[STEP 1] Q, K, V Projections")

with torch.no_grad():
    # PyTorch
    x_2d_torch = x_torch.reshape(batch_size * seq_len, hidden_size)
    Q_torch = layer.attention.self.query(x_2d_torch.reshape(batch_size, seq_len, -1))
    K_torch = layer.attention.self.key(x_2d_torch.reshape(batch_size, seq_len, -1))
    V_torch = layer.attention.self.value(x_2d_torch.reshape(batch_size, seq_len, -1))

# Zenith
x_2d = x_np.reshape(batch_size * seq_len, hidden_size)
x_gpu = cuda.to_gpu(np.ascontiguousarray(x_2d))

q_w = np.ascontiguousarray(layer.attention.self.query.weight.detach().cpu().numpy())
q_b = np.ascontiguousarray(layer.attention.self.query.bias.detach().cpu().numpy())
k_w = np.ascontiguousarray(layer.attention.self.key.weight.detach().cpu().numpy())
k_b = np.ascontiguousarray(layer.attention.self.key.bias.detach().cpu().numpy())
v_w = np.ascontiguousarray(layer.attention.self.value.weight.detach().cpu().numpy())
v_b = np.ascontiguousarray(layer.attention.self.value.bias.detach().cpu().numpy())

# Linear: output = input @ weight.T + bias
Q_np = x_2d @ q_w.T + q_b
K_np = x_2d @ k_w.T + k_b
V_np = x_2d @ v_w.T + v_b

# Compare
Q_diff = np.max(
    np.abs(Q_np.reshape(batch_size, seq_len, -1) - Q_torch.detach().cpu().numpy())
)
K_diff = np.max(
    np.abs(K_np.reshape(batch_size, seq_len, -1) - K_torch.detach().cpu().numpy())
)
V_diff = np.max(
    np.abs(V_np.reshape(batch_size, seq_len, -1) - V_torch.detach().cpu().numpy())
)

print(f"  Q max_diff: {Q_diff:.2e} {'[PASS]' if Q_diff < 1e-5 else '[FAIL]'}")
print(f"  K max_diff: {K_diff:.2e} {'[PASS]' if K_diff < 1e-5 else '[FAIL]'}")
print(f"  V max_diff: {V_diff:.2e} {'[PASS]' if V_diff < 1e-5 else '[FAIL]'}")

# ============================================================================
# STEP 2: Attention Scores & Softmax
# ============================================================================
print("\n[STEP 2] Attention Scores & Softmax")

num_heads = 12
head_dim = hidden_size // num_heads

with torch.no_grad():
    # Run full self-attention to get attention probs
    attn_output_torch, attn_probs_torch = layer.attention.self(
        x_torch, output_attentions=True
    )

# Zenith attention computation
Q_reshaped = Q_np.reshape(batch_size, seq_len, num_heads, head_dim)
K_reshaped = K_np.reshape(batch_size, seq_len, num_heads, head_dim)
V_reshaped = V_np.reshape(batch_size, seq_len, num_heads, head_dim)

Q_reshaped = Q_reshaped.transpose(0, 2, 1, 3)  # [B, H, S, D]
K_reshaped = K_reshaped.transpose(0, 2, 1, 3)
V_reshaped = V_reshaped.transpose(0, 2, 1, 3)

# Attention scores
scale = 1.0 / np.sqrt(head_dim)
attn_scores = np.matmul(Q_reshaped, K_reshaped.transpose(0, 1, 3, 2)) * scale

# Compare attention scores (before softmax)
scores_torch = (
    (
        torch.matmul(
            Q_torch.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2),
            K_torch.reshape(batch_size, seq_len, num_heads, head_dim)
            .transpose(1, 2)
            .transpose(-2, -1),
        )
        * scale
    )
    .detach()
    .cpu()
    .numpy()
)

scores_diff = np.max(np.abs(attn_scores - scores_torch))
print(
    f"  Attention scores max_diff: {scores_diff:.2e} {'[PASS]' if scores_diff < 1e-5 else '[FAIL]'}"
)

# Softmax
attn_probs = np.zeros_like(attn_scores)
for b in range(batch_size):
    for h in range(num_heads):
        # Standard softmax
        row_max = np.max(attn_scores[b, h], axis=-1, keepdims=True)
        exp_scores = np.exp(attn_scores[b, h] - row_max)
        attn_probs[b, h] = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

probs_diff = np.max(np.abs(attn_probs - attn_probs_torch[1].detach().cpu().numpy()))
print(
    f"  Attention probs max_diff: {probs_diff:.2e} {'[PASS]' if probs_diff < 1e-5 else '[FAIL]'}"
)

# ============================================================================
# STEP 3: Apply Attention to Values
# ============================================================================
print("\n[STEP 3] Attention * Values")

attn_output = np.matmul(attn_probs, V_reshaped)  # [B, H, S, D]
attn_output = attn_output.transpose(0, 2, 1, 3)  # [B, S, H, D]
attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)

# Compare with PyTorch self-attention output
attn_out_diff = np.max(
    np.abs(attn_output - attn_output_torch[0].detach().cpu().numpy())
)
print(
    f"  Self-attention output max_diff: {attn_out_diff:.2e} {'[PASS]' if attn_out_diff < 1e-5 else '[FAIL]'}"
)

# ============================================================================
# STEP 4: Attention Output Projection
# ============================================================================
print("\n[STEP 4] Attention Output Projection")

out_w = np.ascontiguousarray(layer.attention.output.dense.weight.detach().cpu().numpy())
out_b = np.ascontiguousarray(layer.attention.output.dense.bias.detach().cpu().numpy())

attn_output_2d = attn_output.reshape(batch_size * seq_len, hidden_size)
projected = attn_output_2d @ out_w.T + out_b

with torch.no_grad():
    projected_torch = (
        layer.attention.output.dense(attn_output_torch[0]).detach().cpu().numpy()
    )

proj_diff = np.max(np.abs(projected.reshape(batch_size, seq_len, -1) - projected_torch))
print(
    f"  Projected output max_diff: {proj_diff:.2e} {'[PASS]' if proj_diff < 1e-5 else '[FAIL]'}"
)

# ============================================================================
# STEP 5: Residual + LayerNorm 1
# ============================================================================
print("\n[STEP 5] Residual + LayerNorm 1")

# Residual: projected + original_input
residual_1 = projected.reshape(batch_size, seq_len, hidden_size) + x_np

# LayerNorm
ln1_w = layer.attention.output.LayerNorm.weight.detach().cpu().numpy()
ln1_b = layer.attention.output.LayerNorm.bias.detach().cpu().numpy()
ln1_eps = layer.attention.output.LayerNorm.eps

# Manual LayerNorm
mean = np.mean(residual_1, axis=-1, keepdims=True)
var = np.var(residual_1, axis=-1, keepdims=True)
hidden_1 = (residual_1 - mean) / np.sqrt(var + ln1_eps) * ln1_w + ln1_b

with torch.no_grad():
    # PyTorch path
    proj_torch = layer.attention.output.dense(attn_output_torch[0])
    dropped = layer.attention.output.dropout(
        proj_torch
    )  # In eval mode, dropout is identity
    residual_torch = dropped + x_torch
    hidden_1_torch = (
        layer.attention.output.LayerNorm(residual_torch).detach().cpu().numpy()
    )

hidden1_diff = np.max(np.abs(hidden_1 - hidden_1_torch))
print(
    f"  After LN1 max_diff: {hidden1_diff:.2e} {'[PASS]' if hidden1_diff < 1e-4 else '[FAIL]'}"
)

# ============================================================================
# STEP 6: FFN (Intermediate)
# ============================================================================
print("\n[STEP 6] FFN Intermediate (Up Projection + GELU)")

ffn_up_w = np.ascontiguousarray(layer.intermediate.dense.weight.detach().cpu().numpy())
ffn_up_b = np.ascontiguousarray(layer.intermediate.dense.bias.detach().cpu().numpy())

hidden_1_2d = hidden_1.reshape(batch_size * seq_len, hidden_size)
ffn_hidden = hidden_1_2d @ ffn_up_w.T + ffn_up_b

# GELU (tanh approximation)
sqrt_2_pi = 0.7978845608028654
coef = 0.044715
x3 = ffn_hidden**3
inner = sqrt_2_pi * (ffn_hidden + coef * x3)
ffn_hidden_gelu = 0.5 * ffn_hidden * (1.0 + np.tanh(inner))

with torch.no_grad():
    ffn_hidden_torch = (
        layer.intermediate(torch.from_numpy(hidden_1_torch).cuda())
        .detach()
        .cpu()
        .numpy()
    )

ffn_diff = np.max(
    np.abs(ffn_hidden_gelu.reshape(batch_size, seq_len, -1) - ffn_hidden_torch)
)
print(
    f"  FFN hidden max_diff: {ffn_diff:.2e} {'[PASS]' if ffn_diff < 1e-4 else '[FAIL]'}"
)

# ============================================================================
# STEP 7: FFN Output (Down Projection)
# ============================================================================
print("\n[STEP 7] FFN Output (Down Projection)")

ffn_down_w = np.ascontiguousarray(layer.output.dense.weight.detach().cpu().numpy())
ffn_down_b = np.ascontiguousarray(layer.output.dense.bias.detach().cpu().numpy())

ffn_output = ffn_hidden_gelu @ ffn_down_w.T + ffn_down_b

with torch.no_grad():
    ffn_output_torch = (
        layer.output.dense(torch.from_numpy(ffn_hidden_torch).cuda())
        .detach()
        .cpu()
        .numpy()
    )

ffn_out_diff = np.max(
    np.abs(ffn_output.reshape(batch_size, seq_len, -1) - ffn_output_torch)
)
print(
    f"  FFN output max_diff: {ffn_out_diff:.2e} {'[PASS]' if ffn_out_diff < 1e-4 else '[FAIL]'}"
)

# ============================================================================
# STEP 8: Residual + LayerNorm 2
# ============================================================================
print("\n[STEP 8] Residual + LayerNorm 2")

# Residual: ffn_output + hidden_1
residual_2 = ffn_output.reshape(batch_size, seq_len, hidden_size) + hidden_1

# LayerNorm
ln2_w = layer.output.LayerNorm.weight.detach().cpu().numpy()
ln2_b = layer.output.LayerNorm.bias.detach().cpu().numpy()
ln2_eps = layer.output.LayerNorm.eps

mean2 = np.mean(residual_2, axis=-1, keepdims=True)
var2 = np.var(residual_2, axis=-1, keepdims=True)
output = (residual_2 - mean2) / np.sqrt(var2 + ln2_eps) * ln2_w + ln2_b

with torch.no_grad():
    # Full PyTorch layer output
    full_output_torch = layer(x_torch)[0].detach().cpu().numpy()

final_diff = np.max(np.abs(output - full_output_torch))
print(
    f"  Final output max_diff: {final_diff:.2e} {'[PASS]' if final_diff < 1e-3 else '[FAIL]'}"
)

print("\n" + "=" * 70)
print("LAYER-BY-LAYER DEBUG COMPLETE")
print("=" * 70)
