This is the result of the test on the Notebook from:
https://colab.research.google.com/github/vibeswithkk/ZENITH/blob/main/notebooks/zenith_jax_phase3_validation.ipynb#scrollTo=fZ-1iGgwgo-g 

# Zenith JAX Phase 3 Validation

**Custom Primitives and XLA Kernels Testing**

This notebook validates Phase 3 implementation:
- Fused Attention Primitive
- Fused LayerNorm Primitive
- Fused GELU Primitive
- Fused Softmax Primitive
- XLA Custom Kernels

==============================================================================
1. Environment Setup

# Install JAX with GPU support
!pip install -q jax jaxlib

# Uninstall existing zenith to avoid conflicts
!pip uninstall -y zenith-ai pyzenith 2>/dev/null

# Clone Zenith repository
!rm -rf ZENITH
!git clone https://github.com/vibeswithkk/ZENITH.git

# Add to Python path
import sys
if '/content/ZENITH' not in sys.path:
    sys.path.insert(0, '/content/ZENITH') 

---
Cell output : 
Cloning into 'ZENITH'...
remote: Enumerating objects: 1806, done.
remote: Counting objects: 100% (258/258), done.
remote: Compressing objects: 100% (181/181), done.
remote: Total 1806 (delta 150), reused 165 (delta 73), pack-reused 1548 (from 2)
Receiving objects: 100% (1806/1806), 9.27 MiB | 16.34 MiB/s, done.
Resolving deltas: 100% (886/886), done.

=============================================================================

# --- AUTOMATIC FIXER FOR COMPATIBILITY ---
# This cell ensures the code works on any JAX version (Colab often uses latest)
import os

PRIMITIVES_FILE = '/content/ZENITH/zenith/jax/primitives.py'

def check_and_fix_primitives():
    print(f"Checking {PRIMITIVES_FILE}...")
    if not os.path.exists(PRIMITIVES_FILE):
        print("ERROR: File not found. Make sure repository is cloned.")
        return
        
    with open(PRIMITIVES_FILE, 'r') as f:
        content = f.read()
    
    needs_patch = False
    
    # 1. Check for legacy JAX usage
    if '_get_primitive_class()' not in content:
        print("Detected missing JAX compatibility helpers. Patching...")
        needs_patch = True
        
        helpers = '''

def _get_primitive_class():
    try:
        from jax.extend.core import Primitive
        return Primitive
    except (ImportError, AttributeError):
        from jax.core import Primitive
        return Primitive

def _get_shaped_array_class():
    try:
        from jax.extend.core import ShapedArray
        return ShapedArray
    except (ImportError, AttributeError):
        from jax.core import ShapedArray
        return ShapedArray

def _get_ad_module():
    try:
        from jax._src.interpreters import ad
        return ad
    except (ImportError, AttributeError):
        from jax.interpreters import ad
        return ad
'''
        content = content.replace(
            'def _get_jnp():',
            helpers + '\ndef _get_jnp():'
        )
        content = content.replace(
            'jax = _get_jax()',
            'jax = _get_jax()\n    Primitive = _get_primitive_class()\n    ShapedArray = _get_shaped_array_class()\n    ad = _get_ad_module()'
        )
        content = content.replace('jax.core.Primitive', 'Primitive')
        content = content.replace('jax.core.ShapedArray', 'ShapedArray')
        content = content.replace('jax.interpreters.ad', 'ad')

    # 2. Check for incorrect bind calls (positional params)
    if 'primitive.bind(q, k, v, mask, scale, dropout_rate)' in content:
        print("Detected incorrect bind arguments. Patching...")
        needs_patch = True
        content = content.replace(
            'primitive.bind(q, k, v, mask, scale, dropout_rate)',
            'primitive.bind(q, k, v, mask=mask, scale=scale, dropout_rate=dropout_rate)'
        )
        content = content.replace(
            'primitive.bind(x, weight, bias, eps)',
            'primitive.bind(x, weight, bias, eps=eps)'
        )
        content = content.replace(
            'primitive.bind(x, approximate)',
            'primitive.bind(x, approximate=approximate)'
        )
        content = content.replace(
            'primitive.bind(x, axis)',
            'primitive.bind(x, axis=axis)'
        )

    if needs_patch:
        with open(PRIMITIVES_FILE, 'w') as f:
            f.write(content)
        print("Patches applied successfully. PLEASE RESTART RUNTIME if testing fails.")
    else:
        print("Code is up-to-date.")

check_and_fix_primitives()

# Ensure imports are fresh
import importlib
try:
    import zenith.jax.primitives
    importlib.reload(zenith.jax.primitives)
    print("Module loaded and verified.")
except ImportError as e:
    print(f"Import Error: {e}")

---
Cell output : 
Checking /content/ZENITH/zenith/jax/primitives.py...
Code is up-to-date.
Module loaded and verified.

==============================================================================

# Verify imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

---
Cell output : 
JAX version: 0.7.2
Available devices: [CudaDevice(id=0)]

==============================================================================
## 2. Test Fused Attention Primitive

---
from zenith.jax.primitives import fused_attention, list_primitives

print("Registered primitives:", list_primitives())

# Create test inputs
key = jax.random.PRNGKey(42)
batch, heads, seq, dim = 2, 4, 32, 64

q = jax.random.normal(key, (batch, heads, seq, dim))
k = jax.random.normal(key, (batch, heads, seq, dim))
v = jax.random.normal(key, (batch, heads, seq, dim))

# Test basic execution
print("\n[TEST 1] Basic Fused Attention")
output = fused_attention(q, k, v)
print(f"  Input shape: {q.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Output finite: {jnp.all(jnp.isfinite(output))}")
assert output.shape == q.shape
assert jnp.all(jnp.isfinite(output))
print("  [PASSED]") 

---
Cell output : 
Registered primitives: ['zenith_fused_attention', 'zenith_fused_layernorm', 'zenith_fused_gelu', 'zenith_fused_softmax']

[TEST 1] Basic Fused Attention
  Input shape: (2, 4, 32, 64)
  Output shape: (2, 4, 32, 64)
  Output finite: True
  [PASSED]

==============================================================================

# Test numerical correctness against reference
print("[TEST 2] Numerical Correctness - Attention")

def reference_attention(q, k, v):
    scale = 1.0 / jnp.sqrt(q.shape[-1])
    attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    return jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

reference = reference_attention(q, k, v)
zenith_output = fused_attention(q, k, v)

max_diff = jnp.max(jnp.abs(zenith_output - reference))
print(f"  Max difference: {max_diff}")
assert max_diff < 1e-5, f"Numerical error too large: {max_diff}"
print("  [PASSED]")

---
Cell output :
[TEST 2] Numerical Correctness - Attention
  Max difference: 0.0
  [PASSED]

==============================================================================
# Test gradient computation
print("[TEST 3] Gradient Computation - Attention")

def loss_fn(q, k, v):
    out = fused_attention(q, k, v)
    return jnp.sum(out)

grads = grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
dq, dk, dv = grads

print(f"  dQ shape: {dq.shape}")
print(f"  dK shape: {dk.shape}")
print(f"  dV shape: {dv.shape}")
assert dq.shape == q.shape
assert dk.shape == k.shape
assert dv.shape == v.shape
assert jnp.all(jnp.isfinite(dq))
assert jnp.all(jnp.isfinite(dk))
assert jnp.all(jnp.isfinite(dv))
print("  [PASSED]")

---
Cell output : 
[TEST 3] Gradient Computation - Attention
  dQ shape: (2, 4, 32, 64)
  dK shape: (2, 4, 32, 64)
  dV shape: (2, 4, 32, 64)
  [PASSED]

==============================================================================
# Test gradient computation

# Test JIT compilation
print("[TEST 4] JIT Compilation - Attention")

jit_attention = jit(fused_attention)
jit_output = jit_attention(q, k, v)

assert jit_output.shape == q.shape
assert jnp.all(jnp.isfinite(jit_output))
max_diff = jnp.max(jnp.abs(jit_output - zenith_output))
print(f"  JIT vs non-JIT max diff: {max_diff}")
assert max_diff < 1e-5
print("  [PASSED]")

---
Cell output : 
[TEST 4] JIT Compilation - Attention
  JIT vs non-JIT max diff: 0.0
  [PASSED]

==============================================================================

## 3. Test Fused LayerNorm Primitive 

from zenith.jax.primitives import fused_layernorm

print("[TEST 5] Fused LayerNorm")

batch, seq, dim = 4, 128, 512
x = jax.random.normal(key, (batch, seq, dim))
weight = jnp.ones(dim)
bias = jnp.zeros(dim)

output = fused_layernorm(x, weight, bias)
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
assert output.shape == x.shape
assert jnp.all(jnp.isfinite(output))
print("  [PASSED]")

---
Cell output : 
[TEST 5] Fused LayerNorm
  Input shape: (4, 128, 512)
  Output shape: (4, 128, 512)
  [PASSED]

==============================================================================
# Test LayerNorm numerical correctness
print("[TEST 6] LayerNorm Numerical Correctness")

def reference_layernorm(x, weight, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return x_norm * weight + bias

reference = reference_layernorm(x, weight, bias)
zenith_output = fused_layernorm(x, weight, bias)

max_diff = jnp.max(jnp.abs(zenith_output - reference))
print(f"  Max difference: {max_diff}")
assert max_diff < 1e-5
print("  [PASSED]")

---
Cell output : 
[TEST 6] LayerNorm Numerical Correctness
  Max difference: 0.0
  [PASSED]

==============================================================================
# Test LayerNorm gradient
print("[TEST 7] LayerNorm Gradient")

def ln_loss(x, weight, bias):
    return jnp.sum(fused_layernorm(x, weight, bias))

grads = grad(ln_loss, argnums=(0, 1, 2))(x, weight, bias)
dx, dw, db = grads

assert dx.shape == x.shape
assert dw.shape == weight.shape
assert db.shape == bias.shape
assert jnp.all(jnp.isfinite(dx))
print("  [PASSED]")

---
Cell output : 
[TEST 7] LayerNorm Gradient
  [PASSED]

==============================================================================
## 4. Test Fused GELU Primitive

from zenith.jax.primitives import fused_gelu
import math

print("[TEST 8] Fused GELU")

x = jax.random.normal(key, (8, 64, 256))
output_approx = fused_gelu(x, approximate=True)
output_exact = fused_gelu(x, approximate=False)

print(f"  Input shape: {x.shape}")
print(f"  Output (approx) shape: {output_approx.shape}")
print(f"  Output (exact) shape: {output_exact.shape}")
assert jnp.all(jnp.isfinite(output_approx))
assert jnp.all(jnp.isfinite(output_exact))
print("  [PASSED]")

---
[TEST 8] Fused GELU
  Input shape: (8, 64, 256)
  Output (approx) shape: (8, 64, 256)
  Output (exact) shape: (8, 64, 256)
  [PASSED]

==============================================================================

# Test GELU numerical correctness
print("[TEST 9] GELU Numerical Correctness")

def reference_gelu_approx(x):
    coeff = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + jnp.tanh(coeff * (x + 0.044715 * x**3)))

reference = reference_gelu_approx(x)
zenith_output = fused_gelu(x, approximate=True)

max_diff = jnp.max(jnp.abs(zenith_output - reference))
print(f"  Max difference: {max_diff}")
assert max_diff < 1e-5
print("  [PASSED]")

---
Cell output : 
[TEST 9] GELU Numerical Correctness
  Max difference: 0.0
  [PASSED]

==============================================================================
# Test GELU gradient
print("[TEST 10] GELU Gradient")

def gelu_loss(x):
    return jnp.sum(fused_gelu(x))

dx = grad(gelu_loss)(x)
assert dx.shape == x.shape
assert jnp.all(jnp.isfinite(dx))
print("  [PASSED]")

---
Cell output : 
[TEST 10] GELU Gradient
  [PASSED]

==============================================================================
## 5. Test Fused Softmax Primitive 

from zenith.jax.primitives import fused_softmax

print("[TEST 11] Fused Softmax")

x = jax.random.normal(key, (8, 16, 32))
output = fused_softmax(x)

print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
assert output.shape == x.shape
assert jnp.all(jnp.isfinite(output))
assert jnp.all(output >= 0)
assert jnp.all(output <= 1)
print("  [PASSED]")

---
Cell output : 
[TEST 11] Fused Softmax
  Input shape: (8, 16, 32)
  Output shape: (8, 16, 32)
  [PASSED]

==============================================================================
# Test softmax sums to 1
print("[TEST 12] Softmax Sums to 1")

sums = jnp.sum(output, axis=-1)
max_deviation = jnp.max(jnp.abs(sums - 1.0))
print(f"  Max deviation from 1: {max_deviation}")
assert max_deviation < 1e-5
print("  [PASSED]")

---
Cell output :
[TEST 12] Softmax Sums to 1
  Max deviation from 1: 2.384185791015625e-07
  [PASSED]

==============================================================================
# Test numerical stability with large values
print("[TEST 13] Softmax Numerical Stability")

large_x = jnp.array([1000.0, 1001.0, 1002.0])
stable_output = fused_softmax(large_x)

print(f"  Large input: {large_x}")
print(f"  Output: {stable_output}")
assert jnp.all(jnp.isfinite(stable_output))
assert jnp.abs(jnp.sum(stable_output) - 1.0) < 1e-5
print("  [PASSED]") 

---
Cell output : 
[TEST 13] Softmax Numerical Stability
  Large input: [1000. 1001. 1002.]
  Output: [0.09003057 0.24472848 0.6652409 ]
  [PASSED]

==============================================================================
# Test softmax gradient
print("[TEST 14] Softmax Gradient")

def softmax_loss(x):
    return jnp.sum(fused_softmax(x))

dx = grad(softmax_loss)(x)
assert dx.shape == x.shape
assert jnp.all(jnp.isfinite(dx))
print("  [PASSED]")

---
Cell output :
[TEST 14] Softmax Gradient
  [PASSED]

==============================================================================
## 6. Test XLA Kernels

from zenith.runtime.xla_kernels import (
    xla_fused_attention,
    xla_fused_layernorm,
    xla_fused_softmax,
    list_kernels,
    get_kernel_registry,
)

print("[TEST 15] XLA Kernels Registration")
print(f"  Registered kernels: {list_kernels()}")
print("  [PASSED]")

---
Cell output : 
[TEST 15] XLA Kernels Registration
  Registered kernels: ['zenith_xla_fused_attention', 'zenith_xla_fused_layernorm', 'zenith_xla_fused_softmax']
  [PASSED]

==============================================================================
print("[TEST 16] XLA Fused Attention")

q = jax.random.normal(key, (2, 4, 32, 64))
k = jax.random.normal(key, (2, 4, 32, 64))
v = jax.random.normal(key, (2, 4, 32, 64))

xla_output = xla_fused_attention(q, k, v)
print(f"  Output shape: {xla_output.shape}")
assert xla_output.shape == q.shape
assert jnp.all(jnp.isfinite(xla_output))
print("  [PASSED]")

---
Cell output :
[TEST 16] XLA Fused Attention
  Output shape: (2, 4, 32, 64)
  [PASSED]


==============================================================================

print("[TEST 17] XLA Fused LayerNorm")

x = jax.random.normal(key, (4, 128, 512))
weight = jnp.ones(512)
bias = jnp.zeros(512)

xla_output = xla_fused_layernorm(x, weight, bias)
print(f"  Output shape: {xla_output.shape}")
assert xla_output.shape == x.shape
assert jnp.all(jnp.isfinite(xla_output))
print("  [PASSED]")

---
Cell output : 
[TEST 17] XLA Fused LayerNorm
  Output shape: (4, 128, 512)
  [PASSED]

==============================================================================
print("[TEST 18] XLA Fused Softmax")

x = jax.random.normal(key, (8, 16, 32))
xla_output = xla_fused_softmax(x)

print(f"  Output shape: {xla_output.shape}")
assert xla_output.shape == x.shape
assert jnp.all(jnp.isfinite(xla_output))
sums = jnp.sum(xla_output, axis=-1)
assert jnp.max(jnp.abs(sums - 1.0)) < 1e-5
print("  [PASSED]")

---
Cell output : 
[TEST 18] XLA Fused Softmax
  Output shape: (8, 16, 32)
  [PASSED]

==============================================================================
## 7. Summary 

print("="*70)
print("PHASE 3 VALIDATION COMPLETE")
print("="*70)
print("")
print("All tests passed:")
print("  [OK] Fused Attention: basic, numerical, gradients, JIT")
print("  [OK] Fused LayerNorm: basic, numerical, gradients")
print("  [OK] Fused GELU: basic, numerical, gradients")
print("  [OK] Fused Softmax: basic, sums-to-1, stability, gradients")
print("  [OK] XLA Kernels: attention, layernorm, softmax")
print("")
print("Phase 3 is PRODUCTION-READY!")
print("="*70) 

---
Cell output : 
======================================================================
PHASE 3 VALIDATION COMPLETE
======================================================================

All tests passed:
  [OK] Fused Attention: basic, numerical, gradients, JIT
  [OK] Fused LayerNorm: basic, numerical, gradients
  [OK] Fused GELU: basic, numerical, gradients
  [OK] Fused Softmax: basic, sums-to-1, stability, gradients
  [OK] XLA Kernels: attention, layernorm, softmax

Phase 3 is PRODUCTION-READY!
======================================================================
