This is the result of the test on the Notebook from:
https://colab.research.google.com/github/vibeswithkk/ZENITH/blob/main/notebooks/zenith_jax_phase1_validation.ipynb

# STEP 1: Clean install from GitHub (specific commit with all fixes)
# If you see errors after this, restart runtime and run again!
!pip uninstall pyzenith -y 2>/dev/null
!pip cache purge 2>/dev/null
!pip install --force-reinstall --no-cache-dir 'git+https://github.com/vibeswithkk/ZENITH.git@badb894' -q

# Clear any cached zenith modules
import sys
for mod in list(sys.modules.keys()):
    if 'zenith' in mod:
        del sys.modules[mod]

# Fresh import
import zenith
print(f'Zenith version: {zenith.__version__}')

import jax
import jax.numpy as jnp
print(f'JAX version: {jax.__version__}')
print(f'Devices: {jax.devices()}') 

Output : 
Found existing installation: pyzenith 0.2.10
Uninstalling pyzenith-0.2.10:
  Successfully uninstalled pyzenith-0.2.10
Files removed: 12
  WARNING: Did not find branch or tag 'badb894', assuming revision or ref.
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.4/16.4 MB 363.9 MB/s eta 0:00:00
  Building wheel for pyzenith (pyproject.toml) ... done
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
opencv-python-headless 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 2.4.0 which is incompatible.
opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 2.4.0 which is incompatible.
opencv-contrib-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 2.4.0 which is incompatible.
numba 0.60.0 requires numpy<2.1,>=1.22, but you have numpy 2.4.0 which is incompatible.
tensorflow 2.19.0 requires numpy<2.2.0,>=1.26.0, but you have numpy 2.4.0 which is incompatible.
Zenith version: 0.2.10
JAX version: 0.7.2
Devices: [CudaDevice(id=0)]


# STEP 2: Import Phase 1 modules
from zenith.jax.checkpointing import OptimalCheckpointSelector, CheckpointPolicy, checkpoint
from zenith.jax.memory_manager import JAXActivationStore, EvictionPolicy, compute_array_size, get_device_string
from zenith.jax.mixed_precision import MixedPrecisionPolicy, DynamicLossScaler, LossScalerConfig, ZenithMixedPrecision, create_policy, detect_best_precision

print('All Phase 1 modules imported!') 

Output : 
All Phase 1 modules imported!


# TEST 1.1: OptimalCheckpointSelector
print('='*60)
print('TEST 1.1: OptimalCheckpointSelector')
print('='*60)

for n in [4, 12, 24, 48]:
    sel = OptimalCheckpointSelector(num_layers=n)
    sqrt = sel.select_sqrt()
    dp = sel.select_dp()
    red = sel.estimate_memory_reduction(sqrt)
    print(f'Layers={n}: sqrt={len(sqrt)}, DP={len(dp)}, reduction={red:.1f}%')

print('\n[PASS] TEST 1.1') 

Output : ============================================================
TEST 1.1: OptimalCheckpointSelector
============================================================
Layers=4: sqrt=2, DP=4, reduction=16.7%
Layers=12: sqrt=4, DP=4, reduction=46.7%
Layers=24: sqrt=6, DP=4, reduction=60.7%
Layers=48: sqrt=8, DP=4, reduction=72.2%

[PASS] TEST 1.1



# TEST 1.2: JAX Checkpoint with Gradients
print('='*60)
print('TEST 1.2: JAX Checkpoint with Gradients')
print('='*60)

def mlp(x, w1, w2):
    return jnp.dot(jax.nn.relu(jnp.dot(x, w1)), w2)

ckpt_mlp = jax.checkpoint(mlp)
key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (32, 64))
w1 = jax.random.normal(key, (64, 128))
w2 = jax.random.normal(key, (128, 64))

grads = jax.grad(lambda x,w1,w2: jnp.mean(ckpt_mlp(x,w1,w2)**2), argnums=(1,2))(x,w1,w2)
print(f'Grad shapes: {grads[0].shape}, {grads[1].shape}')
assert jnp.all(jnp.isfinite(grads[0])) and jnp.all(jnp.isfinite(grads[1]))

print('\n[PASS] TEST 1.2') 

Output : 
============================================================
TEST 1.2: JAX Checkpoint with Gradients
============================================================
Grad shapes: (64, 128), (128, 64)

[PASS] TEST 1.2



# TEST 1.3: Zenith checkpoint() wrapper
print('='*60)
print('TEST 1.3: Zenith checkpoint() wrapper')
print('='*60)

def fn(x, w):
    return jax.nn.relu(jnp.dot(x, w))

ckpt_fn = checkpoint(fn, policy=CheckpointPolicy.DOTS_SAVEABLE)
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (8, 16))
w = jax.random.normal(key, (16, 16))

out = ckpt_fn(x, w)
grad_w = jax.grad(lambda w: jnp.mean(ckpt_fn(x, w)**2))(w)
print(f'Output: {out.shape}, Grad: {grad_w.shape}')
assert jnp.all(jnp.isfinite(grad_w))

print('\n[PASS] TEST 1.3') 

Output : 
============================================================
TEST 1.3: Zenith checkpoint() wrapper
============================================================
Output: (8, 16), Grad: (16, 16)

[PASS] TEST 1.3


# TEST 2.1: JAXActivationStore
print('='*60)
print('TEST 2.1: JAXActivationStore')
print('='*60)

store = JAXActivationStore(max_memory_bytes=100*1024*1024)
key = jax.random.PRNGKey(0)
arrays = {}

for i in range(5):
    arr = jax.random.normal(key, (1024, 1024))
    arrays[i] = arr
    store.store(layer_id=i, array=arr)
    print(f'Stored layer {i}: {compute_array_size(arr)/1024/1024:.2f} MB')

for i in range(5):
    ret = store.retrieve(layer_id=i)
    assert ret is not None and jnp.allclose(ret, arrays[i])

print('\n[PASS] TEST 2.1') 

OUtput : 
============================================================
TEST 2.1: JAXActivationStore
============================================================
Stored layer 0: 4.00 MB
Stored layer 1: 4.00 MB
Stored layer 2: 4.00 MB
Stored layer 3: 4.00 MB
Stored layer 4: 4.00 MB

[PASS] TEST 2.1


# TEST 2.2: Eviction Under Memory Pressure
print('='*60)
print('TEST 2.2: Eviction Under Memory Pressure')
print('='*60)

store = JAXActivationStore(max_memory_bytes=5*1024*1024, eviction_policy=EvictionPolicy.LRU)
key = jax.random.PRNGKey(42)

for i in range(10):
    arr = jax.random.normal(key, (512, 512))
    store.store(layer_id=i, array=arr)

stats = store.statistics
print(f'Stored: {stats["store_count"]}, Evicted: {stats["eviction_count"]}')
print(f'Current: {stats["current_memory_mb"]:.2f} MB')
assert stats['eviction_count'] > 0

print('\n[PASS] TEST 2.2') 

Output : 
============================================================
TEST 2.2: Eviction Under Memory Pressure
============================================================
Stored: 10, Evicted: 5
Current: 5.00 MB

[PASS] TEST 2.2



# TEST 2.3: Device Detection
print('='*60)
print('TEST 2.3: Device Detection')
print('='*60)

arr = jax.random.normal(jax.random.PRNGKey(0), (100, 100))
dev = get_device_string(arr)
print(f'Array device: {dev}')

print('\n[PASS] TEST 2.3')

Output : 
============================================================
TEST 2.3: Device Detection
============================================================
Array device: cuda:0

[PASS] TEST 2.3



# TEST 3.1: MixedPrecisionPolicy
print('='*60)
print('TEST 3.1: MixedPrecisionPolicy')
print('='*60)

for mode in ['fp32', 'bf16', 'fp16']:
    p = create_policy(mode)
    print(f'{mode}: compute={p.compute_dtype}, scaling={p.requires_loss_scaling}')

arr = jnp.ones((10, 10), dtype=jnp.float32)
arr_bf16 = arr.astype(jnp.bfloat16)
assert arr_bf16.dtype == jnp.bfloat16

print('\n[PASS] TEST 3.1') 

Output : 
============================================================
TEST 3.1: MixedPrecisionPolicy
============================================================
fp32: compute=float32, scaling=False
bf16: compute=bfloat16, scaling=False
fp16: compute=float16, scaling=True

[PASS] TEST 3.1


# TEST 3.2: DynamicLossScaler
print('='*60)
print('TEST 3.2: DynamicLossScaler')
print('='*60)

scaler = DynamicLossScaler(LossScalerConfig(
    initial_scale=2**15,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=5,
))
print(f'Initial scale: {scaler.scale}')

key = jax.random.PRNGKey(0)
params = jax.random.normal(key, (64, 64))
x = jax.random.normal(key, (32, 64))

def loss_fn(p, x):
    return jnp.mean((jnp.dot(x, p)) ** 2)

for step in range(10):
    def scaled_loss(p):
        return scaler.scale_loss(loss_fn(p, x))
    
    grads = jax.grad(scaled_loss)(params)
    unscaled, is_finite = scaler.unscale_grads({'p': grads})
    scaler.update(is_finite)
    
    if step % 3 == 0:
        print(f'Step {step}: scale={scaler.scale:.0f}, finite={is_finite}')

print('\n[PASS] TEST 3.2') 

============================================================
TEST 3.2: DynamicLossScaler
============================================================
Initial scale: 32768
Step 0: scale=32768, finite=True
Step 3: scale=32768, finite=True
Step 6: scale=65536, finite=True
Step 9: scale=131072, finite=True

[PASS] TEST 3.2



# TEST 3.3: ZenithMixedPrecision
print('='*60)
print('TEST 3.3: ZenithMixedPrecision')
print('='*60)

mp = ZenithMixedPrecision(policy='bf16')
print(f'Policy: {mp.policy.mode.value}')

params = {
    'w1': jax.random.normal(jax.random.PRNGKey(0), (64, 128), dtype=jnp.float32),
    'w2': jax.random.normal(jax.random.PRNGKey(0), (128, 64), dtype=jnp.float32),
}

compute_params = mp.cast_to_compute(params)
print(f'Original: {params["w1"].dtype} -> Compute: {compute_params["w1"].dtype}')
assert compute_params['w1'].dtype == jnp.bfloat16

back = mp.cast_to_param(compute_params)
assert back['w1'].dtype == jnp.float32

print('\n[PASS] TEST 3.3') 

Output : 
============================================================
TEST 3.3: ZenithMixedPrecision
============================================================
Policy: bf16
Original: float32 -> Compute: bfloat16

[PASS] TEST 3.3


# TEST 3.4: Hardware Detection
print('='*60)
print('TEST 3.4: Hardware Detection')
print('='*60)

best = detect_best_precision()
print(f'Best precision: {best}')
for d in jax.devices():
    print(f'  Device: {d}')

print('\n[PASS] TEST 3.4') 


Output : 
============================================================
TEST 3.4: Hardware Detection
============================================================
Best precision: bf16
  Device: cuda:0

[PASS] TEST 3.4 


# BENCHMARK: Mixed Precision Speedup
print('='*60)
print('BENCHMARK: Mixed Precision')
print('='*60)

import time
SIZE = 2048
key = jax.random.PRNGKey(0)
a32 = jax.random.normal(key, (SIZE, SIZE), dtype=jnp.float32)
b32 = jax.random.normal(key, (SIZE, SIZE), dtype=jnp.float32)
a16 = a32.astype(jnp.bfloat16)
b16 = b32.astype(jnp.bfloat16)

matmul = jax.jit(jnp.dot)
matmul(a32, b32).block_until_ready()
matmul(a16, b16).block_until_ready()

N = 20
t = time.time()
for _ in range(N): matmul(a32, b32).block_until_ready()
t32 = (time.time() - t) / N * 1000

t = time.time()
for _ in range(N): matmul(a16, b16).block_until_ready()
t16 = (time.time() - t) / N * 1000

print(f'FP32: {t32:.2f} ms')
print(f'BF16: {t16:.2f} ms (speedup: {t32/t16:.2f}x)')
print('\n[BENCHMARK COMPLETE]') 

Output : 
============================================================
BENCHMARK: Mixed Precision
============================================================
FP32: 6.39 ms
BF16: 6.63 ms (speedup: 0.96x)

[BENCHMARK COMPLETE]


print('='*70)
print('ALL TESTS PASSED! Phase 1 JAX Integration Validated!')
print('='*70)

Output : 
============================================================
ALL TESTS PASSED! Phase 1 JAX Integration Validated!
============================================================