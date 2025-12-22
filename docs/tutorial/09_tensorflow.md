# Zenith + TensorFlow (Hybrid Mode)

Zenith mendukung integrasi dengan TensorFlow/Keras.

## Installation

```bash
pip install pyzenith[tensorflow]
```

## Quick Start

```python
import zenith
import tensorflow as tf

# Create Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with Zenith
optimized = zenith.compile(
    model,
    target="cuda",
    precision="fp32"
)

# Inference
x = tf.random.normal((32, 784))
output = optimized(x)
print(f"Output shape: {output.shape}")
```

## Functional API

```python
import tensorflow as tf
import zenith

# Functional model
inputs = tf.keras.Input(shape=(768,))
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
x = tf.keras.layers.LayerNormalization()(x)
x = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, x)

# Compile
optimized = zenith.compile(model, target="cuda")
```

## Export to ONNX

```python
import tf2onnx
import tensorflow as tf

# Convert to ONNX
spec = (tf.TensorSpec((None, 784), tf.float32, name="input"),)
output_path = "model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Exported to {output_path}")
```

## Mixed Precision

```python
from tensorflow.keras import mixed_precision

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

# Model will use FP16 for compute, FP32 for storage
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## Best Practices

1. **Use `model.compile()` before inference** for optimal graph construction
2. **Set `training=False`** in model call for inference
3. **Use TensorRT** for production deployment on NVIDIA GPUs
4. **Batch inference** for better GPU utilization

---

[← PyTorch](05_pytorch.md) | [Next: JAX →](09_jax.md)
