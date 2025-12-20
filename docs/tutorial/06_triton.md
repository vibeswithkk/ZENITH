# 6. Triton Deployment

Deploy model dengan Triton Inference Server.

## MockTritonClient

Untuk testing tanpa server:

```python
from zenith.serving.triton_client import MockTritonClient

client = MockTritonClient("localhost:8000")
```

## Register Model

```python
from zenith.serving.triton_client import ModelMetadata

# Register dengan metadata
client.register_model(
    "my_model",
    metadata=ModelMetadata(name="my_model", platform="python")
)
```

## Check Server Status

```python
print(f"Server Live: {client.is_server_live()}")
print(f"Server Ready: {client.is_server_ready()}")
print(f"Model Ready: {client.is_model_ready('my_model')}")
```

**Output:**
```
Server Live: True
Server Ready: True
Model Ready: True
```

## Run Inference

```python
from zenith.serving.triton_client import InferenceInput
import numpy as np

# Prepare input
data = np.array([1.0, 2.0, 3.0]).astype(np.float32)
inputs = [InferenceInput(name="input", data=data)]

# Run inference
result = client.infer("my_model", inputs)

print(f"Success: {result.success}")
print(f"Latency: {result.latency_ms:.3f} ms")
```

**Output:**
```
Success: True
Latency: 0.031 ms
```

## Custom Inference Handler

```python
def my_handler(inputs):
    """Custom model logic."""
    x = inputs[0].data
    return {"output": x * 2}

client.register_model("doubler", handler=my_handler)

result = client.infer("doubler", inputs)
output = result.get_output("output")
print(f"Output: {output}")
```

---

[← PyTorch](05_pytorch.md) | [Next: Autotuner →](07_autotuner.md)
