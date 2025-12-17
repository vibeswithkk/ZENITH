# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Benchmark Model Definitions

Provides standard benchmark models for performance testing.
Per CetakBiru Section 5.3: ResNet, BERT, GPT-2 as standard benchmarks.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
import importlib


@dataclass
class ModelConfig:
    """Configuration for a benchmark model."""

    name: str
    description: str
    framework: str  # pytorch, tensorflow, jax
    input_shape: Tuple[int, ...]
    num_classes: int = 1000
    precision: str = "fp32"
    batch_sizes: Tuple[int, ...] = (1, 4, 8, 16, 32)


# ============================================================================
# Model Registry
# ============================================================================

BENCHMARK_MODELS = {
    # Computer Vision
    "resnet18": ModelConfig(
        name="resnet18",
        description="ResNet-18 (11.7M params)",
        framework="pytorch",
        input_shape=(3, 224, 224),
        num_classes=1000,
    ),
    "resnet50": ModelConfig(
        name="resnet50",
        description="ResNet-50 (25.6M params)",
        framework="pytorch",
        input_shape=(3, 224, 224),
        num_classes=1000,
    ),
    "resnet101": ModelConfig(
        name="resnet101",
        description="ResNet-101 (44.5M params)",
        framework="pytorch",
        input_shape=(3, 224, 224),
        num_classes=1000,
    ),
    "mobilenet_v2": ModelConfig(
        name="mobilenet_v2",
        description="MobileNet V2 (3.5M params)",
        framework="pytorch",
        input_shape=(3, 224, 224),
        num_classes=1000,
    ),
    "efficientnet_b0": ModelConfig(
        name="efficientnet_b0",
        description="EfficientNet-B0 (5.3M params)",
        framework="pytorch",
        input_shape=(3, 224, 224),
        num_classes=1000,
    ),
    # NLP Models
    "bert_base": ModelConfig(
        name="bert_base",
        description="BERT Base (110M params)",
        framework="pytorch",
        input_shape=(512,),  # sequence length
        num_classes=2,  # classification
        batch_sizes=(1, 4, 8, 16),
    ),
    "distilbert": ModelConfig(
        name="distilbert",
        description="DistilBERT (66M params)",
        framework="pytorch",
        input_shape=(512,),
        num_classes=2,
        batch_sizes=(1, 4, 8, 16),
    ),
    # Generative Models
    "gpt2": ModelConfig(
        name="gpt2",
        description="GPT-2 Small (117M params)",
        framework="pytorch",
        input_shape=(256,),  # sequence length
        batch_sizes=(1, 2, 4),
    ),
}


def list_models() -> list:
    """List all available benchmark models."""
    return list(BENCHMARK_MODELS.keys())


def get_model_config(name: str) -> Optional[ModelConfig]:
    """Get model configuration by name."""
    return BENCHMARK_MODELS.get(name)


def get_model(name: str, device: str = "cpu") -> Optional[Any]:
    """
    Load a benchmark model.

    Args:
        name: Model name from BENCHMARK_MODELS
        device: Target device ("cpu", "cuda")

    Returns:
        Loaded model or None if not available
    """
    config = get_model_config(name)
    if config is None:
        return None

    if config.framework == "pytorch":
        return _load_pytorch_model(name, device)
    elif config.framework == "tensorflow":
        return _load_tensorflow_model(name, device)
    elif config.framework == "jax":
        return _load_jax_model(name, device)

    return None


def _load_pytorch_model(name: str, device: str) -> Optional[Any]:
    """Load a PyTorch model."""
    try:
        import torch
        import torchvision.models as models

        # Map model names to torchvision functions
        model_map = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "mobilenet_v2": models.mobilenet_v2,
            "efficientnet_b0": models.efficientnet_b0,
        }

        if name in model_map:
            model = model_map[name](pretrained=False)
            model.eval()
            model.to(device)
            return model

        # NLP models
        if name in ("bert_base", "distilbert"):
            try:
                from transformers import AutoModel

                hf_name = (
                    "bert-base-uncased"
                    if name == "bert_base"
                    else "distilbert-base-uncased"
                )
                model = AutoModel.from_pretrained(hf_name)
                model.eval()
                model.to(device)
                return model
            except ImportError:
                return None

        if name == "gpt2":
            try:
                from transformers import GPT2Model

                model = GPT2Model.from_pretrained("gpt2")
                model.eval()
                model.to(device)
                return model
            except ImportError:
                return None

    except ImportError:
        return None

    return None


def _load_tensorflow_model(name: str, device: str) -> Optional[Any]:
    """Load a TensorFlow model."""
    try:
        import tensorflow as tf

        # For now, only support ResNet from tf.keras.applications
        if name == "resnet50":
            return tf.keras.applications.ResNet50(weights=None)
        elif name == "resnet101":
            return tf.keras.applications.ResNet101(weights=None)
        elif name == "mobilenet_v2":
            return tf.keras.applications.MobileNetV2(weights=None)

    except ImportError:
        pass

    return None


def _load_jax_model(name: str, device: str) -> Optional[Any]:
    """Load a JAX/Flax model."""
    # JAX model loading would go here
    return None


def get_sample_input(
    name: str, batch_size: int = 1, device: str = "cpu"
) -> Optional[Any]:
    """
    Get sample input for a model.

    Args:
        name: Model name
        batch_size: Batch size
        device: Target device

    Returns:
        Sample input tensor
    """
    config = get_model_config(name)
    if config is None:
        return None

    if config.framework == "pytorch":
        try:
            import torch

            if name in (
                "resnet18",
                "resnet50",
                "resnet101",
                "mobilenet_v2",
                "efficientnet_b0",
            ):
                shape = (batch_size,) + config.input_shape
                return torch.randn(shape, device=device)

            if name in ("bert_base", "distilbert", "gpt2"):
                seq_len = config.input_shape[0]
                input_ids = torch.randint(
                    0, 30000, (batch_size, seq_len), device=device
                )
                attention_mask = torch.ones(batch_size, seq_len, device=device)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

        except ImportError:
            pass

    return None
