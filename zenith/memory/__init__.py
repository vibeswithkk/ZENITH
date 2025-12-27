# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith Memory Optimization Module.

Provides memory optimization techniques for training and inference:
- Gradient Checkpointing Phase 1 (PyTorch Integration)
- Gradient Checkpointing Phase 2 (Native Implementation)
- Memory-efficient attention patterns
- Activation memory management
"""

# Phase 1: PyTorch Integration
from .gradient_checkpointing import (
    CheckpointPolicy,
    CheckpointConfig,
    CheckpointingContext,
    checkpoint,
    checkpoint_sequential,
    checkpoint_wrapper,
    unwrap_checkpoint,
    auto_checkpoint,
    estimate_memory_savings,
    get_memory_stats,
    SegmentCheckpointer,
    ModuleCheckpointer,
)

# Phase 2: Native Implementation
from .native_checkpointing import (
    native_checkpoint,
    native_checkpoint_sequential,
    NativeCheckpointer,
    get_native_checkpointer,
    ActivationStore,
    OptimalCheckpointSelector,
    EvictionPolicy,
    RematerializationStrategy,
    CheckpointFunction,
)

__all__ = [
    # Phase 1: Core API (PyTorch Integration)
    "checkpoint",
    "checkpoint_sequential",
    "checkpoint_wrapper",
    "unwrap_checkpoint",
    "auto_checkpoint",
    # Phase 1: Configuration
    "CheckpointPolicy",
    "CheckpointConfig",
    "CheckpointingContext",
    # Phase 1: Helpers
    "estimate_memory_savings",
    "get_memory_stats",
    # Phase 1: Advanced
    "SegmentCheckpointer",
    "ModuleCheckpointer",
    # Phase 2: Native Core API
    "native_checkpoint",
    "native_checkpoint_sequential",
    "NativeCheckpointer",
    "get_native_checkpointer",
    # Phase 2: Memory Management
    "ActivationStore",
    "OptimalCheckpointSelector",
    "EvictionPolicy",
    "RematerializationStrategy",
    "CheckpointFunction",
]
