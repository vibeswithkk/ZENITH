# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
TorchDynamo Integration - Auto-registration of Zenith backend.

This module provides seamless integration with PyTorch 2.x's torch.compile()
by automatically registering the Zenith backend when the module is imported.

Usage:
    import zenith  # Auto-registers 'zenith' backend
    import torch

    model = torch.nn.Linear(10, 5)
    compiled = torch.compile(model, backend="zenith")

The backend leverages:
    - PyTorchAdapter.create_compile_backend() for real optimization
    - ZenithEngine for kernel dispatch
    - Custom CUDA kernels for accelerated execution
"""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("zenith.integrations.torch_dynamo")

# Track registration status
_ZENITH_BACKEND_REGISTERED = False
_ZENITH_BACKEND_INSTANCE = None


def _create_zenith_backend(
    target: str = "cuda",
    precision: str = "fp32",
    opt_level: int = 2,
) -> Callable:
    """
    Create the Zenith backend for TorchDynamo.

    This function creates a backend that properly routes through
    PyTorchAdapter's optimization pipeline.

    Args:
        target: Target device ("cuda", "cpu")
        precision: Precision level ("fp32", "fp16", "bf16")
        opt_level: Optimization level (1-3)

    Returns:
        Backend callable for torch.compile
    """
    try:
        from zenith.adapters.pytorch_adapter import PyTorchAdapter, ZenithPyTorchConfig

        # Create adapter with proper configuration
        config = ZenithPyTorchConfig(
            target=target,
            precision=precision,
            opt_level=opt_level,
        )
        adapter = PyTorchAdapter(config=config)

        # Return the proper compile backend
        return adapter.create_compile_backend(
            target=target,
            precision=precision,
            opt_level=opt_level,
        )
    except Exception as e:
        logger.warning(f"Failed to create optimized Zenith backend: {e}")
        logger.warning("Falling back to pass-through backend")

        # Fallback to minimal pass-through
        def fallback_backend(gm: Any, example_inputs: list) -> Callable:
            return gm.forward

        return fallback_backend


def register_backend(
    name: str = "zenith",
    target: str = "cuda",
    precision: str = "fp32",
    opt_level: int = 2,
    force: bool = False,
) -> bool:
    """
    Register the Zenith backend with TorchDynamo.

    Args:
        name: Backend name to register (default: "zenith")
        target: Target device
        precision: Precision level
        opt_level: Optimization level
        force: Force re-registration even if already registered

    Returns:
        True if registration successful, False otherwise
    """
    global _ZENITH_BACKEND_REGISTERED, _ZENITH_BACKEND_INSTANCE

    if _ZENITH_BACKEND_REGISTERED and not force:
        logger.debug(f"Zenith backend '{name}' already registered")
        return True

    try:
        import torch

        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "_dynamo"):
            logger.debug("torch._dynamo not available (PyTorch < 2.0)")
            return False

        # Check if already registered
        if name in torch._dynamo.list_backends() and not force:
            logger.debug(f"Backend '{name}' already in TorchDynamo")
            _ZENITH_BACKEND_REGISTERED = True
            return True

        # Create and register the backend
        backend = _create_zenith_backend(
            target=target,
            precision=precision,
            opt_level=opt_level,
        )

        # Reset dynamo and register
        torch._dynamo.reset()
        torch._dynamo.register_backend(compiler_fn=backend, name=name)

        _ZENITH_BACKEND_REGISTERED = True
        _ZENITH_BACKEND_INSTANCE = backend

        logger.info(f"Zenith backend '{name}' registered successfully")
        logger.info(f"Usage: torch.compile(model, backend='{name}')")

        return True

    except ImportError:
        logger.debug("PyTorch not available")
        return False
    except Exception as e:
        logger.warning(f"Failed to register Zenith backend: {e}")
        return False


def is_registered() -> bool:
    """Check if Zenith backend is registered."""
    return _ZENITH_BACKEND_REGISTERED


def get_backend() -> Optional[Callable]:
    """Get the registered backend instance."""
    return _ZENITH_BACKEND_INSTANCE


# Auto-register on module import
def _auto_register():
    """Automatically register Zenith backend when module loads."""
    try:
        import torch

        if hasattr(torch, "_dynamo"):
            register_backend()
    except ImportError:
        pass  # PyTorch not installed, skip auto-registration


# Perform auto-registration
_auto_register()
