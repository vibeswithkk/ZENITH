# Zenith Migration Guide

This document provides upgrade guidance for Zenith versions.

## Version Compatibility

| Zenith | Python | PyTorch | JAX |
|--------|--------|---------|-----|
| 0.3.x  | 3.9+   | 1.13+   | 0.4+ |
| 0.2.x  | 3.9+   | 1.13+   | - |
| 0.1.x  | 3.9+   | 1.12+   | - |

---

## Upgrading to 0.3.0

### New Features
- JAX integration with custom primitives
- FX graph pattern replacement
- Terminal and Grafana dashboards
- torch.compile backend support

### Breaking Changes
None in this release.

### Deprecations
None in this release.

---

## Deprecation Policy

Zenith follows a phased deprecation policy:

| Phase | Timeline | Behavior |
|-------|----------|----------|
| **Announce** | Version N | Add deprecation warning |
| **Warning** | Version N+2 | Warning on every use |
| **Remove** | Version N+4 | Feature removed |

### Checking for Deprecations

```python
from zenith.compat import get_deprecation_list, print_deprecation_report

# List all deprecations
print_deprecation_report()

# Check programmatically  
for dep in get_deprecation_list():
    print(f"{dep['name']}: deprecated in {dep['since']}")
```

---

## Handling Deprecation Warnings

```python
import warnings
from zenith.compat import ZenithDeprecationWarning

# Show all deprecation warnings
warnings.filterwarnings("default", category=ZenithDeprecationWarning)

# Suppress warnings (not recommended)
warnings.filterwarnings("ignore", category=ZenithDeprecationWarning)

# Raise errors on deprecation (for testing)
warnings.filterwarnings("error", category=ZenithDeprecationWarning)
```

---

## Version History

### 0.3.0 (Current)
- Added JAX integration
- Added FX pattern replacement
- Added monitoring dashboards
- Added backward compatibility utilities

### 0.2.x
- Initial PyTorch support
- Basic CUDA kernels
- Graph optimization passes

### 0.1.x
- Initial release
- Core GraphIR implementation
