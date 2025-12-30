# Contributing to Zenith

Thank you for your interest in contributing to Zenith! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment.

## Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/vibeswithkk/ZENITH.git
cd ZENITH

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Building Native CUDA Kernels

```bash
# Requires CUDA toolkit and PyTorch with CUDA
python zenith/build_cuda.py
```

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Python and PyTorch versions
   - CUDA version (if applicable)
   - Minimal reproducible example
   - Full error traceback

### Suggesting Features

1. Open a feature request issue
2. Describe the use case
3. Explain expected behavior

### Submitting Code

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Write tests** for new functionality
4. **Ensure tests pass**: `pytest tests/ -v`
5. **Format code**: `black . && ruff check --fix .`
6. **Commit** with clear messages
7. **Push** and open a Pull Request

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters (Black default)
- Use docstrings for all public functions

```python
def my_function(input_tensor: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Brief description.

    Args:
        input_tensor: Description of input
        eps: Small value for numerical stability

    Returns:
        Processed tensor
    """
    pass
```

### CUDA/C++ Style

- Use C++17 features
- Follow Google C++ Style Guide
- Document kernel parameters

### Commit Messages

```
type(scope): brief description

Longer explanation if needed.

- Bullet points for details
- Reference issues: Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_quantization.py -v

# With coverage
pytest tests/ --cov=zenith --cov-report=html
```

### Security Testing

Zenith prioritizes security. We use **Bandit** for Static Application Security Testing (SAST) and **Hypothesis** for property-based fuzzing.

```bash
# Run Fuzzing Tests (Property-Based)
pytest security/fuzz_tests.py -v --hypothesis-show-statistics

# Run Static Security Analysis
bandit -r zenith/ -nn -ii
```

### Writing Tests

- Place tests in `tests/` directory
- Name files `test_*.py`
- Use descriptive test names
- Test edge cases

## Documentation

- Update docstrings for API changes
- Update README for new features
- Add tutorials for complex features

## Release Process

1. Update `CHANGELOG.md`
2. Bump version in `pyproject.toml`
3. Create GitHub release
4. PyPI publish via CI

## Questions?

Open a GitHub Discussion or Issue.

---

Thank you for contributing to Zenith!
