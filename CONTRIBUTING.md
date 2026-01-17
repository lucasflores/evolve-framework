# Contributing to Evolve Framework

Thank you for your interest in contributing to Evolve Framework! This document provides guidelines and best practices for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Architecture Guidelines](#architecture-guidelines)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all backgrounds and experience levels.

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- A virtual environment manager (venv, conda, etc.)

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/evolve-framework.git
cd evolve-framework
```

## Development Setup

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or with all optional dependencies
pip install -e ".[all,dev]"
```

### Verify Installation

```bash
# Run tests
pytest

# Run type checking
mypy evolve/

# Run linting
ruff check evolve/
```

## Code Style

### General Guidelines

1. **Type Hints**: All public functions must have complete type annotations
2. **Docstrings**: Use Google-style docstrings for all public APIs
3. **Line Length**: Maximum 100 characters
4. **Imports**: Use absolute imports, organized by stdlib → third-party → local

### Example

```python
"""
Module docstring explaining purpose.

NO ML FRAMEWORK IMPORTS ALLOWED in core modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from evolve.core.types import Individual


@dataclass
class MyClass:
    """
    Brief description of the class.
    
    Longer description if needed.
    
    Attributes:
        name: Description of name attribute
        value: Description of value attribute
        
    Example:
        >>> obj = MyClass(name="test", value=42)
        >>> obj.compute()
        84
    """
    
    name: str
    value: int
    
    def compute(self) -> int:
        """
        Compute something.
        
        Returns:
            The computed value
        """
        return self.value * 2
```

### Critical Rule: No ML Framework Imports

The following modules must NOT import PyTorch, TensorFlow, JAX, or other ML frameworks:

- `evolve/core/`
- `evolve/representation/`
- `evolve/evaluation/` (except optional backends)
- `evolve/experiment/`

This ensures the framework remains model-agnostic and lightweight.

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=evolve --cov-report=html

# Specific module
pytest tests/unit/test_engine.py

# Integration tests
pytest tests/integration/ -m integration

# Skip slow tests
pytest -m "not slow"
```

### Writing Tests

1. **Unit tests** go in `tests/unit/`
2. **Integration tests** go in `tests/integration/`
3. **Benchmarks** go in `tests/benchmarks/`

```python
import pytest
from evolve.core.types import Fitness


class TestFitness:
    """Tests for Fitness class."""
    
    def test_scalar_creation(self) -> None:
        """Scalar fitness creates single-element array."""
        fitness = Fitness.scalar(0.5)
        assert fitness.values[0] == 0.5
        assert len(fitness.values) == 1
    
    def test_dominates(self) -> None:
        """Dominance comparison works correctly."""
        f1 = Fitness.scalar(0.5)
        f2 = Fitness.scalar(0.3)
        assert f2.dominates(f1)  # Lower is better by default
    
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_valid_range(self, value: float) -> None:
        """Various fitness values are valid."""
        fitness = Fitness.scalar(value)
        assert fitness.is_valid
```

### Property-Based Testing

We use Hypothesis for property-based tests:

```python
from hypothesis import given, strategies as st


@given(st.lists(st.floats(allow_nan=False), min_size=1, max_size=100))
def test_best_is_actually_best(fitness_values: list[float]) -> None:
    """Population.best() returns the actual best."""
    # Create population from values
    # Assert best matches manual check
    pass
```

## Pull Request Process

### Before Submitting

1. **Create a branch**: `git checkout -b feature/my-feature`
2. **Write tests**: All new features need tests
3. **Update docs**: Document public APIs
4. **Run checks**:
   ```bash
   pytest
   mypy evolve/
   ruff check evolve/
   ```

### PR Guidelines

1. **Clear title**: "Add NSGA-III selection operator"
2. **Description**: Explain what and why
3. **Link issues**: "Fixes #123"
4. **Small PRs**: Prefer focused, reviewable changes

### Review Process

1. All PRs require at least one review
2. CI must pass (tests, type checking, linting)
3. Address reviewer feedback
4. Squash commits on merge

## Architecture Guidelines

### Module Organization

```
evolve/
├── core/           # Core types, engine, operators
├── representation/ # Genome types (Vector, Tree, Graph)
├── evaluation/     # Evaluators, benchmark functions
├── multiobjective/ # NSGA-II, Pareto utilities
├── diversity/      # Islands, speciation, novelty
├── rl/             # Gym adapters, RL evaluation
├── experiment/     # Config, checkpointing, tracking
├── backends/       # Optional acceleration
└── utils/          # Shared utilities
```

### Adding New Features

1. **Protocols first**: Define the interface
2. **Core implementation**: NumPy-only, no ML frameworks
3. **Optional backends**: ML framework integrations in `backends/`
4. **Tests**: Unit + integration tests
5. **Documentation**: API docs + tutorial if complex

### Design Principles

1. **Explicit over implicit**: Pass RNG, no global state
2. **Composition over inheritance**: Prefer protocols and dataclasses
3. **Immutability**: Prefer immutable data structures
4. **Reproducibility**: All randomness through seeded RNGs

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

Thank you for contributing! 🎉
