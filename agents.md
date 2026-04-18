# Repository Governance — agents.md

## Project-Wide Rules

- **Language**: Python 3.10+, full type hints (mypy strict)
- **Build**: Hatchling (`pyproject.toml`)
- **Dependencies**: NumPy for numeric ops; no ML framework deps in core
- **Style**: Conventional Commits; atomic commits with tests included
- **Testing**: pytest; TDD (Red-Green-Refactor); tests in `tests/` mirroring `evolve/` structure
- **Naming**: snake_case for modules/functions, PascalCase for classes, UPPER_SNAKE for constants
- **Imports**: `from __future__ import annotations`; absolute imports preferred
- **Immutability**: Prefer frozen dataclasses for data structures
- **No global mutable state**: Dependencies via constructor injection
- **Protocols over ABCs**: Use `typing.Protocol` for interfaces
- **Registry pattern**: All behavioral components registered via typed registries
- **Config**: `UnifiedConfig` is the single source of truth for experiments
