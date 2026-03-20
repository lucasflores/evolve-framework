# evolve-framework Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-01-13

## Active Technologies
- File-based checkpoints (pickle/JSON), optional MLflow artifact store (001-core-framework-architecture)
- Python 3.11+ + NumPy (existing), dataclasses (stdlib) (002-evolvable-reproduction)
- N/A (in-memory, uses existing checkpoint infrastructure) (002-evolvable-reproduction)
- Python 3.11+ + NumPy (array ops), NetworkX (graph representation, cycle detection) (003-scm-representation)
- N/A (in-memory, checkpoint via existing infrastructure) (003-scm-representation)
- Python 3.10+ + evolve (core framework), numpy, matplotlib, plotly, beautiful-mermaid, gymnasium (RL notebook) (004-tutorial-notebooks)
- N/A (notebooks are self-contained, synthetic data generated at runtime) (004-tutorial-notebooks)
- Python 3.10+ (supports 3.10, 3.11, 3.12) + numpy>=1.24.0, networkx>=3.0, typing_extensions (for <3.11) (005-unified-config-meta-evolution)
- JSON file serialization; no database required (005-unified-config-meta-evolution)
- Python 3.10+ (matches existing framework requirements in pyproject.toml) (006-mlflow-metrics-tracking)
- MLflow tracking server (remote) or local filesystem artifact store (006-mlflow-metrics-tracking)

- Python 3.10+ + NumPy (core); Optional: PyTorch, JAX, MLflow, Ray (001-core-framework-architecture)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.10+: Follow standard conventions

## Recent Changes
- 006-mlflow-metrics-tracking: Added Python 3.10+ (matches existing framework requirements in pyproject.toml)
- 005-unified-config-meta-evolution: Added Python 3.10+ (supports 3.10, 3.11, 3.12) + numpy>=1.24.0, networkx>=3.0, typing_extensions (for <3.11)
- 004-tutorial-notebooks: Added Python 3.10+ + evolve (core framework), numpy, matplotlib, plotly, beautiful-mermaid, gymnasium (RL notebook)


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
