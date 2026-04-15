# Quickstart: Documentation Refactor

**Branch**: `011-docs-unifiedconfig-refactor` | **Date**: 2026-04-13

## Getting Started with This Feature

This feature is documentation-only. No framework code changes are required.

### Prerequisites

- Python 3.11+
- evolve-framework installed in development mode: `pip install -e ".[dev]"`
- Jupyter (for notebook validation): `pip install jupyter`
- Sphinx (for API docs validation): `pip install sphinx`

### Validation Commands

```bash
# Run tutorial notebooks (verify they execute)
cd docs/tutorials && jupyter nbconvert --execute --to notebook *.ipynb

# Run examples
python examples/mlflow_tracking_demo.py
python examples/sexual_selection.py
python examples/protocol_evolution.py
python examples/speciation_demo.py

# Build Sphinx docs
cd docs && make html

# Run existing tests (ensure no regressions)
pytest tests/
```

### Key Patterns to Follow

Every code example MUST follow this pattern:

```python
from evolve.config import UnifiedConfig
from evolve.factory import create_engine

config = UnifiedConfig(
    name="experiment_name",
    population_size=100,
    max_generations=50,
    selection="tournament",
    crossover="single_point",
    mutation="gaussian",
    genome_type="vector",
    genome_params={"size": 10, "bounds": (-5.0, 5.0)},
    seed=42,
)

engine = create_engine(config, evaluator=my_fitness_function)
result = engine.run()
```

For ERP features that cannot be fully declarative, use this pattern:

```python
config = UnifiedConfig(
    name="erp_experiment",
    # ... standard params ...
).with_erp(
    # ... ERP settings ...
)

engine = create_engine(config, evaluator=my_fitness_function)

# === Advanced: Manual Override ===
# Custom matchability function cannot be expressed declaratively.
# This is a known limitation documented in the ERP guide.
engine.set_matchability(custom_matchability_fn)

result = engine.run()
```
