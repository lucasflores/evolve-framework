# Quickstart: Dry-Run Statistics Tool

**Feature**: 014-dry-run-statistics

## Basic Usage

```python
from evolve.config import UnifiedConfig
from evolve.experiment.dry_run import dry_run

# Load or create your experiment config
config = UnifiedConfig.from_file("experiment.json")

# Get cost estimate
report = dry_run(config)

# Print the formatted summary table
print(report)
```

**Output**:

```
╔══════════════════╦═══════════════╦═════════╦════════════╗
║ Phase            ║ Est. Time     ║ % Total ║ Bottleneck ║
╠══════════════════╬═══════════════╬═════════╬════════════╣
║ Initialization   ║     0.12s     ║   0.1%  ║            ║
║ Evaluation       ║   312.50s     ║  72.3%  ║     ★      ║
║ Selection        ║    23.40s     ║   5.4%  ║            ║
║ Variation        ║    89.00s     ║  20.6%  ║            ║
║ Merge            ║     7.10s     ║   1.6%  ║            ║
╠══════════════════╬═══════════════╬═════════╬════════════╣
║ TOTAL            ║   432.12s     ║ 100.0%  ║            ║
╚══════════════════╩═══════════════╩═════════╩════════════╝

Resources: 8 CPUs, 32.0 GB RAM, No GPU | Backend: parallel (8 workers)
Memory:    ~48.2 MB population, ~1.2 MB history | Total: ~49.4 MB
Note:      Stopping criteria configured — run may terminate before 500 generations.
```

## With an Explicit Evaluator

```python
from evolve.experiment.dry_run import dry_run
from evolve.evaluation import FunctionEvaluator
from evolve.evaluation.reference.functions import rastrigin

evaluator = FunctionEvaluator(rastrigin)
report = dry_run(config, evaluator=evaluator)

# Access structured data programmatically
print(f"Total estimated time: {report.total_estimated_ms / 1000:.1f}s")
print(f"Bottleneck: {next(p.name for p in report.phase_estimates if p.is_bottleneck)}")
print(f"GPUs detected: {report.resources.gpu_available}")
```

## Comparing Configurations

```python
configs = [
    config.with_params(population_size=100),
    config.with_params(population_size=500),
    config.with_params(population_size=1000),
]

for c in configs:
    r = dry_run(c)
    print(f"pop={c.population_size}: {r.total_estimated_ms / 1000:.0f}s estimated")
```

## Meta-Evolution Cost Estimation

```python
# Config with meta-evolution enabled
meta_config = UnifiedConfig.from_file("meta_experiment.json")

report = dry_run(meta_config)
print(report)

# Inspect the meta-evolution breakdown
if report.meta_estimate:
    m = report.meta_estimate
    print(f"Inner run cost: {m.inner_run_estimate_ms / 1000:.1f}s")
    print(f"Outer generations: {m.outer_generations}")
    print(f"Trials per config: {m.trials_per_config}")
    print(f"Total inner runs (worst case): {m.total_inner_runs}")
    print(f"Total meta-evolution cost: {m.total_estimated_ms / 1000:.0f}s")
```

## Inspecting Active Subsystems and Caveats

```python
report = dry_run(config)

# See which optional subsystems are included in the estimate
print(f"Active subsystems: {report.active_subsystems}")
# → ('erp', 'multiobjective', 'decoder', 'tracking')

# Check caveats about unbenchmarkable overheads
for caveat in report.caveats:
    print(f"⚠ {caveat}")
# → ⚠ ERP recovery overhead not included; triggers only if mating success < 0.1
# → ⚠ Remote MLflow tracking latency not benchmarked (50-500ms per logged generation)
```
