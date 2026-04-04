# Experiment Execution Workflow

## Overview

Execute experiments based on approved hypotheses, tracking everything in MLflow.

**Important:** The specifics of *how* to run experiments come from `framework-context.md` 
after framework discovery. This document covers **principles** that apply regardless of 
framework, plus the MLflow integration patterns that are constant.

## Core Principles

### 1. Reproducibility Through Seeds

Every experiment should be reproducible via random seed control.

- **Minimum replication:** 5 seeds for basic statistical validity
- **Recommended:** 10+ seeds for tight confidence intervals
- **Seed strategy:** Use consistent seed ranges (e.g., 42-51) across related experiments

### 2. Treatment vs Baseline

Always compare against a well-defined baseline:

- **Baseline:** Current best known configuration OR default settings
- **Treatment:** Configuration with hypothesis-specific changes
- **Isolation:** Change one variable at a time when possible

### 3. Batch Execution

Group related experiments for efficiency:

- All seeds for a single hypothesis
- Parameter sweeps across a range
- Comparison across alternative configurations

## MLflow Integration (Framework-Agnostic)

These patterns work regardless of the underlying framework.

### Tagging Runs

**Always tag runs for retrieval.** This is the only reliable way to group experiments.

```python
import mlflow

mlflow.set_tags({
    "hypothesis_id": "H3",        # Links run to hypothesis
    "batch_id": "h3_20260320_1",  # Groups runs from same batch
    "seed": 42,                   # Reproducibility
    "treatment": "name_of_change", # What's being tested
})
```

### Logging Parameters

Log all configuration that affects the experiment:

```python
mlflow.log_params({
    "param_name": value,
    # ... all hypothesis-relevant parameters
})
```

### Logging Metrics

**The specific metrics depend on your framework.** Common patterns:

- **Iterative processes:** Log metrics at each step/epoch/iteration
- **Single-run processes:** Log final metrics only
- **Use `step=` parameter** for time-series data

```python
# Example: iterative logging (adapt to your framework's loop structure)
mlflow.log_metrics({"metric_name": value}, step=iteration)

# Example: final summary
mlflow.log_metrics({"final_metric": result})
```

### Logging Artifacts

Preserve anything needed for later analysis:

```python
mlflow.log_artifact("config.yaml")      # Configuration used
mlflow.log_artifact("results.json")     # Raw results
mlflow.log_artifact("figure.png")       # Visualizations
```

## Experimental Design Principles

### Parameter Studies

**Single parameter:**
- Hold all else constant
- Vary one parameter across meaningful range
- Consider log-scale for parameters with wide ranges

**Interaction studies:**
- Test combinations only when you suspect interactions
- Grows combinatorially — be selective

### Control Variables

Document what's held constant:

- Random seeds (reproducibility)
- Hardware/environment (if relevant)
- Data splits (if applicable)
- Framework version

## State Management

### Before Execution

Update `experiment-state.md` with planned experiments:

```markdown
## Queued Experiments

| Hypothesis | Treatment | Seeds | Status |
|------------|-----------|-------|--------|
| H3 | adaptive_rate | 42-51 | queued |
```

### During Execution

Track active runs:

```markdown
## Active Experiments

| Run ID | Hypothesis | Config | Status | Started |
|--------|------------|--------|--------|---------|
| abc123 | H3 | seed=42 | running | 2026-03-20 15:30 |
```

### After Execution

Record completed runs for interpretation phase.

## Error Handling Principles

### Common Issues

| Category | Examples | General Approach |
|----------|----------|------------------|
| Resource exhaustion | OOM, disk full | Reduce scale, batch differently |
| Connection issues | MLflow server down | Ensure server running, check URI |
| Numerical issues | NaN, overflow | Check inputs, add bounds/clipping |
| Timeout | Long-running jobs | Adjust limits, add checkpointing |

### Recovery Strategy

1. **Identify failed seeds/runs** from MLflow or logs
2. **Diagnose root cause** before retrying
3. **Rerun only failed experiments** — don't waste successful runs
4. **Tag retries** distinctly (e.g., `retry=1`)

## Framework-Specific Details

**These are discovered during framework analysis and documented in `framework-context.md`:**

- Entry point command (CLI, Python API, notebook)
- Configuration format (YAML, Python dict, CLI args)
- Iteration structure (if any)
- Native metrics logged
- Checkpointing support

**Do not assume these patterns.** Consult `framework-context.md` for the specific framework.

## Post-Execution Checklist

- [ ] All planned runs completed (or failures documented)
- [ ] MLflow runs tagged with `hypothesis_id` and `batch_id`
- [ ] Parameters and metrics logged appropriately
- [ ] Artifacts saved for reproducibility
- [ ] `experiment-state.md` updated with run IDs
- [ ] Ready for result interpretation
