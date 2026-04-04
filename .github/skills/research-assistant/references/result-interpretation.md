# Result Interpretation Workflow

## Overview

Analyze experiment results to determine hypothesis verdict and inform next steps.

**Important:** The specific metrics to analyze depend on your framework and research
domain. This document covers **principles** of statistical interpretation and the 
MLflow patterns for retrieving results. Consult `framework-context.md` for 
domain-specific metric names and meanings.

## CRITICAL: Data Source Requirements

**MANDATORY:** All result interpretation MUST use data fetched from MLflow.

**DO NOT:**
- Parse terminal output from experiment runs
- Extract metrics from print statements
- Rely on experiment script return values
- Interpret results before explicitly querying MLflow

**WHY:** Terminal output is ephemeral, may be truncated, and doesn't capture 
the full metric history. MLflow contains the authoritative record with:
- Full metric time-series (per-generation values)
- Logged parameters and tags
- Artifacts (best solutions, checkpoints)
- Consistent schema for comparison

**Before ANY interpretation, you MUST:**
1. Query MLflow for run data (see "Fetching Results" below)
2. Load metrics into a DataFrame or structured format
3. Only then proceed with statistical analysis

## Fetching Results from MLflow

### Python API (Execute in Terminal)

**Step 1: Query runs**

```python
import mlflow
import pandas as pd
from scipy import stats
import numpy as np

# Set tracking URI (match your config.yaml)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Search for runs with specific tags
runs_df = mlflow.search_runs(
    experiment_ids=["1"],  # Get experiment ID from mlflow.get_experiment_by_name()
    filter_string="tags.hypothesis_id = 'H1'",  # Or however you tagged runs
)

# If no tags, search by run name pattern or time range
# runs_df = mlflow.search_runs(search_all_experiments=True)
# runs_df = runs_df[runs_df['tags.mlflow.runName'].str.contains('h1_')]

print(f"Found {len(runs_df)} runs")
print(runs_df[['run_id', 'params.mutation_rate', 'metrics.best_fitness']].head(10))
```

**Step 2: Aggregate by condition**

```python
# Separate conditions based on parameters
control = runs_df[runs_df['params.mutation_rate'] == '0.1']
treatment = runs_df[runs_df['params.mutation_rate'] == '0.01']

# Compute summary statistics for primary metric
print("Control (mut=0.1):")
print(f"  n={len(control)}, mean={control['metrics.best_fitness'].mean():.4f}, "
      f"std={control['metrics.best_fitness'].std():.4f}")

print("Treatment (mut=0.01):")
print(f"  n={len(treatment)}, mean={treatment['metrics.best_fitness'].mean():.4f}, "
      f"std={treatment['metrics.best_fitness'].std():.4f}")
```

**Step 3: Statistical test**

```python
# t-test for significant difference
t_stat, p_value = stats.ttest_ind(
    control['metrics.best_fitness'].dropna(),
    treatment['metrics.best_fitness'].dropna()
)
print(f"t-test: t={t_stat:.3f}, p={p_value:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((control['metrics.best_fitness'].var() + 
                       treatment['metrics.best_fitness'].var()) / 2)
effect_size = (treatment['metrics.best_fitness'].mean() - 
               control['metrics.best_fitness'].mean()) / pooled_std
print(f"Cohen's d: {effect_size:.3f}")
```

### Discovering Available Metrics

```python
# See what metrics were logged
print("Available metrics:", [c for c in runs_df.columns if c.startswith('metrics.')])
print("Available params:", [c for c in runs_df.columns if c.startswith('params.')])
print("Available tags:", [c for c in runs_df.columns if c.startswith('tags.')])
```

### CLI Queries

```bash
# List runs in experiment
mlflow runs list --experiment-name research_H3

# Get specific run details
mlflow runs describe --run-id <run_id>

# Compare runs
mlflow runs compare --run-ids <id1>,<id2>,<id3>
```

## Statistical Analysis Principles

### Aggregation Across Seeds

For any primary metric you're comparing:

```python
import numpy as np
from scipy import stats

# Generic aggregation pattern
primary_metric = "metrics.YOUR_PRIMARY_METRIC"  # e.g., accuracy, reward, loss

results = {
    "mean": metrics_df[primary_metric].mean(),
    "std": metrics_df[primary_metric].std(),
    "median": metrics_df[primary_metric].median(),
    "min": metrics_df[primary_metric].min(),
    "max": metrics_df[primary_metric].max(),
}

# Confidence interval (95%)
n = len(metrics_df)
se = results["std"] / np.sqrt(n)
ci = stats.t.interval(0.95, n-1, loc=results["mean"], scale=se)
results["ci_95"] = ci
```

### Treatment vs Baseline Comparison

```python
# Separate treatment and baseline runs
treatment = metrics_df[metrics_df["tags.treatment"] == "TREATMENT_NAME"]
baseline = metrics_df[metrics_df["tags.treatment"] == "baseline"]

# Statistical test (t-test for means)
t_stat, p_value = stats.ttest_ind(
    treatment[primary_metric],
    baseline[primary_metric]
)

# Effect size (Cohen's d)
pooled_std = np.sqrt((treatment[primary_metric].var() + baseline[primary_metric].var()) / 2)
effect_size = (treatment[primary_metric].mean() - baseline[primary_metric].mean()) / pooled_std
```

### When to Use Which Test

| Scenario | Test | Notes |
|----------|------|-------|
| 2 groups, normal data | t-test | Most common |
| 2 groups, non-normal | Mann-Whitney U | Rank-based |
| >2 groups | ANOVA / Kruskal-Wallis | Then post-hoc |
| Paired samples | Paired t-test | Same seeds, different configs |

## Verdict Classification

### Supported

Hypothesis is **supported** when:
- Treatment significantly outperforms baseline (p < 0.05)
- Effect size is meaningful (Cohen's d > 0.5 medium, > 0.8 large)
- Results are consistent across seeds (low variance)
- Predicted success criteria from hypothesis are met

### Refuted

Hypothesis is **refuted** when:
- Treatment significantly underperforms baseline
- Or: no significant difference AND sufficient statistical power (n ≥ 10, power ≥ 0.8)
- Predicted mechanisms did not materialize in the data

### Inconclusive

Verdict is **inconclusive** when:
- High variance obscures true effect
- Insufficient runs for statistical power
- Mixed results (some seeds support, others refute)
- External factors may have confounded results

## Interpretation Template

Use this structure for documenting interpretations. **Replace metric names with 
those relevant to your domain.**

```markdown
## Interpretation: Hypothesis H[N]

### Summary Statistics

| Metric | Treatment | Baseline | Δ | p-value | Effect Size |
|--------|-----------|----------|---|---------|-------------|
| [Primary] | X.XX ± X.XX | X.XX ± X.XX | +X.X% | 0.XXX | d=X.XX |
| [Secondary] | X.XX ± X.XX | X.XX ± X.XX | +X.X% | 0.XXX | d=X.XX |

### Verdict: **[SUPPORTED / REFUTED / INCONCLUSIVE]**

**Rationale:** 
[1-2 sentences explaining why this verdict, citing p-values and effect sizes]

### Key Observations

1. **[Observation 1]:** [What the data shows]
2. **[Observation 2]:** [What the data shows]
3. **[Observation 3]:** [What the data shows]

### Unexpected Findings

- [Anything that differed from predictions in the hypothesis]
- [Surprises worth investigating]

### Limitations

- [Constraints on generalizability]
- [What wasn't tested]
- [Potential confounds]

### Suggested Next Directions

Based on these results:

1. **H[N+1]:** [Follow-up hypothesis]
2. **H[N+2]:** [Alternative direction]

### Raw Data

Treatment runs: [list of run_ids]
Baseline runs: [list of run_ids]
```

## Updating State Files

### experiment-state.md

Move hypothesis from "Active" to "Completed":

```markdown
## Completed Hypotheses

### H[N]: [Hypothesis Title] ([VERDICT])
- Tested: [date]
- Verdict: [Supported/Refuted/Inconclusive] (p = X.XXX)
- Key finding: [1-line summary]
- Next: See H[N+1]
```

### research-log.md

Append full iteration record:

```markdown
---

## [Date] [Time] — Iteration [N] Complete

**Hypothesis:** H[N] — [Full hypothesis statement]

**Literature consulted:**
- [Citation 1] — [Key insight from NotebookLM]
- [Citation 2] — [Key insight from NotebookLM]

**Experiments run:**
| Run ID | Seed | Treatment | [Primary Metric] | [Secondary Metric] |
|--------|------|-----------|------------------|---------------------|
| abc123 | 42 | treatment | X.XX | X.XX |
| def456 | 43 | treatment | X.XX | X.XX |
| ... | ... | ... | ... | ... |

**Interpretation:**
Hypothesis **[verdict]**. Treatment achieved:
- [Key result 1 with p-value]
- [Key result 2 with p-value]

**Next steps:** [What follows from this result]
```

## Common Interpretation Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| p-hacking | Cherry-picking significant results | Pre-register hypotheses, report all metrics |
| Underpowered | Claiming "no effect" with n=3 | Run more seeds, compute power |
| Ignoring variance | Reporting means without uncertainty | Always report std/CI |
| Overgeneralizing | "X always beats Y" | Note specific conditions tested |
| Confirmation bias | Seeing what you expect | Let the data speak, document surprises |

**Verdict:** SUPPORTED

**Next direction:** 
Validate on additional benchmark functions (H4) before recommending
as default configuration.

---
```

## Visualization (Optional)

### Convergence Curves

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for run_id in treatment_runs:
    history = mlflow.get_run(run_id).data.metrics
    ax.plot(history["best_fitness"], alpha=0.5, color="blue")
for run_id in baseline_runs:
    history = mlflow.get_run(run_id).data.metrics
    ax.plot(history["best_fitness"], alpha=0.5, color="red")
ax.set_xlabel("Generation")
ax.set_ylabel("Best Fitness")
ax.legend(["Treatment", "Baseline"])
plt.savefig("convergence_comparison.png")
```

### Box Plots

```python
import seaborn as sns

data = pd.concat([treatment.assign(group="Treatment"), 
                  baseline.assign(group="Baseline")])
sns.boxplot(data=data, x="group", y="metrics.final_fitness")
plt.savefig("fitness_boxplot.png")
```

## Anti-Patterns

Avoid:
- **Cherry-picking:** Report all seeds, not just successful ones
- **P-hacking:** Define success criteria before running experiments
- **Overfitting interpretation:** Don't over-explain noise
- **Ignoring variance:** High variance means low confidence
- **Confirmation bias:** Actively look for refuting evidence
