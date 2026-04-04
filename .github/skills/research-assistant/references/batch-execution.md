# Batch Execution Mode

## Overview

Run pre-approved experiment batches autonomously, with progress tracking 
and failure handling.

## When to Use Batch Mode

Batch mode is appropriate when:
- User has approved a set of experiments and wants them to run unattended
- Running multiple seeds or configurations for a single hypothesis
- Executing a parameter sweep with known bounds
- Overnight or long-running experiment sessions

**Interactive mode is preferred when:**
- Exploring a new hypothesis direction
- Results from early experiments should inform later ones
- User wants to pause and discuss after each run

## Batch Approval Flow

### Step 1: Generate Experiment Plan

Present the full batch to user before execution:

```markdown
## Proposed Experiment Batch

**Hypothesis:** H[N] — [statement]
**Batch ID:** h[n]_[date]_[sequence]

| # | Config | Seeds | Est. Time | Purpose |
|---|--------|-------|-----------|---------|
| 1 | baseline | 42-51 | ~10 min | Control group |
| 2 | treatment_A | 42-51 | ~10 min | Test hypothesis |
| 3 | treatment_B | 42-51 | ~10 min | Alternative |

**Total runs:** 30
**Estimated duration:** ~30 minutes

Approve this batch? (yes/no/modify)
```

### Step 2: Pre-flight Checks

Before starting batch:

- [ ] MLflow tracking URI accessible
- [ ] **Dependencies installed** (run `require_tracking()` or `uv sync`)
- [ ] Framework dependencies installed
- [ ] Configuration files validated
- [ ] Sufficient disk space for artifacts
- [ ] No conflicting processes

**Dependency Check Pattern**

Experiments should include at the top:
```python
from evolve.utils import require_tracking
require_tracking()  # Exits with clear error if mlflow not installed
```

This fails fast with a clear error instead of silently skipping tracking.

### Step 3: Store Batch State

Create batch record in `experiment-state.md`:

```markdown
## Active Batch: h3_20260320_1

**Status:** Running
**Started:** 2026-03-20 15:30
**Total runs:** 30
**Completed:** 0
**Failed:** 0

| Run # | Config | Seed | Status | Run ID | Started |
|-------|--------|------|--------|--------|---------|
| 1 | baseline | 42 | running | abc123 | 15:30 |
| 2 | baseline | 43 | queued | - | - |
...
```

## Execution Strategy

### Sequential Execution (Default)

Run one experiment at a time, wait for completion:

```
for each (config, seed) in batch:
    run_id = start_run(config, seed)
    wait_for_completion(run_id)
    update_batch_state(run_id, status)
    if failed and not recoverable:
        log_failure(run_id)
        continue  # Don't halt batch for single failures
```

**Advantages:**
- Simple resource management
- Easy to track progress
- Can detect issues early

### Parallel Execution (Optional)

Run multiple experiments concurrently:

```
with concurrent_limit(max_workers):
    for each (config, seed) in batch:
        submit(run_experiment, config, seed)
    wait_all()
```

**When to use:**
- Independent experiments (no shared state)
- Sufficient compute resources
- Framework supports concurrent execution

**Risks:**
- Resource contention (CPU, GPU, memory)
- Harder to debug failures
- May need framework-specific configuration

## Progress Tracking

### During Execution

Update `experiment-state.md` as runs complete:

```markdown
**Status:** Running (50% complete)
**Completed:** 15/30
**Failed:** 1
**ETA:** ~15 minutes remaining
```

### On Completion

```markdown
## Batch Complete: h3_20260320_1

**Status:** Complete
**Duration:** 28 minutes
**Results:** 29 successful, 1 failed

**Failed runs:**
- Run #7 (seed 48): [error summary] — see MLflow run xyz789

**Ready for interpretation.** Run IDs: [list]
```

## Failure Handling

### Recoverable Failures

Some failures can be automatically retried:

| Failure Type | Action |
|--------------|--------|
| Connection timeout | Retry with backoff |
| Transient resource error | Wait and retry |
| MLflow connection lost | Reconnect and retry |

### Non-Recoverable Failures

Log and continue with batch:

| Failure Type | Action |
|--------------|--------|
| Invalid configuration | Log error, skip run |
| Framework crash | Log with traceback, skip |
| Persistent resource issue | Pause batch, notify user |

### Batch Failure (Abort)

Stop entire batch for:

- >50% failure rate
- Critical resource exhaustion
- User interruption

```markdown
## Batch Aborted: h3_20260320_1

**Reason:** High failure rate (15/20 runs failed)
**Completed:** 5 successful runs
**Partial results saved:** [run IDs]

**Common failure:** [pattern if identifiable]

Recommend: Review configuration before retry.
```

## Resuming Interrupted Batches

If a batch is interrupted:

1. Load batch state from `experiment-state.md`
2. Identify completed vs remaining runs
3. Offer to resume:

```markdown
Found interrupted batch: h3_20260320_1

**Completed:** 15/30 runs
**Remaining:** 15 runs (no failures recorded)

Would you like to:
1. Resume remaining runs
2. Restart entire batch
3. Abandon batch
```

## Batch Configuration Options

In `config.yaml`:

```yaml
batch:
  max_concurrent: 1            # Sequential by default
  retry_attempts: 2            # For recoverable failures
  retry_backoff_seconds: 30
  failure_threshold: 0.5       # Abort if >50% fail
  checkpoint_interval: 5       # Update state every N runs
  
  # Resource limits (framework-specific)
  timeout_per_run_minutes: 60
```

## Post-Batch Workflow

After batch completes:

1. **Summary notification** (execution complete, NOT interpretation)
   ```markdown
   Batch h3_20260320_1 complete.
   - 29/30 runs successful
   - Run IDs logged to experiment-state.md
   - **Ready for interpretation via MLflow query**
   ```

2. **STOP — DO NOT interpret terminal output**
   - Terminal output is ephemeral and may be incomplete
   - The batch execution phase is DONE
   - Interpretation is a SEPARATE workflow

3. **Automatic next step prompt**
   ```markdown
   Experiments complete. Next steps:
   1. Query MLflow for metrics → Interpret results
   2. Run additional experiments
   3. Save state and continue later
   ```

4. **Handoff to interpretation**
   - Collect all run IDs from batch
   - Transition to `result-interpretation.md` workflow
   - **MUST query MLflow** before any statistical analysis
   - See result-interpretation.md for mandatory data fetch pattern

## IMPORTANT: Execution vs Interpretation Boundary

```
┌─────────────────────┐
│  BATCH EXECUTION    │  ← You are here
│  - Creates runs     │
│  - Logs to MLflow   │
│  - Terminal output  │  ← DO NOT USE FOR ANALYSIS
│  - Updates state    │
└─────────────────────┘
         │
         ▼
    [STOP & QUERY]  ← Mandatory boundary
         │
         ▼
┌─────────────────────┐
│  INTERPRETATION     │
│  - Query MLflow     │  ← REQUIRED FIRST STEP
│  - Load DataFrame   │
│  - Statistical test │
│  - Verdict          │
└─────────────────────┘
```

**Never skip the MLflow query step.**

## Monitoring During Batch

### Lightweight Status Check

User can ask for status at any time:

```markdown
## Batch Status: h3_20260320_1

Running: seed 47, treatment_A
Progress: 18/30 (60%)
ETA: ~12 minutes
Failures: 0

Last completed: abc123 (success)
```

### Detailed Progress

```markdown
## Detailed Batch Progress

| Config | Seeds Done | Avg Time | Status |
|--------|------------|----------|--------|
| baseline | 10/10 | 58s | ✓ Complete |
| treatment_A | 5/10 | 61s | Running |
| treatment_B | 0/10 | - | Queued |

Current run: treatment_A/seed_46
Elapsed: 45s
```
