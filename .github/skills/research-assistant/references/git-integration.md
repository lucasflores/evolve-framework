# Git Integration

## Overview

Track the relationship between research activities and code changes. Detect when
framework changes may affect research validity, and link commits to proposals.

## Framework Change Detection

### On Session Start

Check if framework code has changed since last analysis:

```bash
# Get current HEAD
current_sha=$(git rev-parse HEAD)

# Compare to stored SHA
stored_sha=$(grep "git_sha:" .research-assistant/config.yaml | cut -d'"' -f2)

if [ "$current_sha" != "$stored_sha" ]; then
    echo "Framework changed since last analysis"
fi
```

### Detailed Change Analysis

When changes detected, identify what changed:

```bash
# Files changed since last analysis
git diff --name-only $stored_sha HEAD

# Summarize changes
git log --oneline $stored_sha..HEAD
```

### Change Impact Assessment

| Changed Area | Impact | Action |
|--------------|--------|--------|
| Config defaults | May affect baseline | Consider re-running baselines |
| Core algorithms | High | Re-analyze framework, may invalidate results |
| Utilities/helpers | Low | Usually safe to continue |
| Tests only | None | No action needed |
| Documentation | None | No action needed |

### Prompt User When Stale

```markdown
## Framework Changed

Since your last session, the framework has changed:

**Commits:**
- abc123 — Fix mutation rate bounds
- def456 — Add speciation callback

**Files affected:**
- evolve/core/mutation.py
- evolve/core/speciation.py

This may affect experiments related to:
- Mutation parameters (H2, H3)
- Speciation behavior (H5 in queue)

Would you like to:
1. Re-analyze framework context
2. Continue with current context (acknowledge risk)
3. Review changes in detail
```

## Linking Commits to Research

### In Research Log

When proposals are implemented, record git info:

```markdown
---

## 2026-03-20 18:00 — P001 Implemented

**Proposal:** P001 — Update tournament default to 5
**Supporting research:** H2 (SUPPORTED)

**Commits:**
- `a1b2c3d` — Update tournament_size default from 3 to 5
- `e4f5g6h` — Add benchmark results to docs

**Validation:** Queued H4 to verify no regression
```

### In Proposal Files

Update proposal status with commit links:

```markdown
## Implementation

**Status:** Implemented
**Date:** 2026-03-20

**Commits:**
- [`a1b2c3d`](../../commit/a1b2c3d) — Update tournament_size default

**Validation runs:** H4 batch (pending)
```

## Committing Research State

### What to Commit

The `.research-assistant/` directory should be version controlled:

```gitignore
# .gitignore additions for research-assistant

# Commit these
!.research-assistant/
!.research-assistant/**

# But not temporary state
.research-assistant/*.tmp
.research-assistant/cache/
```

### Commit Patterns

**After completing a hypothesis:**
```bash
git add .research-assistant/
git commit -m "research: Complete H3 — adaptive mutation (SUPPORTED)"
```

**After proposing a change:**
```bash
git add .research-assistant/proposals/P001-*.md
git commit -m "research: Propose P001 — tournament default update"
```

**Session checkpoint:**
```bash
git add .research-assistant/
git commit -m "research: Session checkpoint — H3 awaiting interpretation"
```

## Research History in Git

### Viewing Research Timeline

```bash
# All research commits
git log --oneline --grep="research:"

# Research commits with file changes
git log --oneline --stat .research-assistant/

# When was hypothesis H3 completed?
git log --oneline --grep="H3"
```

### Bisecting Research Issues

If a later hypothesis seems inconsistent with earlier results:

```bash
# Find when results diverged
git log --oneline .research-assistant/experiment-state.md

# Check framework changes between hypotheses
git diff <h2-commit>..<h5-commit> -- evolve/
```

## Tagging Research Milestones

For significant findings, create git tags:

```bash
# Tag a major discovery
git tag -a research-v1 -m "Baseline established, tournament optimization complete"

# Tag before major framework changes
git tag -a pre-refactor-research -m "Research state before algorithm refactor"
```

## Branch Strategy

### Research Branches

For exploratory research that may not merge:

```bash
# Create research branch
git checkout -b research/speciation-effects

# Do experiments...
# If results are valuable, merge
git checkout main
git merge research/speciation-effects

# Or if exploration didn't pan out
git branch -D research/speciation-effects
```

### Parallel Research Tracks

When testing independent hypotheses:

```
main
├── research/mutation-rates      (H3, H4)
└── research/selection-pressure  (H5, H6)
```

## Config.yaml Git Fields

Track git state in config:

```yaml
framework:
  # Last analyzed commit
  git_sha: "abc123def456"
  last_analyzed: "2026-03-20T14:00:00Z"
  
  # Branch at time of analysis
  git_branch: "main"
  
  # Dirty state warning
  had_uncommitted_changes: false
```

## Detecting Uncommitted Changes

Before running experiments, check for clean state:

```bash
if ! git diff-index --quiet HEAD --; then
    echo "Warning: Uncommitted changes in working directory"
    echo "Results may not be reproducible"
fi
```

Present warning to user:

```markdown
**Warning:** Uncommitted changes detected

Running experiments with uncommitted changes means results may not be 
fully reproducible. Consider committing or stashing changes first.

Files changed:
- evolve/core/mutation.py (modified)
- experiments/config.yaml (modified)

Continue anyway? (yes/no/stash)
```

## Reproducibility Links

### In MLflow Tags

Tag runs with git info:

```python
mlflow.set_tags({
    "git_sha": current_sha,
    "git_branch": current_branch,
    "git_dirty": "true" if has_uncommitted else "false",
})
```

### In Experiment State

Record git context for each batch:

```markdown
## Batch h3_20260320_1

**Git context:**
- SHA: abc123def4
- Branch: main
- Clean: Yes

**Runs:** [run IDs]
```

## Handling Framework Updates

When framework is updated after research started:

### Option 1: Continue (Document Risk)

```markdown
**Note:** Framework updated after H1-H3 baseline.
Commit `xyz789` changed mutation bounds.
Results after H4 use updated code.
```

### Option 2: Re-validate Baseline

```markdown
**Action:** Re-running H1 baseline with updated framework
to establish new reference point.
```

### Option 3: Branch for Comparison

```bash
# Keep old baseline on a branch
git checkout -b research-baseline-v1 <old-sha>

# New research on main with updated code
git checkout main
```

## Quick Reference Commands

```bash
# Current framework SHA
git rev-parse HEAD

# Changes since last analysis
git diff $(cat .research-assistant/config.yaml | grep git_sha | cut -d'"' -f2)..HEAD --stat

# Research commit history
git log --oneline --grep="research:"

# Framework changes affecting experiments
git log --oneline -- evolve/core/

# Tag current research state
git tag -a research-checkpoint -m "Before major refactor"
```
