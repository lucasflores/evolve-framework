# Status Checking Workflow

## Overview

When user asks for status (e.g., "what's the current state?", "status", "where are we?"),
provide a concise summary of the research session state.

## Status Levels

### Quick Status

Default response — fit in one screen:

```markdown
## Research Status

**Session:** evolve-framework research
**Current phase:** Awaiting interpretation

### Active Hypothesis
H3: [Brief statement] — 30 runs complete, ready for analysis

### Recent Activity
- Batch h3_20260320_1 completed (29/30 successful)
- Last interpretation: H2 — SUPPORTED (2 hours ago)

### Next Action
Interpret H3 results, or modify hypothesis direction
```

### Detailed Status

When user asks for more detail:

```markdown
## Detailed Research Status

### Framework Context
- **Framework:** evolve-framework
- **Last analyzed:** 2026-03-20 (current)
- **MLflow:** sqlite:///mlflow.db (6 experiments, 142 runs)

### Hypothesis History

| ID | Statement | Status | Verdict |
|----|-----------|--------|---------|
| H1 | Initial baseline | Complete | Baseline established |
| H2 | Tournament size effect | Complete | SUPPORTED |
| H3 | Adaptive mutation | Awaiting interpretation | - |

### Active Experiments

| Batch | Hypothesis | Progress | Status |
|-------|------------|----------|--------|
| h3_20260320_1 | H3 | 30/30 | Complete |

### Pending Proposals

| ID | Title | Status |
|----|-------|--------|
| P001 | Update tournament default | Proposed |

### Literature Consulted
- EA, GP, NE, and Causal Discovery Resources (3 queries this session)

### Session Timeline
- 14:00 — Session started, framework analyzed
- 14:15 — H1 baseline experiments complete
- 15:00 — H2 proposed and approved
- 15:30 — H2 batch complete, interpreted (SUPPORTED)
- 16:00 — H3 proposed, literature consulted
- 16:15 — H3 batch started
- 16:45 — H3 batch complete ← current
```

## What to Check

### Framework State

```python
# Check config.yaml
config = load_config(".research-assistant/config.yaml")

# Check if framework context is stale
if git_sha_changed(config.framework.git_sha):
    warn("Framework has changed since last analysis")
```

### MLflow State

```python
# Quick MLflow health check
tracking_uri = config.mlflow.tracking_uri
experiments = list_experiments()
recent_runs = search_runs(filter_string="start_time > 'today'")
```

### Hypothesis State

From `experiment-state.md`:
- Active hypotheses awaiting experiments
- Active batches in progress
- Completed hypotheses awaiting interpretation
- Fully resolved hypotheses

### Proposal State

From `proposals/` directory:
- Draft proposals being prepared
- Proposed changes awaiting approval
- Approved changes pending implementation

## Status Triggers

### Explicit Request

User says:
- "status"
- "where are we?"
- "what's the current state?"
- "summarize progress"

### Session Start

When resuming a session:
```markdown
## Resuming Research Session

Last active: 2 hours ago
**Picking up from:** H3 batch complete, awaiting interpretation

Ready to continue?
```

### After Long Pause

If significant time has passed (e.g., >1 hour):
```markdown
---
_Note: 2 hours since last activity. Run `status` for full context._
---
```

## State Consistency Checks

When generating status, verify:

### 1. MLflow Consistency

```python
# Runs referenced in experiment-state.md exist in MLflow
for run_id in state.referenced_runs:
    assert mlflow.get_run(run_id) exists
```

### 2. File Existence

```python
# All referenced files exist
for proposal_id in state.proposals:
    assert exists(f".research-assistant/proposals/{proposal_id}.md")
```

### 3. Git Consistency

```python
# Framework code hasn't changed unexpectedly
current_sha = git_rev_parse("HEAD")
if current_sha != config.framework.git_sha:
    warn("Framework code has changed since analysis")
```

## Warnings and Alerts

Include warnings in status when:

| Condition | Warning |
|-----------|---------|
| Stale framework context | "Framework changed since analysis — consider re-analyzing" |
| High failure rate | "N runs failed in current batch" |
| Incomplete interpretation | "H2 has results but no recorded interpretation" |
| Long-running batch | "Batch running for >2 hours" |
| MLflow unreachable | "Cannot connect to MLflow tracking server" |

## Status Response Format

### For Chat Context

Keep responses concise. User can ask follow-ups.

```markdown
**Status:** Awaiting H3 interpretation (30 runs complete)

Recent: H2 SUPPORTED, P001 proposed
Next: Analyze H3 results

Need details on a specific area?
```

### For Session Handoff

When user will return later, provide full context:

```markdown
## Session Checkpoint: 2026-03-20 17:00

### Where We Left Off
H3 experiments complete. 30 runs ready for interpretation.

### Key Context
- H2 showed tournament_size=5 improves convergence
- H3 tests if adaptive mutation further helps
- P001 proposes updating tournament default

### To Resume
1. Interpret H3 results
2. If supported, consider combined approach (H4)
3. Review P001 with full evidence

### Quick Resume Command
"Interpret H3 results"
```

## Generating Status Report

When status is requested:

1. **Load state files**
   - `.research-assistant/config.yaml`
   - `.research-assistant/experiment-state.md`
   - `.research-assistant/research-log.md`
   - `.research-assistant/proposals/*.md`

2. **Query MLflow**
   - Count experiments and runs
   - Check for recent activity
   - Verify run IDs from state

3. **Determine phase**
   - What's the next logical action?
   - What's blocking progress (if anything)?

4. **Compose response**
   - Lead with current phase and next action
   - Summarize recent activity
   - Note any warnings
   - Offer to expand on specific areas
