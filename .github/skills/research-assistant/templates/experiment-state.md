# Experiment State

> Last updated: {{date}}
> Current iteration: {{iteration_number}}

## Current Hypothesis

**ID:** H{{hypothesis_id}}
**Status:** {{draft|approved|testing|completed}}

### Statement
{{hypothesis_statement}}

### Rationale
{{why_this_hypothesis}}

### Literature Support
{{citations_from_notebooklm}}

### Predictions
- If true: {{expected_outcome}}
- Metric thresholds: {{specific_thresholds}}

## Active Experiments

| Run ID | Config | Status | Started |
|--------|--------|--------|---------|
| {{run_id}} | {{config_summary}} | {{running|completed|failed}} | {{timestamp}} |

## Recent Results

### Latest Batch ({{date}})

**Hypothesis tested:** H{{id}}
**Verdict:** {{supported|refuted|inconclusive}}

| Run ID | Key Metrics | Notes |
|--------|-------------|-------|
| {{run_id}} | {{metrics}} | {{notes}} |

**Summary:** {{brief_interpretation}}

## Queued Experiments

Approved for execution:

1. **Config:** {{config_description}}
   **Rationale:** {{why_run_this}}

## Observations & Priors

### Updated Beliefs
- {{observation_1}} → {{updated_belief}}
- {{observation_2}} → {{updated_belief}}

### Unexplored Directions
- {{direction_1}}
- {{direction_2}}

## Failed Experiments

| Run ID | Error | Resolution |
|--------|-------|------------|
| {{run_id}} | {{error_summary}} | {{how_resolved_or_ignored}} |
