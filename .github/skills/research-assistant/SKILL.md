---
name: research-assistant
description: Automates ML experiments and the scientific process. Use when the user asks to run experiments, generate hypotheses, interpret results, analyze MLflow metrics, or manage a research workflow. Triggers on "run experiment", "analyze results", "next hypothesis", "research session", "interpret these metrics", "what should I try next".
---

# Research Assistant Skill

This skill provides the scientific loop workflow for automated ML experimentation.

## State Directory

All state lives in `.research-assistant/` at project root:

```
.research-assistant/
├── config.yaml           # MLflow location, toolkits, NotebookLM notebooks
├── framework-context.md  # Project-level analysis
├── toolkit-context/      # Per-toolkit analysis (if toolkits configured)
│   └── {toolkit}.md
├── experiment-state.md   # Current hypotheses, results
├── research-log.md       # Append-only decisions
└── proposals/            # Code change proposals
```

## Workflow: Initial Setup

When invoked on a new project (no `.research-assistant/` exists):

1. **Discover framework**
   - Read pyproject.toml, README, config files
   - Identify experiment entry points, config patterns
   - Find existing MLflow usage

2. **Ask user for config**
   - NotebookLM notebooks for literature grounding
   - Confirm MLflow location (default: local mlflow.db)

3. **Initialize state**
   - Create `.research-assistant/` directory
   - Write `config.yaml` with user inputs
   - Write `framework-context.md` with discoveries
   - Initialize empty `experiment-state.md` and `research-log.md`

## Workflow: Scientific Iteration

Each iteration follows this structure:

### Phase 1: Hypothesis Generation

```
hypothesis_init = generate_hypothesis(context, user_steering?)
```

1. Load `framework-context.md` and `experiment-state.md`
2. Review recent results (if any)
3. Propose 1-3 testable hypotheses
4. Present to user for approval/refinement

### Phase 2: Literature Grounding (if NotebookLM configured)

```
literature = query_notebooklm(context, hypothesis_init)
hypothesis_final = refine_hypothesis(literature, hypothesis_init)
```

1. Load notebooklm skill
2. Query configured notebooks with hypothesis context
3. Extract relevant citations and insights
4. Refine hypothesis based on literature
5. Present refined hypothesis to user

### Phase 3: Experiment Execution

1. Generate experiment configurations
2. Present batch to user for approval
3. Execute approved experiments (track in MLflow)
4. Monitor progress, handle failures
5. Update `experiment-state.md` with run IDs

**STOP HERE** — Do not interpret results from terminal output.  
Execution outputs are for monitoring only, not analysis.

### Phase 4: Result Interpretation

See [references/result-interpretation.md](references/result-interpretation.md) for the complete workflow.

**MANDATORY FIRST STEP:** Query MLflow before any analysis.

```
# REQUIRED: Fetch data from MLflow
runs_df = mlflow.search_runs(experiment_ids=[...], filter_string="tags.hypothesis_id = 'H1'")

# Only THEN perform analysis
results = aggregate_metrics(runs_df)
interpretation = analyze(results, hypothesis, context)
```

1. **Query MLflow** using `mlflow.search_runs()` — see reference file for patterns
2. Load metrics into DataFrame — this is the authoritative data source
3. Compute statistics (mean, std, CI) from the DataFrame
4. Run statistical tests (t-test, effect size)
5. Generate interpretation with:
   - Support/refute/inconclusive verdict
   - Key observations from the *queried* data
   - Suggested next directions
6. Update `research-log.md` with full record

**DO NOT:**
- Parse terminal output, script return values, or print statements
- Load `querying-mlflow-metrics` skill — that is for GenAI trace metrics (tokens, latency), not ML experiment metrics like `best_fitness`

### Phase 5: Context Update

```
context_{i+1} = update_context(results, hypothesis, interpretation)
```

1. Check if git diff shows framework changes
2. If stale, prompt user to re-run framework discovery
3. Update `experiment-state.md` with:
   - Completed hypotheses
   - New observations
   - Updated priors

## Supporting Skills

Load these as needed:

| Skill | When | NOT for |
|-------|------|--------|
| `querying-mlflow-metrics` | GenAI/LLM trace metrics (tokens, latency) | ML experiment metrics (`best_fitness`, `mean_fitness`) |
| `retrieving-mlflow-traces` | Debugging execution traces | — |

**For ML experiment metrics**: Use `mlflow.search_runs()` directly — see [references/result-interpretation.md](references/result-interpretation.md)
| `notebooklm` | Literature grounding |

## Detailed Workflow References

For in-depth guidance on each phase, see:

### Core Scientific Loop

| Reference | Contents |
|-----------|----------|
| [references/hypothesis-generation.md](references/hypothesis-generation.md) | Prompts, quality criteria, hypothesis categories |
| [references/notebooklm-integration.md](references/notebooklm-integration.md) | Query formulation, citation extraction, follow-ups |
| [references/experiment-execution.md](references/experiment-execution.md) | MLflow tagging, error handling, principles |
| [references/result-interpretation.md](references/result-interpretation.md) | Statistical analysis, verdict criteria, templates |

### Operations & Integration

| Reference | Contents |
|-----------|----------|
| [references/project-bootstrap.md](references/project-bootstrap.md) | Toolkit linking, two-level analysis, proposal routing |
| [references/batch-execution.md](references/batch-execution.md) | Autonomous batch runs, progress tracking, failure handling |
| [references/code-proposals.md](references/code-proposals.md) | Proposing framework changes, linking to evidence |
| [references/status-reporting.md](references/status-reporting.md) | Status checks, session summaries, state inspection |
| [references/git-integration.md](references/git-integration.md) | Change detection, reproducibility, commit linking |

## State File Schemas

### config.yaml

```yaml
mlflow:
  tracking_uri: "sqlite:///mlflow.db"

toolkits:
  - name: "evolve-framework"
    path: "../evolve-framework"
    scope: [causal, optimization]
  
notebooklm:
  notebooks:
    - name: "Evolutionary Algorithms"
      id: "<notebook-id>"
    - name: "Optimization Theory"
      id: "<notebook-id>"
      
framework:
  last_analyzed: "2026-03-20T14:00:00Z"
  git_sha: "abc123"
```

### framework-context.md

See `templates/framework-context.md` for full schema.

Key sections:
- Framework overview
- Config system
- Experiment entry points
- Key abstractions
- MLflow integration patterns

### experiment-state.md

See `templates/experiment-state.md` for full schema.

Key sections:
- Current hypothesis
- Active experiments (run IDs)
- Recent results summary
- Queued experiments

### research-log.md

Append-only format:

```markdown
## YYYY-MM-DD HH:MM — Iteration N

**Hypothesis:** [statement]
**Literature:** [key citations]
**Results:** [summary]
**Interpretation:** [analysis]
**Next:** [direction]
```

## Code Change Proposals

When experiments suggest code changes, create a formal proposal.
See [references/code-proposals.md](references/code-proposals.md) for full workflow.

Quick reference:
1. Create `proposals/P[NNN]-<description>.md`
2. Link to supporting hypothesis and evidence
3. Never modify code directly without approval
4. User reviews and approves via conversation

## Batch Execution Mode

For running multiple experiments autonomously:
See [references/batch-execution.md](references/batch-execution.md) for details.

Triggers:
- User approves a set of experiments to run unattended
- Parameter sweep or multi-seed replication
- "Run these experiments and come back with results"

## Status Commands

When user asks "status", "where are we?", or similar:
See [references/status-reporting.md](references/status-reporting.md) for response format.

Quick response includes:
- Current phase (hypothesis/experiments/interpretation)
- Active hypothesis status
- Recent activity summary
- Suggested next action

## Git Integration

Track relationship between code and research:
See [references/git-integration.md](references/git-integration.md) for full workflow.

Key features:
- Detect framework changes since last analysis
- Link proposals to commits
- Tag runs with git SHA for reproducibility

## Staleness Detection

Before each session:

```bash
git diff --stat $(cat .research-assistant/framework-context.md | grep git_sha | cut -d: -f2)
```

If significant changes detected:
- Prompt user: "Framework has changed since last analysis. Re-discover? [y/n]"
- If yes, re-run framework discovery workflow

## Example Session

**User:** "Start a research session on the evolve-framework"

**Agent:**
1. Check for `.research-assistant/` — not found
2. Run initial setup workflow
3. Analyze evolve-framework structure
4. Ask: "Which NotebookLM notebooks should I use for literature?"
5. Initialize state files
6. "Ready. Based on the framework, I see opportunities in selection operators. Want me to propose hypotheses?"

**User:** "Yes, focus on tournament selection"

**Agent:**
1. Generate hypotheses about tournament selection
2. Query NotebookLM for relevant literature
3. Present refined hypotheses with citations
4. User approves hypothesis H1
5. Generate experiment configs
6. User approves batch
7. Execute experiments
8. Fetch results, interpret
9. Log to research-log.md
10. "Results support H1. Suggest exploring adaptive tournament size next. Continue?"
