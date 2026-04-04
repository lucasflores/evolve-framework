# Project Bootstrap Reference

How to set up a research project that depends on external toolkits.

## The Problem

A fresh research project repo contains only:
- Experiment scripts that *import* a toolkit
- Minimal config/data
- No context for the agent to understand experiment patterns

The agent needs to analyze the toolkit(s) to understand:
- How experiments are structured
- What metrics are meaningful
- How to propose valid changes

## Toolkit Linking

Add toolkits to `.research-assistant/config.yaml`:

```yaml
toolkits:
  - name: my-toolkit
    path: ../my-toolkit              # Relative or absolute path
    # OR
    url: https://github.com/org/repo  # Clone if not present
    scope:                            # Which modules are relevant
      - causal
      - optimization
      - core

mlflow:
  tracking_uri: "sqlite:///mlflow.db"
```

### Path Resolution

1. **Relative paths** — Resolved from project root
2. **Absolute paths** — Used as-is
3. **Git URLs** — Cloned to `.research-assistant/.toolkits/` on first use

### Scope Filtering

The `scope` array limits which parts of the toolkit the agent analyzes:
- Reduces noise from irrelevant modules
- Speeds up framework discovery
- Focuses proposals on relevant code

If omitted, the agent scans the entire toolkit.

## Two-Level Framework Analysis

### Level 1: Toolkit Analysis

When the agent encounters a linked toolkit for the first time:

1. **Traverse into toolkit path**
   - Read README, pyproject.toml, docstrings
   - Identify experiment patterns, config structure
   - Map key abstractions (what can be evolved, optimized, etc.)

2. **Generate toolkit context**
   - Store in `.research-assistant/toolkit-context/`
   - One file per toolkit: `{toolkit-name}.md`
   - Contains:
     - Entry points for running experiments
     - Key classes/functions and their roles
     - Config patterns and parameter spaces
     - Metric definitions and interpretations

3. **Scope-aware filtering**
   - If `scope` specified, analyze only matching modules
   - Otherwise, infer scope from project imports

### Level 2: Project Scoping

After toolkit analysis:

1. **Analyze project code**
   - Which toolkit modules does the project actually use?
   - What research questions drive the experiments?
   - What's the relationship between project configs and toolkit params?

2. **Generate project context**
   - Update `framework-context.md` with:
     - Research goal/question
     - Relevant toolkit subset
     - Project-specific patterns
     - How results map to toolkit metrics

3. **Create linkage**
   - `framework-context.md` references `toolkit-context/*.md`
   - Agent understands both layers

## Bootstrap Workflow

When invoked on a project with `toolkits:` in config:

### Step 1: Verify Toolkit Access

```
for each toolkit in config.toolkits:
  if toolkit.path exists:
    proceed with local analysis
  elif toolkit.url provided:
    clone to .research-assistant/.toolkits/
  else:
    error: cannot access toolkit
```

### Step 2: Run Toolkit Analysis (if not cached)

```
for each toolkit:
  if toolkit-context/{name}.md is stale or missing:
    run Level 1 analysis
    write toolkit-context/{name}.md
```

### Step 3: Run Project Scoping

```
analyze project imports and configs
cross-reference with toolkit-context/
update framework-context.md with scoped view
```

### Step 4: Continue Normal Workflow

With both levels populated, the agent has full context for:
- Generating meaningful hypotheses
- Understanding experiment results
- Proposing valid code changes

## Staleness Detection

Re-run toolkit analysis when:

1. **Git changes in toolkit** — `git diff` shows modifications
2. **User requests** — "refresh toolkit context"
3. **Proposal requires it** — Agent realizes context is incomplete

Check staleness via:

```yaml
# In toolkit-context/{name}.md frontmatter
---
toolkit: my-toolkit
analyzed_at: 2026-03-20T14:30:00
git_sha: abc123
---
```

Compare `git_sha` to current HEAD.

## Proposal Routing

When the agent proposes code changes, it must decide:

| Change Type | Route To |
|-------------|----------|
| Research-question-specific experiment | Project repo |
| Project config/scripts | Project repo |
| Bug fix affecting experiments | Toolkit (if contributor) OR workaround in project |
| New feature needed for research | Toolkit proposal + project workaround |
| Optimization/enhancement | Depends on scope |

### Toolkit Proposals

If a change belongs in the toolkit:

1. Create proposal in `.research-assistant/proposals/` as usual
2. Mark with `target: toolkit/{name}`
3. Include:
   - Which toolkit module to modify
   - Why this enables the research
   - Whether a project workaround is possible

```markdown
<!-- .research-assistant/proposals/P003.md -->
# P003: Add causal metric to SCM evaluation

**Target:** toolkit/evolve-framework  
**Module:** `evolve/multiobjective/metrics.py`

## Evidence

- [H002.1] showed current metrics don't capture causal validity
- Result interpretation suggested causal fidelity measure
- NotebookLM citation: "Structural Hamming Distance is standard for DAG comparison"

## Proposed Change

Add `structural_hamming_distance()` metric to SCM evaluation...

## Project Workaround

Until merged, implement locally in `experiments/metrics_patch.py`
```

### Project-Only Proposals

Most proposals stay within the project:

```markdown
<!-- .research-assistant/proposals/P004.md -->
# P004: Sweep population sizes for convergence analysis

**Target:** project

## Evidence

- [H003] hypothesizes population size affects convergence rate
- Need parameter sweep to test

## Proposed Change

Add sweep config `configs/pop_size_sweep.yaml`...
```

## Multi-Toolkit Projects

For projects depending on multiple toolkits:

```yaml
toolkits:
  - name: evolve-framework
    path: ../evolve-framework
    scope: [causal, scm]
    
  - name: my-eval-lib
    path: ../eval-lib
    scope: [metrics]
```

The agent:
- Analyzes each toolkit independently
- Generates separate `toolkit-context/*.md` files
- Cross-references when needed
- Routes proposals to correct toolkit

## User Commands

| Command | Action |
|---------|--------|
| "bootstrap project" | Run full two-level analysis |
| "refresh toolkit X" | Re-run Level 1 for specific toolkit |
| "show toolkit context" | Display toolkit-context/ contents |
| "scope to module X" | Update scope filter in config |

## Directory Structure with Toolkits

```
my-research-project/
├── experiments/
├── configs/
├── .research-assistant/
│   ├── config.yaml              # With toolkits: section
│   ├── framework-context.md     # Project-level context
│   ├── toolkit-context/         # Per-toolkit analysis
│   │   └── evolve-framework.md
│   ├── experiment-state.md
│   ├── research-log.md
│   └── proposals/
```

## Example: Setting Up a New Project

```bash
# 1. Create project
mkdir my-causal-study && cd my-causal-study

# 2. Initialize with toolkit link
# (Agent creates this on first run, or user creates manually)
cat > .research-assistant/config.yaml << EOF
toolkits:
  - name: evolve-framework
    path: ../evolve-framework
    scope: [causal]

mlflow:
  tracking_uri: "sqlite:///mlflow.db"
EOF

# 3. Invoke research assistant
# Agent will:
#   - Traverse into ../evolve-framework
#   - Analyze causal module
#   - Generate toolkit-context/evolve-framework.md
#   - Ask about research goals
#   - Generate framework-context.md with project scope
```

## Integration with Scientific Loop

After bootstrap:

1. **Hypothesis generation** — Uses both toolkit and project context
2. **Literature grounding** — Queries with toolkit-aware terminology
3. **Experiment execution** — Knows how to call toolkit correctly
4. **Result interpretation** — Understands toolkit metrics
5. **Proposals** — Correctly routes to toolkit or project
