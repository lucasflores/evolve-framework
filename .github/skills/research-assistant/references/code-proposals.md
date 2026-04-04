# Code Change Proposal Workflow

## Overview

When experiments suggest framework improvements, create structured proposals 
that link research findings to code changes.

## When to Propose Changes

Generate a proposal when:
- Experiments consistently show a parameter setting outperforms defaults
- A hypothesis reveals a missing feature that would enable new research directions
- Results suggest a bug or unexpected behavior in the framework
- User explicitly requests a change based on research findings

**Do not propose changes for:**
- One-off experiment configurations (use config files instead)
- Changes unrelated to research findings
- Speculative improvements without supporting data

## Proposal Structure

Store proposals in `.research-assistant/proposals/`:

```
.research-assistant/proposals/
├── P001-adaptive-learning-rate.md
├── P002-config-validation.md
└── P003-checkpoint-frequency.md
```

### Proposal Template

```markdown
# P[NNN]: [Descriptive Title]

**Status:** Draft | Proposed | Approved | Implemented | Rejected
**Created:** [date]
**Hypothesis:** H[N] — [link to supporting hypothesis]
**Priority:** Low | Medium | High | Critical

## Summary

[1-2 sentence description of the proposed change]

## Motivation

### Research Findings

**Supporting experiments:**
| Run ID | Result | Key Observation |
|--------|--------|-----------------|
| [id] | [metric] | [what it shows] |

**Statistical evidence:**
- [Summary of statistical analysis]
- [Effect size and confidence]

### Expected Impact

- [Benefit 1]
- [Benefit 2]

### Risks

- [Risk 1 and mitigation]
- [Risk 2 and mitigation]

## Proposed Implementation

### Option A: [Brief name]

**Description:** [What changes]

**Files affected:**
- `path/to/file.py` — [what changes]
- `path/to/config.yaml` — [what changes]

**Effort estimate:** [Small/Medium/Large]

### Option B (Alternative): [Brief name]

[If applicable, describe alternative approach]

## Acceptance Criteria

- [ ] [Testable criterion 1]
- [ ] [Testable criterion 2]
- [ ] [Performance criterion based on research]

## Follow-up Experiments

After implementation, run:
- H[N+1]: [Hypothesis to validate change]
- [Any regression tests needed]

## Discussion

[Space for user feedback and iteration notes]
```

## Linking Proposals to Research

### From Research Log

When findings suggest a change, document in `research-log.md`:

```markdown
**Interpretation:**
Hypothesis **supported**. Results suggest framework improvement:
- Created proposal P[NNN] — [title]
```

### From Experiment State

Track proposal status in `experiment-state.md`:

```markdown
## Active Proposals

| ID | Title | Status | Related Hypothesis |
|----|-------|--------|-------------------|
| P001 | Adaptive learning rate default | Proposed | H3 |
| P002 | Config validation errors | Approved | H5 |
```

## Proposal Lifecycle

### 1. Draft

Agent creates proposal based on research findings.
- Gather supporting evidence from MLflow runs
- Identify affected files (use codebase exploration)
- Estimate effort and risks

### 2. Proposed

Present to user for review:

```markdown
## Proposal: P[NNN] — [Title]

Based on experiments from H[N], I recommend:

[Brief summary]

**Evidence:** [Key statistic]
**Risk:** [Primary concern]
**Effort:** [Estimate]

Would you like to:
1. Approve this proposal
2. Request modifications
3. Defer for more experiments
4. Reject with reason
```

### 3. Approved

User approves the proposal.
- Mark status as "Approved" in proposal file
- Add to implementation queue
- Schedule follow-up experiments

### 4. Implemented

After code changes are merged:
- Mark status as "Implemented"
- Link to git commit(s)
- Queue follow-up hypothesis to validate

### 5. Rejected

If user rejects:
- Mark status as "Rejected"
- Document reason
- May revisit with more evidence

## Categories of Proposals

### Configuration Defaults

Changes to default parameter values based on empirical performance.

**Evidence needed:**
- Performance improvement across multiple seeds
- Tested on representative problem set
- No regression on edge cases

### New Features

Addition of functionality suggested by research needs.

**Evidence needed:**
- Clear use case from research workflow
- Workaround currently exists but is cumbersome
- Aligns with framework design principles

### Bug Fixes

Corrections to unexpected behavior discovered during research.

**Evidence needed:**
- Reproducible issue from experiment logs
- Root cause identified
- Clear fix available

### Performance Improvements

Optimizations discovered through profiling or benchmarking.

**Evidence needed:**
- Measurable improvement (time, memory, etc.)
- No correctness regression
- Tested at scale

## Proposal Review Principles

When reviewing proposals, consider:

| Factor | Questions |
|--------|-----------|
| Evidence strength | How many runs? Statistical significance? |
| Generalizability | Does this apply broadly or just to one problem? |
| Breaking changes | Will this affect existing users? |
| Reversibility | Can we roll back if issues emerge? |
| Maintenance burden | Does this add complexity? |

## Integration with Git

After implementation, link back:

```markdown
## Implementation

**Commits:**
- [sha1] — [commit message]
- [sha2] — [commit message]

**PR:** #[number]

**Validation experiments:** [Run IDs from follow-up]
```
