# Hypothesis Generation Workflow

## Overview

Generate testable hypotheses grounded in framework context and prior results.

## Hypothesis Quality Criteria

A good hypothesis must be:
1. **Testable** — Can be verified/falsified with available experiments
2. **Specific** — Predicts concrete, measurable outcomes
3. **Grounded** — Based on framework capabilities and prior observations
4. **Actionable** — Suggests clear experimental design

## Generation Prompts

### Initial Hypothesis (No Prior Results)

When starting fresh, analyze:
1. Framework capabilities from `framework-context.md`
2. User's stated research goals
3. Common optimization patterns in the domain

**Prompt template:**
```
Given the framework context:
[Insert framework-context.md summary]

The user wants to explore: [user_goal]

Generate 3 testable hypotheses that:
- Leverage specific framework features
- Have clear success metrics
- Can be tested with available config parameters

For each hypothesis, specify:
- H[N]: [Statement]
- Rationale: [Why this might be true]
- Test: [Config changes needed]
- Success metric: [Specific threshold]
- Risk: [What could invalidate this]
```

### Iteration Hypothesis (After Prior Results)

When building on previous experiments:
1. Load recent results from `experiment-state.md`
2. Identify surprising or incomplete findings
3. Generate hypotheses that explain or extend observations

**Prompt template:**
```
Previous hypothesis H[N]: [statement]
Result: [supported/refuted/inconclusive]
Key observations:
- [observation_1]
- [observation_2]

Generate 2-3 follow-up hypotheses that:
- Explain unexpected observations
- Extend successful approaches
- Address gaps in understanding

Consider:
- Parameter sensitivity (what if we vary X?)
- Interaction effects (does X affect Y differently?)
- Boundary conditions (when does this break?)
```

### User-Steered Hypothesis

When user provides direction:
1. Extract key concepts from user input
2. Map to framework capabilities
3. Formalize as testable hypothesis

**Prompt template:**
```
User suggestion: "[user_input]"
Framework capabilities: [relevant_features]
Prior context: [recent_results_if_any]

Formalize this into a testable hypothesis:
- Clarify any ambiguous terms
- Identify measurable outcomes
- Propose specific parameter values
- Note any constraints or assumptions
```

## Hypothesis Categories

These categories apply regardless of domain. **Examples will be domain-specific** — 
consult `framework-context.md` for your framework's terminology.

### Performance Hypotheses
Test whether a change improves key metrics.
- Pattern: "Changing [parameter] from [A] to [B] will improve [metric] by >[threshold]%"
- Pattern: "[Method X] will reduce [undesirable outcome] compared to baseline"

### Mechanism Hypotheses
Test how/why something works.
- Pattern: "[Component A] achieves [outcome] by [proposed mechanism]"
- Pattern: "[Component A] contributes more than [Component B] to [outcome] in [phase/condition]"

### Boundary Hypotheses
Test limits and edge cases.
- Pattern: "[Parameter] below [threshold] causes [failure mode]"
- Pattern: "Performance degrades gracefully/sharply as [variable] increases"

### Comparative Hypotheses
Test relative performance of alternatives.
- Pattern: "[Method A] outperforms [Method B] on [problem type]"
- Pattern: "[Config A] produces more consistent results than [Config B]"

## Presentation Format

Present hypotheses to user as:

```markdown
## Proposed Hypotheses

### H[N]: [Title]

**Statement:** [Precise testable claim]

**Rationale:** [Why this is worth testing]
Based on: [specific observations or framework features]

**Experimental Design:**
- Parameter changes: [specific config values]
- Seeds/replicates: [recommended count]
- Key metrics: [what to measure]

**Success Criteria:**
- Supported if: [specific threshold, e.g., "[primary_metric] > X for >80% of runs"]
- Refuted if: [opposite condition]
- Inconclusive if: [unclear outcome criteria]

**Literature:** [pending NotebookLM query]

---

Which hypothesis would you like to pursue? (or suggest modifications)
```

## Anti-Patterns

Avoid:
- **Vague claims:** "This might improve performance" → "Will improve fitness by >5%"
- **Untestable statements:** "This is theoretically optimal" → "Will reach optimum faster"
- **Compound hypotheses:** "A and B together will..." → Test A and B separately
- **Confirmation bias:** Only testing what you expect to work
