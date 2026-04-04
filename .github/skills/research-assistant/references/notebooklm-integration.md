# NotebookLM Integration Workflow

## Overview

Ground hypotheses in literature using NotebookLM's source-grounded answers.

## Prerequisites

1. NotebookLM skill must be loaded
2. User must have notebooks registered in library
3. Notebook IDs configured in `.research-assistant/config.yaml`

## Configuration

In `.research-assistant/config.yaml`:

```yaml
notebooklm:
  notebooks:
    - name: "EA, GP, NE, and Causal Discovery Resources"
      id: "ea,-gp,-ne,-and-causal-discovery-resources"
```

## Query Workflow

### Step 1: Formulate Research Question

Transform hypothesis into literature query:

**Hypothesis:** "[Your specific hypothesis statement]"

**Query formulation:**
```
What does the literature say about:
1. [Core concept from hypothesis]
2. [Expected behavior or outcome]
3. Evidence for or against [hypothesis claim]

Provide citations for each claim.
```

*Example: If hypothesis involves "adaptive learning rates improving convergence," 
query for learning rate schedules, convergence analysis, and empirical comparisons.*

### Step 2: Execute Query

```bash
cd /Users/lucasflores/.agents/skills/notebooklm
python scripts/run.py ask_question.py \
  --question "[formulated_question]" \
  --notebook-id "ea,-gp,-ne,-and-causal-discovery-resources"
```

### Step 3: Extract Relevant Citations

From the response, extract:
- **Supporting evidence:** Literature that supports the hypothesis
- **Contradicting evidence:** Literature that challenges the hypothesis
- **Related mechanisms:** Theoretical explanations for expected behavior
- **Boundary conditions:** Known limitations from prior work

### Step 4: Refine Hypothesis

Update hypothesis based on literature:

```markdown
**H[N] (Refined):** [Updated statement]

**Literature support:**
- [Citation 1]: [Key finding]
- [Citation 2]: [Key finding]

**Literature caveats:**
- [Citation 3]: [Important limitation or condition]

**Refined predictions:**
- Based on [citation], expect [specific outcome]
- Watch for [known issue from literature]
```

## Query Patterns

These patterns are domain-agnostic — substitute terms from your research area.

### Mechanism Queries
"What are the known mechanisms by which [X] affects [Y]?"

### Evidence Queries
"What experimental evidence exists for [hypothesis statement]? Include any negative results."

### Parameter Queries
"What are recommended values for [parameter] in [context]? What tradeoffs exist?"

### Comparison Queries
"How does [method A] compare to [method B] for [task]? Under what conditions does each excel?"

### Historical Queries
"What approaches have been tried for [problem]? What worked and what didn't?"

### Gap Queries
"What are open questions or underexplored areas in [research domain]?"

## Follow-Up Strategy

Per the notebooklm skill: **always check if more information is needed**.

After initial response, ask follow-ups for:
1. Missing citations on key claims
2. Clarification on contradictory findings
3. Specific parameter recommendations
4. Known failure modes

Example follow-up pattern:
```
You mentioned [claim from response]. 
What specific [mechanisms/approaches] have been studied?
What [parameter values/configurations] are recommended?
Are there cases where [proposed approach] performed worse than [baseline]?
```

## Integration with Hypothesis Generation

Use NotebookLM at different stages of the scientific loop:

### Before Generating Hypotheses
Query: "What are open research questions in [research domain]?"
Goal: Identify gaps and promising directions

### After Initial Hypothesis
Query: "What does literature say about [hypothesis topic]? What experimental designs have been used?"
Goal: Ground hypothesis in prior work, borrow experimental methodology

### After Surprising Results
Query: "Has [unexpected observation] been reported in literature? What explanations exist?"
Goal: Understand if observation is known phenomenon or novel finding

## Recording Literature in State

Update `experiment-state.md`:

```markdown
### Literature Support

**Sources consulted:** EA, GP, NE, and Causal Discovery Resources

**Key citations:**
1. [Author, Year] — [Main finding relevant to hypothesis]
2. [Author, Year] — [Supporting or contradicting evidence]

**Gaps identified:**
- No literature found on [specific aspect]
- Conflicting reports on [topic]
```

Update `research-log.md`:

```markdown
**Literature consulted:**
- [Citation 1]: [How it informed the hypothesis]
- [Citation 2]: [How it informed experimental design]
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No relevant results | Broaden query terms, try different phrasings |
| Too general results | Add specific constraints (algorithm type, problem domain) |
| Contradictory sources | Query for experimental conditions that explain differences |
| Missing notebook | Run `notebook_manager.py list` to verify registration |
