---
name: idea-to-speckit
description: "Transform fuzzy ideas into well-crafted prompts for spec-driven development using Spec Kit. Use when users have a vague concept, rough feature idea, or half-formed project notion that needs to become actionable /speckit.* commands. Triggers: 'I have an idea for...', 'I want to build something that...', 'help me spec out...', 'turn this idea into a spec', or any request to convert concepts into structured development plans."
---

# Idea to Spec Kit Prompts

Transform vague ideas into well-crafted prompts ready for spec-driven development.

## Overview

This skill bridges the gap between "I have a rough idea" and "here's the prompt for /speckit.specify". It uses the brainstorming methodology to explore, refine, and structure ideas into prompts optimized for Spec Kit's workflow.

## The Process

### Phase 0: Project Context Gathering

Before brainstorming, gather context about the existing project (if any):

**Quick Reconnaissance (always do first):**
```
1. list_dir on project root → understand structure
2. Read key files if they exist:
   - package.json / pyproject.toml / Cargo.toml / go.mod (dependencies, scripts)
   - README.md (project purpose, setup)
   - .env.example or config files (integrations, services)
3. file_search for patterns like **/*.md, **/schema.*, **/models/**
4. git log --oneline -10 (recent activity context)
```

**What to extract:**
- **Tech stack**: Languages, frameworks, databases, key libraries
- **Architecture patterns**: Monolith, microservices, plugin system, etc.
- **Existing entities/models**: What data structures exist
- **Integration points**: APIs, external services, auth systems
- **Conventions**: Naming patterns, file organization, test structure

**For deeper analysis** (optional, if project is complex):
- Use the `analyze` skill for comprehensive tech stack detection and structure analysis
- Review output from `analyze` before proceeding to brainstorming

**Context Summary Template:**
```markdown
## Project Context
- **Stack**: [e.g., Python/Flask, PostgreSQL, Redis]
- **Structure**: [e.g., plugin architecture, monorepo]
- **Key Entities**: [e.g., User, Event, Booking]
- **Integrations**: [e.g., Stripe, Auth0, S3]
- **Conventions**: [e.g., snake_case, /api/v1/ routes]
- **Relevant Prior Work**: [e.g., existing auth system to leverage]
```

This context directly informs the brainstorming questions and resulting prompts.

---

### Phase 1: Idea Exploration (Brainstorming)

Start by understanding the fuzzy idea, **informed by project context**:

#### 1a. Idea Research (Optional)

If the idea references external concepts, technologies, or prior art, gather information:

**Ask the user:**
- "Do you have any reference URLs (docs, articles, competitor products)?"
- "Have you done research in NotebookLM I should query?"
- "Is there a GitHub repo that does something similar you'd like me to look at?"

**Available research tools:**

| Trigger | Tool | Use |
|---------|------|-----|
| User provides URLs | `fetch_webpage` | Extract key concepts from articles/docs |
| Similar project exists | `github_repo` | Search for patterns, approaches in public repos |
| Tech/library questions | `mcp_context7_*` | Get current docs for frameworks/libraries |
| User has research notes | `notebooklm` skill | Query their NotebookLM for source-grounded answers |

**Research Summary Template:**
```markdown
## Idea Research
- **Reference Material**: [URLs, repos reviewed]
- **Key Patterns Found**: [common approaches from research]
- **Library Capabilities**: [relevant features from docs]
- **User's Prior Research**: [insights from NotebookLM if used]
- **Differentiators**: [how this idea differs from existing solutions]
```

#### 1b. Clarifying Questions

1. **Gather context** - What sparked this idea? What problem does it solve?
2. **Ask one question at a time** - Prefer multiple choice when possible
3. **Leverage project knowledge** - Reference existing patterns, entities, and conventions
4. **Leverage research findings** - Reference discovered patterns and capabilities
5. **Focus on**:
   - Core purpose (the "why")
   - Target users (the "who")
   - Key capabilities (the "what")
   - How it fits with existing system (the "where")
   - Constraints or must-haves

Sample questions to consider (ask only the most relevant):
- "What problem are you trying to solve?" (open)
- "Who will use this?" (A) Internal team (B) External customers (C) Both (D) Just me
- "What's the one thing this MUST do to be useful?"
- "I see you have [Entity X] - will this feature interact with it?" (context-informed)
- "Your project uses [Framework Y] - should this follow the same patterns?"
- "I found [Pattern Z] in similar projects - does that align with your vision?" (research-informed)
- "Are there existing tools you're replacing or complementing?"
- "What's the scope?" (A) Quick prototype (B) MVP for users (C) Production system

### Phase 2: Idea Synthesis

Once the idea is understood, synthesize into key components:

**Feature Statement**: One sentence describing what this is  
**User Stories**: 2-4 core user actions  
**Success Criteria**: How we know it works  
**Scope Boundaries**: What's explicitly OUT  
**Assumptions**: Reasonable defaults we're making  
**Integration Points**: How this connects to existing system (from Phase 0)  

### Phase 3: Generate Spec Kit Prompts

Produce prompts for the Spec Kit workflow:

#### 1. Constitution Prompt (if new project)

```text
/speckit.constitution [principles focused on X, Y, Z qualities]
```

Only needed for new projects. Focus on 3-5 governing principles.

#### 2. Specify Prompt (always needed)

```text
/speckit.specify [Feature description with WHAT and WHY. Include: user roles, core actions, data involved, key constraints. Be explicit about scope. 2-4 sentences.]
```

**Good specify prompts include**:
- Clear user roles/personas
- Core actions users can take
- Data entities involved
- Key constraints or requirements
- What's explicitly out of scope

**Avoid**:
- Tech stack details (save for /speckit.plan)
- Implementation specifics
- Vague adjectives without criteria

#### 3. Clarify Focus Areas (optional)

```text
/speckit.clarify Focus on [specific areas needing attention: security, edge cases, user flows, etc.]
```

Only if there are known ambiguities worth addressing upfront.

#### 4. Plan Prompt

```text
/speckit.plan [Tech stack choices: framework, database, key libraries. Architecture preferences. Integration requirements.]
```

**Good plan prompts include**:
- Specific technology choices
- Architecture style (monolith, microservices, etc.)
- Database preferences
- Key libraries or frameworks
- Deployment targets

## Output Format

After exploration, deliver:

```markdown
## Your Spec Kit Prompts

### 1. Specification
/speckit.specify [full prompt]

### 2. Technical Plan  
/speckit.plan [full prompt]

### 3. Optional Refinements
- /speckit.clarify [focus areas if needed]

## Summary
- **Feature**: [one-liner]
- **Core Stories**: [bullet list]
- **Key Decisions Made**: [assumptions we locked in]
- **Open Questions**: [things to revisit in /speckit.clarify]
```

## Key Principles

- **One question at a time** - Don't overwhelm
- **Multiple choice preferred** - Easier to answer
- **YAGNI ruthlessly** - Cut scope aggressively
- **Separate WHAT from HOW** - Spec is WHAT, plan is HOW
- **Make reasonable defaults** - Don't ask about everything
- **Document assumptions** - So they can be revisited

## Anti-Patterns to Avoid

❌ Asking about tech stack during specification phase  
❌ Including implementation details in /speckit.specify  
❌ Vague specs like "build a good user experience"  
❌ Over-scoped specs trying to do everything  
❌ Under-specified specs missing core user actions  

## Examples

### Input: "I want to build something to track my workouts"

**After exploration, output might be:**

```text
/speckit.specify Build a personal workout tracker for individual use. Users can log exercises with sets, reps, and weight. Users can view workout history by date and exercise type. Users can see progress over time for specific exercises. Data is stored locally. Single user, no authentication needed. Out of scope: social features, pre-built workout plans, video guides.

/speckit.plan Use a local-first approach with SQLite for persistence. Build as a web app using vanilla HTML/CSS/JS for simplicity. No backend server - runs entirely in browser with IndexedDB or localStorage fallback. Mobile-responsive design.
```

### Input: "Some kind of team dashboard thing"

**After exploration clarifying it's for engineering metrics:**

```text
/speckit.specify Build an engineering team dashboard showing deployment frequency, lead time, and incident counts. Team leads can view metrics for the past 30/60/90 days. Dashboard pulls from GitHub (deployments) and PagerDuty (incidents). Read-only display, no data entry. Refreshes hourly. Single team view initially, multi-team is out of scope.

/speckit.plan React frontend with Chart.js for visualizations. Node.js backend with scheduled jobs to fetch from GitHub and PagerDuty APIs. PostgreSQL to cache metrics data. Deploy as containerized app. OAuth for GitHub, API key for PagerDuty.
```
