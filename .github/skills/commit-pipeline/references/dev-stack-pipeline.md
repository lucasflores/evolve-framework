# Dev-Stack Pipeline — Artifact Map & Silent Failure Guide

The dev-stack pipeline fires inside two git hooks for every commit. Understanding what each stage produces (and what can silently skip) is critical for catching incomplete commits.

---

## Hook Split

| Hook | Stages | Timing |
|---|---|---|
| `pre-commit` | 1 lint, 2 typecheck | Before commit object is created |
| `prepare-commit-msg` | 3 test, 4 security, 5 docs-api, 6 docs-narrative, 7 infra-sync, 8 visualize, 9 commit-message | After commit object is created, message editable |

Stages outside the current hook context are filtered and logged as skipped — this is normal.

---

## Stage-by-Stage Artifact Map

### Stage 1 — `lint` (HARD gate)

**Tool**: `ruff format` + `ruff check`  
**Artifacts generated**:
- Reformatted Python source files (in-place) — **any staged Python file may be reformatted**
- On first brownfield commit (`.dev-stack/brownfield-init` marker present): ALL Python files in the repo are formatted

**Action required after lint**:
```bash
git diff --name-only    # look for newly modified (formatted) files
git add <reformatted-files>
git commit --amend --no-edit
```

**Silent skip conditions**:
- `ruff` not installed in project venv → `SKIP` with message
- Fix: `uv sync --extra dev --extra docs`

---

### Stage 2 — `typecheck` (HARD gate)

**Tool**: `mypy`  
**Artifacts generated**: none  
**Silent skip conditions**:
- `mypy` not installed → `SKIP`
- No Python packages found by layout detection → `SKIP`
- Fix: `uv sync --extra dev --extra docs`

---

### Stage 3 — `test` (HARD gate)

**Tool**: `pytest`  
**Artifacts generated**: none (coverage reports if configured separately)  
**Silent skip conditions**:
- `pytest` not available → `SKIP`
- No test files found → `SKIP`

---

### Stage 4 — `security` (HARD gate)

**Tool**: `detect-secrets scan`  
**Artifacts generated**:
- `.secrets.baseline` — updated if new findings differ from current baseline (findings-only diff, timestamp changes are ignored to avoid noise)
- If findings unchanged: original file is restored (no dirty working tree)

**Action required**:
```bash
git add .secrets.baseline    # if modified — auto-staged by pipeline, but verify
```

**Failure modes**:
- Unaudited or confirmed-real secrets → HARD FAIL
- Fix: `detect-secrets audit .secrets.baseline` — mark each finding as `false_positive` or investigate

**Silent skip conditions**:
- `detect-secrets` not installed → `SKIP`

---

### Stage 5 — `docs-api` (HARD gate)

**Tool**: `sphinx-apidoc` + `sphinx-build`  
**Artifacts generated**:
- `docs/api/*.rst` — one file per Python module, auto-staged by pipeline runner
- `docs/_build/` — HTML output (**typically gitignored** — verify `.gitignore` has `docs/_build/`)

**Action required**:
```bash
# Verify docs/api/ .rst files are staged
git diff --cached --name-only | grep 'docs/api/'
# If not auto-staged:
git add docs/api/
git commit --amend --no-edit
```

**Failure modes**:
- Sphinx build errors (RST syntax, missing references, import errors) → HARD FAIL
- `strict_docs = true` in `pyproject.toml` → warnings are treated as errors
- Fix: correct docstrings / RST, re-run `sphinx-build docs docs/_build -W --keep-going` locally

**Silent skip conditions**:
- `sphinx-build` not installed → `SKIP`
- `docs/` directory missing → `SKIP`
- No Python packages found → `SKIP` (passes vacuously)
- Fix: `uv sync --extra dev --extra docs`

---

### Stage 6 — `docs-narrative` (SOFT gate)

**Tool**: Coding agent (Copilot / Claude)  
**Artifacts generated** (behavior depends on mode):

| Mode | Artifact | Notes |
|---|---|---|
| Direct (CLI invocation) | `docs/guides/index.md` + other `docs/guides/*.md` | Written in place |
| Sandbox (hook context) | `.dev-stack/pending-docs.md` | Advisory suggestions only — NOT written to `docs/guides/` |

> **Important**: When running inside `prepare-commit-msg` hook, the stage runs in **sandbox mode** and writes to `.dev-stack/pending-docs.md`, not `docs/guides/`. This is by design (the hook cannot modify the staging area). After the commit, review the advisory and apply manually if desired.

**Silent skip conditions**:
- Agent unavailable (`DEV_STACK_AGENT=none` or no agent detected) → `SKIP`
- No staged changes detected → `SKIP`
- `docs_update.txt` prompt template missing → `SKIP`

---

### Stage 7 — `infra-sync` (SOFT gate)

**Tool**: File hash comparison  
**Artifacts generated**: none (read-only check)  
**Output**: `WARN` if `scripts/hooks/pre-commit` has drifted from the dev-stack template

**Action if WARN**:
```bash
dev-stack update --modules hooks    # re-sync hook scripts from template
git add scripts/hooks/pre-commit
git commit --amend --no-edit
```

---

### Stage 8 — `visualize` (SOFT gate)

**Tool**: Understand-Anything graph freshness policy  
**Artifacts generated**: depends on policy scope

| Scope | Behavior |
|---|---|
| `pre_commit` | Checks if staged changes require a graph refresh; warns if stale |
| `ci_required_check` | Stricter enforcement |

**Silent skip conditions**:
- `visualization` module not in `dev-stack.toml` manifest → `SKIP`
- `visualize = false` in `[tool.dev-stack.pipeline]` in `pyproject.toml` → `SKIP`

---

### Stage 9 — `commit-message` (SOFT gate)

**Tool**: Coding agent  
**Artifacts generated**:
- Rewrites the commit message in-place via `prepare-commit-msg` hook
- Appends or updates trailers: `Spec-Ref`, `Task-Ref`, `Agent`, `Pipeline`, `Edited`

**Action required**: After commit completes, verify the message with `git log -1`. If the agent's message is incorrect, `git commit --amend` to fix.

**Silent skip conditions**:
- Agent unavailable → `SKIP`
- `commit_message.txt` prompt template missing → `SKIP`

---

## Post-Commit Artifact Checklist

After every `git commit` on a dev-stack repo, verify:

```bash
git status    # should be clean; if not, unstaged pipeline artifacts remain

# Check pipeline last-run state
cat .dev-stack/pipeline/last-run.json | python3 -m json.tool

# Per-stage checks:
git show --name-only HEAD    # which files are in the commit
```

Expect the commit to include (when applicable):
- [ ] `.secrets.baseline` (if security stage found changes)
- [ ] `docs/api/*.rst` files (if docs-api ran successfully)
- [ ] `docs/guides/index.md` (if docs-narrative ran in direct/CLI mode)
- [ ] Any reformatted `.py` files (if lint auto-formatted)

Do **not** expect:
- `docs/_build/` (gitignored)
- `__pycache__/` (gitignored)
- `.dev-stack/logs/` (typically gitignored)
- `.dev-stack/pending-docs.md` (advisory, not committed — stage manually if you want to track it)

---

## Pipeline Skip Flag

If `.dev-stack/pipeline-skipped` exists at the repo root, the **entire pipeline is bypassed**. This is an escape hatch for emergencies.

```bash
ls .dev-stack/pipeline-skipped    # check if skip flag is active
rm .dev-stack/pipeline-skipped    # remove to re-enable pipeline
```

Never leave the skip flag in place after resolving an emergency.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `DEV_STACK_AGENT=none` | Disables agent-dependent stages (docs-narrative, commit-message, visualize) |
| `DEV_STACK_AGENT=copilot` or `claude` | Pins a specific agent |
| `DEV_STACK_DEBUG=1` | Writes debug logs to `.dev-stack/logs/pipeline-<timestamp>.log` |
| `DEV_STACK_HOOK_CONTEXT=pre-commit` | Restricts to stages 1–2 (set automatically by the hook) |

---

## Pyproject.toml Pipeline Config

```toml
[tool.dev-stack.pipeline]
strict_docs = true     # Sphinx warnings → errors (default: true)
visualize = true       # Enable visualize stage (default: true)
```
