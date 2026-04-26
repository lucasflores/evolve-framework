---
name: dev-stack-update
description: "Update a repo that was previously set up with dev-stack when new artifacts, modules, or pipeline changes land in a newer CLI version. Use when the user says 'update dev-stack', 'sync dev-stack to the latest version', 'upgrade the pipeline', 'add a new dev-stack module', 'a new module was added to dev-stack', 'dev-stack has new artifacts I need', or 'my dev-stack is out of date'. Handles module version diffs, deprecated module migration (e.g. speckit → apm), new-default-module opt-in, conflict resolution, and safe rollback."
argument-hint: "Optional: '--modules <name1,name2>' to target specific modules, or 'rollback' to undo the last update."
---

# Dev-Stack Update

Brings an already-initialized repository up to date with the current dev-stack CLI version. Covers module version bumps, newly added default modules, deprecated module removal, pipeline artifact regeneration, and safe rollback when things go wrong.

## When to Use

- `dev-stack.toml` already exists and `dev-stack --json status` reports stale module versions
- A new module was added to dev-stack and you want to install it into an existing repo
- A module was deprecated (e.g., `speckit` → `apm`) and the manifest needs cleaning
- The pipeline script (`scripts/hooks/pre-commit`), hook templates, or CI workflows changed
- Hook checksums in `dev-stack --json hooks status` show `modified: true` (drift detected)
- `infra-sync` stage warns about template drift on commit

## Expected Output

- `dev-stack.toml` manifest with all module versions bumped to current
- Regenerated managed artifacts for updated modules
- `dev-stack --json status` reporting `healthy: true` for every module
- A new rollback tag (`dev-stack/rollback/<timestamp>`) in case you need to revert
- First post-update commit passes all hard-gate pipeline stages without `--no-verify`

---

## Procedure

### Step 0 — Assess Current State

Before touching anything, capture a baseline:

```bash
# Confirm dev-stack is initialized
cat dev-stack.toml   # must exist; if absent, run dev-stack init instead

# Verify the active binary is the version you expect
dev-stack --version
which -a dev-stack          # list all executables in PATH order
pyenv which dev-stack       # shows which pyenv-managed binary resolves

# See what the CLI considers stale
dev-stack --json status

# Check hook health — look for modified: true entries
dev-stack --json hooks status

# Preview the full update diff without writing anything
dev-stack --dry-run update
```

> **Pyenv shim masking**: If `which -a dev-stack` shows both `~/.pyenv/shims/dev-stack` and `~/.local/bin/dev-stack`, pyenv's shim wins because `~/.pyenv/shims` is higher in `$PATH`. This means `dev-stack --version` may report the old version even after a `uv tool install --force` upgrade. Fix: install the updated wheel into the active pyenv Python (`pyenv which python` → note the Python, then install via that Python's `pip`) **or** install the wheel via `uv tool install --force` and then also copy/link it inside the pyenv environment. See Step 1 for full upgrade instructions.

> **Stale `pipeline` block in `--json status`**: The `pipeline.stages` section in `dev-stack --json status` reflects the *most recent pipeline run* stored on disk — it is not a live health check. Stale warnings (e.g., `infra-sync: warn`) may linger from a previous run and no longer apply. For current stage state, always run `dev-stack --json pipeline run --stage <stage>` directly rather than relying on the cached result.

The dry-run output includes:
- `modules_added` — new modules in `DEFAULT_GREENFIELD_MODULES` not yet in the manifest
- `modules_updated` — modules whose version in the manifest is lower than the CLI's current version
- `modules_removed` — modules deprecated since last init (e.g., `speckit`)
- `conflicts` — existing files that will need resolution

> If `modules_updated` and `modules_added` are both empty and no deprecated modules are flagged, do **not** automatically stop. First cross-check with `dev-stack --json status`. If any module is `healthy: false` despite the dry-run no-op, you have template-content drift without a version bump — proceed to the ["Status unhealthy but update is a no-op"](#status-unhealthy-but-update-is-a-no-op-template-drift) speedbump rather than stopping.

---

### Step 1 — Upgrade the dev-stack CLI itself (if needed)

The `dev-stack update` command updates the **repo's artifacts** to match the **installed CLI version**. Before upgrading, determine how dev-stack is installed — the correct action depends on install type:

```bash
# Identify install type via direct_url.json
find ~/.local/share/uv/tools/dev-stack ~/.pyenv -path '*/dev_stack*.dist-info/direct_url.json' \
  -exec cat {} + 2>/dev/null
```

**Editable install** — output contains `"editable": true`. The CLI binary already reflects the live source tree at all times. There is nothing to rebuild; it is always current. Skip directly to Step 2.

**Wheel install** — output contains a `file:///.../dev_stack-*.whl` URL. A specific wheel was installed. Check if newer source commits exist and rebuild:

```bash
# The URL shows the source path — navigate to its parent directory
cd /path/shown/in/direct_url  # e.g. /Users/you/dev-stack
git log --oneline -5           # confirm newer commits exist

# Rebuild and reinstall
uv build
uv tool install --force ./dist/dev_stack-<version>-py3-none-any.whl

# Verify
dev-stack --version
```

**PyPI install** — no `direct_url.json` found, or URL starts with `https://`. Upgrade normally:

```bash
uv tool upgrade dev-stack
dev-stack --version
```

---

### Step 2 — Run the Update

```bash
cd /path/to/your-repo

# Standard update — updates all modules listed in dev-stack.toml to current versions
dev-stack update
```

**What happens automatically:**
1. Reads `dev-stack.toml` to determine installed module versions
2. Computes a `ModuleDelta` (added / updated / removed / unchanged) by diffing manifest versions against CLI's current versions
3. **New default modules** — if any `DEFAULT_GREENFIELD_MODULES` are absent from the manifest, the CLI prompts interactively: `Install 'docker'? [y/N]`. New modules are NEVER auto-installed without confirmation.
4. **Deprecated modules** (e.g., `speckit`) — emits a migration notice, marks the entry as `deprecated: true` in `dev-stack.toml`, and excludes it from installation. No files are deleted.
5. Creates a new rollback tag before touching anything
6. Writes `.dev-stack/update-in-progress` marker for crash safety (removed on success)
7. Calls `module.update()` for each updated module (re-renders managed templates)
8. Calls `module.install(force=True)` for each newly added module
9. Bumps `last_updated` and module versions in `dev-stack.toml`

---

### Step 3 — Selective Module Update (Optional)

To update or add only specific modules without touching the rest:

```bash
# Update only the hooks and vcs_hooks modules
dev-stack update --modules hooks,vcs_hooks

# Add a net-new module to an existing repo
dev-stack update --modules docker

# Add apm after migrating away from deprecated speckit
dev-stack update --modules apm
```

Module dependencies are resolved automatically — adding `sphinx_docs` will also ensure `uv_project` is present.

---

### Step 4 — Handle Conflicts

When updated artifacts collide with locally modified files, the CLI prompts interactively:

```
.pre-commit-config.yaml already exists.
  [a] Accept proposed (overwrite with new template)
  [s] Skip (keep current, skip this file)
  [m] Merge (open diff in $EDITOR)
Choice:
```

**Non-interactive / JSON mode** — conflicts auto-resolve to overwrite:
```bash
dev-stack --json update          # auto-overwrites all pending conflicts
dev-stack --force update         # same but human-readable output
```

**Recommended for hooks files** — always accept proposed (`a`). Hook templates are checksum-tracked; a locally modified hook will keep firing `modified: true` in `hooks status` until overwritten.

**Recommended for `pyproject.toml`** — use merge (`m`) to review changes before accepting.

---

### Step 5 — Post-Update Validation

```bash
# Module health
dev-stack --json status   # all modules: healthy: true

# Hook integrity
dev-stack --json hooks status   # no modified: true entries

# Pipeline smoke test
dev-stack --json pipeline run --force   # stages 1-5 must pass

# Infra-sync drift check (stage 7)
dev-stack --json pipeline run --stage infra-sync   # should report no drift
```

> **Side effect**: Running `pipeline run --force` executes the `visualize` stage, which writes or updates `.understand-anything/` (4 files: `knowledge-graph.json`, `fingerprints.json`, `meta.json`, `summary.json`). These will appear as untracked or modified in `git status`. They should be committed — do not discard them. Stage them as part of Step 6.

---

### Step 6 — Commit the Updated Artifacts

```bash
git add -A
git commit -m "chore: update dev-stack artifacts to vX.Y.Z"
```

The pre-commit pipeline will run. All hard-gate stages (1–5) must pass. If they don't, see the speedbumps section below.

---

## Deprecated Module Migration

### `speckit` → `apm`

The `speckit` module was removed. Agent/MCP dependencies are now managed by `apm`.

```bash
# 1. Update to mark speckit deprecated and install apm
dev-stack update --modules apm

# 2. Set up the .specify/ directory (one-time, replaces speckit scaffolding)
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
specify init --here --ai copilot
```

The `apm` module manages MCP servers (`context7`, `github`, `huggingface`) via `apm.yml`. The `speckit` entry in `dev-stack.toml` is marked `deprecated = true` but not deleted.

---

## Known Speedbumps & Fixes

### "dev-stack.toml not found"
`dev-stack update` requires an initialized repo. Run `dev-stack init` first.

### "A previous dev-stack update did not complete"
The `.dev-stack/update-in-progress` marker exists from a crashed update.
```bash
# Option A: Roll back to the pre-update state, then retry
dev-stack rollback
dev-stack update

# Option B: Confirm and continue from where it left off
dev-stack update   # answers "Continue anyway? [y/N]" with y
```

### "No modules require updates" (unexpected)
The manifest versions already match the CLI. Either the CLI was not upgraded, or the repo is truly current — but also cross-check whether any modules are still unhealthy.
```bash
dev-stack --version        # confirm CLI version
cat dev-stack.toml         # compare [modules.*] versions by hand
dev-stack --dry-run update # authoritative diff
dev-stack --json status    # cross-check: are any modules healthy: false?
```
If `status` shows unhealthy modules despite the no-op, see "Status unhealthy but update is a no-op" below.

### Package version bumped but update is still a no-op — module VERSION constants not bumped

This happens when `pyproject.toml` was bumped (e.g., `0.1.0` → `1.0.0`) but the per-module `VERSION = "..."` constants inside `src/dev_stack/modules/*.py` were **not** updated. `dev-stack update` diffs those module-level constants against the versions stored in `dev-stack.toml`, not the package version — so the delta is empty and nothing is regenerated.

**Diagnosis:**
```bash
dev-stack --version           # shows the new version, e.g. 1.0.0
dev-stack --dry-run update    # reports "No modules require updates" — this is the tell

# Check module constants in the dev-stack source
find ~/.local/share/uv/tools/dev-stack ~/.pyenv -path '*/dev_stack*.dist-info/direct_url.json' \
  -exec cat {} + 2>/dev/null  # note the source path
grep -R 'VERSION\s*=' /path/to/dev-stack-source/src/dev_stack/modules/*.py
```
If those constants are still the old version (e.g., `0.1.x`) while `dev-stack --version` shows `1.0.0`, this is the cause.

**Fix (repo side):** Use the template-copy approach from ["Status unhealthy but update is a no-op"](#status-unhealthy-but-update-is-a-no-op-template-drift) to manually refresh files you know changed. This unblocks the repo without waiting for a dev-stack patch.

**Fix (dev-stack side — file a bug):** Module `VERSION` constants must be kept in sync with the package version on every release, or derived from the package version dynamically.

### Hook checksums still show `modified: true` after update
A locally edited hook was skipped during conflict resolution. Force-overwrite:
```bash
dev-stack update --modules hooks,vcs_hooks --force
dev-stack --json hooks status   # should now show modified: false
```

### `uv sync` fails after adding `uv_project` module
```bash
uv sync --extra dev --extra docs   # explicit extras
# — or if optional extras are broken —
uv sync
```

### Pipeline stage failures after update

| Failing stage | Likely cause | Fix |
|--------------|-------------|-----|
| `lint` | Ruff rules tightened in new template | `ruff check --fix . && ruff format .` |
| `typecheck` | mypy config updated | Review new `[tool.mypy]` section in `pyproject.toml` |
| `security` | New `detect-secrets` plugin or `pip-audit` findings | Review findings; update `.secrets.baseline` with `detect-secrets scan > .secrets.baseline` |
| `docs-api` | Sphinx conf updated | Check `docs/conf.py` diff; for brownfield `strict_docs = false` should already be set |
| `infra-sync` | Drift from skipped conflict | Re-run `dev-stack update --force` for affected module |

> **Misleading "missing tools" warning**: The message `"⚠ No substantive validation: lint, typecheck, test all skipped due to missing tools"` fires whenever those stages are absent from the current run — including when they were intentionally filtered via `--stage`. If you ran `--stage docs-api` (or any single-stage run) and see this warning, the tools are not actually missing; the stages were filtered out. Confirm tools are present with `uv run ruff --version && uv run mypy --version && uv run pytest --version`.

### New module install fails mid-update (partial state)
```bash
dev-stack rollback             # restore pre-update state
dev-stack update --dry-run     # confirm what will be applied
dev-stack update               # retry from clean state
```

### Status unhealthy but update is a no-op (template drift)
This happens when the dev-stack source repo has commits that changed managed template files (e.g., CI workflows, hook scripts) **without bumping the module's `VERSION` constant**. The CLI sees matching versions and reports nothing to do, but the on-disk artifacts are out of date with the template.

**Diagnosis:**
```bash
# Find the dev-stack source path from direct_url.json
find ~/.local/share/uv/tools/dev-stack ~/.pyenv -path '*/dev_stack*.dist-info/direct_url.json' \
  -exec cat {} + 2>/dev/null
# Note the source path from the URL field

# Identify which module is unhealthy
dev-stack --json status

# Diff the on-disk managed file against the current source template
# ci-workflows example:
diff /path/to/dev-stack-source/src/dev_stack/templates/ci/dev-stack-tests.yml \
     .github/workflows/dev-stack-tests.yml
# hooks example:
diff /path/to/dev-stack-source/src/dev_stack/templates/hooks/pre-commit \
     scripts/hooks/pre-commit
```

Template locations follow the pattern `src/dev_stack/templates/<module-name>/<file>`.

**Fix — copy the current template over the drifted file:**
```bash
# ci-workflows
cp /path/to/dev-stack-source/src/dev_stack/templates/ci/dev-stack-tests.yml \
   .github/workflows/dev-stack-tests.yml

# hooks
cp /path/to/dev-stack-source/src/dev_stack/templates/hooks/pre-commit \
   scripts/hooks/pre-commit
```

After copying, verify: `dev-stack --json status` — the module should now be `healthy: true`.

---

### `uv_project` unhealthy — `.python-version` missing
The `uv_project` health check requires `.python-version` to exist at the repo root. In brownfield mode, this file is gitignored but must be present on disk. `dev-stack update` does not create it automatically.

```bash
# Check the active Python version
pyenv version-name 2>/dev/null || python3 --version

# Create the file — major.minor only, not patch
echo "3.12" > .python-version

# Verify
dev-stack --json status
```

The file is gitignored by default; re-create it once per fresh checkout.

---

### Deprecated module warning keeps appearing
The deprecated entry remains in `dev-stack.toml` with `deprecated = true`. It is safe to leave as-is — it will not be re-installed. The warning only appears when explicitly named in `--modules`.

---

## Key Behavioral Differences: `update` vs `init --force`

| Behavior | `dev-stack update` | `dev-stack init --force` |
|----------|--------------------|--------------------------|
| Scope | Only modules with version delta | All modules |
| New default modules | Prompts user (opt-in) | Installs all defaults |
| Rollback tag | Always created | Created on first init |
| `brownfield-init` marker | Not written | Written for brownfield repos |
| Deprecated module handling | Marks deprecated, skips | Not applicable |
| Safe for CI/automation | Yes (auto-overwrites conflicts) | Risky (overwrites everything) |

Prefer `update` for routine maintenance. Use `init --force` only for a full re-scaffold.

---

## Rollback

Every `update` run creates a rollback tag before making changes:

```bash
# List available rollback points
git tag -l 'dev-stack/rollback/*'

# Restore to the most recent rollback (undoes last update)
dev-stack rollback

# Restore to a specific rollback point
dev-stack rollback --ref dev-stack/rollback/20260423T120000Z
```

Rollback restores all managed files to their pre-update state and removes intermediate rollback tags. The manifest's `rollback_ref` is cleared after a successful rollback.
