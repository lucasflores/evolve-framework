---
name: research-project-init
description: "Scaffold a new ML research project repository with standardized structure, MLflow tracking, and research-assistant integration. Use when the user wants to create a new research project, init a research repo, start a new experiment repo, or bootstrap a research workspace. Triggers: 'create a research project', 'new research project', 'init research repo', 'research skeleton', 'start a new experiment project'."
---

# Research Project Initializer

Create a standardized ML research project repository.

## Workflow

### Step 1: Gather inputs

Ask these questions **one at a time**:

1. **Project name** — kebab-case (e.g., `causal-discovery-experiments`). This becomes the directory name and Python package name (underscored).
2. **Target directory** — Where to create it. Default: `~/` (so the project lands at `~/<project-name>/`).
3. **Extra dependencies** — Any Python packages beyond the defaults? (Default deps: mlflow, pyyaml, numpy, scipy, matplotlib, pandas, seaborn). User can say "none" or list extras.
4. **Python version** — Default: `3.12`. User can override.
5. **Author name and email** — For pyproject.toml.

### Step 2: Create all files

Use the **exact templates** below. Substitute only these variables:

| Variable | Source | Example |
|----------|--------|---------|
| `{{PROJECT_NAME}}` | Question 1, as-is | `causal-discovery-experiments` |
| `{{PACKAGE_NAME}}` | Question 1, replace `-` with `_` | `causal_discovery_experiments` |
| `{{EXTRA_DEPS}}` | Question 3, as TOML list entries | `"torch>=2.0",\n    "scikit-learn",` |
| `{{PYTHON_VERSION}}` | Question 4 | `3.12` |
| `{{PYTHON_VERSION_NODOT}}` | Question 4, remove `.` | `312` |
| `{{AUTHOR_NAME}}` | Question 5 | `Lucas Flores` |
| `{{AUTHOR_EMAIL}}` | Question 5 | `user@example.com` |

**CRITICAL**: Write every file exactly as shown. Do NOT improvise, add features, or modify templates.

Create these files in this order:

#### 1. `pyproject.toml`

```toml
[project]
name = "{{PROJECT_NAME}}"
version = "0.1.0"
description = ""
readme = "README.md"
authors = [
    { name = "{{AUTHOR_NAME}}", email = "{{AUTHOR_EMAIL}}" },
]
requires-python = ">={{PYTHON_VERSION}}"
dependencies = [
    "mlflow>=2.0",
    "pyyaml>=6.0",
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "seaborn",
    {{EXTRA_DEPS}}
]

[project.optional-dependencies]
dev = [
    "mypy>=1.10",
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.3",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{{PACKAGE_NAME}}"]

[tool.ruff]
target-version = "py{{PYTHON_VERSION_NODOT}}"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers -v"

[tool.coverage.run]
source = ["src/{{PACKAGE_NAME}}"]
omit = ["tests/*"]

[tool.mypy]
python_version = "{{PYTHON_VERSION}}"
strict = false
warn_return_any = true
warn_unused_configs = true
check_untyped_defs = true
mypy_path = "src"
```

#### 2. `apm.yml`

```yaml
name: "{{PROJECT_NAME}}"
version: "0.1.0"
dependencies:
  apm:
    - lucasflores/agent-skills
```

#### 3. `configs/base.yaml`

```yaml
# Base experiment configuration
# Each config should be self-contained — all parameters needed to reproduce a run.
# Create per-hypothesis variants: configs/<hypothesis_name>/control_seed42.yaml

# --- Experiment identity (used by MLflow) ---
name: "baseline"
description: "Baseline experiment"
tags: [baseline]

# --- Reproducibility ---
seed: 42

# --- Add framework-specific parameters below ---
# Examples:
#   learning_rate: 0.001
#   batch_size: 32
#   max_epochs: 100
```

#### 4. `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
build/
dist/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/

# Type checking / linting / testing
.mypy_cache/
.ruff_cache/
.pytest_cache/
htmlcov/
.coverage
.coverage.*
coverage.xml

# MLflow (tracking data lives in MLflow, not git)
mlflow.db
mlruns/

# Data and models (too large for git)
data/
!data/.gitkeep
*.pt
*.pth
*.ckpt
*.h5
*.safetensors
checkpoints/

# Notebooks
.ipynb_checkpoints/

# Environment / secrets
.env

# Editors
.vscode/
!.vscode/settings.json
.idea/

# OS
.DS_Store
Thumbs.db

# Dev tooling (if used)
.dev-stack/pipeline/
.dev-stack/viz/
docs/_build/
```

#### 5. `.python-version`

```
{{PYTHON_VERSION}}
```

#### 6. `README.md`

````markdown
# {{PROJECT_NAME}}

> TODO: One-line description of the research question.

## Setup

```bash
uv sync
apm install
```

## Running Experiments

```bash
python experiments/<script>.py configs/base.yaml
```

## Project Structure

```
configs/          ← Experiment configurations (one file per run)
experiments/      ← Entry point scripts
analysis/         ← Result interpretation scripts/notebooks
benchmarks/       ← Standard benchmarks and scaling tests
src/              ← Reusable project-specific code
tests/            ← Tests
```

## Config Convention

Each config in `configs/` is self-contained. Organize variants by hypothesis:

```
configs/
├── base.yaml
├── h1_<name>/
│   ├── control_seed42.yaml
│   └── treatment_seed42.yaml
```

## Citation

<!-- Added when publishing -->
````

#### 7. `src/{{PACKAGE_NAME}}/__init__.py`

```python
"""{{PROJECT_NAME}}."""
```

#### 8. `tests/__init__.py`

```python
```

(Empty file.)

#### 9. `tests/test_placeholder.py`

```python
def test_import() -> None:
    import {{PACKAGE_NAME}}  # noqa: F401
```

#### 10. Empty directory markers

Create `.gitkeep` files (empty) in:
- `experiments/.gitkeep`
- `analysis/.gitkeep`
- `benchmarks/standard/.gitkeep`
- `benchmarks/scaling/.gitkeep`

### Step 3: Initialize the project

Run these commands in the terminal:

```bash
cd <target_directory>/{{PROJECT_NAME}}
git init
uv sync
```

If `apm` is on PATH, also run:
```bash
apm install
```

### Step 4: Confirm and guide next steps

After creation, tell the user:

> Project created at `<path>`. Next steps:
>
> 1. Open the project in VS Code
> 2. Add your ML framework dependencies to `pyproject.toml` and run `uv sync`
> 3. Create your first experiment script in `experiments/`
> 4. Invoke the **Research Assistant** to bootstrap `.research-assistant/` and start the scientific loop

## What This Skill Does NOT Do

- Does **not** create `.research-assistant/` — the Research Assistant agent does that on first invocation
- Does **not** install dev-stack — that's a separate concern
- Does **not** create notebooks/, docs/, or data/ — add those per-project as needed
- Does **not** guess at framework-specific parameters
