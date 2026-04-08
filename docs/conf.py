"""Sphinx configuration for evolve_framework."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "evolve_framework"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
html_theme = "alabaster"

# Continue building docs even when optional dependencies (JAX, torch, etc.)
# are not installed — just skip the modules that fail to import.
autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "torch",
    "transformers",
    "gymnasium",
    "wandb",
    "ray",
    "mlflow",
]

# Suppress autodoc warnings for modules that fail to import (JAX/Torch
# backends with optional native deps).  Without this, Sphinx's -W flag
# (used by dev-stack) treats them as errors.
suppress_warnings = ["autodoc.import_object"]

# Deterministic build: suppress dynamic timestamps.
# For full CI reproducibility also set SOURCE_DATE_EPOCH=0 in the
# build environment.
html_last_updated_fmt = None
