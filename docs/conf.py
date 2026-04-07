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

# Deterministic build: suppress dynamic timestamps.
# For full CI reproducibility also set SOURCE_DATE_EPOCH=0 in the
# build environment.
html_last_updated_fmt = None
