"""Signature helper shared by the registries."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """
    True if ``fn(**{name: value})`` binds ``name`` to a named parameter.

    Registries call factories and operator classes with keyword arguments
    only, so a positional-only parameter does not count, and neither does a
    bare ``**kwargs`` (it may forward to something that rejects the name).
    Callables without an inspectable signature report False.
    """
    try:
        parameter = inspect.signature(fn).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in _KEYWORD_KINDS
