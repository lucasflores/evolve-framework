"""
Logging one candidate as its own MLflow run, nested under a tracked run.

For evaluators that want each candidate recorded: its tags, its metrics and
any documents describing it, as a child of the run's tracked parent.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any


def log_candidate(
    run_name: str,
    *,
    tags: Mapping[str, str] | None = None,
    metrics: Mapping[str, float] | None = None,
    dicts: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """
    Log one candidate as a nested MLflow run under the active run.

    The caller decides what to log and when: once per new candidate, say,
    so copies aren't logged again, and only the metrics that mean something
    (no placeholder objectives for a candidate that wasn't scored). Call it
    from the run's thread, since MLflow's active run is per thread. The
    nested run goes in the parent's experiment, whichever is current.

    Args:
        run_name: The nested run's name.
        tags: Tags to set on it.
        metrics: Metrics to log on it.
        dicts: Documents to log as JSON artifacts, by artifact file name.

    Returns:
        True if logged. False, doing nothing, when MLflow isn't installed or
        no run is active; False with a RuntimeWarning when logging fails,
        since tracking must never fail a candidate. What was written before
        the failure stays, in a nested run ended as FAILED.
    """
    try:
        import mlflow
    except ImportError:
        return False
    parent = mlflow.active_run()
    if parent is None:
        return False
    try:
        with mlflow.start_run(
            nested=True,
            run_name=run_name,
            experiment_id=parent.info.experiment_id,
        ):
            if tags:
                mlflow.set_tags(dict(tags))
            for name, document in (dicts or {}).items():
                mlflow.log_dict(dict(document), name)
            if metrics:
                mlflow.log_metrics(dict(metrics))
    except Exception as exc:
        warnings.warn(f"candidate {run_name!r} was not logged: {exc}", RuntimeWarning, stacklevel=2)
        return False
    return True
