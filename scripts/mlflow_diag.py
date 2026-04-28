#!/usr/bin/env python3
"""MLflow diagnostic and cleanup script."""

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

client = MlflowClient()

print("=== Experiments ===")
exps = client.search_experiments(view_type=ViewType.ALL)
for e in exps:
    print(f"  [{e.experiment_id}] {e.name!r}  lifecycle={e.lifecycle_stage}")

print("\n=== Active Run ===")
active = mlflow.active_run()
if active:
    print(f"  STALE active run: {active.info.run_id}  (exp={active.info.experiment_id})")
    mlflow.end_run()
    print("  Ended stale run.")
else:
    print("  No active run.")

print("\n=== Runs in each experiment ===")
for e in exps:
    runs = client.search_runs(experiment_ids=[e.experiment_id])
    print(f"  Exp [{e.experiment_id}] {e.name!r}: {len(runs)} run(s)")
    for r in runs[:3]:
        metric_keys = list(r.data.metrics.keys())
        ensemble_keys = [k for k in metric_keys if "ensemble" in k]
        print(
            f"    run={r.info.run_id[:8]}  status={r.info.status}  metrics={len(metric_keys)}  ensemble_keys={ensemble_keys[:5]}"
        )
