#!/usr/bin/env python3
"""Debug script to test MLflow tracking directly."""

import os

os.chdir("/Users/lucasflores/evolve-framework")

print("=" * 60)
print("Step 1: Test MLflow import")
print("=" * 60)

try:
    import mlflow

    print(f"MLflow version: {mlflow.__version__}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
except ImportError as e:
    print(f"MLflow not available: {e}")
    exit(1)

print("\n" + "=" * 60)
print("Step 2: Test direct MLflow logging")
print("=" * 60)

mlflow.set_experiment("evolve_debug_test")
with mlflow.start_run(run_name="debug_run"):
    mlflow.log_metric("test_metric", 42.0, step=0)
    mlflow.log_metric("best_fitness", 100.0, step=0)
    mlflow.log_metric("best_fitness", 95.0, step=1)
    mlflow.log_metric("best_fitness", 80.0, step=2)
    print("Logged test metrics directly to MLflow")

print("\n" + "=" * 60)
print("Step 3: Check mlruns directory")
print("=" * 60)

import subprocess  # noqa: E402

result = subprocess.run(["ls", "-la", "mlruns"], capture_output=True, text=True)
print(result.stdout or result.stderr)

print("\n" + "=" * 60)
print("Done! Now start MLflow UI:")
print("  cd /Users/lucasflores/evolve-framework")
print("  .venv/bin/mlflow ui --port 5000")
print("=" * 60)
