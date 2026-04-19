#!/usr/bin/env python3
"""
Dry-Run Estimate vs. Real Run — Live E2E Comparison

Runs dry_run() to get cost estimates, then executes the same config
for real and compares predicted vs actual wall-clock time per phase.
"""

from __future__ import annotations

import time
from random import Random

import numpy as np

from evolve.config.erp import ERPSettings
from evolve.config.meta import MetaEvolutionConfig, ParameterSpec
from evolve.config.tracking import TrackingConfig
from evolve.config.unified import UnifiedConfig
from evolve.core.population import Population
from evolve.core.types import Individual
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.experiment.dry_run import dry_run
from evolve.factory import create_engine
from evolve.meta.evaluator import run_meta_evolution
from evolve.representation.vector import VectorGenome

# ── Fitness functions ───────────────────────────────────────────────────────


def sphere(genes: np.ndarray) -> float:
    return float(np.sum(genes**2))


def rastrigin(genes: np.ndarray) -> float:
    A = 10.0
    n = len(genes)
    return float(A * n + np.sum(genes**2 - A * np.cos(2 * np.pi * genes)))


# ── Helpers ─────────────────────────────────────────────────────────────────


def create_population(size: int, dims: int, bounds: tuple[float, float], seed: int) -> Population:
    rng = Random(seed)
    individuals = [
        Individual(genome=VectorGenome(genes=np.array([rng.uniform(*bounds) for _ in range(dims)])))
        for _ in range(size)
    ]
    return Population(individuals=individuals)


def run_scenario(
    label: str,
    config: UnifiedConfig,
    fitness_fn,
    seed: int = 42,
    n_trials: int = 3,
) -> None:
    """Run one estimate-vs-actual scenario with multiple trials for stability.

    Runs dry_run once to get the estimate, then runs the actual evolution
    n_trials times and uses the **median** actual time to reduce system
    load noise.
    """

    print(f"\n{'=' * 72}")
    print(f"  Scenario: {label}")
    print(
        f"  pop={config.population_size}  gens={config.max_generations}  "
        f"dims={config.genome_params.get('dimensions', '?')}  "
        f"sel={config.selection}  cx={config.crossover}  mut={config.mutation}"
    )
    flags = []
    if config.is_tracking_enabled:
        flags.append(f"tracking({config.tracking.backend})")
    if config.is_erp_enabled:
        flags.append("erp")
    if config.is_merge_enabled:
        flags.append("merge")
    if config.is_meta_evolution:
        flags.append("meta")
    if flags:
        print(f"  subsystems: {', '.join(flags)}")
    print(f"{'=' * 72}")

    # Wrap callable into a proper evaluator for dry_run
    evaluator = FunctionEvaluator(fitness_fn)

    # ── DRY RUN ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    report = dry_run(config, evaluator=evaluator, seed=seed)
    dry_run_time = (time.perf_counter() - t0) * 1000

    print(f"\n  Dry-run completed in {dry_run_time:.0f} ms")
    print(f"\n{report.summary()}")

    estimated_total_s = report.total_estimated_ms / 1000

    # ── REAL RUN (multiple trials, median) ──────────────────────────────
    actual_times = []
    best_fitness = None
    for _trial in range(n_trials):
        engine = create_engine(config, fitness_fn, seed=seed)
        initial_pop = create_population(
            config.population_size,
            config.genome_params["dimensions"],
            config.genome_params["bounds"],
            seed,
        )
        t0 = time.perf_counter()
        result = engine.run(initial_pop)
        actual_times.append(time.perf_counter() - t0)
        if best_fitness is None:
            best_fitness = result.best.fitness.values[0]

    actual_s = float(np.median(actual_times))

    print(
        f"\n  Running real evolution ({config.max_generations} gens, {n_trials} trials, median)..."
    )
    print(f"  Trial times: {', '.join(f'{t:.3f}s' for t in actual_times)}")

    # ── COMPARISON TABLE ────────────────────────────────────────────────
    ratio = estimated_total_s / actual_s if actual_s > 0 else float("inf")
    error_pct = abs(estimated_total_s - actual_s) / actual_s * 100 if actual_s > 0 else 0

    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │  COMPARISON                                 │")
    print("  ├─────────────────────────────────────────────┤")
    print(f"  │  Estimated total : {estimated_total_s:>10.3f} s              │")
    print(f"  │  Actual (median) : {actual_s:>10.3f} s              │")
    print(f"  │  Ratio (est/act) : {ratio:>10.2f} x              │")
    print(f"  │  Abs error       : {error_pct:>9.1f} %               │")
    print(f"  │  Best fitness    : {best_fitness:>10.6f}              │")
    print("  └─────────────────────────────────────────────┘")

    return ratio, error_pct, estimated_total_s, actual_s


# ── Scenarios ───────────────────────────────────────────────────────────────


def run_scenario_meta(
    label: str,
    config: UnifiedConfig,
    fitness_fn,
    seed: int = 42,
    n_trials: int = 3,
) -> tuple[float, float, float, float]:
    """Run meta-evolution: dry-run estimate vs real meta loop.

    Runs the actual meta loop n_trials times and uses the median for
    comparison, reducing system load noise.
    """

    meta = config.meta
    assert meta is not None

    print(f"\n{'=' * 72}")
    print(f"  Scenario: {label}")
    print(
        f"  inner: pop={config.population_size}  gens={config.max_generations}  "
        f"dims={config.genome_params.get('dimensions', '?')}"
    )
    print(
        f"  outer: pop={meta.outer_population_size}  gens={meta.outer_generations}  "
        f"trials={meta.trials_per_config}"
    )
    print(f"  evolvable params: {', '.join(p.path for p in meta.evolvable_params)}")
    print(f"{'=' * 72}")

    evaluator = FunctionEvaluator(fitness_fn)

    # ── DRY RUN ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    report = dry_run(config, evaluator=evaluator, seed=seed)
    dry_run_time = (time.perf_counter() - t0) * 1000

    print(f"\n  Dry-run completed in {dry_run_time:.0f} ms")
    print(f"\n{report.summary()}")

    # Use meta estimate as the total estimate
    if report.meta_estimate is not None:
        estimated_total_s = report.meta_estimate.total_estimated_ms / 1000
    else:
        estimated_total_s = report.total_estimated_ms / 1000

    # ── REAL META LOOP (multiple trials, median) ────────────────────────
    print(
        f"\n  Running real meta-evolution "
        f"({meta.outer_generations} outer gens × "
        f"{meta.outer_population_size} configs × "
        f"{meta.trials_per_config} trials, {n_trials} e2e trials)..."
    )

    actual_times = []
    last_result = None
    for _ in range(n_trials):
        t0 = time.perf_counter()
        last_result = run_meta_evolution(config, fitness_fn, seed=seed)
        actual_times.append(time.perf_counter() - t0)

    actual_s = float(np.median(actual_times))
    meta_result = last_result

    print(f"  Trial times: {', '.join(f'{t:.3f}s' for t in actual_times)}")

    # ── COMPARISON TABLE ────────────────────────────────────────────────
    ratio = estimated_total_s / actual_s if actual_s > 0 else float("inf")
    error_pct = abs(estimated_total_s - actual_s) / actual_s * 100 if actual_s > 0 else 0

    print("\n  ┌──────────────────────────────────────────────────┐")
    print("  │  META-EVOLUTION COMPARISON                       │")
    print("  ├──────────────────────────────────────────────────┤")
    print(f"  │  Estimated meta cost : {estimated_total_s:>10.3f} s              │")
    print(f"  │  Actual (median)     : {actual_s:>10.3f} s              │")
    print(f"  │  Ratio (est/act)     : {ratio:>10.2f} x              │")
    print(f"  │  Abs error           : {error_pct:>9.1f} %               │")
    print(f"  │  Trials run          : {meta_result.trials_run:>10d}                │")
    print(f"  │  Best meta fitness   : {meta_result.best_fitness:>10.6f}              │")
    print("  └──────────────────────────────────────────────────┘")

    return ratio, error_pct, estimated_total_s, actual_s


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  DRY-RUN  vs  REAL RUN  —  E2E Accuracy Comparison     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results = []

    # ── Section 1: Baseline scenarios (selection + crossover variations) ─────

    # Scenario 1 — Tournament + Uniform (baseline small)
    cfg1 = UnifiedConfig(
        name="small_sphere",
        population_size=50,
        max_generations=30,
        elitism=2,
        selection="tournament",
        selection_params={"tournament_size": 3},
        crossover="uniform",
        crossover_rate=0.9,
        mutation="gaussian",
        mutation_rate=0.1,
        mutation_params={"sigma": 0.5},
        genome_type="vector",
        genome_params={"dimensions": 10, "bounds": (-5.0, 5.0)},
        minimize=True,
        seed=42,
    )
    r = run_scenario("Tournament + Uniform (50×30×10D)", cfg1, sphere)
    results.append(("Tourn+Uniform", *r))

    # Scenario 2 — Roulette + Blend crossover
    cfg2 = UnifiedConfig(
        name="roulette_blend",
        population_size=100,
        max_generations=50,
        elitism=3,
        selection="roulette",
        crossover="blend",
        crossover_rate=0.9,
        crossover_params={"alpha": 0.5},
        mutation="gaussian",
        mutation_rate=0.1,
        mutation_params={"sigma": 0.3},
        genome_type="vector",
        genome_params={"dimensions": 20, "bounds": (-5.12, 5.12)},
        minimize=True,
        seed=42,
    )
    r = run_scenario("Roulette + Blend (100×50×20D)", cfg2, sphere)
    results.append(("Roulette+Blend", *r))

    # Scenario 3 — Rank + Single-point crossover
    cfg3 = UnifiedConfig(
        name="rank_singlepoint",
        population_size=100,
        max_generations=50,
        elitism=3,
        selection="rank",
        selection_params={"selection_pressure": 1.5},
        crossover="single_point",
        crossover_rate=0.9,
        mutation="gaussian",
        mutation_rate=0.15,
        mutation_params={"sigma": 0.4},
        genome_type="vector",
        genome_params={"dimensions": 20, "bounds": (-5.0, 5.0)},
        minimize=True,
        seed=42,
    )
    r = run_scenario("Rank + SinglePoint (100×50×20D)", cfg3, sphere)
    results.append(("Rank+SinglePt", *r))

    # Scenario 4 — Tournament + Two-point crossover
    cfg4 = UnifiedConfig(
        name="tourn_twopoint",
        population_size=100,
        max_generations=50,
        elitism=3,
        selection="tournament",
        selection_params={"tournament_size": 5},
        crossover="two_point",
        crossover_rate=0.9,
        mutation="gaussian",
        mutation_rate=0.1,
        mutation_params={"sigma": 0.3},
        genome_type="vector",
        genome_params={"dimensions": 20, "bounds": (-5.12, 5.12)},
        minimize=True,
        seed=42,
    )
    r = run_scenario("Tournament + TwoPoint (100×50×20D)", cfg4, rastrigin)
    results.append(("Tourn+TwoPt", *r))

    # Scenario 5 — Tournament + SBX (large, high accuracy expected)
    cfg5 = UnifiedConfig(
        name="large_sbx",
        population_size=200,
        max_generations=100,
        elitism=5,
        selection="tournament",
        selection_params={"tournament_size": 5},
        crossover="sbx",
        crossover_rate=0.9,
        crossover_params={"eta": 20.0},
        mutation="gaussian",
        mutation_rate=0.2,
        mutation_params={"sigma": 1.0},
        genome_type="vector",
        genome_params={"dimensions": 50, "bounds": (-10.0, 10.0)},
        minimize=True,
        seed=42,
    )
    r = run_scenario("Tournament + SBX (200×100×50D)", cfg5, sphere)
    results.append(("Tourn+SBX", *r))

    # ── Section 2: Subsystem scenarios ──────────────────────────────────────

    # Scenario 6 — Tracking enabled (null backend, no I/O overhead)
    cfg6 = UnifiedConfig(
        name="tracking_null",
        population_size=100,
        max_generations=50,
        elitism=3,
        selection="tournament",
        selection_params={"tournament_size": 3},
        crossover="uniform",
        crossover_rate=0.9,
        mutation="gaussian",
        mutation_rate=0.1,
        mutation_params={"sigma": 0.5},
        genome_type="vector",
        genome_params={"dimensions": 20, "bounds": (-5.0, 5.0)},
        minimize=True,
        seed=42,
        tracking=TrackingConfig(backend="null"),
    )
    r = run_scenario("Tracking (null) (100×50×20D)", cfg6, sphere)
    results.append(("Tracking(null)", *r))

    # Scenario 7 — Tracking with local backend
    cfg7 = UnifiedConfig(
        name="tracking_local",
        population_size=100,
        max_generations=50,
        elitism=3,
        selection="rank",
        selection_params={"selection_pressure": 1.8},
        crossover="blend",
        crossover_rate=0.9,
        crossover_params={"alpha": 0.3},
        mutation="gaussian",
        mutation_rate=0.1,
        mutation_params={"sigma": 0.3},
        genome_type="vector",
        genome_params={"dimensions": 30, "bounds": (-5.0, 5.0)},
        minimize=True,
        seed=42,
        tracking=TrackingConfig(backend="local"),
    )
    r = run_scenario("Rank+Blend+Tracking(local) (100×50×30D)", cfg7, sphere)
    results.append(("Track(local)", *r))

    # Scenario 8 — ERP enabled
    cfg8 = UnifiedConfig(
        name="erp_sphere",
        population_size=100,
        max_generations=50,
        elitism=3,
        selection="tournament",
        selection_params={"tournament_size": 3},
        crossover="sbx",
        crossover_rate=0.9,
        crossover_params={"eta": 15.0},
        mutation="gaussian",
        mutation_rate=0.1,
        mutation_params={"sigma": 0.5},
        genome_type="vector",
        genome_params={"dimensions": 20, "bounds": (-5.0, 5.0)},
        minimize=True,
        seed=42,
        erp=ERPSettings(),
    )
    r = run_scenario("ERP + Tournament + SBX (100×50×20D)", cfg8, sphere)
    results.append(("ERP+Tourn+SBX", *r))

    # Scenario 9 — Meta-evolution (full comparison)
    cfg9 = UnifiedConfig(
        name="meta_sphere",
        population_size=50,
        max_generations=10,
        elitism=2,
        selection="tournament",
        selection_params={"tournament_size": 3},
        crossover="uniform",
        crossover_rate=0.9,
        mutation="gaussian",
        mutation_rate=0.1,
        mutation_params={"sigma": 0.5},
        genome_type="vector",
        genome_params={"dimensions": 10, "bounds": (-5.0, 5.0)},
        minimize=True,
        seed=42,
        meta=MetaEvolutionConfig(
            evolvable_params=(
                ParameterSpec(path="mutation_rate", bounds=(0.01, 0.3)),
                ParameterSpec(path="mutation_params.sigma", bounds=(0.1, 2.0)),
            ),
            outer_population_size=10,
            outer_generations=5,
            trials_per_config=2,
        ),
    )
    r = run_scenario_meta("Meta-Evolution (50×10×10D, 10×5×2)", cfg9, sphere)
    results.append(("Meta-Evo", *r))

    # ── Section 3: Large-scale scenarios (minutes, compounding errors) ──────

    # Scenario 10 — Large population (1000×200×50D, ~60s)
    #   Tests whether per-gen overhead compounds with 5x population increase.
    cfg10 = UnifiedConfig(
        name="large_pop",
        population_size=1000,
        max_generations=200,
        elitism=10,
        selection="tournament",
        selection_params={"tournament_size": 5},
        crossover="sbx",
        crossover_rate=0.9,
        crossover_params={"eta": 20.0},
        mutation="gaussian",
        mutation_rate=0.2,
        mutation_params={"sigma": 1.0},
        genome_type="vector",
        genome_params={"dimensions": 50, "bounds": (-10.0, 10.0)},
        minimize=True,
        seed=42,
    )
    r = run_scenario("Large Pop (1000×200×50D)", cfg10, sphere, n_trials=2)
    results.append(("LargePop-1000", *r))

    # Scenario 11 — High dimensionality (500×200×200D, ~60s)
    #   Tests scaling in genome dimension — affects crossover, mutation,
    #   evaluation, and diversity metrics.
    cfg11 = UnifiedConfig(
        name="high_dim",
        population_size=500,
        max_generations=200,
        elitism=5,
        selection="tournament",
        selection_params={"tournament_size": 5},
        crossover="sbx",
        crossover_rate=0.9,
        crossover_params={"eta": 20.0},
        mutation="gaussian",
        mutation_rate=0.2,
        mutation_params={"sigma": 0.5},
        genome_type="vector",
        genome_params={"dimensions": 200, "bounds": (-5.0, 5.0)},
        minimize=True,
        seed=42,
    )
    r = run_scenario("High Dim (500×200×200D)", cfg11, sphere, n_trials=2)
    results.append(("HighDim-200D", *r))

    # Scenario 12 — Large ERP (500×100×30D with ERP, ~90s)
    #   ERP per-mating overhead is O(pop_size) — tests compounding at scale.
    cfg12 = UnifiedConfig(
        name="large_erp",
        population_size=500,
        max_generations=100,
        elitism=5,
        selection="tournament",
        selection_params={"tournament_size": 5},
        crossover="sbx",
        crossover_rate=0.9,
        crossover_params={"eta": 15.0},
        mutation="gaussian",
        mutation_rate=0.1,
        mutation_params={"sigma": 0.5},
        genome_type="vector",
        genome_params={"dimensions": 30, "bounds": (-5.0, 5.0)},
        minimize=True,
        seed=42,
        erp=ERPSettings(),
    )
    r = run_scenario("Large ERP (500×100×30D)", cfg12, sphere, n_trials=2)
    results.append(("LargeERP-500", *r))

    # ── FINAL SUMMARY ───────────────────────────────────────────────────
    print(f"\n\n{'=' * 72}")
    print("  SUMMARY: Estimate Accuracy Across All Scenarios")
    print(f"{'=' * 72}")
    print(f"  {'Scenario':<30s} {'Estimated':>10s} {'Actual':>10s} {'Ratio':>8s} {'Error':>8s}")
    print(f"  {'-' * 68}")
    for name, ratio, error, est, act in results:
        print(f"  {name:<30s} {est:>9.3f}s {act:>9.3f}s {ratio:>7.2f}x {error:>6.1f}%")

    avg_error = sum(r[2] for r in results) / len(results)
    print(f"  {'-' * 68}")
    print(f"  {'Average error':<30s} {'':>10s} {'':>10s} {'':>8s} {avg_error:>6.1f}%")
    print()


if __name__ == "__main__":
    main()
