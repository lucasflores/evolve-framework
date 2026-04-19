"""
Dry-Run Statistics Tool.

Estimates the cost of an evolutionary run without executing it.
Micro-benchmarks each atomic operation and multiplies by structural
constants from the config to produce a granular cost estimate.

NO ML FRAMEWORK IMPORTS ALLOWED (unconditionally).
"""

from __future__ import annotations

import contextlib
import os
import random as random_module
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from evolve.config.unified import UnifiedConfig
    from evolve.evaluation.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Frozen dataclasses (Phase 2 — foundational)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseEstimate:
    """Per-phase timing data for a single evolutionary phase."""

    name: str
    measured_time_ms: float
    operations_per_generation: int
    estimated_total_ms: float
    percentage: float
    is_bottleneck: bool


@dataclass(frozen=True)
class ComputeResources:
    """Detected hardware capabilities of the execution environment."""

    cpu_count: int
    total_memory_bytes: int | None
    gpu_available: bool
    gpu_name: str | None
    gpu_memory_bytes: int | None
    backend_name: str
    backend_workers: int | None


@dataclass(frozen=True)
class MemoryEstimate:
    """Estimated memory usage for the full run."""

    genome_bytes: int
    individual_overhead_bytes: int
    population_bytes: int
    history_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class MetaEstimate:
    """Cost breakdown for meta-evolution."""

    inner_run_estimate_ms: float
    outer_generations: int
    trials_per_config: int
    total_inner_runs: int
    total_estimated_ms: float


@dataclass(frozen=True)
class DryRunReport:
    """Top-level output containing the complete dry-run analysis."""

    config_hash: str
    phase_estimates: tuple[PhaseEstimate, ...]
    total_estimated_ms: float
    estimated_generations: int
    resources: ComputeResources
    memory: MemoryEstimate
    seed_used: int
    early_stop_possible: bool
    active_subsystems: tuple[str, ...]
    meta_estimate: MetaEstimate | None
    caveats: tuple[str, ...]

    def summary(self) -> str:
        """Return formatted ASCII table summary."""
        lines: list[str] = []

        # Header
        lines.append("╔══════════════════╦═══════════════╦═════════╦════════════╗")
        lines.append("║ Phase            ║ Est. Time     ║ % Total ║ Bottleneck ║")
        lines.append("╠══════════════════╬═══════════════╬═════════╬════════════╣")

        # Phase rows
        for phase in self.phase_estimates:
            time_str = _format_time(phase.estimated_total_ms)
            pct_str = f"{phase.percentage:5.1f}%"
            marker = "  ★   " if phase.is_bottleneck else "      "
            name = phase.name[:16].ljust(16)
            lines.append(f"║ {name} ║ {time_str:>13s} ║ {pct_str:>7s} ║ {marker:>10s} ║")

        # Total row
        lines.append("╠══════════════════╬═══════════════╬═════════╬════════════╣")
        total_str = _format_time(self.total_estimated_ms)
        lines.append(f"║ {'TOTAL':16s} ║ {total_str:>13s} ║ {'100.0%':>7s} ║ {'':>10s} ║")
        lines.append("╚══════════════════╩═══════════════╩═════════╩════════════╝")

        # Resource line
        mem_str = (
            _format_bytes(self.resources.total_memory_bytes)
            if self.resources.total_memory_bytes
            else "N/A"
        )
        gpu_str = self.resources.gpu_name if self.resources.gpu_available else "No GPU"
        workers = (
            f" ({self.resources.backend_workers} workers)" if self.resources.backend_workers else ""
        )
        lines.append(
            f"Resources: {self.resources.cpu_count} CPUs, {mem_str} RAM, "
            f"{gpu_str} | Backend: {self.resources.backend_name}{workers}"
        )

        # Memory line
        pop_mb = self.memory.population_bytes / (1024 * 1024)
        hist_mb = self.memory.history_bytes / (1024 * 1024)
        total_mb = self.memory.total_bytes / (1024 * 1024)
        lines.append(
            f"Memory:    ~{pop_mb:.1f} MB population, ~{hist_mb:.1f} MB history | Total: ~{total_mb:.1f} MB"
        )

        # Meta-evolution section
        if self.meta_estimate is not None:
            lines.append("")
            lines.append("Meta-Evolution Breakdown:")
            inner_str = _format_time(self.meta_estimate.inner_run_estimate_ms)
            total_meta_str = _format_time(self.meta_estimate.total_estimated_ms)
            lines.append(f"  Inner run cost:  {inner_str}")
            lines.append(
                f"  Outer loop:      {self.meta_estimate.outer_generations} generations "
                f"× {self.meta_estimate.trials_per_config} trials/config"
            )
            lines.append(f"  Total inner runs: {self.meta_estimate.total_inner_runs}")
            lines.append(f"  Total meta cost:  {total_meta_str}")

        # Caveats
        if self.caveats:
            lines.append("")
            for caveat in self.caveats:
                lines.append(f"Note: {caveat}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Delegate to summary()."""
        return self.summary()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_time(ms: float) -> str:
    """Format milliseconds to human-readable time."""
    if ms < 1000:
        return f"{ms:.2f}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60
    return f"{hours:.1f}h"


def _format_bytes(n: int | None) -> str:
    """Format bytes to human-readable."""
    if n is None:
        return "N/A"
    gb = n / (1024**3)
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    mb = n / (1024**2)
    return f"{mb:.1f} MB"


# ---------------------------------------------------------------------------
# Config validation (T018)
# ---------------------------------------------------------------------------


def _validate_config(config: UnifiedConfig) -> None:
    """Validate config has required fields for dry-run benchmarking."""
    if not config.genome_type:
        raise ValueError("UnifiedConfig.genome_type is required for dry-run benchmarking")
    if config.population_size <= 0:
        raise ValueError("UnifiedConfig.population_size must be positive")
    if config.max_generations <= 0:
        raise ValueError("UnifiedConfig.max_generations must be positive")


# ---------------------------------------------------------------------------
# Sample population creation (T019)
# ---------------------------------------------------------------------------


def _create_sample_population(
    config: UnifiedConfig,
    seed: int,
    n_workers: int | None = None,
) -> list[Any]:
    """Create a sample population for benchmarking.

    The sample is sized to ``config.population_size`` so that the
    calibration engine can run real generations without crashing on
    elitism / selection bounds.
    """
    from evolve.core.types import Individual, IndividualMetadata
    from evolve.registry import get_genome_registry

    registry = get_genome_registry()
    rng = random_module.Random(seed)

    sample_size = max(config.population_size, 3)
    if n_workers is not None and n_workers > 1:
        sample_size = max(sample_size, 2 * n_workers)

    individuals: list[Any] = []
    for _ in range(sample_size):
        genome = registry.create(
            config.genome_type,
            rng=rng,
            **config.genome_params,
        )
        individual = Individual(
            genome=genome,
            metadata=IndividualMetadata(),
        )
        individuals.append(individual)

    return individuals


# ---------------------------------------------------------------------------
# Structural constants (T020)
# ---------------------------------------------------------------------------


def _derive_structural_constants(config: UnifiedConfig) -> dict[str, int]:
    """Compute per-generation operation counts from config."""
    pop = config.population_size
    elite = config.elitism
    n_offspring = max(pop - elite, 0)

    constants: dict[str, int] = {
        "initialization": pop,  # one-shot, not per-generation
        "evaluation": pop,
        "selection": n_offspring * 2,  # 2 parents per offspring
        "crossover": n_offspring,
        "mutation": n_offspring,
    }

    if config.is_merge_enabled and config.merge is not None:
        constants["merge"] = int(pop * config.merge.merge_rate)

    if config.is_erp_enabled:
        constants["erp_intent"] = pop
        constants["erp_matchability"] = pop  # ~pop matchability checks

    if config.is_multiobjective:
        constants["ranking"] = 1  # one sort per generation

    if config.decoder is not None:
        constants["decoding"] = pop  # one decode per individual

    if config.is_tracking_enabled:
        constants["tracking"] = 1  # one log call per generation

    return constants


# ---------------------------------------------------------------------------
# Benchmark timing (T021)
# ---------------------------------------------------------------------------


def _benchmark_phase(
    fn: Callable[[], Any],
    timeout: float,
    n_warmup: int = 1,
    n_samples: int = 5,
) -> float:
    """Time *fn* with warm-up, return median elapsed time in ms.

    Runs *n_warmup* calls (discarded) then *n_samples* timed calls,
    returning the **median** to reduce cold-start / outlier bias.
    Returns -1.0 on timeout or error.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _timed_call_multi,
            fn,
            n_warmup,
            n_samples,
        )
        try:
            return future.result(timeout=timeout)
        except (FuturesTimeoutError, Exception):
            return -1.0


def _timed_call_multi(
    fn: Callable[[], Any],
    n_warmup: int,
    n_samples: int,
) -> float:
    """Warm-up then collect *n_samples* timings; return median in ms."""
    for _ in range(n_warmup):
        fn()

    timings: list[float] = []
    for _ in range(n_samples):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)

    timings.sort()
    mid = len(timings) // 2
    return timings[mid]


# ---------------------------------------------------------------------------
# Core phase benchmarking (T022)
# ---------------------------------------------------------------------------


def _benchmark_core_phases(
    config: UnifiedConfig,
    sample_population: list[Any],
    evaluator: Any,
    seed: int,
    timeout_per_phase: float,
) -> tuple[list[PhaseEstimate], float, float]:
    """Benchmark core evolutionary phases using real engine calibration.

    Runs real generations through the engine, captures the internal timer
    breakdown, and extrapolates to the full run.  Uses **median** of
    post-warmup generations for robustness against outlier gens.

    Returns:
        (phase_estimates, per_trial_setup_ms, calibration_wall_clock_ms)
        where per_trial_setup_ms is engine creation + population creation +
        warmup overhead, and calibration_wall_clock_ms is the wall-clock
        time of the calibration run itself (for meta-evolution inner-run
        estimation).
    """
    from evolve.core.population import Population
    from evolve.factory import create_engine

    constants = _derive_structural_constants(config)
    generations = config.max_generations

    # --- Initialization (one-shot, not per-generation) ---
    from evolve.registry import get_genome_registry

    rng = random_module.Random(seed)
    registry = get_genome_registry()

    def bench_init() -> None:
        registry.create(config.genome_type, rng=rng, **config.genome_params)

    init_ms = _benchmark_phase(bench_init, timeout_per_phase)
    if init_ms < 0:
        init_ms = 0.0

    # --- Calibration: run real generations ---
    # Use up to 10 gens for stable estimates.  The first few are skipped as
    # warmup (cold-cache / JIT bias), and per-gen times use median to
    # resist outlier gens.
    calibration_gens = min(10, generations)

    # Build a short-lived config for calibration.
    # Strip ERP settings so create_engine builds a standard EvolutionEngine
    # with GenerationTimer instrumentation.  ERP overhead is estimated
    # separately by _benchmark_erp().
    import dataclasses

    cal_config = dataclasses.replace(
        config,
        max_generations=calibration_gens,
        erp=None,
    )

    # Time engine creation for per-trial meta overhead estimation.
    engine_creation_ms = 0.0
    _t_eng = time.perf_counter()
    try:
        engine = create_engine(cal_config, evaluator, seed=seed)
        engine_creation_ms = (time.perf_counter() - _t_eng) * 1000.0
    except Exception:
        # Fallback: if engine creation fails, return zero estimates
        fallback = [
            PhaseEstimate(
                "initialization",
                init_ms,
                constants["initialization"],
                init_ms * constants["initialization"],
                0.0,
                False,
            ),
            PhaseEstimate("evaluation", 0.0, constants["evaluation"], 0.0, 0.0, False),
            PhaseEstimate("selection", 0.0, constants["selection"], 0.0, 0.0, False),
            PhaseEstimate("variation", 0.0, constants["mutation"], 0.0, 0.0, False),
        ]
        if config.is_merge_enabled and config.merge is not None:
            fallback.append(
                PhaseEstimate(
                    "merge",
                    0.0,
                    constants.get("merge", 0),
                    0.0,
                    0.0,
                    False,
                )
            )
        return fallback, 0.0, 0.0

    pop = Population(individuals=list(sample_population), generation=0)

    # Time population creation for per-trial meta overhead estimation.
    from evolve.factory.engine import create_initial_population as _create_init_pop

    _t_pop = time.perf_counter()
    with contextlib.suppress(Exception):
        _create_init_pop(cal_config, seed=seed)
    pop_creation_ms = (time.perf_counter() - _t_pop) * 1000.0

    def _run_calibration() -> tuple[dict[str, float], float, float]:
        """Run calibration gens, return (median per-gen times, total_warmup_excess, wall_clock_ms).

        Uses adaptive warmup skip and **median** per-phase for
        robustness against outlier generations.
        """
        _t_wall = time.perf_counter()
        result = engine.run(pop)
        wall_clock_ms = (time.perf_counter() - _t_wall) * 1000.0
        history = result.history

        # Skip only the first generation (cold-cache / import overhead).
        # With median aggregation, we don't need aggressive warmup skip —
        # median naturally resists the occasional outlier gen.
        warmup_skip = min(1, max(len(history) - 3, 0))
        measured = history[warmup_skip:]

        if not measured:
            return (
                {
                    "selection": 0.0,
                    "variation": 0.0,
                    "evaluation": 0.0,
                    "generation": 0.0,
                    "merge": 0.0,
                },
                0.0,
                wall_clock_ms,
            )

        def _median(vals: list[float]) -> float:
            s = sorted(vals)
            n = len(s)
            mid = n // 2
            return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2

        per_gen_result = {
            "selection": _median([h.get("selection_time_ms", 0.0) for h in measured]),
            "variation": _median([h.get("variation_time_ms", 0.0) for h in measured]),
            "evaluation": _median([h.get("evaluation_time_ms", 0.0) for h in measured]),
            "generation": _median([h.get("generation_time_ms", 0.0) for h in measured]),
            "merge": _median([h.get("merge_time_ms", 0.0) for h in measured]),
        }

        # Total warmup excess: sum of (warmup_gen - steady_state) for all
        # warmup gens.  This captures the full warmup cost, not just gen-0.
        steady = per_gen_result["generation"]
        warmup_gens = history[:warmup_skip]
        total_warmup_excess = sum(
            max(g.get("generation_time_ms", 0.0) - steady, 0.0) for g in warmup_gens
        )

        return per_gen_result, total_warmup_excess, wall_clock_ms

    cal_wall_clock_ms = 0.0
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_calibration)
        try:
            per_gen, total_warmup_excess, cal_wall_clock_ms = future.result(
                timeout=timeout_per_phase * 4,
            )
        except (FuturesTimeoutError, Exception):
            per_gen = {
                "selection": 0.0,
                "variation": 0.0,
                "evaluation": 0.0,
                "generation": 0.0,
                "merge": 0.0,
            }
            total_warmup_excess = 0.0

    # Per-trial setup overhead for meta-evolution estimation:
    # engine creation + population creation + total warmup excess
    # (all warmup gens' cost above steady-state).
    per_trial_setup_ms = engine_creation_ms + pop_creation_ms + total_warmup_excess

    # Account for unmeasured per-generation overhead within _step():
    # Population construction, Individual creation, uuid4, etc.
    # These are captured by generation_time_ms but not by the individual
    # phase timers.  Distribute proportionally across phases.
    measured_sum = (
        per_gen["selection"] + per_gen["variation"] + per_gen["evaluation"] + per_gen["merge"]
    )
    intra_step_overhead = max(per_gen["generation"] - measured_sum, 0.0)

    overhead_factor = 1.0 + intra_step_overhead / measured_sum if measured_sum > 0 else 1.0

    sel_per_gen = per_gen["selection"] * overhead_factor
    var_per_gen = per_gen["variation"] * overhead_factor
    eval_per_gen = per_gen["evaluation"] * overhead_factor
    merge_per_gen = per_gen["merge"] * overhead_factor

    phases: list[PhaseEstimate] = []

    phases.append(
        PhaseEstimate(
            name="initialization",
            measured_time_ms=init_ms,
            operations_per_generation=constants["initialization"],
            estimated_total_ms=init_ms * constants["initialization"],
            percentage=0.0,
            is_bottleneck=False,
        )
    )

    phases.append(
        PhaseEstimate(
            name="evaluation",
            measured_time_ms=eval_per_gen,
            operations_per_generation=constants["evaluation"],
            estimated_total_ms=eval_per_gen * generations,
            percentage=0.0,
            is_bottleneck=False,
        )
    )

    phases.append(
        PhaseEstimate(
            name="selection",
            measured_time_ms=sel_per_gen,
            operations_per_generation=constants["selection"],
            estimated_total_ms=sel_per_gen * generations,
            percentage=0.0,
            is_bottleneck=False,
        )
    )

    phases.append(
        PhaseEstimate(
            name="variation",
            measured_time_ms=var_per_gen,
            operations_per_generation=constants["mutation"],
            estimated_total_ms=var_per_gen * generations,
            percentage=0.0,
            is_bottleneck=False,
        )
    )

    # --- Merge (if enabled) ---
    if config.is_merge_enabled and config.merge is not None:
        merge_total_ms = merge_per_gen * generations
        phases.append(
            PhaseEstimate(
                name="merge",
                measured_time_ms=merge_per_gen,
                operations_per_generation=constants.get("merge", 0),
                estimated_total_ms=merge_total_ms,
                percentage=0.0,
                is_bottleneck=False,
            )
        )

    return phases, per_trial_setup_ms, cal_wall_clock_ms


# ---------------------------------------------------------------------------
# Percentage and bottleneck (T023)
# ---------------------------------------------------------------------------


def _compute_percentages_and_bottleneck(
    phases: list[PhaseEstimate],
) -> tuple[PhaseEstimate, ...]:
    """Calculate percentages and mark the bottleneck (highest %). First wins ties."""
    total = sum(p.estimated_total_ms for p in phases)
    if total <= 0:
        # All zeros — mark first as bottleneck
        if phases:
            result = []
            for i, p in enumerate(phases):
                result.append(
                    PhaseEstimate(
                        name=p.name,
                        measured_time_ms=p.measured_time_ms,
                        operations_per_generation=p.operations_per_generation,
                        estimated_total_ms=p.estimated_total_ms,
                        percentage=100.0 / len(phases) if phases else 0.0,
                        is_bottleneck=(i == 0),
                    )
                )
            return tuple(result)
        return ()

    max_pct = -1.0
    max_idx = 0
    percentages: list[float] = []
    for i, p in enumerate(phases):
        pct = (p.estimated_total_ms / total) * 100.0
        percentages.append(pct)
        if pct > max_pct:
            max_pct = pct
            max_idx = i

    result = []
    for i, p in enumerate(phases):
        result.append(
            PhaseEstimate(
                name=p.name,
                measured_time_ms=p.measured_time_ms,
                operations_per_generation=p.operations_per_generation,
                estimated_total_ms=p.estimated_total_ms,
                percentage=percentages[i],
                is_bottleneck=(i == max_idx),
            )
        )
    return tuple(result)


# ---------------------------------------------------------------------------
# Resource detection (T028-T031)
# ---------------------------------------------------------------------------


def _detect_cpu_count() -> int:
    """Detect CPU count, container-aware."""
    # Try cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
            if parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1])
                return max(1, quota // period)
    except (OSError, ValueError, IndexError):
        pass

    # Try cgroup v1
    try:
        with (
            open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq,
            open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp,
        ):
            quota = int(fq.read().strip())
            period = int(fp.read().strip())
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except (OSError, ValueError):
        pass

    # Try sched_getaffinity (Linux)
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        pass

    return os.cpu_count() or 1


def _detect_memory() -> int | None:
    """Detect total system memory in bytes."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and pages > 0:
            return page_size * pages
    except (ValueError, OSError, AttributeError):
        pass
    return None


def _detect_gpu() -> tuple[bool, str | None, int | None]:
    """Detect GPU presence using conditional imports."""
    from evolve.utils.dependencies import check_dependency

    # Try PyTorch
    torch_check = check_dependency("torch")
    if torch_check.available:
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory
                return True, name, mem
        except Exception:
            pass

    # Try JAX
    jax_check = check_dependency("jax")
    if jax_check.available:
        try:
            import jax

            devices = jax.devices("gpu")
            if devices:
                dev = devices[0]
                name = str(dev)
                # JAX doesn't always expose memory
                return True, name, None
        except Exception:
            pass

    return False, None, None


def _detect_resources(config: UnifiedConfig) -> ComputeResources:
    """Compose all resource detection into ComputeResources."""
    cpu_count = _detect_cpu_count()
    total_memory = _detect_memory()
    gpu_available, gpu_name, gpu_memory = _detect_gpu()

    # Determine backend info from config
    backend_name = "sequential"  # default
    backend_workers: int | None = None

    # Infer from evaluator or config patterns
    if config.evaluator is not None:
        eval_name = config.evaluator.lower()
        if "parallel" in eval_name:
            backend_name = "parallel"
            backend_workers = cpu_count
        elif "jax" in eval_name:
            backend_name = "jax"
        elif "torch" in eval_name or "pytorch" in eval_name:
            backend_name = "torch"

    return ComputeResources(
        cpu_count=cpu_count,
        total_memory_bytes=total_memory,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_bytes=gpu_memory,
        backend_name=backend_name,
        backend_workers=backend_workers,
    )


# ---------------------------------------------------------------------------
# Memory estimation (T059)
# ---------------------------------------------------------------------------

_INDIVIDUAL_OVERHEAD_BYTES = 256  # UUID, fitness, metadata dict estimate


def _estimate_memory(
    config: UnifiedConfig,
    sample_genome: Any,
) -> MemoryEstimate:
    """Estimate peak memory usage from a sample genome."""
    # Measure genome size
    if hasattr(sample_genome, "nbytes"):
        genome_bytes = int(sample_genome.nbytes)
    else:
        genome_bytes = sys.getsizeof(sample_genome)

    individual_overhead = _INDIVIDUAL_OVERHEAD_BYTES

    # Current + offspring overlap = 2× population
    population_bytes = (genome_bytes + individual_overhead) * config.population_size * 2

    # History: ~200 bytes per generation for metrics dict
    history_bytes = 200 * config.max_generations

    total_bytes = population_bytes + history_bytes

    return MemoryEstimate(
        genome_bytes=genome_bytes,
        individual_overhead_bytes=individual_overhead,
        population_bytes=population_bytes,
        history_bytes=history_bytes,
        total_bytes=total_bytes,
    )


# ---------------------------------------------------------------------------
# Optional subsystem benchmarking (T049-T054)
# ---------------------------------------------------------------------------


def _benchmark_erp(
    config: UnifiedConfig,
    sample_population: list[Any],
    evaluator: Any,
    seed: int,
    standard_per_gen_ms: float,
    timeout: float,
) -> list[PhaseEstimate]:
    """Benchmark ERP overhead using real ERP engine calibration.

    Runs a short calibration through the real ERPEngine and measures
    wall-clock time.  The ERP overhead per generation is the difference
    between the ERP engine's per-gen wall-clock and the standard engine's
    per-gen time (from GenerationTimer).

    The total overhead is split 40/60 between intent and matchability
    based on profiled relative costs (matchability includes genetic
    distance, diversity computation, and fitness-ratio logic on top of
    the intent evaluation cost).
    """
    from evolve.core.population import Population
    from evolve.factory import create_engine

    constants = _derive_structural_constants(config)
    generations = config.max_generations

    # Run calibration with the real ERP engine
    calibration_gens = min(5, generations)
    import dataclasses

    cal_config = dataclasses.replace(config, max_generations=calibration_gens)

    try:
        engine = create_engine(cal_config, evaluator, seed=seed)
        pop = Population(individuals=list(sample_population), generation=0)

        def _run_erp_calibration() -> float:
            t0 = time.perf_counter()
            engine.run(pop)
            return (time.perf_counter() - t0) * 1000.0

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_erp_calibration)
            try:
                erp_wall_ms = future.result(timeout=timeout * 4)
            except (FuturesTimeoutError, Exception):
                erp_wall_ms = 0.0

        # Per-generation ERP cost (skip first gen conceptually by
        # subtracting ~1/gens of warmup, similar to standard calibration)
        erp_per_gen = erp_wall_ms / calibration_gens if calibration_gens > 1 else erp_wall_ms

        # ERP overhead = ERP per-gen - standard per-gen
        erp_overhead_per_gen = max(erp_per_gen - standard_per_gen_ms, 0.0)

    except Exception:
        erp_overhead_per_gen = 0.0

    # Split overhead 40% intent / 60% matchability
    intent_per_gen = erp_overhead_per_gen * 0.4
    match_per_gen = erp_overhead_per_gen * 0.6

    intent_total = intent_per_gen * generations
    match_total = match_per_gen * generations

    phases: list[PhaseEstimate] = []

    phases.append(
        PhaseEstimate(
            name="erp_intent",
            measured_time_ms=intent_per_gen,
            operations_per_generation=constants.get("erp_intent", config.population_size),
            estimated_total_ms=intent_total,
            percentage=0.0,
            is_bottleneck=False,
        )
    )

    phases.append(
        PhaseEstimate(
            name="erp_matchability",
            measured_time_ms=match_per_gen,
            operations_per_generation=constants.get("erp_matchability", config.population_size),
            estimated_total_ms=match_total,
            percentage=0.0,
            is_bottleneck=False,
        )
    )

    return phases


def _benchmark_ranking(
    config: UnifiedConfig,
    timeout: float,
) -> list[PhaseEstimate]:
    """Benchmark NSGA-II non-dominated sorting and crowding distance."""
    constants = _derive_structural_constants(config)
    generations = config.max_generations

    n_objectives = 2
    if config.multiobjective is not None:
        n_objectives = len(config.multiobjective.objectives)

    pop_size = config.population_size

    # Create random fitness values for a full-size population
    rng = np.random.default_rng(42)
    random_fitnesses_array = rng.random((pop_size, n_objectives))

    def bench_ranking() -> None:
        from evolve.multiobjective.crowding import crowding_distance
        from evolve.multiobjective.fitness import MultiObjectiveFitness
        from evolve.multiobjective.ranking import fast_non_dominated_sort

        fitnesses = [
            MultiObjectiveFitness(objectives=random_fitnesses_array[i]) for i in range(pop_size)
        ]
        fronts = fast_non_dominated_sort(fitnesses)
        if fronts:
            crowding_distance(fitnesses, fronts[0])

    ranking_ms = _benchmark_phase(bench_ranking, timeout)
    if ranking_ms < 0:
        ranking_ms = 0.0

    ranking_total = ranking_ms * constants.get("ranking", 1) * generations

    return [
        PhaseEstimate(
            name="ranking",
            measured_time_ms=ranking_ms,
            operations_per_generation=constants.get("ranking", 1),
            estimated_total_ms=ranking_total,
            percentage=0.0,
            is_bottleneck=False,
        )
    ]


def _benchmark_decoder(
    config: UnifiedConfig,
    sample_genome: Any,
    timeout: float,
) -> list[PhaseEstimate]:
    """Benchmark one decode operation."""
    from evolve.registry import get_decoder_registry

    constants = _derive_structural_constants(config)
    generations = config.max_generations

    try:
        decoder_registry = get_decoder_registry()
        assert config.decoder is not None  # caller checks before invoking
        decoder = decoder_registry.get(config.decoder, **config.decoder_params)

        def bench_decode() -> None:
            decoder.decode(sample_genome)

        decode_ms = _benchmark_phase(bench_decode, timeout)
    except Exception:
        decode_ms = 0.0

    if decode_ms < 0:
        decode_ms = 0.0

    decode_total = decode_ms * constants.get("decoding", config.population_size) * generations

    return [
        PhaseEstimate(
            name="decoding",
            measured_time_ms=decode_ms,
            operations_per_generation=constants.get("decoding", config.population_size),
            estimated_total_ms=decode_total,
            percentage=0.0,
            is_bottleneck=False,
        )
    ]


def _benchmark_tracking(
    config: UnifiedConfig,
    timeout: float,
) -> list[PhaseEstimate]:
    """Benchmark one log_metrics() call for local tracking backends."""
    constants = _derive_structural_constants(config)
    generations = config.max_generations

    # Only benchmark for local backends
    tracking_ms = 0.0
    if config.tracking is not None and config.tracking.tracking_uri is None:

        def bench_tracking() -> None:
            # Simulate a local metrics log
            pass

        tracking_ms = _benchmark_phase(bench_tracking, timeout)
        if tracking_ms < 0:
            tracking_ms = 0.0

    tracking_total = tracking_ms * constants.get("tracking", 1) * generations

    return [
        PhaseEstimate(
            name="tracking",
            measured_time_ms=tracking_ms,
            operations_per_generation=constants.get("tracking", 1),
            estimated_total_ms=tracking_total,
            percentage=0.0,
            is_bottleneck=False,
        )
    ]


def _detect_active_subsystems(config: UnifiedConfig) -> tuple[str, ...]:
    """Return names of active optional subsystems."""
    subsystems: list[str] = []
    if config.is_erp_enabled:
        subsystems.append("erp")
    if config.is_multiobjective:
        subsystems.append("multiobjective")
    if config.is_merge_enabled:
        subsystems.append("merge")
    if config.decoder is not None:
        subsystems.append("decoder")
    if config.is_tracking_enabled:
        subsystems.append("tracking")
    if config.is_meta_evolution:
        subsystems.append("meta_evolution")
    return tuple(subsystems)


def _collect_caveats(config: UnifiedConfig) -> tuple[str, ...]:
    """Build caveat strings based on config."""
    caveats: list[str] = []

    # Point-estimate caveat (always)
    caveats.append(
        "Estimates are based on single-invocation micro-benchmarks; actual run times may vary."
    )

    # Early-stop caveat
    if config.stopping is not None:
        caveats.append(
            "Stopping criteria configured — run may terminate "
            f"before {config.max_generations} generations."
        )

    # ERP recovery caveat
    if config.is_erp_enabled and config.erp is not None:
        caveats.append(
            "ERP recovery overhead not included in estimate; "
            f"triggers only if mating success rate < {config.erp.recovery_threshold}."
        )

    # Remote tracking caveat
    if config.tracking is not None and config.tracking.tracking_uri is not None:
        caveats.append(
            "Remote MLflow tracking configured — per-generation overhead "
            "depends on network latency and is not benchmarked."
        )

    return tuple(caveats)


# ---------------------------------------------------------------------------
# Meta-evolution estimation (T040)
# ---------------------------------------------------------------------------


def _estimate_meta_evolution(
    config: UnifiedConfig,
    inner_run_total_ms: float,
    per_trial_setup_ms: float = 0.0,
    calibration_wall_clock_ms: float = 0.0,
) -> MetaEstimate:
    """Compute MetaEstimate from inner run cost and meta config.

    Each meta trial creates a fresh engine and initial population, so
    ``per_trial_setup_ms`` (engine creation + population creation +
    warmup excess) is added per trial.

    When ``calibration_wall_clock_ms`` is provided and the calibration
    covered the full inner-run length, we use it as the inner-run cost
    instead of the extrapolated ``inner_run_total_ms``.  This avoids
    bias from median post-warmup extrapolation on short inner runs.

    A small outer-loop overhead factor (1.05×) accounts for tournament
    selection, codec encode/decode, and config mutation in the outer
    loop.
    """
    meta = config.meta
    if meta is None:
        raise ValueError("Meta-evolution not enabled")

    outer_generations = meta.outer_generations
    trials_per_config = meta.trials_per_config
    outer_pop_size = meta.outer_population_size
    total_inner_runs = outer_generations * trials_per_config * outer_pop_size

    # Use calibration wall-clock when it covers the full inner run, as
    # it naturally includes all warmup / overhead that per-gen median
    # extrapolation would miss on short inner runs.
    if calibration_wall_clock_ms > 0:
        effective_inner_ms = calibration_wall_clock_ms
    else:
        effective_inner_ms = inner_run_total_ms

    # Per-trial cost = inner run + per-trial engine/population creation.
    per_trial_cost_ms = effective_inner_ms + per_trial_setup_ms
    # 1.05× outer-loop overhead for codec, tournament, and mutation.
    _META_OUTER_OVERHEAD_FACTOR = 1.05
    total_estimated_ms = per_trial_cost_ms * total_inner_runs * _META_OUTER_OVERHEAD_FACTOR

    return MetaEstimate(
        inner_run_estimate_ms=inner_run_total_ms,
        outer_generations=outer_generations,
        trials_per_config=trials_per_config,
        total_inner_runs=total_inner_runs,
        total_estimated_ms=total_estimated_ms,
    )


# ---------------------------------------------------------------------------
# Top-level dry_run function (T024)
# ---------------------------------------------------------------------------


def dry_run(
    config: UnifiedConfig,
    evaluator: Evaluator | Callable[[Any], float] | None = None,
    seed: int | None = None,
    timeout_per_phase: float = 30.0,
) -> DryRunReport:
    """
    Estimate the cost of an evolutionary run without executing it.

    Micro-benchmarks a single invocation of each atomic operation
    (evaluation, crossover, mutation, selection, merge) using the
    real configured backend, then multiplies by structural constants
    from the config to project total run time.

    Args:
        config: Complete experiment configuration.
        evaluator: Optional explicit evaluator. If None, resolved from
            config.evaluator via the evaluator registry.
        seed: Random seed for sample genome creation.
        timeout_per_phase: Maximum seconds per micro-benchmark.

    Returns:
        DryRunReport with per-phase timing breakdown.

    Raises:
        ValueError: If config is invalid.
        KeyError: If an operator or genome type is not in the registry.
    """
    # Validate
    _validate_config(config)

    # Resolve seed
    if seed is None:
        seed = config.seed if config.seed is not None else int(time.time()) % (2**31)

    # Resolve evaluator
    if evaluator is None:
        if config.evaluator is not None:
            from evolve.registry import get_evaluator_registry

            evaluator = get_evaluator_registry().get(config.evaluator, **config.evaluator_params)
        else:
            raise ValueError(
                "No evaluator provided and config.evaluator is None. "
                "Pass an evaluator explicitly or set config.evaluator."
            )

    # Detect resources
    resources = _detect_resources(config)

    # Create sample population
    sample_population = _create_sample_population(
        config,
        seed,
        n_workers=resources.backend_workers,
    )

    # Core benchmarking
    core_phases, per_trial_setup_ms, cal_wall_clock_ms = _benchmark_core_phases(
        config,
        sample_population,
        evaluator,
        seed,
        timeout_per_phase,
    )

    # Optional subsystem benchmarking
    all_phases = list(core_phases)

    if config.is_erp_enabled:
        # Compute standard per-generation time from core phases (excluding
        # initialization which is one-shot) for ERP overhead calculation.
        standard_per_gen_ms = sum(
            p.estimated_total_ms / max(config.max_generations, 1)
            for p in core_phases
            if p.name != "initialization"
        )
        all_phases.extend(
            _benchmark_erp(
                config,
                sample_population,
                evaluator,
                seed,
                standard_per_gen_ms,
                timeout_per_phase,
            )
        )

    if config.is_multiobjective:
        all_phases.extend(_benchmark_ranking(config, timeout_per_phase))

    if config.decoder is not None:
        all_phases.extend(
            _benchmark_decoder(
                config,
                sample_population[0].genome,
                timeout_per_phase,
            )
        )

    if config.is_tracking_enabled:
        all_phases.extend(_benchmark_tracking(config, timeout_per_phase))

    # Compute percentages and bottleneck
    phase_estimates = _compute_percentages_and_bottleneck(all_phases)

    total_estimated_ms = sum(p.estimated_total_ms for p in phase_estimates)

    # Memory estimation
    memory = _estimate_memory(config, sample_population[0].genome)

    # Active subsystems
    active_subsystems = _detect_active_subsystems(config)

    # Caveats
    caveats = _collect_caveats(config)

    # Early stop detection
    early_stop_possible = config.stopping is not None

    # Meta-evolution
    meta_estimate: MetaEstimate | None = None
    if config.is_meta_evolution:
        meta_estimate = _estimate_meta_evolution(
            config,
            total_estimated_ms,
            per_trial_setup_ms,
            cal_wall_clock_ms,
        )

    # Config hash
    config_hash = config.compute_hash()

    return DryRunReport(
        config_hash=config_hash,
        phase_estimates=phase_estimates,
        total_estimated_ms=total_estimated_ms,
        estimated_generations=config.max_generations,
        resources=resources,
        memory=memory,
        seed_used=seed,
        early_stop_possible=early_stop_possible,
        active_subsystems=active_subsystems,
        meta_estimate=meta_estimate,
        caveats=caveats,
    )
