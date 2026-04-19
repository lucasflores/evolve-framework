"""
Unit tests for dry-run statistics tool.

Tests cover all frozen dataclasses, core benchmarking, resource detection,
bottleneck identification, meta-evolution estimation, memory estimation,
and the summary output format.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from evolve.config.unified import UnifiedConfig
from evolve.core.types import Fitness
from evolve.experiment.dry_run import (
    ComputeResources,
    DryRunReport,
    MemoryEstimate,
    MetaEstimate,
    PhaseEstimate,
    _collect_caveats,
    _compute_percentages_and_bottleneck,
    _derive_structural_constants,
    _detect_active_subsystems,
    _detect_cpu_count,
    _detect_memory,
    _detect_resources,
    _estimate_memory,
    _estimate_meta_evolution,
    _validate_config,
    dry_run,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_config(**overrides: Any) -> UnifiedConfig:
    """Build a minimal valid config for testing."""
    defaults: dict[str, Any] = {
        "population_size": 20,
        "max_generations": 10,
        "genome_type": "vector",
        "genome_params": {"dimensions": 5, "bounds": (-5.0, 5.0)},
        "selection": "tournament",
        "crossover": "uniform",
        "mutation": "gaussian",
        "evaluator": None,
    }
    defaults.update(overrides)
    return UnifiedConfig(**defaults)


class _DummyEvaluator:
    """Minimal evaluator for testing."""

    @property
    def capabilities(self) -> Any:
        return MagicMock()

    def evaluate(
        self,
        individuals: Sequence[Any],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        return [Fitness.scalar(1.0) for _ in individuals]


# ============================================================================
# Phase 2: Frozen Dataclass Tests (T004, T006, T008, T010, T012)
# ============================================================================


class TestPhaseEstimate:
    """Tests for PhaseEstimate frozen dataclass (T004)."""

    def test_construction(self) -> None:
        """PhaseEstimate can be constructed with all fields."""
        pe = PhaseEstimate(
            name="evaluation",
            measured_time_ms=1.5,
            operations_per_generation=100,
            estimated_total_ms=150.0,
            percentage=60.0,
            is_bottleneck=True,
        )
        assert pe.name == "evaluation"
        assert pe.measured_time_ms == 1.5
        assert pe.operations_per_generation == 100
        assert pe.estimated_total_ms == 150.0
        assert pe.percentage == 60.0
        assert pe.is_bottleneck is True

    def test_immutability(self) -> None:
        """PhaseEstimate is frozen — cannot mutate fields."""
        pe = PhaseEstimate(
            name="evaluation",
            measured_time_ms=1.5,
            operations_per_generation=100,
            estimated_total_ms=150.0,
            percentage=60.0,
            is_bottleneck=True,
        )
        with pytest.raises(FrozenInstanceError):
            pe.name = "selection"  # type: ignore[misc]


class TestComputeResources:
    """Tests for ComputeResources frozen dataclass (T006)."""

    def test_construction(self) -> None:
        """ComputeResources can be constructed with all fields."""
        cr = ComputeResources(
            cpu_count=8,
            total_memory_bytes=16 * 1024**3,
            gpu_available=True,
            gpu_name="NVIDIA RTX 3090",
            gpu_memory_bytes=24 * 1024**3,
            backend_name="parallel",
            backend_workers=8,
        )
        assert cr.cpu_count == 8
        assert cr.gpu_available is True
        assert cr.gpu_name == "NVIDIA RTX 3090"

    def test_gpu_consistency_no_gpu(self) -> None:
        """When gpu_available=False, gpu fields should be None."""
        cr = ComputeResources(
            cpu_count=4,
            total_memory_bytes=8 * 1024**3,
            gpu_available=False,
            gpu_name=None,
            gpu_memory_bytes=None,
            backend_name="sequential",
            backend_workers=None,
        )
        assert cr.gpu_name is None
        assert cr.gpu_memory_bytes is None


class TestMemoryEstimate:
    """Tests for MemoryEstimate frozen dataclass (T008)."""

    def test_construction(self) -> None:
        """MemoryEstimate can be constructed."""
        me = MemoryEstimate(
            genome_bytes=80,
            individual_overhead_bytes=256,
            population_bytes=67200,
            history_bytes=2000,
            total_bytes=69200,
        )
        assert me.genome_bytes == 80
        assert me.total_bytes == 69200

    def test_total_bytes_consistency(self) -> None:
        """total_bytes == population_bytes + history_bytes."""
        pop = 67200
        hist = 2000
        me = MemoryEstimate(
            genome_bytes=80,
            individual_overhead_bytes=256,
            population_bytes=pop,
            history_bytes=hist,
            total_bytes=pop + hist,
        )
        assert me.total_bytes == me.population_bytes + me.history_bytes


class TestMetaEstimate:
    """Tests for MetaEstimate frozen dataclass (T010)."""

    def test_construction(self) -> None:
        """MetaEstimate can be constructed."""
        me = MetaEstimate(
            inner_run_estimate_ms=5000.0,
            outer_generations=10,
            trials_per_config=3,
            total_inner_runs=600,
            total_estimated_ms=3_000_000.0,
        )
        assert me.inner_run_estimate_ms == 5000.0
        assert me.total_inner_runs == 600

    def test_total_inner_runs_consistency(self) -> None:
        """total_inner_runs == outer_generations * trials_per_config * outer_pop_size."""
        outer_gen = 10
        trials = 3
        outer_pop = 20
        total_runs = outer_gen * trials * outer_pop
        me = MetaEstimate(
            inner_run_estimate_ms=100.0,
            outer_generations=outer_gen,
            trials_per_config=trials,
            total_inner_runs=total_runs,
            total_estimated_ms=100.0 * total_runs,
        )
        assert me.total_inner_runs == outer_gen * trials * outer_pop


class TestDryRunReport:
    """Tests for DryRunReport frozen dataclass (T012)."""

    def _make_report(self, **overrides: Any) -> DryRunReport:
        """Build a minimal DryRunReport."""
        defaults = {
            "config_hash": "abc123",
            "phase_estimates": (
                PhaseEstimate("eval", 1.0, 100, 100.0, 80.0, True),
                PhaseEstimate("sel", 0.1, 200, 25.0, 20.0, False),
            ),
            "total_estimated_ms": 125.0,
            "estimated_generations": 10,
            "resources": ComputeResources(4, 8 * 1024**3, False, None, None, "sequential", None),
            "memory": MemoryEstimate(80, 256, 67200, 2000, 69200),
            "seed_used": 42,
            "early_stop_possible": False,
            "active_subsystems": (),
            "meta_estimate": None,
            "caveats": ("Estimates are point-based.",),
        }
        defaults.update(overrides)
        return DryRunReport(**defaults)

    def test_construction(self) -> None:
        """DryRunReport can be constructed."""
        report = self._make_report()
        assert report.config_hash == "abc123"
        assert len(report.phase_estimates) == 2

    def test_summary_returns_string(self) -> None:
        """summary() returns a non-empty string."""
        report = self._make_report()
        text = report.summary()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_str_delegates_to_summary(self) -> None:
        """__str__ delegates to summary()."""
        report = self._make_report()
        assert str(report) == report.summary()


# ============================================================================
# Phase 3: US1 MVP Tests (T013–T017)
# ============================================================================


class TestDryRunCorePhases:
    """Tests for dry_run() core functionality (T013)."""

    def test_returns_report_with_core_phases(self) -> None:
        """dry_run() returns a DryRunReport with core phases."""
        config = _make_config()
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)

        assert isinstance(report, DryRunReport)
        phase_names = {p.name for p in report.phase_estimates}
        assert "initialization" in phase_names
        assert "evaluation" in phase_names
        assert "selection" in phase_names
        assert "variation" in phase_names

    def test_merge_phase_when_enabled(self) -> None:
        """dry_run() with merge enabled includes a merge phase (T014)."""
        from evolve.config.merge import MergeConfig

        config = _make_config(
            merge=MergeConfig(merge_rate=0.1),
        )
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)

        phase_names = {p.name for p in report.phase_estimates}
        assert "merge" in phase_names

    def test_percentages_sum_to_100(self) -> None:
        """Phase percentages sum to ~100% and exactly one bottleneck (T015)."""
        config = _make_config()
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)

        total_pct = sum(p.percentage for p in report.phase_estimates)
        assert abs(total_pct - 100.0) < 0.1

        bottleneck_count = sum(1 for p in report.phase_estimates if p.is_bottleneck)
        assert bottleneck_count == 1

    def test_raises_on_missing_evaluator(self) -> None:
        """dry_run() raises ValueError when no evaluator provided (T016)."""
        config = _make_config(evaluator=None)

        with pytest.raises(ValueError, match="No evaluator"):
            dry_run(config, evaluator=None, seed=42)

    def test_structural_constants_match_config(self) -> None:
        """operations_per_generation matches config constants (T017)."""
        config = _make_config(population_size=50, elitism=5)
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)

        phase_map = {p.name: p for p in report.phase_estimates}

        # Evaluation: population_size
        assert phase_map["evaluation"].operations_per_generation == 50

        # Selection: (pop - elitism) * 2
        assert phase_map["selection"].operations_per_generation == (50 - 5) * 2

        # Variation: pop - elitism
        assert phase_map["variation"].operations_per_generation == 50 - 5


# ============================================================================
# Phase 3: Helpers Tests
# ============================================================================


class TestValidateConfig:
    """Tests for _validate_config (T018)."""

    def test_valid_config_passes(self) -> None:
        """Valid config does not raise."""
        config = _make_config()
        _validate_config(config)  # should not raise

    def test_missing_evaluator_raises(self) -> None:
        """Raises ValueError when no evaluator provided."""
        config = _make_config(evaluator=None)
        with pytest.raises(ValueError, match="No evaluator"):
            dry_run(config, evaluator=None, seed=42)


class TestDeriveStructuralConstants:
    """Tests for _derive_structural_constants (T020)."""

    def test_basic_constants(self) -> None:
        """Basic structural constants are correct."""
        config = _make_config(population_size=100, elitism=5)
        constants = _derive_structural_constants(config)

        assert constants["initialization"] == 100
        assert constants["evaluation"] == 100
        assert constants["selection"] == (100 - 5) * 2
        assert constants["crossover"] == 95
        assert constants["mutation"] == 95

    def test_merge_constants_when_enabled(self) -> None:
        """Merge ops count when merge enabled."""
        from evolve.config.merge import MergeConfig

        config = _make_config(
            population_size=100,
            merge=MergeConfig(merge_rate=0.1),
        )
        constants = _derive_structural_constants(config)
        assert constants["merge"] == 10  # 100 * 0.1


class TestComputePercentagesAndBottleneck:
    """Tests for _compute_percentages_and_bottleneck (T023)."""

    def test_percentages_sum_to_100(self) -> None:
        """Percentages sum to 100%."""
        phases = [
            PhaseEstimate("eval", 1.0, 100, 80.0, 0.0, False),
            PhaseEstimate("sel", 0.5, 200, 20.0, 0.0, False),
        ]
        result = _compute_percentages_and_bottleneck(phases)
        total = sum(p.percentage for p in result)
        assert abs(total - 100.0) < 0.01

    def test_bottleneck_is_highest(self) -> None:
        """Bottleneck is marked on the phase with highest time."""
        phases = [
            PhaseEstimate("eval", 1.0, 100, 80.0, 0.0, False),
            PhaseEstimate("sel", 0.5, 200, 20.0, 0.0, False),
        ]
        result = _compute_percentages_and_bottleneck(phases)
        bottleneck = [p for p in result if p.is_bottleneck]
        assert len(bottleneck) == 1
        assert bottleneck[0].name == "eval"

    def test_equal_percentages_first_wins(self) -> None:
        """When phases have equal time, first wins bottleneck."""
        phases = [
            PhaseEstimate("eval", 1.0, 100, 50.0, 0.0, False),
            PhaseEstimate("sel", 1.0, 100, 50.0, 0.0, False),
        ]
        result = _compute_percentages_and_bottleneck(phases)
        bottleneck = [p for p in result if p.is_bottleneck]
        assert len(bottleneck) == 1
        assert bottleneck[0].name == "eval"

    def test_all_zero_times(self) -> None:
        """Handles all-zero times gracefully."""
        phases = [
            PhaseEstimate("eval", 0.0, 100, 0.0, 0.0, False),
            PhaseEstimate("sel", 0.0, 200, 0.0, 0.0, False),
        ]
        result = _compute_percentages_and_bottleneck(phases)
        assert len(result) == 2
        # First should be bottleneck
        assert result[0].is_bottleneck


# ============================================================================
# Phase 4: US2 Resource Detection Tests (T026–T027)
# ============================================================================


class TestResourceDetection:
    """Tests for resource detection (T026-T027)."""

    def test_detect_resources_returns_valid(self) -> None:
        """_detect_resources returns ComputeResources with cpu_count >= 1 (T026)."""
        config = _make_config()
        resources = _detect_resources(config)

        assert isinstance(resources, ComputeResources)
        assert resources.cpu_count >= 1
        assert resources.backend_name == "sequential"

    def test_no_gpu_fields_none(self) -> None:
        """When gpu_available=False, gpu fields are None (T027)."""
        config = _make_config()
        resources = _detect_resources(config)

        if not resources.gpu_available:
            assert resources.gpu_name is None
            assert resources.gpu_memory_bytes is None

    def test_detect_cpu_count_positive(self) -> None:
        """CPU count is always positive."""
        count = _detect_cpu_count()
        assert count >= 1

    def test_detect_memory_nonnegative(self) -> None:
        """Memory detection returns a positive number or None."""
        mem = _detect_memory()
        if mem is not None:
            assert mem > 0


# ============================================================================
# Phase 5: US3 Bottleneck Tests (T033–T034)
# ============================================================================


class TestBottleneckIdentification:
    """Tests for bottleneck identification (T033-T034)."""

    def test_summary_contains_star(self) -> None:
        """summary() output has ★ on bottleneck row (T033)."""
        config = _make_config()
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)
        text = report.summary()

        assert "★" in text

    def test_evaluation_bottleneck_when_dominant(self) -> None:
        """Evaluation marked as bottleneck when it dominates (T034)."""
        phases = [
            PhaseEstimate("evaluation", 10.0, 100, 1000.0, 0.0, False),
            PhaseEstimate("selection", 0.1, 200, 1.0, 0.0, False),
            PhaseEstimate("variation", 0.1, 100, 1.0, 0.0, False),
        ]
        result = _compute_percentages_and_bottleneck(phases)
        eval_phase = [p for p in result if p.name == "evaluation"][0]
        assert eval_phase.is_bottleneck is True


# ============================================================================
# Phase 6: US5 Meta-Evolution Tests (T037–T039)
# ============================================================================


class TestMetaEvolution:
    """Tests for meta-evolution estimation (T037-T039)."""

    def _meta_config(self) -> UnifiedConfig:
        """Config with meta-evolution enabled."""
        from evolve.config.meta import MetaEvolutionConfig, ParameterSpec

        return _make_config(
            meta=MetaEvolutionConfig(
                evolvable_params=(ParameterSpec(path="mutation_rate", bounds=(0.01, 0.3)),),
                outer_population_size=20,
                outer_generations=5,
                trials_per_config=3,
            ),
        )

    def test_meta_estimate_not_none(self) -> None:
        """dry_run with meta returns meta_estimate (T037)."""
        config = self._meta_config()
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)

        assert report.meta_estimate is not None
        assert report.meta_estimate.total_estimated_ms > 0

    def test_total_inner_runs_formula(self) -> None:
        """total_inner_runs == outer_gen * trials * outer_pop (T038)."""
        config = self._meta_config()
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)

        me = report.meta_estimate
        assert me is not None
        expected = 5 * 3 * 20  # outer_gen * trials * outer_pop
        assert me.total_inner_runs == expected

    def test_summary_includes_meta_section(self) -> None:
        """summary() includes meta-evolution breakdown (T039)."""
        config = self._meta_config()
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)
        text = report.summary()

        assert "Meta-Evolution" in text
        assert "Inner run" in text or "inner run" in text.lower()


# ============================================================================
# Phase 7: US1 Extended Subsystem Tests (T043–T048)
# ============================================================================


class TestExtendedSubsystems:
    """Tests for extended subsystem benchmarking (T043-T048)."""

    def test_erp_phases(self) -> None:
        """ERP enabled includes erp_intent and erp_matchability (T043)."""
        from evolve.config.erp import ERPSettings

        config = _make_config(erp=ERPSettings())
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)
        names = {p.name for p in report.phase_estimates}

        assert "erp_intent" in names
        assert "erp_matchability" in names

    def test_multiobjective_ranking_phase(self) -> None:
        """Multiobjective config includes ranking phase (T044)."""
        from evolve.config.multiobjective import MultiObjectiveConfig, ObjectiveSpec

        config = _make_config(
            multiobjective=MultiObjectiveConfig(
                objectives=(
                    ObjectiveSpec(name="f1"),
                    ObjectiveSpec(name="f2"),
                ),
            ),
        )
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)
        names = {p.name for p in report.phase_estimates}

        assert "ranking" in names

    def test_tracking_phase(self) -> None:
        """Tracking enabled includes tracking phase (T046)."""
        from evolve.config.tracking import TrackingConfig

        config = _make_config(tracking=TrackingConfig())
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)
        names = {p.name for p in report.phase_estimates}

        assert "tracking" in names

    def test_active_subsystems_list(self) -> None:
        """active_subsystems lists all enabled subsystems (T047)."""
        from evolve.config.erp import ERPSettings
        from evolve.config.tracking import TrackingConfig

        config = _make_config(
            erp=ERPSettings(),
            tracking=TrackingConfig(),
        )
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)

        assert "erp" in report.active_subsystems
        assert "tracking" in report.active_subsystems

    def test_caveats_include_erp(self) -> None:
        """Caveats include ERP recovery caveat (T048)."""
        from evolve.config.erp import ERPSettings

        config = _make_config(erp=ERPSettings())
        caveats = _collect_caveats(config)

        assert any("ERP" in c for c in caveats)

    def test_caveats_include_remote_tracking(self) -> None:
        """Caveats include remote tracking caveat (T048)."""
        from evolve.config.tracking import TrackingConfig

        config = _make_config(
            tracking=TrackingConfig(tracking_uri="http://remote:5000"),
        )
        caveats = _collect_caveats(config)

        assert any(
            "Remote" in c or "remote" in c.lower() or "network" in c.lower() for c in caveats
        )

    def test_erp_core_phases_nonzero(self) -> None:
        """ERP config still produces non-zero core phase estimates.

        Regression test: calibration must strip ERP settings so a standard
        EvolutionEngine (with GenerationTimer) is used for the calibration
        run.  Otherwise evaluation/selection/variation estimate as 0.
        """
        from evolve.config.erp import ERPSettings

        config = _make_config(erp=ERPSettings())
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)
        by_name = {p.name: p for p in report.phase_estimates}

        # Core phases must have non-zero estimated_total_ms
        assert by_name["evaluation"].estimated_total_ms > 0
        assert by_name["selection"].estimated_total_ms > 0
        assert by_name["variation"].estimated_total_ms > 0


# ============================================================================
# Phase 8: US4 Memory Tests (T056–T058)
# ============================================================================


class TestMemoryProjections:
    """Tests for memory projections (T056-T058)."""

    def test_genome_bytes_positive(self) -> None:
        """genome_bytes > 0 for vector genome (T056)."""
        config = _make_config()
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)

        assert report.memory.genome_bytes > 0

    def test_total_bytes_equals_sum(self) -> None:
        """total_bytes == population_bytes + history_bytes (T057)."""
        config = _make_config()
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)

        assert report.memory.total_bytes == (
            report.memory.population_bytes + report.memory.history_bytes
        )

    def test_summary_includes_memory_line(self) -> None:
        """summary() includes memory line (T058)."""
        config = _make_config()
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)
        text = report.summary()

        assert "Memory:" in text or "memory" in text.lower()
        assert "MB" in text


# ============================================================================
# Detect Active Subsystems Tests (T053)
# ============================================================================


class TestDetectActiveSubsystems:
    """Tests for _detect_active_subsystems."""

    def test_no_subsystems(self) -> None:
        """No optional subsystems enabled."""
        config = _make_config()
        result = _detect_active_subsystems(config)
        assert result == ()

    def test_erp_subsystem(self) -> None:
        """ERP detected when enabled."""
        from evolve.config.erp import ERPSettings

        config = _make_config(erp=ERPSettings())
        result = _detect_active_subsystems(config)
        assert "erp" in result

    def test_meta_subsystem(self) -> None:
        """Meta-evolution detected when enabled."""
        from evolve.config.meta import MetaEvolutionConfig, ParameterSpec

        config = _make_config(
            meta=MetaEvolutionConfig(
                evolvable_params=(ParameterSpec(path="mutation_rate", bounds=(0.01, 0.3)),),
            ),
        )
        result = _detect_active_subsystems(config)
        assert "meta_evolution" in result


# ============================================================================
# Estimate Meta Evolution Tests (T040)
# ============================================================================


class TestEstimateMetaEvolution:
    """Tests for _estimate_meta_evolution."""

    def test_formula(self) -> None:
        """MetaEstimate formula: total_inner_runs = outer_gen * trials * outer_pop."""
        from evolve.config.meta import MetaEvolutionConfig, ParameterSpec

        config = _make_config(
            meta=MetaEvolutionConfig(
                evolvable_params=(ParameterSpec(path="mutation_rate", bounds=(0.01, 0.3)),),
                outer_population_size=20,
                outer_generations=5,
                trials_per_config=3,
            ),
        )
        inner_ms = 1000.0
        me = _estimate_meta_evolution(config, inner_ms)

        assert me.inner_run_estimate_ms == 1000.0
        assert me.outer_generations == 5
        assert me.trials_per_config == 3
        assert me.total_inner_runs == 5 * 3 * 20
        # total = (inner_run + per_trial_setup) * total_runs * 1.05
        # with per_trial_setup=0 (default): inner_run * total_runs * 1.05
        expected_total = 1000.0 * (5 * 3 * 20) * 1.05
        assert me.total_estimated_ms == expected_total


# ============================================================================
# Estimate Memory Tests (T059)
# ============================================================================


class TestEstimateMemory:
    """Tests for _estimate_memory."""

    def test_numpy_genome(self) -> None:
        """Memory estimate for numpy array genome uses nbytes."""
        config = _make_config(population_size=100, max_generations=50)
        genome = np.zeros(10, dtype=np.float64)  # 80 bytes

        me = _estimate_memory(config, genome)

        assert me.genome_bytes == 80
        assert me.population_bytes > 0
        assert me.history_bytes > 0
        assert me.total_bytes == me.population_bytes + me.history_bytes

    def test_python_object_genome(self) -> None:
        """Memory estimate for plain Python objects uses sys.getsizeof."""
        config = _make_config(population_size=50, max_generations=10)
        genome = [1.0, 2.0, 3.0]

        me = _estimate_memory(config, genome)
        assert me.genome_bytes > 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_seed_determinism(self) -> None:
        """Same seed produces same report hash."""
        config = _make_config()
        evaluator = _DummyEvaluator()

        r1 = dry_run(config, evaluator=evaluator, seed=42)
        r2 = dry_run(config, evaluator=evaluator, seed=42)

        assert r1.config_hash == r2.config_hash
        assert r1.seed_used == r2.seed_used

    def test_early_stop_possible(self) -> None:
        """early_stop_possible reflects stopping config."""
        from evolve.config.stopping import StoppingConfig

        config = _make_config(
            stopping=StoppingConfig(stagnation_generations=10),
        )
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)
        assert report.early_stop_possible is True

    def test_no_stopping_config(self) -> None:
        """early_stop_possible is False when no stopping config."""
        config = _make_config()
        evaluator = _DummyEvaluator()

        report = dry_run(config, evaluator=evaluator, seed=42)
        assert report.early_stop_possible is False
