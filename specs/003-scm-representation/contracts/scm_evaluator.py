"""
SCM Evaluator API Contract

This file defines the interface contracts for SCMEvaluator and
fitness configuration. Implementation MUST satisfy all type signatures
and documented behaviors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evolve.core.types import Fitness, Individual
    from evolve.evaluation.evaluator import EvaluatorCapabilities

    from .scm_decoder import DecodedSCM, SCMDecoder
    from .scm_genome import AcyclicityMode, AcyclicityStrategy, SCMGenome


# === Fitness Configuration ===


@dataclass(frozen=True)
class SCMFitnessConfig:
    """
    Configuration for SCM fitness evaluation.

    Separate from SCMConfig to allow different evaluation
    settings for the same genome configuration.
    """

    # Objectives to compute
    objectives: tuple[str, ...] = ("data_fit", "sparsity", "simplicity")
    """
    Objectives to compute. Available:
    - "data_fit": Negative MSE on observed endogenous variables
    - "sparsity": Negative edge count
    - "simplicity": Negative total AST complexity
    - "coverage": Fraction of observed variables with equations
    - "latent_parsimony": Negative latent variable count
    """

    # Constraints
    constraints: tuple[str, ...] = ("acyclicity",)
    """
    Constraints to check. Available:
    - "acyclicity": Graph must be a DAG
    - "coverage": All observed variables must have equations
    - "conflict_free": No equation conflicts
    """

    # Penalty configuration
    cycle_penalty_per_cycle: float = 1.0
    incomplete_coverage_penalty: float = 10.0
    conflict_penalty: float = 1.0
    div_zero_penalty: float = 5.0
    latent_ancestor_penalty: float = 10.0

    # Acyclicity handling
    acyclicity_mode: AcyclicityMode = None  # Default from SCMConfig
    acyclicity_strategy: AcyclicityStrategy = None  # Default from SCMConfig

    def __post_init__(self) -> None:
        """Set defaults from AcyclicityMode/Strategy if not provided."""
        # Import here to avoid circular dependency
        from .scm_genome import AcyclicityMode, AcyclicityStrategy

        if self.acyclicity_mode is None:
            object.__setattr__(self, "acyclicity_mode", AcyclicityMode.REJECT)
        if self.acyclicity_strategy is None:
            object.__setattr__(self, "acyclicity_strategy", AcyclicityStrategy.ACYCLIC_SUBGRAPH)


# === Evaluation Result ===


@dataclass(frozen=True)
class SCMEvaluationResult:
    """
    Detailed evaluation result for debugging and analysis.

    Contains objective values, penalty breakdown, and evaluation metadata.
    """

    # Core fitness
    objectives: tuple[float, ...]
    """Objective values in order matching config.objectives."""

    total_penalty: float
    """Sum of all applied penalties."""

    is_valid: bool
    """Whether individual passes all hard constraints."""

    # Penalty breakdown
    cycle_penalty: float
    coverage_penalty: float
    conflict_penalty: float
    div_zero_penalty: float
    latent_ancestor_penalty: float

    # Evaluation metadata
    evaluated_variables: frozenset[str]
    """Variables that were successfully evaluated."""

    nan_variables: frozenset[str]
    """Variables with NaN predictions."""

    cycle_count: int
    """Number of cycles detected."""


# === Evaluator ===


class SCMEvaluator:
    """
    Multi-objective fitness evaluator for SCMs.

    Implements Evaluator[SCMGenome] protocol.

    Evaluates decoded SCMs against observed data using
    configurable objectives, constraints, and penalties.

    Thread Safety:
        Safe for concurrent use. All state is read-only
        after construction.
    """

    def __init__(
        self,
        data: np.ndarray,
        variable_names: Sequence[str],
        config: SCMFitnessConfig,
        decoder: SCMDecoder | None = None,
    ) -> None:
        """
        Initialize evaluator with data and configuration.

        Args:
            data: Observed data array, shape (n_samples, n_variables)
            variable_names: Column names matching data columns
            config: Fitness evaluation configuration
            decoder: Optional decoder instance (created if not provided)

        Raises:
            ValueError: If data shape doesn't match variable_names
        """
        if data.shape[1] != len(variable_names):
            raise ValueError(
                f"Data has {data.shape[1]} columns but "
                f"{len(variable_names)} variable names provided"
            )

        self.data = data
        self.variable_names = tuple(variable_names)
        self.config = config
        self.decoder = decoder

        # Pre-compute data dict for evaluation
        self._data_dict: dict[str, np.ndarray] = {
            name: data[:, i] for i, name in enumerate(variable_names)
        }

    @property
    def capabilities(self) -> EvaluatorCapabilities:
        """
        Declare evaluator capabilities.

        Returns:
            EvaluatorCapabilities with:
            - batchable=True
            - stochastic=False
            - n_objectives=len(config.objectives)
            - n_constraints=len(config.constraints)
        """
        from evolve.evaluation.evaluator import EvaluatorCapabilities

        return EvaluatorCapabilities(
            batchable=True,
            stochastic=False,
            stateful=True,  # Holds reference data
            n_objectives=len(self.config.objectives),
            n_constraints=len(self.config.constraints),
        )

    def evaluate(
        self,
        individuals: Sequence[Individual[SCMGenome]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate batch of SCM individuals.

        For each individual:
        1. Decode genome to SCM
        2. Check constraints (reject mode returns None)
        3. Compute objectives
        4. Apply penalties
        5. Return Fitness

        Args:
            individuals: Individuals to evaluate
            seed: Ignored (evaluation is deterministic)

        Returns:
            Fitness values in same order as input.
            Invalid individuals get None fitness.
        """
        ...

    def evaluate_detailed(
        self,
        genome: SCMGenome,
    ) -> tuple[Fitness | None, SCMEvaluationResult]:
        """
        Evaluate single genome with detailed result.

        Useful for debugging and analysis.

        Args:
            genome: Genome to evaluate

        Returns:
            Tuple of (fitness, detailed_result)
        """
        ...

    # === Objective Computation ===

    def _compute_data_fit(
        self,
        scm: DecodedSCM,
    ) -> tuple[float, frozenset[str]]:
        """
        Compute data fit objective.

        Evaluates equations in topological order, computing
        predictions for endogenous variables and comparing
        to observed values via MSE.

        Returns:
            Tuple of:
            - Negative MSE (higher is better)
            - Variables with NaN predictions
        """
        ...

    def _compute_sparsity(self, scm: DecodedSCM) -> float:
        """
        Compute sparsity objective.

        Returns negative edge count (higher is better = fewer edges).
        """
        return float(-scm.edge_count)

    def _compute_simplicity(self, scm: DecodedSCM) -> float:
        """
        Compute simplicity objective.

        Returns negative total AST complexity (higher is better = simpler).
        """
        return float(-scm.total_complexity)

    def _compute_coverage(self, scm: DecodedSCM) -> float:
        """
        Compute coverage objective.

        Returns fraction of observed variables with equations.
        """
        return scm.metadata.coverage

    def _compute_latent_parsimony(self, scm: DecodedSCM) -> float:
        """
        Compute latent parsimony objective.

        Returns negative count of latent variables used.
        """
        return float(-len(scm.metadata.latent_variables_used))

    # === Partial Evaluation Strategies ===

    def _evaluate_acyclic_subgraph(
        self,
        scm: DecodedSCM,
    ) -> tuple[float, frozenset[str]]:
        """
        Evaluate maximal acyclic subgraph.

        Finds largest DAG by removing cycle-creating edges,
        evaluates data fit on remaining equations.
        """
        ...

    def _evaluate_parse_order(
        self,
        scm: DecodedSCM,
    ) -> tuple[float, frozenset[str]]:
        """
        Break cycles by parse order.

        Removes edges that would create cycles based on
        the order equations appear in the genome.
        """
        ...

    def _apply_penalty_only(
        self,
        scm: DecodedSCM,
    ) -> float:
        """
        Compute penalty without partial evaluation.

        Returns cycle_count * cycle_penalty_per_cycle.
        """
        return len(scm.metadata.cycles) * self.config.cycle_penalty_per_cycle

    def _evaluate_composite(
        self,
        scm: DecodedSCM,
    ) -> tuple[float, float, frozenset[str]]:
        """
        Composite strategy: acyclic subgraph + proportional penalty.

        Returns:
            Tuple of:
            - Data fit on acyclic subgraph
            - Penalty for unevaluated portion
            - Variables with NaN
        """
        ...

    # === Constraint Checking ===

    def _check_acyclicity(self, scm: DecodedSCM) -> bool:
        """Check if SCM is acyclic."""
        return not scm.is_cyclic

    def _check_coverage(self, scm: DecodedSCM) -> bool:
        """Check if all observed variables have equations."""
        return scm.metadata.coverage >= 1.0

    def _check_conflict_free(self, scm: DecodedSCM) -> bool:
        """Check if there were no equation conflicts."""
        return scm.metadata.conflict_count == 0
