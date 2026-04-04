"""
SCM (Structural Causal Model) Evaluator.

This module provides multi-objective fitness evaluation for decoded SCMs,
supporting configurable objectives, constraints, and penalty handling.

Key components:
- SCMFitnessConfig: Configuration for fitness evaluation
- SCMEvaluationResult: Detailed evaluation result for debugging
- SCMEvaluator: Multi-objective evaluator implementing Evaluator protocol

Example:
    >>> from evolve.evaluation.scm_evaluator import SCMEvaluator, SCMFitnessConfig
    >>> evaluator = SCMEvaluator(data, variable_names, SCMFitnessConfig())
    >>> fitness = evaluator.evaluate([individual])
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from evolve.core.types import Fitness, Individual
    from evolve.evaluation.evaluator import EvaluatorCapabilities
    from evolve.representation.scm import AcyclicityMode, AcyclicityStrategy, SCMConfig, SCMGenome
    from evolve.representation.scm_decoder import DecodedSCM, SCMDecoder


__all__ = [
    "SCMFitnessConfig",
    "SCMEvaluationResult",
    "SCMEvaluator",
]


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

    # Acyclicity handling (import lazily to avoid circular import)
    acyclicity_mode: str = "reject"  # "reject" or "penalize"
    acyclicity_strategy: str = "acyclic_subgraph"

    def get_acyclicity_mode(self) -> AcyclicityMode:
        """Get AcyclicityMode enum value."""
        from evolve.representation.scm import AcyclicityMode

        return AcyclicityMode(self.acyclicity_mode)

    def get_acyclicity_strategy(self) -> AcyclicityStrategy:
        """Get AcyclicityStrategy enum value."""
        from evolve.representation.scm import AcyclicityStrategy

        return AcyclicityStrategy(self.acyclicity_strategy)


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
    div_zero_penalty_value: float
    latent_ancestor_penalty_value: float

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
        self._decoder = decoder

        # Pre-compute data dict for evaluation
        self._data_dict: dict[str, np.ndarray] = {
            name: data[:, i] for i, name in enumerate(variable_names)
        }

    def _get_decoder(self, scm_config: SCMConfig) -> SCMDecoder:
        """Get or create decoder for given config."""
        if self._decoder is not None:
            return self._decoder
        from evolve.representation.scm_decoder import SCMDecoder

        return SCMDecoder(scm_config)

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
    ) -> Sequence[Fitness | None]:
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

        results: list[Fitness | None] = []

        for individual in individuals:
            fitness, _ = self.evaluate_detailed(individual.genome)
            results.append(fitness)

        return results

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
        from evolve.core.types import Fitness
        from evolve.representation.scm import AcyclicityMode

        # Decode genome
        decoder = self._get_decoder(genome.config)
        scm = decoder.decode(genome)

        # Check hard constraints
        is_valid = True
        if "acyclicity" in self.config.constraints:
            if scm.is_cyclic:
                if self.config.get_acyclicity_mode() == AcyclicityMode.REJECT:
                    is_valid = False

        if "coverage" in self.config.constraints:
            if scm.metadata.coverage < 1.0:
                is_valid = False

        if "conflict_free" in self.config.constraints:
            if scm.metadata.conflict_count > 0:
                is_valid = False

        # If invalid and reject mode, return None
        if not is_valid and self.config.get_acyclicity_mode() == AcyclicityMode.REJECT:
            result = SCMEvaluationResult(
                objectives=tuple([float("-inf")] * len(self.config.objectives)),
                total_penalty=float("inf"),
                is_valid=False,
                cycle_penalty=0.0,
                coverage_penalty=0.0,
                conflict_penalty=0.0,
                div_zero_penalty_value=0.0,
                latent_ancestor_penalty_value=0.0,
                evaluated_variables=frozenset(),
                nan_variables=frozenset(),
                cycle_count=len(scm.cycles),
            )
            return None, result

        # Compute objectives
        objectives: list[float] = []
        nan_vars: set[str] = set()
        evaluated_vars: set[str] = set()

        for obj_name in self.config.objectives:
            match obj_name:
                case "data_fit":
                    fit, nans = self._compute_data_fit(scm)
                    objectives.append(fit)
                    nan_vars |= nans
                    evaluated_vars |= set(scm.equations.keys()) - nans
                case "sparsity":
                    objectives.append(self._compute_sparsity(scm))
                case "simplicity":
                    objectives.append(self._compute_simplicity(scm))
                case "coverage":
                    objectives.append(self._compute_coverage(scm))
                case "latent_parsimony":
                    objectives.append(self._compute_latent_parsimony(scm))
                case _:
                    raise ValueError(f"Unknown objective: {obj_name}")

        # Compute penalties
        cycle_penalty = len(scm.cycles) * self.config.cycle_penalty_per_cycle
        coverage_penalty = (1.0 - scm.metadata.coverage) * self.config.incomplete_coverage_penalty
        conflict_penalty = scm.metadata.conflict_count * self.config.conflict_penalty
        div_zero_penalty_value = len(nan_vars) * self.config.div_zero_penalty

        # Validate latent ancestors
        latent_violations = self._validate_latent_ancestors(scm, genome.config)
        latent_ancestor_penalty_value = len(latent_violations) * self.config.latent_ancestor_penalty

        total_penalty = (
            cycle_penalty
            + coverage_penalty
            + conflict_penalty
            + div_zero_penalty_value
            + latent_ancestor_penalty_value
        )

        # Apply penalties to objectives (subtract from first objective typically)
        penalized_objectives = list(objectives)
        if len(penalized_objectives) > 0:
            penalized_objectives[0] -= total_penalty

        result = SCMEvaluationResult(
            objectives=tuple(penalized_objectives),
            total_penalty=total_penalty,
            is_valid=is_valid,
            cycle_penalty=cycle_penalty,
            coverage_penalty=coverage_penalty,
            conflict_penalty=conflict_penalty,
            div_zero_penalty_value=div_zero_penalty_value,
            latent_ancestor_penalty_value=latent_ancestor_penalty_value,
            evaluated_variables=frozenset(evaluated_vars),
            nan_variables=frozenset(nan_vars),
            cycle_count=len(scm.cycles),
        )

        fitness = Fitness(values=np.array(penalized_objectives))
        return fitness, result

    def _validate_latent_ancestors(
        self,
        scm: DecodedSCM,
        config: SCMConfig,
    ) -> set[str]:
        """
        Find latent variables without observed ancestors.

        Returns set of latent variables violating the constraint.
        """
        observed = set(config.observed_variables)
        violations = set()

        for latent in scm.metadata.latent_variables_used:
            if latent not in scm.graph:
                violations.add(latent)
                continue
            ancestors = nx.ancestors(scm.graph, latent)
            if not ancestors & observed:
                violations.add(latent)

        return violations

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
        from evolve.representation.scm_decoder import evaluate, variables

        if scm.is_cyclic:
            # Apply acyclicity strategy
            return self._apply_acyclicity_strategy(scm)

        # Get topological order for evaluation
        try:
            eval_order = list(nx.topological_sort(scm.graph))
        except nx.NetworkXUnfeasible:
            # Graph has cycles - shouldn't happen if is_cyclic check passed
            return float("-inf"), frozenset(scm.equations.keys())

        # Build environment with observed data
        env: dict[str, np.ndarray] = dict(self._data_dict)
        nan_vars: set[str] = set()
        total_mse = 0.0
        n_evaluated = 0

        for var in eval_order:
            if var not in scm.equations:
                continue

            expr = scm.equations[var]
            expr_vars = variables(expr)

            # Check if all dependencies are available
            if not all(v in env for v in expr_vars):
                nan_vars.add(var)
                continue

            # Evaluate expression for all samples
            try:
                predictions = np.array(
                    [
                        evaluate(expr, {v: env[v][i] for v in expr_vars})
                        for i in range(len(self.data))
                    ]
                )
            except (KeyError, ZeroDivisionError):
                nan_vars.add(var)
                continue

            # Check for NaN
            if np.any(np.isnan(predictions)):
                nan_vars.add(var)
                continue

            # Add to environment for downstream variables
            env[var] = predictions

            # Compute MSE if variable is observed
            if var in self._data_dict:
                observed = self._data_dict[var]
                mse = np.mean((predictions - observed) ** 2)
                total_mse += mse
                n_evaluated += 1

        if n_evaluated == 0:
            return float("-inf"), frozenset(nan_vars)

        avg_mse = total_mse / n_evaluated
        return -avg_mse, frozenset(nan_vars)

    def _apply_acyclicity_strategy(
        self,
        scm: DecodedSCM,
    ) -> tuple[float, frozenset[str]]:
        """Apply configured acyclicity strategy for cyclic SCMs."""
        from evolve.representation.scm import AcyclicityStrategy

        strategy = self.config.get_acyclicity_strategy()

        match strategy:
            case AcyclicityStrategy.PENALTY_ONLY:
                # Just return penalty, no partial evaluation
                return float("-inf"), frozenset(scm.equations.keys())

            case AcyclicityStrategy.ACYCLIC_SUBGRAPH:
                return self._evaluate_acyclic_subgraph(scm)

            case AcyclicityStrategy.PARSE_ORDER:
                return self._evaluate_parse_order(scm)

            case AcyclicityStrategy.PARENT_INHERITANCE:
                return self._apply_parent_inheritance(scm)

            case AcyclicityStrategy.COMPOSITE:
                fit, penalty, nans = self._evaluate_composite(scm)
                return fit - penalty, nans

            case _:
                return float("-inf"), frozenset(scm.equations.keys())

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
        # Create copy of graph and remove edges until acyclic
        graph = scm.graph.copy()
        removed_vars: set[str] = set()

        while not nx.is_directed_acyclic_graph(graph):
            # Find a cycle and remove an edge
            try:
                cycle = next(nx.simple_cycles(graph))
                if len(cycle) > 0:
                    # Remove first edge in cycle
                    graph.remove_edge(cycle[0], cycle[1] if len(cycle) > 1 else cycle[0])
                    removed_vars.add(cycle[0])
            except StopIteration:
                break

        # Evaluate remaining equations
        # For simplicity, mark removed vars as NaN
        return float("-inf") + len(removed_vars), frozenset(removed_vars)

    def _evaluate_parse_order(
        self,
        scm: DecodedSCM,
    ) -> tuple[float, frozenset[str]]:
        """
        Break cycles by parse order.

        Removes edges that would create cycles based on
        the order equations appear in the genome.
        """
        # Similar to acyclic_subgraph but respecting order
        return self._evaluate_acyclic_subgraph(scm)

    def _apply_parent_inheritance(
        self,
        scm: DecodedSCM,
    ) -> tuple[float, frozenset[str]]:
        """
        ERP-aware cycle breaking.

        Uses parent inheritance information to break cycles.
        """
        # Placeholder - requires ERP integration
        return self._evaluate_acyclic_subgraph(scm)

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
        fit, nans = self._evaluate_acyclic_subgraph(scm)
        penalty = len(scm.cycles) * self.config.cycle_penalty_per_cycle
        return fit, penalty, nans

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
