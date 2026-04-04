"""
SCM Decoder API Contract

This file defines the interface contracts for SCMDecoder, DecodedSCM,
and Expression AST types. Implementation MUST satisfy all type signatures
and documented behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

    from .scm_genome import SCMConfig, SCMGenome


# === Expression AST ===


@dataclass(frozen=True)
class Var:
    """
    Variable reference in expression.

    Represents a reference to a variable (observed or latent)
    in an SCM equation's right-hand side.
    """

    name: str


@dataclass(frozen=True)
class Const:
    """
    Numeric constant in expression.

    Represents a literal numeric value, either from the
    standard constant set or an ERC.
    """

    value: float


@dataclass(frozen=True)
class BinOp:
    """
    Binary operation in expression.

    Represents arithmetic operations: +, -, *, /
    """

    op: str  # One of: '+', '-', '*', '/'
    left: Expression
    right: Expression


# Type alias for expression tree
Expression = Var | Const | BinOp


# === Expression Utilities ===


def complexity(expr: Expression) -> int:
    """
    Count total AST nodes.

    Used for simplicity objective (lower is better).

    Examples:
        complexity(Var("A")) == 1
        complexity(BinOp("+", Var("A"), Const(1))) == 3
    """
    ...


def variables(expr: Expression) -> frozenset[str]:
    """
    Extract all variable names from expression.

    Returns set of variable names referenced in expression.
    Does not include constants.
    """
    ...


def evaluate(expr: Expression, env: dict[str, float]) -> float:
    """
    Evaluate expression given variable assignments.

    Args:
        expr: Expression to evaluate
        env: Mapping from variable name to value

    Returns:
        Numeric result (may be NaN for undefined operations)

    Raises:
        KeyError: If required variable not in env
    """
    ...


def to_string(expr: Expression) -> str:
    """
    Convert expression to human-readable infix string.

    Examples:
        to_string(BinOp("+", Var("A"), Var("B"))) == "(A + B)"
        to_string(Var("X")) == "X"
    """
    ...


def expr_to_dict(expr: Expression) -> dict:
    """Convert expression to JSON-serializable dict."""
    ...


def expr_from_dict(data: dict) -> Expression:
    """Reconstruct expression from dict."""
    ...


# === Decoded SCM ===


@dataclass(frozen=True)
class SCMMetadata:
    """
    Metadata from decoding process.

    Contains information about junk genes, conflicts, cycles,
    and variable coverage for fitness evaluation and debugging.
    """

    conflict_count: int
    """Number of variables with multiple equations (before resolution)."""

    junk_gene_indices: tuple[int, ...]
    """Indices of genes that didn't contribute to final equations."""

    is_cyclic: bool
    """Whether the causal graph contains cycles."""

    cycles: tuple[tuple[str, ...], ...]
    """List of cycles as variable name tuples (empty if acyclic)."""

    latent_variables_used: frozenset[str]
    """Latent variables that appear in equations."""

    coverage: float
    """Fraction of observed variables with equations (0.0 to 1.0)."""

    conflicts: dict[str, int]
    """Map from variable to number of conflicting definitions."""


@dataclass(frozen=True)
class DecodedSCM:
    """
    Decoded Structural Causal Model.

    Contains structural equations, causal graph, and metadata
    from the decoding process.

    Invariants:
        - equations.keys() are variables with definitions
        - graph.nodes() includes all referenced variables
        - graph.edges() match equation dependencies
        - metadata accurately reflects decoding results
    """

    equations: dict[str, Expression]
    """Mapping from variable name to its defining expression."""

    graph: nx.DiGraph
    """
    Causal graph where edges point from cause to effect.
    
    For equation X = f(A, B), edges are: A → X, B → X
    
    Compatible with NetworkX and DoWhy.
    """

    metadata: SCMMetadata
    """Decoding metadata for evaluation and debugging."""

    # === Properties ===

    @property
    def is_cyclic(self) -> bool:
        """Whether the causal graph contains cycles."""
        return self.metadata.is_cyclic

    @property
    def cycles(self) -> tuple[tuple[str, ...], ...]:
        """Detected cycles in the graph."""
        return self.metadata.cycles

    @property
    def endogenous_variables(self) -> frozenset[str]:
        """Variables with equations (have incoming edges)."""
        return frozenset(self.equations.keys())

    @property
    def exogenous_variables(self) -> frozenset[str]:
        """Variables without equations (no incoming edges)."""
        ...

    @property
    def edge_count(self) -> int:
        """Number of edges in causal graph."""
        return self.graph.number_of_edges()

    @property
    def total_complexity(self) -> int:
        """Sum of AST complexity across all equations."""
        return sum(complexity(expr) for expr in self.equations.values())

    @property
    def junk_fraction(self) -> float:
        """Fraction of genes that were junk."""
        ...


# === Decoder ===


class SCMDecoder:
    """
    Stack machine decoder for SCM genomes.

    Implements Decoder[SCMGenome, DecodedSCM] protocol.

    The decoder is stateless - all configuration comes from
    the genome's SCMConfig. This enables safe concurrent use.

    Decoding is deterministic: the same genome always produces
    the same DecodedSCM (no random number generation).
    """

    def __init__(self, config: SCMConfig) -> None:
        """
        Initialize decoder with configuration.

        Args:
            config: SCM configuration for decoding behavior
        """
        self.config = config

    def decode(self, genome: SCMGenome) -> DecodedSCM:
        """
        Decode genome to SCM via stack machine execution.

        Algorithm:
        1. Execute stack machine to extract equations
        2. Apply conflict resolution
        3. Build causal graph
        4. Detect cycles
        5. Validate latent variable constraints
        6. Compile metadata

        Args:
            genome: SCMGenome to decode

        Returns:
            DecodedSCM with equations, graph, and metadata

        Guarantees:
            - Deterministic (no RNG)
            - Total function (always returns, never raises for valid genome)
            - O(n) complexity in genome length for decoding
            - O(V+E) for cycle detection
        """
        ...

    def _execute_stack_machine(
        self,
        genes: tuple[str | float, ...],
        erc_values: dict[int, float],
    ) -> tuple[dict[str, list[Expression]], list[int]]:
        """
        Execute genes on stack machine.

        Returns:
            Tuple of:
            - Raw equations (before conflict resolution): {var: [expr, ...]}
            - Junk gene indices
        """
        ...

    def _resolve_conflicts(
        self,
        raw_equations: dict[str, list[Expression]],
    ) -> tuple[dict[str, Expression], dict[str, int], list[int]]:
        """
        Apply conflict resolution strategy.

        Returns:
            Tuple of:
            - Final equations: {var: expr}
            - Conflict counts: {var: count}
            - Additional junk indices from resolution
        """
        ...

    def _build_graph(
        self,
        equations: dict[str, Expression],
    ) -> nx.DiGraph:
        """
        Build causal graph from equations.

        Creates edge from each RHS variable to LHS variable.
        """
        ...

    def _detect_cycles(
        self,
        graph: nx.DiGraph,
    ) -> tuple[bool, list[tuple[str, ...]]]:
        """
        Check for cycles and enumerate them.

        Uses NetworkX algorithms:
        - nx.is_directed_acyclic_graph for check
        - nx.simple_cycles for enumeration (if cyclic)
        """
        ...

    def _validate_latent_ancestors(
        self,
        graph: nx.DiGraph,
        latent_used: set[str],
        observed: set[str],
    ) -> set[str]:
        """
        Find latent variables without observed ancestors.

        Returns set of latent variables violating the constraint.
        """
        ...
