"""
SCM (Structural Causal Model) Decoder.

This module provides the decoder for transforming SCM genomes into
decoded structural causal models with equations and causal graphs.

Key components:
- Expression AST: Var, Const, BinOp for representing equations
- Expression utilities: complexity, variables, evaluate, to_string
- SCMMetadata: Metadata from decoding process
- DecodedSCM: Decoded phenotype with equations and graph
- SCMDecoder: Stack machine decoder implementing Decoder protocol

Example:
    >>> from evolve.representation.scm_decoder import SCMDecoder, Var, BinOp
    >>> decoder = SCMDecoder(config)
    >>> scm = decoder.decode(genome)
    >>> print(scm.equations)  # {"C": BinOp("+", Var("A"), Var("B"))}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import networkx as nx

if TYPE_CHECKING:
    from evolve.representation.scm import SCMConfig, SCMGenome


__all__ = [
    # Expression AST
    "Var",
    "Const",
    "BinOp",
    "Expression",
    # Expression utilities
    "complexity",
    "variables",
    "evaluate",
    "to_string",
    "expr_to_dict",
    "expr_from_dict",
    # Decoded SCM
    "SCMMetadata",
    "DecodedSCM",
    # Decoder
    "SCMDecoder",
]


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
    r"""
    Binary operation in expression.

    Represents arithmetic operations: +, -, \*, /
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
    match expr:
        case Var(_):
            return 1
        case Const(_):
            return 1
        case BinOp(_, left, right):
            return 1 + complexity(left) + complexity(right)


def variables(expr: Expression) -> frozenset[str]:
    """
    Extract all variable names from expression.

    Returns set of variable names referenced in expression.
    Does not include constants.
    """
    match expr:
        case Var(name):
            return frozenset({name})
        case Const(_):
            return frozenset()
        case BinOp(_, left, right):
            return variables(left) | variables(right)


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
    match expr:
        case Var(name):
            return env[name]
        case Const(value):
            return value
        case BinOp(op, left, right):
            left_val = evaluate(left, env)
            right_val = evaluate(right, env)
            match op:
                case "+":
                    return left_val + right_val
                case "-":
                    return left_val - right_val
                case "*":
                    return left_val * right_val
                case "/":
                    if right_val == 0:
                        return math.nan
                    return left_val / right_val
                case _:
                    raise ValueError(f"Unknown operator: {op}")


def to_string(expr: Expression) -> str:
    """
    Convert expression to human-readable infix string.

    Examples:
        to_string(BinOp("+", Var("A"), Var("B"))) == "(A + B)"
        to_string(Var("X")) == "X"
    """
    match expr:
        case Var(name):
            return name
        case Const(value):
            return str(value)
        case BinOp(op, left, right):
            return f"({to_string(left)} {op} {to_string(right)})"


def expr_to_dict(expr: Expression) -> dict[str, Any]:
    """Convert expression to JSON-serializable dict."""
    match expr:
        case Var(name):
            return {"type": "Var", "name": name}
        case Const(value):
            return {"type": "Const", "value": value}
        case BinOp(op, left, right):
            return {
                "type": "BinOp",
                "op": op,
                "left": expr_to_dict(left),
                "right": expr_to_dict(right),
            }


def expr_from_dict(data: dict[str, Any]) -> Expression:
    """Reconstruct expression from dict."""
    expr_type = data["type"]
    match expr_type:
        case "Var":
            return Var(name=data["name"])
        case "Const":
            return Const(value=data["value"])
        case "BinOp":
            return BinOp(
                op=data["op"],
                left=expr_from_dict(data["left"]),
                right=expr_from_dict(data["right"]),
            )
        case _:
            raise ValueError(f"Unknown expression type: {expr_type}")


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
        all_vars = frozenset(self.graph.nodes())
        return all_vars - self.endogenous_variables

    @property
    def edge_count(self) -> int:
        """Number of edges in causal graph."""
        return cast(int, self.graph.number_of_edges())

    @property
    def total_complexity(self) -> int:
        """Sum of AST complexity across all equations."""
        return sum(complexity(expr) for expr in self.equations.values())

    @property
    def junk_fraction(self) -> float:
        """Fraction of genes that were junk."""
        total_genes = len(self.metadata.junk_gene_indices)
        if total_genes == 0:
            return 0.0
        # This is approximate - would need genome length for exact calculation
        return 0.0  # Placeholder - actual calculation needs genome length

    # === Serialization ===

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serializable dict.

        Note: The NetworkX graph is not serialized; it can be
        reconstructed from equations.
        """
        return {
            "type": "DecodedSCM",
            "version": "1.0",
            "equations": {name: expr_to_dict(expr) for name, expr in self.equations.items()},
            "metadata": {
                "conflict_count": self.metadata.conflict_count,
                "junk_gene_indices": list(self.metadata.junk_gene_indices),
                "is_cyclic": self.metadata.is_cyclic,
                "cycles": [list(c) for c in self.metadata.cycles],
                "latent_variables_used": list(self.metadata.latent_variables_used),
                "coverage": self.metadata.coverage,
                "conflicts": self.metadata.conflicts,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecodedSCM:
        """
        Reconstruct from serialized dict.

        Rebuilds the NetworkX graph from equations.
        """
        if data.get("type") != "DecodedSCM":
            raise ValueError(f"Expected type 'DecodedSCM', got {data.get('type')}")

        equations = {
            name: expr_from_dict(expr_data) for name, expr_data in data["equations"].items()
        }

        # Rebuild graph from equations
        graph = nx.DiGraph()
        for target, expr in equations.items():
            graph.add_node(target)
            deps = variables(expr)
            for dep in deps:
                graph.add_node(dep)
                graph.add_edge(dep, target)

        meta = data["metadata"]
        metadata = SCMMetadata(
            conflict_count=meta["conflict_count"],
            junk_gene_indices=tuple(meta["junk_gene_indices"]),
            is_cyclic=meta["is_cyclic"],
            cycles=tuple(tuple(c) for c in meta["cycles"]),
            latent_variables_used=frozenset(meta["latent_variables_used"]),
            coverage=meta["coverage"],
            conflicts=meta["conflicts"],
        )

        return cls(equations=equations, graph=graph, metadata=metadata)


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
        from evolve.representation.scm import SCMAlphabet

        self._alphabet = SCMAlphabet.from_config(config)

    def decode(self, genome: SCMGenome) -> DecodedSCM:
        """
        Decode genome to SCM via stack machine execution.

        Algorithm:
        1. Execute stack machine to extract equations
        2. Apply conflict resolution
        3. Build causal graph
        4. Detect cycles
        5. Compile metadata

        Args:
            genome: SCMGenome to decode

        Returns:
            DecodedSCM with equations, graph, and metadata
        """
        # Build ERC lookup
        erc_lookup = dict(genome.erc_values)

        # Execute stack machine
        raw_equations, junk_indices = self._execute_stack_machine(genome.genes, erc_lookup)

        # Resolve conflicts
        equations, conflicts, conflict_junk = self._resolve_conflicts(raw_equations)
        junk_indices = sorted(set(junk_indices) | set(conflict_junk))

        # Build graph
        graph = self._build_graph(equations)

        # Detect cycles
        is_cyclic, cycles = self._detect_cycles(graph)

        # Find latent variables used
        latent_used = set()
        for var in equations:
            if var.startswith("H"):
                latent_used.add(var)
        for expr in equations.values():
            for var in variables(expr):
                if var.startswith("H"):
                    latent_used.add(var)

        # Compute coverage
        observed_with_eq = sum(1 for var in self.config.observed_variables if var in equations)
        coverage = observed_with_eq / len(self.config.observed_variables)

        # Compile metadata
        conflict_count = sum(1 for c in conflicts.values() if c > 1)
        metadata = SCMMetadata(
            conflict_count=conflict_count,
            junk_gene_indices=tuple(junk_indices),
            is_cyclic=is_cyclic,
            cycles=tuple(tuple(c) for c in cycles),
            latent_variables_used=frozenset(latent_used),
            coverage=coverage,
            conflicts=conflicts,
        )

        return DecodedSCM(
            equations=equations,
            graph=graph,
            metadata=metadata,
        )

    def _execute_stack_machine(
        self,
        genes: tuple[str | float, ...],
        erc_values: dict[int, float],
    ) -> tuple[dict[str, list[Expression]], list[int]]:
        """
        Execute genes on stack machine to extract equations.

        Stack Machine Algorithm:
        ========================
        The decoder processes genes left-to-right, maintaining:
        - A stack of Expression objects (operands and sub-expressions)
        - A dictionary of raw equations {variable -> [expressions]}
        - A list of "junk" gene indices (invalid operations)

        Gene Processing Rules:
        1. NUMERIC/CONSTANT: Push Const(value) onto stack
        2. ERC_n: Push Const(erc_values[n]) if slot exists, else mark junk
        3. VARIABLE (A, B, H1, etc.): Push Var(name) onto stack
        4. OPERATOR (+, -, *, /): Pop 2 operands, push BinOp result
           - If stack has < 2 items: mark operator as junk (underflow)
        5. STORE_X: Pop expression, emit equation X = expr
           - If stack empty: mark as junk

        Junk Handling:
        - Genes that can't execute (underflow, unknown) are marked as junk
        - Junk genes are skipped; decoding continues with remaining genes
        - This "silent junk" strategy allows evolution to explore freely

        Returns:
            Tuple of:
            - Raw equations (before conflict resolution): {var: [expr, ...]}
            - Junk gene indices (for metadata/debugging)
        """
        stack: list[Expression] = []
        raw_equations: dict[str, list[Expression]] = {}
        junk_indices: list[int] = []

        for i, gene in enumerate(genes):
            # Handle numeric gene (constant or ERC value)
            if isinstance(gene, int | float):
                stack.append(Const(float(gene)))
                continue

            # Handle string gene
            gene_str = str(gene)

            # Check for ERC slot
            if gene_str.startswith("ERC_"):
                if i in erc_values:
                    stack.append(Const(erc_values[i]))
                else:
                    # ERC without value - treat as junk
                    junk_indices.append(i)
                continue

            # Check for constant (string representation)
            try:
                const_val = float(gene_str)
                stack.append(Const(const_val))
                continue
            except ValueError:
                pass

            # Check for variable reference
            if gene_str in self._alphabet.variable_refs:
                stack.append(Var(gene_str))
                continue

            # Check for operator
            if gene_str in self._alphabet.operators:
                if len(stack) < 2:
                    # Underflow - mark as junk
                    junk_indices.append(i)
                    continue
                right = stack.pop()
                left = stack.pop()
                stack.append(BinOp(gene_str, left, right))
                continue

            # Check for STORE gene
            if gene_str.startswith("STORE_"):
                var_name = gene_str[6:]  # Remove "STORE_" prefix
                if len(stack) == 0:
                    # Empty stack - mark as junk
                    junk_indices.append(i)
                    continue
                expr = stack.pop()
                if var_name not in raw_equations:
                    raw_equations[var_name] = []
                raw_equations[var_name].append(expr)
                continue

            # Unknown gene - mark as junk
            junk_indices.append(i)

        return raw_equations, junk_indices

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
        from evolve.representation.scm import ConflictResolution

        equations: dict[str, Expression] = {}
        conflicts: dict[str, int] = {}
        junk_indices: list[int] = []

        for var, exprs in raw_equations.items():
            conflicts[var] = len(exprs)

            if len(exprs) == 1:
                equations[var] = exprs[0]
            elif len(exprs) > 1:
                match self.config.conflict_resolution:
                    case ConflictResolution.FIRST_WINS:
                        equations[var] = exprs[0]
                    case ConflictResolution.LAST_WINS:
                        equations[var] = exprs[-1]
                    case ConflictResolution.ALL_JUNK:
                        # Don't add any equation
                        pass

        return equations, conflicts, junk_indices

    def _build_graph(
        self,
        equations: dict[str, Expression],
    ) -> nx.DiGraph:
        """
        Build causal graph from equations.

        Creates edge from each RHS variable to LHS variable.
        """
        graph = nx.DiGraph()

        # Add all variables as nodes
        for var in equations:
            graph.add_node(var)

        # Add edges from dependencies
        for var, expr in equations.items():
            deps = variables(expr)
            for dep in deps:
                graph.add_node(dep)  # Ensure dependency node exists
                graph.add_edge(dep, var)

        return graph

    def _detect_cycles(
        self,
        graph: nx.DiGraph,
    ) -> tuple[bool, list[tuple[str, ...]]]:
        """
        Check for cycles and enumerate them.

        Cycle Detection Algorithm:
        ==========================
        Uses NetworkX's optimized graph algorithms:

        1. Quick Check (O(V+E)):
           - nx.is_directed_acyclic_graph uses DFS-based topological sort
           - Returns immediately if graph is DAG (most common case)

        2. Cycle Enumeration (only if cyclic):
           - nx.simple_cycles uses Johnson's algorithm
           - Finds all elementary circuits in directed graph
           - Time complexity: O((V+E)(C+1)) where C = number of cycles

        Note: Self-loops (A -> A) are detected as cycles of length 1.

        Returns:
            Tuple of (is_cyclic, list of cycles as variable tuples)
        """
        is_acyclic = nx.is_directed_acyclic_graph(graph)

        if is_acyclic:
            return False, []

        # Enumerate cycles
        cycles = list(nx.simple_cycles(graph))
        return True, [tuple(c) for c in cycles]

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
        violations = set()

        for latent in latent_used:
            if latent not in graph:
                continue
            ancestors = nx.ancestors(graph, latent)
            if not ancestors & observed:
                violations.add(latent)

        return violations
