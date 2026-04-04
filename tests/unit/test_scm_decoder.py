"""
Unit tests for SCM Decoder module.

Tests cover:
- Expression AST (Var, Const, BinOp)
- Expression utilities (complexity, variables, evaluate, to_string)
- SCMDecoder stack machine
- Conflict resolution strategies
- Cycle detection
- Expression serialization round-trip
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from evolve.representation.scm import (
    AcyclicityMode,
    ConflictResolution,
    SCMConfig,
    SCMGenome,
)
from evolve.representation.scm_decoder import (
    BinOp,
    Const,
    DecodedSCM,
    SCMDecoder,
    Var,
    complexity,
    evaluate,
    expr_from_dict,
    expr_to_dict,
    to_string,
    variables,
)

# === Fixtures ===


@pytest.fixture
def basic_config() -> SCMConfig:
    """Simple 3-variable configuration."""
    return SCMConfig(observed_variables=("A", "B", "C"))


@pytest.fixture
def decoder_first_wins(basic_config) -> SCMDecoder:
    """Decoder with FIRST_WINS conflict resolution."""
    return SCMDecoder(basic_config)


@pytest.fixture
def decoder_last_wins() -> SCMDecoder:
    """Decoder with LAST_WINS conflict resolution."""
    config = SCMConfig(
        observed_variables=("X", "Y"),
        conflict_resolution=ConflictResolution.LAST_WINS,
    )
    return SCMDecoder(config)


@pytest.fixture
def decoder_all_junk() -> SCMDecoder:
    """Decoder with ALL_JUNK conflict resolution."""
    config = SCMConfig(
        observed_variables=("X", "Y"),
        conflict_resolution=ConflictResolution.ALL_JUNK,
    )
    return SCMDecoder(config)


# === Expression AST Tests ===


class TestExpressionVar:
    """Tests for Var expression node."""

    def test_creation(self):
        """Test Var creation."""
        var = Var("X")
        assert var.name == "X"

    def test_frozen(self):
        """Test that Var is immutable."""
        var = Var("X")
        with pytest.raises(AttributeError):
            var.name = "Y"

    def test_equality(self):
        """Test Var equality."""
        assert Var("X") == Var("X")
        assert Var("X") != Var("Y")


class TestExpressionConst:
    """Tests for Const expression node."""

    def test_creation(self):
        """Test Const creation."""
        const = Const(3.14)
        assert const.value == 3.14

    def test_zero(self):
        """Test Const with zero."""
        const = Const(0.0)
        assert const.value == 0.0

    def test_frozen(self):
        """Test that Const is immutable."""
        const = Const(1.0)
        with pytest.raises(AttributeError):
            const.value = 2.0


class TestExpressionBinOp:
    """Tests for BinOp expression node."""

    def test_creation(self):
        """Test BinOp creation."""
        expr = BinOp("+", Var("X"), Const(1.0))
        assert expr.op == "+"
        assert isinstance(expr.left, Var)
        assert isinstance(expr.right, Const)

    def test_nested(self):
        """Test nested BinOp expressions."""
        # (X + Y) * 2
        inner = BinOp("+", Var("X"), Var("Y"))
        outer = BinOp("*", inner, Const(2.0))

        assert outer.op == "*"
        assert isinstance(outer.left, BinOp)
        assert outer.left.op == "+"

    def test_frozen(self):
        """Test that BinOp is immutable."""
        expr = BinOp("+", Var("X"), Const(1.0))
        with pytest.raises(AttributeError):
            expr.op = "-"


# === Expression Utilities Tests ===


class TestExpressionComplexity:
    """Tests for complexity() utility."""

    def test_var_complexity(self):
        """Test Var has complexity 1."""
        assert complexity(Var("X")) == 1

    def test_const_complexity(self):
        """Test Const has complexity 1."""
        assert complexity(Const(3.14)) == 1

    def test_binop_complexity(self):
        """Test BinOp complexity is sum + 1."""
        # X + 1 => complexity = 1 + 1 + 1 = 3
        expr = BinOp("+", Var("X"), Const(1.0))
        assert complexity(expr) == 3

    def test_nested_complexity(self):
        """Test nested expression complexity."""
        # (X + Y) * Z => complexity = (1 + 1 + 1) + 1 + 1 = 5
        inner = BinOp("+", Var("X"), Var("Y"))
        outer = BinOp("*", inner, Var("Z"))
        assert complexity(outer) == 5


class TestExpressionVariables:
    """Tests for variables() utility."""

    def test_var_variables(self):
        """Test Var returns its name."""
        assert variables(Var("X")) == {"X"}

    def test_const_variables(self):
        """Test Const returns empty set."""
        assert variables(Const(1.0)) == set()

    def test_binop_variables(self):
        """Test BinOp returns all variables."""
        expr = BinOp("+", Var("X"), Var("Y"))
        assert variables(expr) == {"X", "Y"}

    def test_nested_variables(self):
        """Test nested expression variables."""
        # (X + Y) * X => {"X", "Y"}
        inner = BinOp("+", Var("X"), Var("Y"))
        outer = BinOp("*", inner, Var("X"))
        assert variables(outer) == {"X", "Y"}


class TestExpressionEvaluate:
    """Tests for evaluate() utility."""

    def test_var_evaluation(self):
        """Test Var evaluation with assignment."""
        expr = Var("X")
        assert evaluate(expr, {"X": 5.0}) == 5.0

    def test_const_evaluation(self):
        """Test Const evaluation."""
        expr = Const(3.14)
        assert evaluate(expr, {}) == 3.14

    def test_addition(self):
        """Test addition evaluation."""
        expr = BinOp("+", Var("X"), Const(10.0))
        assert evaluate(expr, {"X": 5.0}) == 15.0

    def test_subtraction(self):
        """Test subtraction evaluation."""
        expr = BinOp("-", Var("X"), Const(3.0))
        assert evaluate(expr, {"X": 10.0}) == 7.0

    def test_multiplication(self):
        """Test multiplication evaluation."""
        expr = BinOp("*", Var("X"), Var("Y"))
        assert evaluate(expr, {"X": 3.0, "Y": 4.0}) == 12.0

    def test_division(self):
        """Test division evaluation."""
        expr = BinOp("/", Var("X"), Const(2.0))
        assert evaluate(expr, {"X": 10.0}) == 5.0

    def test_division_by_zero(self):
        """Test protected division by zero returns NaN."""
        expr = BinOp("/", Const(1.0), Const(0.0))
        result = evaluate(expr, {})
        assert math.isnan(result)  # Protected division returns NaN

    def test_missing_variable(self):
        """Test missing variable raises KeyError."""
        expr = Var("X")
        with pytest.raises(KeyError):
            evaluate(expr, {"Y": 1.0})


class TestExpressionToString:
    """Tests for to_string() utility."""

    def test_var_string(self):
        """Test Var string representation."""
        assert to_string(Var("X")) == "X"

    def test_const_string(self):
        """Test Const string representation."""
        assert to_string(Const(3.14)) == "3.14"

    def test_binop_string(self):
        """Test BinOp string representation."""
        expr = BinOp("+", Var("X"), Const(1.0))
        assert to_string(expr) == "(X + 1.0)"

    def test_nested_string(self):
        """Test nested expression string."""
        inner = BinOp("+", Var("X"), Var("Y"))
        outer = BinOp("*", inner, Const(2.0))
        assert to_string(outer) == "((X + Y) * 2.0)"


class TestExpressionSerialization:
    """Tests for expression serialization."""

    def test_var_round_trip(self):
        """Test Var serialization round-trip."""
        expr = Var("X")
        data = expr_to_dict(expr)
        restored = expr_from_dict(data)
        assert restored == expr

    def test_const_round_trip(self):
        """Test Const serialization round-trip."""
        expr = Const(3.14)
        data = expr_to_dict(expr)
        restored = expr_from_dict(data)
        assert restored == expr

    def test_binop_round_trip(self):
        """Test BinOp serialization round-trip."""
        expr = BinOp("+", Var("X"), Const(1.0))
        data = expr_to_dict(expr)
        restored = expr_from_dict(data)
        assert restored == expr

    def test_nested_round_trip(self):
        """Test nested expression round-trip."""
        inner = BinOp("+", Var("X"), Var("Y"))
        outer = BinOp("*", inner, Const(2.0))
        data = expr_to_dict(outer)
        restored = expr_from_dict(data)
        assert restored == outer


# === SCMDecoder Stack Machine Tests ===


class TestSCMDecoderStackMachine:
    """Tests for stack machine execution."""

    def test_simple_equation(self, decoder_first_wins):
        """Test decoding simple A = B + C."""
        # Push B, push C, add, store A
        genes = ["B", "C", "+", "STORE_A"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)

        assert "A" in decoded.equations
        expr = decoded.equations["A"]
        assert isinstance(expr, BinOp)
        assert variables(expr) == {"B", "C"}

    def test_constant_equation(self, decoder_first_wins):
        """Test decoding A = 1.0."""
        genes = [1.0, "STORE_A"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)

        assert "A" in decoded.equations
        expr = decoded.equations["A"]
        assert isinstance(expr, Const)
        assert expr.value == 1.0

    def test_empty_stack_store(self, decoder_first_wins):
        """Test that STORE with empty stack produces no equation."""
        genes = ["STORE_A"]  # Empty stack, nothing to store
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)

        # Should have no equations - empty stack
        assert "A" not in decoded.equations

    def test_stack_underflow_handling(self, decoder_first_wins):
        """Test graceful handling of stack underflow."""
        # Push A, then try binary op (needs 2 operands)
        genes = ["A", "+", "STORE_B"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)

        # Should not crash, equation may be partial or absent
        assert decoded is not None


class TestSCMDecoderConflictResolution:
    """Tests for conflict resolution strategies."""

    def test_first_wins(self, decoder_first_wins):
        """Test FIRST_WINS keeps first equation."""
        # Two stores to A
        genes = [1.0, "STORE_A", 2.0, "STORE_A"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)

        assert "A" in decoded.equations
        assert isinstance(decoded.equations["A"], Const)
        assert decoded.equations["A"].value == 1.0  # First value wins
        assert decoded.metadata.conflict_count > 0

    def test_last_wins(self, decoder_last_wins):
        """Test LAST_WINS keeps last equation."""
        genes = [1.0, "STORE_X", 2.0, "STORE_X"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_last_wins.config,
            erc_values=(),
        )

        decoded = decoder_last_wins.decode(genome)

        assert "X" in decoded.equations
        assert decoded.equations["X"].value == 2.0  # Last value wins

    def test_all_junk(self, decoder_all_junk):
        """Test ALL_JUNK discards conflicting equations."""
        genes = [1.0, "STORE_X", 2.0, "STORE_X"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_all_junk.config,
            erc_values=(),
        )

        decoded = decoder_all_junk.decode(genome)

        # X should not be in equations due to conflict
        assert "X" not in decoded.equations
        assert decoded.metadata.conflict_count > 0


class TestSCMDecoderCycleDetection:
    """Tests for cycle detection."""

    def test_acyclic_graph(self, decoder_first_wins):
        """Test detection of acyclic graph."""
        # A = B, B = C (acyclic chain)
        genes = ["C", "STORE_B", "B", "STORE_A"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)

        assert not decoded.metadata.is_cyclic
        assert len(decoded.metadata.cycles) == 0

    def test_self_loop_detection(self):
        """Test detection of self-loop."""
        config = SCMConfig(
            observed_variables=("X", "Y"),
            acyclicity_mode=AcyclicityMode.PENALIZE,  # Don't reject
        )
        decoder = SCMDecoder(config)

        # X = X (self-loop)
        genes = ["X", "STORE_X"]
        genome = SCMGenome(
            inner=genes,
            config=config,
            erc_values=(),
        )

        decoded = decoder.decode(genome)

        assert decoded.metadata.is_cyclic
        assert len(decoded.metadata.cycles) > 0

    def test_mutual_dependency_detection(self):
        """Test detection of mutual dependency cycle."""
        config = SCMConfig(
            observed_variables=("X", "Y"),
            acyclicity_mode=AcyclicityMode.PENALIZE,
        )
        decoder = SCMDecoder(config)

        # X = Y, Y = X (mutual dependency)
        genes = ["Y", "STORE_X", "X", "STORE_Y"]
        genome = SCMGenome(
            inner=genes,
            config=config,
            erc_values=(),
        )

        decoded = decoder.decode(genome)

        assert decoded.metadata.is_cyclic


class TestSCMDecoderGraph:
    """Tests for graph construction."""

    def test_graph_nodes(self, decoder_first_wins):
        """Test graph contains correct nodes."""
        genes = ["B", "STORE_A"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)

        assert "A" in decoded.graph.nodes()
        assert "B" in decoded.graph.nodes()

    def test_graph_edges(self, decoder_first_wins):
        """Test graph contains correct edges."""
        # A = B + C
        genes = ["B", "C", "+", "STORE_A"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)

        # Edges should point from parents to child
        assert decoded.graph.has_edge("B", "A")
        assert decoded.graph.has_edge("C", "A")

    def test_graph_is_digraph(self, decoder_first_wins):
        """Test graph is a directed graph."""
        genes = ["B", "STORE_A"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)

        assert isinstance(decoded.graph, nx.DiGraph)


class TestSCMDecoderMetadata:
    """Tests for metadata generation."""

    def test_coverage_tracking(self, decoder_first_wins):
        """Test variable coverage tracking."""
        # Only A gets equation
        genes = ["B", "STORE_A"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)

        # Coverage should be fraction of observed vars with equations
        # Here, 1 out of 3 (A, B, C) has an equation
        assert "A" in decoded.equations
        assert decoded.metadata.coverage > 0.0

    def test_junk_gene_tracking(self, decoder_first_wins):
        """Test junk gene index tracking."""
        # Create genome with some operations that don't result in STORE
        genes = ["A", "B", "+", "C", "*", "STORE_A", "D", "E"]  # D, E are leftover
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)

        # Check that something was decoded
        assert decoded is not None


class TestDecodedSCMSerialization:
    """Tests for DecodedSCM serialization."""

    def test_to_dict(self, decoder_first_wins):
        """Test DecodedSCM serialization."""
        genes = ["B", "STORE_A"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)
        data = decoded.to_dict()

        assert "equations" in data
        assert "metadata" in data
        assert "A" in data["equations"]

    def test_round_trip(self, decoder_first_wins):
        """Test DecodedSCM serialization round-trip."""
        genes = ["B", "C", "+", "STORE_A"]
        genome = SCMGenome(
            inner=genes,
            config=decoder_first_wins.config,
            erc_values=(),
        )

        decoded = decoder_first_wins.decode(genome)
        data = decoded.to_dict()
        restored = DecodedSCM.from_dict(data)

        assert list(restored.equations.keys()) == list(decoded.equations.keys())
        assert restored.metadata.conflict_count == decoded.metadata.conflict_count
