"""
SCM Genome API Contract

This file defines the interface contracts for SCMGenome and related types.
Implementation MUST satisfy all type signatures and documented behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evolve.representation.sequence import SequenceGenome

# === Enums ===


class ConflictResolution(Enum):
    """How to handle multiple STORE_X for same variable."""

    FIRST_WINS = "first_wins"
    LAST_WINS = "last_wins"
    ALL_JUNK = "all_junk"


class AcyclicityMode(Enum):
    """How to handle cyclic SCMs during evaluation."""

    REJECT = "reject"
    PENALIZE = "penalize"


class AcyclicityStrategy(Enum):
    """Partial evaluation strategy when mode=PENALIZE."""

    ACYCLIC_SUBGRAPH = "acyclic_subgraph"
    PARSE_ORDER = "parse_order"
    PENALTY_ONLY = "penalty_only"
    PARENT_INHERITANCE = "parent_inheritance"
    COMPOSITE = "composite"


# === Configuration ===


@dataclass(frozen=True)
class SCMConfig:
    """
    Configuration for SCM genome creation and evolution.

    All fields have sensible defaults except observed_variables,
    which must be provided.

    Invariants:
        - len(observed_variables) > 0
        - max_latent_variables >= 0
        - All penalty values >= 0
        - erc_sigma_init > 0, erc_sigma_perturb > 0
    """

    # Required
    observed_variables: tuple[str, ...]

    # Variable configuration
    max_latent_variables: int = 3

    # Decoding behavior
    conflict_resolution: ConflictResolution = ConflictResolution.FIRST_WINS

    # Evaluation behavior
    acyclicity_mode: AcyclicityMode = AcyclicityMode.REJECT
    acyclicity_strategy: AcyclicityStrategy = AcyclicityStrategy.ACYCLIC_SUBGRAPH

    # Objectives and constraints
    objectives: tuple[str, ...] = ("data_fit", "sparsity", "simplicity")
    constraints: tuple[str, ...] = ("acyclicity",)

    # Penalty weights
    cycle_penalty_per_cycle: float = 1.0
    incomplete_coverage_penalty: float = 10.0
    conflict_penalty: float = 1.0
    div_zero_penalty: float = 5.0

    # ERC parameters
    erc_sigma_init: float = 1.0
    erc_sigma_perturb: float = 0.1
    erc_count: int = 5

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if len(self.observed_variables) == 0:
            raise ValueError("observed_variables must be non-empty")
        if self.max_latent_variables < 0:
            raise ValueError("max_latent_variables must be >= 0")
        if self.erc_sigma_init <= 0 or self.erc_sigma_perturb <= 0:
            raise ValueError("ERC sigma values must be positive")


# === Alphabet ===


@dataclass(frozen=True)
class SCMAlphabet:
    """
    Symbol set for SCM genomes.

    Generated from SCMConfig. Contains all valid gene values
    organized by category for mutation and interpretation.
    """

    symbols: frozenset[str]
    variable_refs: frozenset[str]
    store_genes: frozenset[str]
    operators: frozenset[str]
    constants: frozenset[float]
    erc_slots: frozenset[str]

    @classmethod
    def from_config(cls, config: SCMConfig) -> SCMAlphabet:
        """
        Generate alphabet from configuration.

        Creates:
        - Variable refs for observed + latent variables
        - STORE_* genes for all variables
        - Standard operators: +, -, *, /
        - Standard constants: 0, 1, 2, -1, 0.5, PI
        - ERC slots: ERC_0, ERC_1, ... ERC_{n-1}
        """
        ...

    @property
    def all_variables(self) -> frozenset[str]:
        """All variable names (observed + latent)."""
        ...


# === Genome ===


@dataclass(frozen=True)
class SCMGenome:
    """
    SCM genome encoding potential causal model.

    Wraps SequenceGenome[str | float] with SCM-specific semantics.
    Implements Genome and SerializableGenome protocols.

    Invariants:
        - inner.genes contains only symbols from alphabet
        - erc_values indices match ERC slots in genes
        - config matches alphabet used to generate genes
    """

    inner: SequenceGenome[str | float]  # From evolve.representation.sequence
    config: SCMConfig
    erc_values: tuple[tuple[int, float], ...]

    # === Genome Protocol ===

    def copy(self) -> SCMGenome:
        """
        Create deep copy of genome.

        Returns new SCMGenome with copied inner sequence.
        Config and erc_values are immutable, so shared.
        """
        ...

    def __eq__(self, other: object) -> bool:
        """
        Structural equality.

        Two SCMGenomes are equal if they have:
        - Same genes in inner sequence
        - Same ERC values
        - Same config (by value)
        """
        ...

    def __hash__(self) -> int:
        """Hash based on genes and ERC values."""
        ...

    # === SerializableGenome Protocol ===

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serializable dict.

        Returns:
            {
                "type": "SCMGenome",
                "version": "1.0",
                "genes": [...],
                "erc_values": [[slot, value], ...],
                "config": {...}
            }
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCMGenome:
        """
        Reconstruct from dict.

        Validates version compatibility and config consistency.

        Raises:
            ValueError: If version unsupported or data invalid
        """
        ...

    # === Convenience Methods ===

    @property
    def genes(self) -> tuple[str | float, ...]:
        """Access genes from inner sequence."""
        return self.inner.genes

    @property
    def alphabet(self) -> SCMAlphabet:
        """Get alphabet for this genome's config."""
        ...

    @classmethod
    def random(
        cls,
        config: SCMConfig,
        length: int,
        rng: Random,
    ) -> SCMGenome:
        """
        Create random genome.

        - Samples genes uniformly from alphabet
        - Samples ERC values from N(0, erc_sigma_init)

        Args:
            config: SCM configuration
            length: Number of genes
            rng: Random instance for reproducibility

        Returns:
            New random SCMGenome
        """
        ...

    def get_erc_value(self, slot: int) -> float:
        """Get ERC value for slot index."""
        ...

    def with_erc_values(self, new_values: tuple[tuple[int, float], ...]) -> SCMGenome:
        """Return copy with updated ERC values."""
        ...
