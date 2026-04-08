"""
SCM (Structural Causal Model) Genome Representation.

This module provides evolutionary representations for causal discovery
through the evolution of Structural Causal Models (SCMs).

Key components:
- SCMConfig: Configuration for SCM genome creation and evolution
- SCMAlphabet: Symbol set factory for SCM genomes
- SCMGenome: Genome wrapper composing SequenceGenome with SCM semantics
- ConflictResolution: Enum for handling equation conflicts
- AcyclicityMode: Enum for handling cyclic SCMs
- AcyclicityStrategy: Enum for partial evaluation strategies

Example:
    >>> from evolve.representation.scm import SCMConfig, SCMGenome
    >>> config = SCMConfig(observed_variables=("A", "B", "C"))
    >>> genome = SCMGenome.random(config, length=50, rng=Random(42))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import TYPE_CHECKING, Any

from evolve.representation.sequence import SequenceGenome

if TYPE_CHECKING:
    from evolve.representation.scm_decoder import SCMDecoder


__all__ = [
    "ConflictResolution",
    "AcyclicityMode",
    "AcyclicityStrategy",
    "SCMConfig",
    "SCMAlphabet",
    "SCMGenome",
    # Distance functions for ERP integration
    "scm_sequence_distance",
    "scm_structural_distance",
    "scm_distance",
]


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
    latent_ancestor_penalty: float = 10.0

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
        # Validate penalty values
        for name in (
            "cycle_penalty_per_cycle",
            "incomplete_coverage_penalty",
            "conflict_penalty",
            "div_zero_penalty",
            "latent_ancestor_penalty",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


# Standard constants for SCM alphabet
_STANDARD_CONSTANTS: tuple[float, ...] = (0.0, 1.0, 2.0, -1.0, 0.5, math.pi)
_STANDARD_OPERATORS: tuple[str, ...] = ("+", "-", "*", "/")


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
        # Variable references
        observed = set(config.observed_variables)
        latent = {f"H{i + 1}" for i in range(config.max_latent_variables)}
        variable_refs = observed | latent

        # STORE genes
        store_genes = {f"STORE_{var}" for var in variable_refs}

        # Operators and constants
        operators = set(_STANDARD_OPERATORS)
        constants = set(_STANDARD_CONSTANTS)

        # ERC slots
        erc_slots = {f"ERC_{i}" for i in range(config.erc_count)}

        # All symbols (as strings for SequenceGenome)
        symbols = variable_refs | store_genes | operators | erc_slots

        return cls(
            symbols=frozenset(symbols),
            variable_refs=frozenset(variable_refs),
            store_genes=frozenset(store_genes),
            operators=frozenset(operators),
            constants=frozenset(constants),
            erc_slots=frozenset(erc_slots),
        )

    @property
    def all_variables(self) -> frozenset[str]:
        """All variable names (observed + latent)."""
        return self.variable_refs

    @property
    def all_gene_symbols(self) -> tuple[str, ...]:
        """All gene symbols as tuple for random sampling."""
        # Include constants as string representations for gene sampling
        const_strs = {str(c) for c in self.constants}
        return tuple(self.symbols | const_strs)


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

    # Accept list at init, but converted to SequenceGenome in __post_init__
    inner: SequenceGenome[str | float]
    config: SCMConfig
    erc_values: tuple[tuple[int, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Convert raw lists to SequenceGenome if needed."""
        if isinstance(self.inner, list):
            object.__setattr__(self, "inner", SequenceGenome(genes=tuple(self.inner)))

    # === Genome Protocol ===

    def copy(self) -> SCMGenome:
        """
        Create deep copy of genome.

        Returns new SCMGenome with copied inner sequence.
        Config and erc_values are immutable, so shared.
        """
        return SCMGenome(
            inner=self.inner.copy(),
            config=self.config,
            erc_values=self.erc_values,
        )

    def __eq__(self, other: object) -> bool:
        """
        Structural equality.

        Two SCMGenomes are equal if they have:
        - Same genes in inner sequence
        - Same ERC values
        - Same config (by value)
        """
        if not isinstance(other, SCMGenome):
            return NotImplemented
        return (
            self.inner.genes == other.inner.genes
            and self.erc_values == other.erc_values
            and self.config == other.config
        )

    def __hash__(self) -> int:
        """Hash based on genes and ERC values."""
        return hash((self.inner.genes, self.erc_values))

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
        return {
            "type": "SCMGenome",
            "version": "1.0",
            "genes": list(self.inner.genes),
            "erc_values": [list(ev) for ev in self.erc_values],
            "config": {
                "observed_variables": list(self.config.observed_variables),
                "max_latent_variables": self.config.max_latent_variables,
                "conflict_resolution": self.config.conflict_resolution.value,
                "acyclicity_mode": self.config.acyclicity_mode.value,
                "acyclicity_strategy": self.config.acyclicity_strategy.value,
                "objectives": list(self.config.objectives),
                "constraints": list(self.config.constraints),
                "cycle_penalty_per_cycle": self.config.cycle_penalty_per_cycle,
                "incomplete_coverage_penalty": self.config.incomplete_coverage_penalty,
                "conflict_penalty": self.config.conflict_penalty,
                "div_zero_penalty": self.config.div_zero_penalty,
                "latent_ancestor_penalty": self.config.latent_ancestor_penalty,
                "erc_sigma_init": self.config.erc_sigma_init,
                "erc_sigma_perturb": self.config.erc_sigma_perturb,
                "erc_count": self.config.erc_count,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCMGenome:
        """
        Reconstruct from dict.

        Validates version compatibility and config consistency.

        Raises:
            ValueError: If version unsupported or data invalid
        """
        if data.get("type") != "SCMGenome":
            raise ValueError(f"Expected type 'SCMGenome', got {data.get('type')}")

        version = data.get("version", "1.0")
        if version != "1.0":
            raise ValueError(f"Unsupported version: {version}")

        config_data = data["config"]
        config = SCMConfig(
            observed_variables=tuple(config_data["observed_variables"]),
            max_latent_variables=config_data["max_latent_variables"],
            conflict_resolution=ConflictResolution(config_data["conflict_resolution"]),
            acyclicity_mode=AcyclicityMode(config_data["acyclicity_mode"]),
            acyclicity_strategy=AcyclicityStrategy(config_data["acyclicity_strategy"]),
            objectives=tuple(config_data["objectives"]),
            constraints=tuple(config_data["constraints"]),
            cycle_penalty_per_cycle=config_data["cycle_penalty_per_cycle"],
            incomplete_coverage_penalty=config_data["incomplete_coverage_penalty"],
            conflict_penalty=config_data["conflict_penalty"],
            div_zero_penalty=config_data["div_zero_penalty"],
            latent_ancestor_penalty=config_data.get("latent_ancestor_penalty", 10.0),
            erc_sigma_init=config_data["erc_sigma_init"],
            erc_sigma_perturb=config_data["erc_sigma_perturb"],
            erc_count=config_data["erc_count"],
        )

        genes = tuple(data["genes"])
        erc_values = tuple(tuple(ev) for ev in data["erc_values"])

        return cls(
            inner=SequenceGenome(genes=genes),
            config=config,
            erc_values=erc_values,
        )

    # === Convenience Methods ===

    @property
    def genes(self) -> tuple[str | float, ...]:
        """Access genes from inner sequence."""
        return self.inner.genes

    @property
    def alphabet(self) -> SCMAlphabet:
        """Get alphabet for this genome's config."""
        return SCMAlphabet.from_config(self.config)

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
        alphabet = SCMAlphabet.from_config(config)
        gene_pool = alphabet.all_gene_symbols

        # Sample genes
        genes = tuple(rng.choice(gene_pool) for _ in range(length))

        # Find ERC slots in genes and sample values
        erc_values: list[tuple[int, float]] = []
        for i, gene in enumerate(genes):
            if isinstance(gene, str) and gene.startswith("ERC_"):
                value = rng.gauss(0.0, config.erc_sigma_init)
                erc_values.append((i, value))

        return cls(
            inner=SequenceGenome(genes=genes),
            config=config,
            erc_values=tuple(erc_values),
        )

    def get_erc_value(self, slot: int) -> float:
        """Get ERC value for slot index."""
        for idx, value in self.erc_values:
            if idx == slot:
                return value
        raise KeyError(f"No ERC value at slot {slot}")

    def with_erc_values(self, new_values: tuple[tuple[int, float], ...]) -> SCMGenome:
        """Return copy with updated ERC values."""
        return SCMGenome(
            inner=self.inner,
            config=self.config,
            erc_values=new_values,
        )

    def mutate_erc(self, rng: Random, slot: int | None = None) -> SCMGenome:
        """
        Apply perturbation mutation to ERC values.

        Adds Gaussian noise N(0, erc_sigma_perturb) to ERC values.

        Args:
            rng: Random instance for reproducibility
            slot: If specified, mutate only this slot; otherwise mutate all

        Returns:
            New SCMGenome with perturbed ERC values
        """
        new_values: list[tuple[int, float]] = []
        for idx, value in self.erc_values:
            if slot is None or idx == slot:
                new_value = value + rng.gauss(0.0, self.config.erc_sigma_perturb)
                new_values.append((idx, new_value))
            else:
                new_values.append((idx, value))

        return self.with_erc_values(tuple(new_values))


# === Distance Functions for ERP Integration ===


def scm_sequence_distance(genome_a: SCMGenome, genome_b: SCMGenome) -> float:
    """
    Compute normalized Hamming distance between SCM genome sequences.

    Returns a value in [0, 1] representing the fraction of positions
    where the genes differ. If genomes have different lengths, returns 1.0.

    Args:
        genome_a: First SCM genome
        genome_b: Second SCM genome

    Returns:
        Normalized distance in [0, 1]
    """
    genes_a = genome_a.genes
    genes_b = genome_b.genes

    if len(genes_a) != len(genes_b):
        return 1.0

    if len(genes_a) == 0:
        return 0.0

    differences = sum(1 for a, b in zip(genes_a, genes_b) if a != b)
    return differences / len(genes_a)


def scm_structural_distance(
    genome_a: SCMGenome,
    genome_b: SCMGenome,
    decoder: SCMDecoder,
) -> float:
    """
    Compute normalized structural distance between decoded SCM graphs.

    Uses graph edit distance normalized by the maximum possible edits
    (sum of nodes and edges in both graphs). Returns a value in [0, 1].

    Args:
        genome_a: First SCM genome
        genome_b: Second SCM genome
        decoder: SCMDecoder instance for decoding genomes

    Returns:
        Normalized structural distance in [0, 1]
    """

    # Decode both genomes
    scm_a = decoder.decode(genome_a)
    scm_b = decoder.decode(genome_b)

    graph_a = scm_a.graph
    graph_b = scm_b.graph

    # For empty graphs, check if both are empty
    if graph_a.number_of_edges() == 0 and graph_b.number_of_edges() == 0:
        return 0.0

    # Compute symmetric edge difference
    edges_a = set(graph_a.edges())
    edges_b = set(graph_b.edges())

    # Count edge differences
    edges_only_in_a = edges_a - edges_b
    edges_only_in_b = edges_b - edges_a
    edge_diff = len(edges_only_in_a) + len(edges_only_in_b)

    # Normalize by total possible edges
    # Maximum difference is all edges different = edges_a + edges_b
    total_edges = len(edges_a) + len(edges_b)

    if total_edges == 0:
        return 0.0

    return edge_diff / total_edges


def scm_distance(
    genome_a: SCMGenome,
    genome_b: SCMGenome,
    decoder: SCMDecoder,
    structural_weight: float = 0.5,
) -> float:
    """
    Compute combined distance between SCM genomes.

    Combines sequence distance and structural distance using a weighted sum:
        distance = (1 - structural_weight) * seq_dist + structural_weight * struct_dist

    Both component distances are normalized to [0, 1], so the result is also
    in [0, 1].

    Args:
        genome_a: First SCM genome
        genome_b: Second SCM genome
        decoder: SCMDecoder instance for decoding genomes
        structural_weight: Weight for structural distance [0, 1].
            0.0 = sequence only, 1.0 = structure only, 0.5 = equal weight

    Returns:
        Combined distance in [0, 1]

    Raises:
        ValueError: If structural_weight is not in [0, 1]
    """
    if not 0.0 <= structural_weight <= 1.0:
        raise ValueError(f"structural_weight must be in [0, 1], got {structural_weight}")

    seq_dist = scm_sequence_distance(genome_a, genome_b)
    struct_dist = scm_structural_distance(genome_a, genome_b, decoder)

    return (1.0 - structural_weight) * seq_dist + structural_weight * struct_dist
