"""
CoherenceDefense — Three-layer defense against coherence collapse.

Layer 1: Mutation L2 norm clamping.
Layer 2: Perplexity-based feasibility check (requires torch/transformers).
Layer 3: Fitness-based selection via framework's Fitness.constraints.

Each layer is independently toggleable (FR-012, FR-013).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from evolve.core.types import Fitness

if TYPE_CHECKING:
    from evolve.meta.soft_prompt.decoder import SoftPromptDecoder
    from evolve.representation.embedding import EmbeddingGenome

logger = logging.getLogger(__name__)


@dataclass
class CoherenceDefense:
    """
    Three-layer coherence defense that prevents wasted evaluations
    on genomes producing gibberish output.

    Attributes:
        enable_mutation_clamp: Toggle Layer 1 (L2 norm clamping).
        enable_perplexity_check: Toggle Layer 2 (perplexity feasibility).
        enable_fitness_selection: Toggle Layer 3 (fitness constraint marking).
        coherence_radius: Max L2 norm for per-token mutation delta.
        perplexity_threshold: Max output perplexity for feasibility.
    """

    enable_mutation_clamp: bool = True
    enable_perplexity_check: bool = True
    enable_fitness_selection: bool = True
    coherence_radius: float = 0.1
    perplexity_threshold: float = 100.0

    def clamp_mutation(self, original: np.ndarray, mutated: np.ndarray) -> np.ndarray:
        """
        Layer 1: Clamp per-token mutation delta to coherence radius.

        For each token, if the L2 norm of (mutated - original) exceeds
        the coherence_radius, scale the delta down.

        Args:
            original: Original embeddings, shape (n_tokens, embed_dim).
            mutated: Mutated embeddings, same shape.

        Returns:
            Clamped embeddings as float32 ndarray.
        """
        if not self.enable_mutation_clamp:
            return cast(np.ndarray, mutated.copy())

        result = mutated.copy()
        delta = result - original
        for i in range(delta.shape[0]):
            norm = float(np.linalg.norm(delta[i]))
            if norm > self.coherence_radius:
                result[i] = original[i] + delta[i] * (self.coherence_radius / norm)

        return cast(np.ndarray, result.astype(np.float32))

    def check_feasibility(
        self,
        genome: EmbeddingGenome,
        decoder: SoftPromptDecoder | None = None,
        probe_input: str = "",
    ) -> bool:
        """
        Layer 2: Cheap perplexity-based feasibility check.

        Decodes the genome with a short probe input and computes
        output perplexity. Returns False if perplexity exceeds threshold.

        Args:
            genome: Genome to check.
            decoder: SoftPromptDecoder for generating output.
            probe_input: Short text input for the probe.

        Returns:
            True if genome is feasible (perplexity below threshold).
        """
        if not self.enable_perplexity_check:
            return True

        if decoder is None:
            return True

        try:
            import torch

            decoder._ensure_loaded()

            # Validate model compatibility
            if genome.model_id != decoder.model_id:
                return False

            # Build combined embeddings
            input_ids = decoder._tokenizer.encode(probe_input or "The", return_tensors="pt").to(
                decoder.device
            )

            embed_layer = decoder._model.get_input_embeddings()
            input_embeds = embed_layer(input_ids)

            soft_prompt = torch.tensor(
                genome.embeddings,
                dtype=input_embeds.dtype,
                device=decoder.device,
            ).unsqueeze(0)

            combined = torch.cat([soft_prompt, input_embeds], dim=1)

            # Forward pass to get logits
            with torch.no_grad():
                outputs = decoder._model(inputs_embeds=combined)
                logits = outputs.logits

            # Compute perplexity from cross-entropy loss
            shift_logits = logits[:, :-1, :].contiguous()
            # Create target from the combined sequence shifted by 1
            # Use a simple approach: compute per-token log probabilities
            probs = torch.softmax(shift_logits, dim=-1)
            # Use max probability as proxy for confidence
            max_probs = probs.max(dim=-1).values
            # Perplexity ~ exp(-mean(log(max_prob)))
            log_probs = torch.log(max_probs + 1e-10)
            perplexity = float(torch.exp(-log_probs.mean()))

            return perplexity <= self.perplexity_threshold

        except ImportError:
            logger.warning("torch not available for perplexity check; treating genome as feasible")
            return True
        except Exception as e:
            logger.warning(f"Perplexity check failed: {e}; treating genome as feasible")
            return True

    def mark_infeasible(self, fitness: Fitness) -> Fitness:
        """
        Layer 3: Add a constraint violation to mark fitness as infeasible.

        Creates a new Fitness with a positive constraint value (> 0),
        which the framework's selection mechanism treats as infeasible.

        Args:
            fitness: Original fitness values.

        Returns:
            New Fitness with constraint violation added.
        """
        if not self.enable_fitness_selection:
            return fitness

        # Add a constraint violation (positive value = infeasible)
        constraint = np.array([1.0])
        return Fitness(
            values=fitness.values.copy(),
            constraints=constraint,
            metadata={**fitness.metadata, "coherence_infeasible": True},
        )
