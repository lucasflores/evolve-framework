"""
Text-mediated cross-model transfer for evolved soft prompts.

Decodes a best-performing genome into a text interpretation
suitable for use as a hard prompt on a different model (FR-014, FR-017).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evolve.meta.soft_prompt.decoder import SoftPromptDecoder
    from evolve.representation.embedding import EmbeddingGenome


def text_mediated_transfer(
    genome: EmbeddingGenome,
    decoder: SoftPromptDecoder,
    probe_input: str = "",
    max_tokens: int | None = None,
) -> str:
    """
    Transfer an evolved soft prompt to text via decoding.

    Decodes the genome using the original model's decoder to produce
    a text interpretation that can be used as a hard prompt (seed_text)
    for a new ESPO run on a different model.

    Args:
        genome: Best-performing EmbeddingGenome from an ESPO run.
        decoder: SoftPromptDecoder configured for the original model.
        probe_input: Optional input text to condition decoding.
        max_tokens: Max generation tokens for decoding.

    Returns:
        Non-empty text string suitable for use as seed_text.

    Raises:
        ValueError: If genome and decoder model IDs don't match,
                   or if decoding produces empty text.
    """
    if genome.model_id != decoder.model_id:
        raise ValueError(
            f"Model mismatch: genome targets '{genome.model_id}' "
            f"but decoder loaded '{decoder.model_id}'"
        )

    text = decoder.decode(genome, probe_input, max_tokens=max_tokens)

    if not text or not text.strip():
        raise ValueError(
            "Text-mediated transfer produced empty text. The genome may be degenerate."
        )

    return text.strip()
