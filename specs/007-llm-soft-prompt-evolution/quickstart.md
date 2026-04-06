# Quickstart: Evolutionary Soft-Prompt Optimization (ESPO)

**Feature Branch**: `007-llm-soft-prompt-evolution`

## Prerequisites

```bash
# Install evolve-framework with PyTorch support
pip install evolve-framework[pytorch]

# Ensure you have a local model (e.g., via huggingface-cli)
# huggingface-cli download meta-llama/Llama-2-7b-hf
```

## Minimal Example: Optimize a QA Soft Prompt

```python
from random import Random

import numpy as np

from evolve.core.engine import EvolutionConfig, EvolutionEngine
from evolve.core.operators.token_crossover import TokenLevelCrossover
from evolve.core.operators.token_mutation import TokenAwareMutator
from evolve.core.operators.selection import TournamentSelection
from evolve.evaluation.benchmark import GroundTruthEvaluator
from evolve.evaluation.task_spec import TaskSpec
from evolve.meta.soft_prompt.decoder import SoftPromptDecoder
from evolve.meta.soft_prompt.initializer import PopulationInitializer
from evolve.representation.embedding import EmbeddingGenome
from evolve.representation.embedding_config import (
    DimensionalityStrategy,
    EmbeddingGenomeConfig,
)

# 1. Configure the genome
config = EmbeddingGenomeConfig(
    n_tokens=8,  # Default: minimal tokens strategy
    embed_dim=4096,  # Must match your target model
    model_id="meta-llama/Llama-2-7b-hf",
    strategy=DimensionalityStrategy.MINIMAL_TOKENS,
    seed_text="Answer the following question accurately:",
    coherence_radius=0.15,
)

# 2. Create the decoder (loads model lazily)
decoder = SoftPromptDecoder(
    model_id=config.model_id,
    device="cuda",  # or "cpu"
)

# 3. Define the task
task = TaskSpec(
    task_type="qa",
    inputs=[
        {"input": "What is the capital of France?"},
        {"input": "What is 2 + 2?"},
        # ... more questions
    ],
    ground_truth=["Paris", "4"],
    metrics=["accuracy"],
    max_generation_tokens=64,
)

# 4. Set up the evaluator
evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=task)

# 5. Initialize population from seed text
rng = Random(42)
initializer = PopulationInitializer(config=config, decoder=decoder)
initial_population = initializer.noise_init(
    seed_text=config.seed_text,
    n=50,
    rng=rng,
)

# 6. Configure operators
mutation = TokenAwareMutator(
    mutation_rate=0.1,
    sigma=0.05,
    coherence_radius=config.coherence_radius,
)
crossover = TokenLevelCrossover(crossover_type="single_point")
selection = TournamentSelection(tournament_size=5)

# 7. Run evolution
engine = EvolutionEngine(
    config=EvolutionConfig(
        population_size=50,
        max_generations=50,
        mutation_rate=0.8,
        crossover_rate=0.7,
        minimize=False,  # Higher accuracy = better
        elitism=2,
    ),
    evaluator=evaluator,
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    seed=42,
)

result = engine.run(initial_population=initial_population)

# 8. Inspect best result
best = result.best_individual
print(f"Best fitness: {best.fitness.values[0]:.3f}")
decoded_text = decoder.decode(best.genome, "What is the capital of Germany?")
print(f"Model output: {decoded_text}")
```

## Using LLM-as-Judge for Open-Ended Tasks

```python
from evolve.evaluation.llm_judge import LLMJudgeEvaluator
from evolve.evaluation.task_spec import RubricCriterion, TaskSpec

# Define rubric for creative writing
task = TaskSpec(
    task_type="generation",
    inputs=[{"input": "Write a haiku about programming."}],
    rubric=[
        RubricCriterion(name="creativity", description="Novel and surprising imagery", scale_min=0, scale_max=1),
        RubricCriterion(name="form", description="Correct 5-7-5 syllable structure", scale_min=0, scale_max=1),
    ],
    metrics=[],  # No ground-truth metrics
)

# LLM-judge returns multi-objective fitness (one per criterion)
judge_evaluator = LLMJudgeEvaluator(
    decoder=decoder,
    task_spec=task,
    judge_model_id="meta-llama/Llama-2-70b-hf",
    temperature=0.0,
)
```

## Text-Mediated Cross-Model Transfer

```python
from evolve.meta.soft_prompt.transfer import text_mediated_transfer

# After evolution on model A, get the best genome
best_genome = result.best_individual.genome

# Transfer to text (works with any model)
transferred_text = text_mediated_transfer(
    genome=best_genome,
    decoder=decoder,
)
print(f"Transferred prompt: {transferred_text}")

# Use as seed for new ESPO run on model B
config_b = EmbeddingGenomeConfig(
    n_tokens=8,
    embed_dim=2048,  # Different model = different embed_dim
    model_id="mistralai/Mistral-7B-v0.1",
    seed_text=transferred_text,  # Use transferred text as seed
)
```

## Using the Registry / Unified Config

```python
from evolve.registry.genomes import get_genome_registry
from evolve.registry.operators import get_operator_registry

# Create genome via registry
registry = get_genome_registry()
genome = registry.create("embedding", rng=rng, config=config)

# Create operators via registry
op_registry = get_operator_registry()
mutation = op_registry.get("mutation", "token_gaussian", sigma=0.05)
crossover = op_registry.get("crossover", "token_single_point")
```
