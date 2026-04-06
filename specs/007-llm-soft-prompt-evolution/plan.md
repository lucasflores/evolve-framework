# Implementation Plan: Evolutionary Soft-Prompt Optimization (ESPO)

**Branch**: `007-llm-soft-prompt-evolution` | **Date**: 2026-04-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/007-llm-soft-prompt-evolution/spec.md`

## Summary

Implement Evolutionary Soft-Prompt Optimization (ESPO) — a new representation module that evolves natural language artifacts as continuous soft-prompt embeddings using evolutionary operators. The core contribution is an `EmbeddingGenome` that stores a 2D continuous array `(n_tokens, embed_dim)` conforming to the framework's `Genome`/`SerializableGenome` protocols, with a `SoftPromptDecoder` that injects embeddings into open-weight LLMs, benchmark and LLM-as-judge evaluators, token-aware mutation/crossover operators, multi-layer coherence defense, and text-mediated cross-model transfer. The genome module uses only numpy (no ML imports); torch/transformers are confined to decoder and evaluator modules.

## Technical Context

**Language/Version**: Python ≥3.10 (project targets 3.10–3.12)
**Primary Dependencies**: numpy ≥1.24.0 (core genome), torch ≥2.0.0 (optional — decoder/evaluator), transformers (optional — tokenizer/model loading), evolve-framework core protocols
**Storage**: N/A (checkpoints via framework serialization + MLflow artifacts)
**Testing**: pytest ≥7.0.0, pytest-cov, hypothesis ≥6.0.0, mypy strict
**Target Platform**: Linux/macOS, CPU reference + optional CUDA GPU
**Project Type**: Library (extension module for evolve-framework)
**Performance Goals**: ≤60s wall-clock per individual per generation for benchmark evaluation on single GPU with 7–8B parameter model (SC-010)
**Constraints**: Genome module zero ML imports (FR-016, SC-009); GPU optional per Principle III; memory bounded by model + population soft-prompt batch
**Scale/Scope**: Single-experiment scope; populations of 20–200 individuals, 4–128 virtual tokens, embedding dims 768–4096

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Design Gate

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| I | Model-Agnostic Architecture | ✅ PASS | EmbeddingGenome uses only numpy; torch/transformers confined to decoder/evaluator as optional-dep plugin modules (FR-016, SC-009) |
| II | Separation of Concerns | ✅ PASS | Genome, decoder, evaluator, operators, initializer, coherence defense are independent composable modules with protocol interfaces |
| III | Acceleration as Optional | ✅ PASS | GPU is used only in decoder/evaluator; CPU reference implementations required; genome + operators are pure numpy |
| IV | Determinism and Reproducibility | ✅ PASS | All operators accept explicit RNG; benchmark evaluator is deterministic (FR-005, SC-006); LLM-as-judge uses temperature=0 |
| V | Extensibility Over Premature Optimization | ✅ PASS | Three dimensionality strategies via configuration, not hardcoding; operators pluggable via existing protocol interfaces |
| VI | Multi-Domain Algorithm Support | ✅ PASS | ESPO extends the framework to LLM prompt optimization domain via existing abstractions (Genome, Evaluator, operators) |
| VII | Observability and Experiment Tracking | ✅ PASS | FR-018 requires comprehensive per-generation MLflow logging; all operators emit structured events |

**Gate Result**: ✅ ALL PASS — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/007-llm-soft-prompt-evolution/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── embedding_genome.md
│   ├── soft_prompt_decoder.md
│   ├── evaluators.md
│   └── operators.md
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
evolve/
├── representation/
│   ├── embedding.py          # EmbeddingGenome (numpy-only, Genome + SerializableGenome)
│   └── embedding_config.py   # DimensionalityStrategy, EmbeddingGenomeConfig
├── evaluation/
│   ├── benchmark.py          # GroundTruthEvaluator (torch optional)
│   ├── llm_judge.py          # LLMJudgeEvaluator (torch optional)
│   └── task_spec.py          # TaskSpec configuration
├── core/
│   └── operators/
│       ├── token_mutation.py  # TokenAwareMutator
│       └── token_crossover.py # TokenLevelCrossover
├── meta/
│   └── soft_prompt/
│       ├── decoder.py         # SoftPromptDecoder (torch/transformers)
│       ├── initializer.py     # PopulationInitializer (torch/transformers)
│       ├── coherence.py       # CoherenceDefense (torch optional for perplexity)
│       ├── callback.py        # ESPOCallback (MLflow per-generation logging)
│       └── transfer.py        # Text-mediated cross-model transfer
└── config/
    └── espo.py                # ESPOConfig aggregating all ESPO settings

tests/
├── unit/
│   ├── test_embedding_genome.py
│   ├── test_token_mutation.py
│   ├── test_token_crossover.py
│   ├── test_coherence_defense.py
│   ├── test_task_spec.py
│   ├── test_initializer.py
│   └── test_transfer.py
├── integration/
│   ├── test_soft_prompt_decoder.py
│   ├── test_benchmark_evaluator.py
│   ├── test_llm_judge_evaluator.py
│   └── test_espo_pipeline.py
└── conftest.py            # Existing — add ESPO fixtures
```

**Structure Decision**: Follows the existing evolve-framework layout convention. The genome (`embedding.py`) goes in `representation/` alongside `vector.py`, `sequence.py`, `graph.py`. Operators go in `core/operators/` alongside existing mutation/crossover. ML-dependent components (decoder, initializer, coherence, transfer) are isolated in `meta/soft_prompt/` to enforce the no-ML-imports boundary for core modules. Evaluators go in `evaluation/` alongside the existing `evaluator.py`.

## Complexity Tracking

> No constitution violations detected. All principles pass.
