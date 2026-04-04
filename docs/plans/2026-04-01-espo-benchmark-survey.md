# ESPO Benchmark Survey: Evaluation Datasets for Soft-Prompt Evolution

**Date:** 2026-04-01
**Status:** Research Note
**Parent:** [LLM Representation Design (ESPO)](2026-04-01-llm-representation-design.md)

---

## 1. Purpose

Catalog the benchmarks used in the prompt optimization literature, assess their suitability for ESPO, and recommend a prioritized evaluation stack. Literature conventions take priority — we want apples-to-apples comparison with existing work.

---

## 2. Literature Map

The benchmarks below are organized by the papers that established them as standard evaluation targets for prompt optimization. There are two distinct research lineages:

- **Continuous / soft-prompt papers** (Lester 2021, BBT, BBTv2, P-Tuning) — optimize embedding vectors; closest methodological relatives to ESPO
- **Discrete / text-prompt papers** (EvoPrompt, PromptBreeder, OPRO, APE) — optimize natural language instructions; relevant for comparison but different search space

### 2.1 Lester et al. 2021 — "The Power of Scale for Parameter-Efficient Prompt Tuning"
**Venue:** EMNLP 2021 | **Method:** Continuous soft-prompt tuning | **Model:** T5 (Small → XXL)

**Benchmarks:** All 8 SuperGLUE tasks:

| Dataset | Task | Metric | # Test | Notes |
|---------|------|--------|--------|-------|
| **BoolQ** | Boolean question answering | Accuracy | 3,245 | Yes/no questions from Wikipedia passages |
| **CB** (CommitmentBank) | Natural language inference (3-class) | Accuracy / F1 | 56 | Very small test set; high variance |
| **COPA** | Causal reasoning | Accuracy | 100 | Choose most plausible cause/effect |
| **MultiRC** | Multi-sentence reading comprehension | F1a / EM | 4,848 | Multiple correct answers per question |
| **ReCoRD** | Reading comprehension with commonsense | F1 / EM | 10,000 | Cloze-style, entity selection |
| **RTE** | Recognizing textual entailment (2-class) | Accuracy | 277 | Entailment vs. not-entailment |
| **WiC** | Word-in-context (word sense disambiguation) | Accuracy | 638 | Same sense in two sentences? |
| **WSC** (Winograd Schema Challenge) | Coreference resolution | Accuracy | 104 | Pronoun disambiguation |

**Why this matters for ESPO:** This is the canonical soft-prompt benchmark. Lester showed that prompt tuning closes the gap to full fine-tuning as model scale increases. Direct comparison target for our embedding-space evolution.

**Key result:** With T5-XXL (11B params), 20 soft tokens match full fine-tuning on SuperGLUE aggregate.

---

### 2.2 BBT — "Black-Box Tuning for Language-Model-as-a-Service"
**Venue:** ICML 2022 | **Method:** CMA-ES over random-projected soft-prompt embeddings | **Model:** RoBERTa-large

**Benchmarks:** Few-shot text classification (16-shot per class):

| Dataset | Task | Metric | Notes |
|---------|------|--------|-------|
| **SST-2** | Binary sentiment (positive/negative) | Accuracy | Stanford Sentiment Treebank, movie reviews |
| **Yelp** (Polarity) | Binary sentiment | Accuracy | Yelp restaurant reviews |
| **AG's News** | 4-class topic classification (World/Sports/Business/Tech) | Accuracy | News articles |
| **DBPedia** | 14-class ontology classification | Accuracy | Wikipedia first paragraphs; budget 20K forward passes |
| **SNLI** | 3-class NLI (entailment/contradiction/neutral) | Accuracy | Stanford Natural Language Inference |
| **MNLI** | 3-class NLI (matched) | Accuracy | Multi-Genre NLI |
| **RTE** | 2-class NLI | Accuracy | Same task as SuperGLUE, different formatting |

**Optimization budget:** ~8,000 forward passes (20,000 for DBPedia). Population size 20, CMA-ES with intrinsic dimensionality reduction to 500.

**Why this matters for ESPO:** BBT is the closest existing work to our approach — it uses CMA-ES (an evolution strategy) to optimize soft prompts in a random-projected subspace. Direct ancestor. Their intrinsic dimensionality reduction (500d subspace from ~50×1024 = 51K dims) validates our compressed-subspace strategy.

**Key results:** With 16-shot, 8K forward passes, BBT matches gradient-based prompt tuning on SST-2 (~90% accuracy).

---

### 2.3 BBTv2 — "Towards a Gradient-Free Future with Large Language Models"
**Venue:** EMNLP 2022 | **Method:** CMA-ES with deep prompt (multi-layer injection) | **Model:** RoBERTa-large, T5, GPT-2, BERT, BART

**Benchmarks:** Same datasets as BBT plus extended model coverage:

| Dataset | Task | Metric | Notes |
|---------|------|--------|-------|
| SST-2 | Binary sentiment | Accuracy | Same as BBT |
| Yelp | Binary sentiment | Accuracy | Same as BBT |
| AG's News | 4-class topic | Accuracy | Same as BBT |
| DBPedia | 14-class ontology | Accuracy | Same as BBT |
| SNLI | 3-class NLI | Accuracy | Same as BBT |
| MNLI | 3-class NLI | Accuracy | Same as BBT |
| RTE | 2-class NLI | Accuracy | Same as BBT |
| **MRPC** | Paraphrase detection | Accuracy / F1 | Microsoft Research Paraphrase Corpus |

**Key advance over BBT:** Deep prompt tuning (inject at every transformer layer, not just input). Alternating layer-by-layer optimization via divide-and-conquer.

**arxiv:** 2205.11200

---

### 2.4 EvoPrompt — "Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers"
**Venue:** ICLR 2024 | **Method:** GA + DE on discrete text prompts via LLM operators | **Models:** Alpaca-7B, GPT-3.5

**Benchmarks (31 total):**

#### Language Understanding (7 datasets):

| Dataset | Task | Metric | # Test |
|---------|------|--------|--------|
| **SST-2** | Binary sentiment | Accuracy | 1,821 |
| **MR** | Binary sentiment (movie reviews) | Accuracy | 2,000 |
| **CR** | Binary sentiment (customer reviews) | Accuracy | 2,000 |
| **SST-5** | 5-class sentiment (terrible/bad/okay/good/great) | Accuracy | 2,210 |
| **AG's News** | 4-class topic | Accuracy | 7,600 |
| **TREC** | 6-class question type | Accuracy | 500 |
| **Subj** | Binary subjectivity | Accuracy | 2,000 |

#### Language Generation (2 datasets):

| Dataset | Task | Metric | # Test |
|---------|------|--------|--------|
| **SAMSum** | Dialogue summarization | ROUGE-1/2/L | 819 |
| **ASSET** | Text simplification | SARI | 359 |

#### BIG-Bench Hard (22 tasks):

All 23 BBH tasks minus "web of lies" (already 100% baseline accuracy). These include:

| Task ID | Task | Domain |
|---------|------|--------|
| 01 | Hyperbaton | Syntax / adjective ordering |
| 02 | Temporal Sequences | Temporal reasoning |
| 03 | Object Counting | Enumeration |
| 04 | Disambiguation QA | Pronoun resolution |
| 05 | Logical Deduction (3/5/7 objects) | Logical reasoning |
| 06 | Causal Judgment | Causal reasoning |
| 07 | Date Understanding | Temporal reasoning |
| 08 | Ruin Names | Humor / wordplay |
| 09 | Word Sorting | String manipulation |
| 10 | Geometric Shapes | Spatial reasoning (SVG paths) |
| 11 | Movie Recommendation | Preference reasoning |
| 12 | Salient Translation Error Detection | Cross-lingual |
| 13 | Formal Fallacies | Deductive logic |
| 14 | Penguins in a Table | Table reasoning |
| 15 | Dyck Languages | Formal language theory |
| 16 | Multistep Arithmetic | Arithmetic reasoning |
| 17 | Navigate | Spatial reasoning |
| 18 | Reasoning about Colored Objects | Visual reasoning |
| 19 | Boolean Expressions | Logic |
| 20 | Tracking Shuffled Objects (3/5/7) | State tracking |
| 21 | Sports Understanding | Commonsense |
| 22 | Snarks | Sarcasm detection |

**EvoPrompt settings:** Population size 10, 10 iterations, dev set = 200 samples (classification) or 100 (generation) or 50 (BBH). 3-shot CoT for BBH tasks.

**Why this matters for ESPO:** Different search space (discrete text vs. continuous embedding), but the benchmark suite is comprehensive and directly comparable. EvoPrompt's DE variant outperforms GA on most BBH tasks — validates the value of differential evolution operators in prompt optimization.

---

### 2.5 PromptBreeder — "Self-Referential Self-Improvement via Prompt Evolution"
**Venue:** arXiv 2023 (DeepMind) | **Method:** LLM-driven mutation of discrete prompts + self-referential mutation-prompt evolution | **Model:** PaLM 2-L

**Benchmarks:**

#### Arithmetic Reasoning:

| Dataset | Task | Metric | Notes |
|---------|------|--------|-------|
| **GSM8K** | Grade-school math word problems | Accuracy (final answer) | 8.5K training, 1.3K test; chain-of-thought required |
| **MultiArith** | Multi-step arithmetic word problems | Accuracy | 600 examples |
| **AQuA-RAT** | Algebraic word problems (multiple choice) | Accuracy | ~100K training, 254 test |
| **SVAMP** | Simple variations on arithmetic math problems | Accuracy | 1,000 test examples |
| **AddSub** | Addition/subtraction word problems | Accuracy | 395 examples |
| **SingleEq** | Single-equation word problems | Accuracy | 508 examples |

#### Commonsense Reasoning:

| Dataset | Task | Metric | Notes |
|---------|------|--------|-------|
| **CommonsenseQA** | 5-choice commonsense questions | Accuracy | ~1.2K dev |
| **StrategyQA** | Yes/no questions requiring implicit reasoning | Accuracy | 2.3K questions |
| **Last Letters** | Concatenate last letters of N words | Accuracy | Synthetic benchmark |
| **Coin Flip** | Track state after N coin flips | Accuracy | Synthetic benchmark |

#### Classification:

| Dataset | Task | Metric | Notes |
|---------|------|--------|-------|
| **ETHOS** | Hate speech detection (binary) | Accuracy / F1 | ~1K examples; PromptBreeder's novel contribution |

**Why this matters for ESPO:** PromptBreeder's self-referential mechanism (evolving the mutation operators, not just the solutions) maps directly to our ERP system — ESPO could evolve the reproduction protocols alongside the prompts. Also expands evaluation to math reasoning, which is a harder test of prompt quality than classification.

---

### 2.6 OPRO — "Large Language Models as Optimizers"
**Venue:** ICLR 2024 (Google DeepMind) | **Method:** LLM generates new prompt candidates conditioned on (prompt, score) history | **Models:** PaLM 2-L, GPT-3.5, GPT-4

**Benchmarks:**

| Dataset | Task | Metric | Notes |
|---------|------|--------|-------|
| **GSM8K** | Math word problems | Accuracy | Same as PromptBreeder |
| **BBH (23 tasks)** | Multi-step reasoning | Accuracy | Same suite as EvoPrompt; uses 3-shot CoT |

**Key result:** OPRO finds a prompt ("Let's think step by step" → "Take a deep breath and work on this problem step-by-step") that improves GSM8K by ~8% over baseline, and up to 50% on some BBH tasks.

**Why this matters for ESPO:** OPRO's prompt trajectory analysis shows how optimization converges — useful reference for understanding convergence dynamics in embedding space.

---

### 2.7 APE — "Large Language Models Are Human-Level Prompt Engineers"
**Venue:** ICLR 2023 | **Method:** LLM-generated prompt candidates + iterative Monte Carlo search | **Models:** InstructGPT, GPT-3

**Benchmarks:** 24 NLI/instruction induction tasks from the BIG-Bench / Instruction Induction suite. Significant overlap with EvoPrompt and OPRO benchmarks.

---

## 3. Benchmark Taxonomy

Across all papers, the benchmarks cluster into five categories:

### 3.1 Text Classification (NLU)
| Dataset | Classes | Domain | Used By |
|---------|---------|--------|---------|
| SST-2 | 2 | Sentiment (movies) | BBT, BBTv2, EvoPrompt |
| SST-5 | 5 | Sentiment (movies) | EvoPrompt |
| MR | 2 | Sentiment (movies) | EvoPrompt |
| CR | 2 | Sentiment (products) | EvoPrompt |
| Subj | 2 | Subjectivity | EvoPrompt |
| AG's News | 4 | Topic (news) | BBT, BBTv2, EvoPrompt |
| DBPedia | 14 | Ontology (Wikipedia) | BBT, BBTv2 |
| TREC | 6 | Question type | EvoPrompt |
| ETHOS | 2 | Hate speech | PromptBreeder |
| Yelp | 2 | Sentiment (reviews) | BBT, BBTv2 |

### 3.2 Natural Language Inference (NLI)
| Dataset | Classes | Used By |
|---------|---------|---------|
| SNLI | 3 | BBT, BBTv2 |
| MNLI | 3 | BBT, BBTv2 |
| RTE | 2 | BBT, BBTv2, Lester 2021 |
| CB | 3 | Lester 2021 |

### 3.3 Reading Comprehension / QA
| Dataset | Task | Used By |
|---------|------|---------|
| BoolQ | Boolean QA | Lester 2021 |
| MultiRC | Multi-sentence RC | Lester 2021 |
| ReCoRD | Commonsense RC | Lester 2021 |
| COPA | Causal reasoning | Lester 2021 |
| WiC | Word sense | Lester 2021 |
| WSC | Coreference | Lester 2021 |
| CommonsenseQA | 5-choice commonsense | PromptBreeder |
| StrategyQA | Implicit reasoning | PromptBreeder |

### 3.4 Arithmetic / Math Reasoning
| Dataset | Task | Used By |
|---------|------|---------|
| GSM8K | Grade-school math | PromptBreeder, OPRO |
| MultiArith | Multi-step arithmetic | PromptBreeder |
| AQuA-RAT | Algebraic word problems | PromptBreeder |
| SVAMP | Simple arithmetic variations | PromptBreeder |
| AddSub | Addition/subtraction | PromptBreeder |
| SingleEq | Single-equation | PromptBreeder |

### 3.5 Language Generation
| Dataset | Task | Metric | Used By |
|---------|------|--------|---------|
| SAMSum | Dialogue summarization | ROUGE | EvoPrompt |
| ASSET | Text simplification | SARI | EvoPrompt |

### 3.6 Multi-Step Reasoning (BBH)
22 diverse tasks from BIG-Bench Hard — used by both EvoPrompt and OPRO. Covers logic, spatial reasoning, temporal reasoning, arithmetic, string manipulation, and commonsense.

---

## 4. Recommendations for ESPO

### 4.1 Tier 1: Must-Have (apples-to-apples with closest work)

These are the benchmarks we MUST run to be taken seriously against BBT/BBTv2 (our methodological ancestors):

| Dataset | Why | Budget |
|---------|-----|--------|
| **SST-2** | Universal baseline — every paper uses it | Low (1.8K test, binary) |
| **AG's News** | 4-class; tests whether soft prompts capture topic | Medium (7.6K test) |
| **DBPedia** | 14-class; highest difficulty in BBT suite | Medium-High (needs 20K forward passes in BBT) |
| **SNLI** | NLI pair task; different structure than classification | Medium |
| **RTE** | Small but appears in BBT, SuperGLUE, and EvoPrompt | Low |

**Model for Tier 1:** RoBERTa-large (to match BBT directly), then replicate with a more modern model (e.g., Llama-3-8B).

**Evaluation protocol:** 16-shot per class, report accuracy. Track optimization budget (forward passes) as a primary comparison metric — BBT reports 8K passes to converge.

### 4.2 Tier 2: Strong Differentiators

These push beyond classification into reasoning, where the value of evolved prompts is most visible:

| Dataset | Why | Budget |
|---------|-----|--------|
| **GSM8K** | Gold standard for math reasoning; used by PromptBreeder + OPRO | High (1.3K test, needs CoT) |
| **BBH subset (5–8 tasks)** | Multi-step reasoning diversity; used by EvoPrompt + OPRO | High per task |
| **SST-5** | 5-class; harder than SST-2; EvoPrompt reports big gains here | Low |
| **Subj** | EvoPrompt's DE shines here (DE > GA by 5%); good for operator comparison | Low |

**Model for Tier 2:** Open-weight instruction-following model (Llama-3-8B-Instruct or Mistral-7B-Instruct).

### 4.3 Tier 3: Nice-to-Have (novel contributions)

| Dataset | Why |
|---------|-----|
| **ETHOS** (hate speech) | Classification + social impact angle; only PromptBreeder used it |
| **SAMSum** (summarization) | Tests generation quality, not just classification; ROUGE metric |
| **CommonsenseQA** | Commonsense reasoning beyond arithmetic |
| **MRPC** (paraphrase) | Classic GLUE task; BBTv2 uses it |
| **Full SuperGLUE** | Complete replication of Lester 2021 (only if we have T5 infrastructure) |

---

## 5. Evaluation Protocol Design

### 5.1 What to Report

For every benchmark, report:

1. **Task accuracy** (or F1/ROUGE as appropriate) — the headline number
2. **Forward-pass budget** — how many model inference calls to reach that accuracy (BBT's primary comparison axis)
3. **Convergence curve** — accuracy vs. generation number (EvoPrompt-style)
4. **Population diversity** — prompt length variance, embedding distance metrics (EvoPrompt reports this)
5. **Best-of-run vs. mean-of-runs** — report both (EvoPrompt uses 3 seeds ± std)

### 5.2 Baselines to Compare Against

| Baseline | Type | Source |
|----------|------|--------|
| Zero-shot prompting | No optimization | Manual prompt |
| Few-shot prompting (k-shot) | No optimization | k examples per class |
| Manual prompt tuning (best-known prompt) | Human-crafted | Literature |
| BBT (CMA-ES) | Gradient-free soft-prompt | Sun et al. 2022 |
| BBTv2 (deep CMA-ES) | Gradient-free soft-prompt | Sun et al. 2022 |
| Full fine-tuning | Upper bound | Standard |
| Gradient-based prompt tuning | Gradient soft-prompt | Lester et al. 2021 |
| EvoPrompt (DE) | Evolutionary discrete prompt | Guo et al. 2023 |

### 5.3 Experimental Design for ESPO-Specific Questions

Beyond standard benchmark scores, ESPO needs experiments that answer our unique research questions:

| Research Question | Experiment | Benchmark |
|-------------------|------------|-----------|
| Does evolutionary search outperform CMA-ES for soft prompts? | ESPO vs. BBT on same tasks, same budget | SST-2, AG's News, SNLI |
| Does population-based search find better prompts than single-trajectory? | ESPO (pop=20) vs. BBT (pop=20) vs. gradient-based (single trajectory) | Tier 1 suite |
| Do token-aware operators beat flat-vector operators? | Token crossover/mutation vs. Gaussian mutation on flat vector | SST-2, GSM8K |
| Does ERP help? (evolving reproduction alongside prompts) | ERP-enabled vs. fixed operators | Any Tier 1 task |
| Dimensionality: full vs. projected vs. minimal tokens? | Three genome configs, same task | SST-2, AG's News |
| How quickly does coherence collapse appear? | Track decoded text quality over generations | SST-2 (easy), GSM8K (hard) |
| Multi-model transfer via text mediation | Evolve on Llama-3-8B, evaluate on Mistral-7B | SST-2, GSM8K |

---

## 6. Data Access Summary

All recommended benchmarks are freely available:

| Dataset | Source | Format |
|---------|--------|--------|
| SST-2, SST-5 | HuggingFace `glue/sst2`, Stanford NLP | Text + label |
| AG's News | HuggingFace `ag_news` | Text + 4 labels |
| DBPedia | HuggingFace `dbpedia_14` | Text + 14 labels |
| SNLI | HuggingFace `snli` | Premise + hypothesis + label |
| MNLI | HuggingFace `multi_nli` | Premise + hypothesis + label |
| RTE, BoolQ, CB, COPA, MultiRC, ReCoRD, WiC, WSC | HuggingFace `super_glue` | Various |
| MRPC | HuggingFace `glue/mrpc` | Sentence pair + label |
| MR, CR, Subj | Available via NLTK / standard NLP packages | Text + label |
| TREC | HuggingFace `trec` | Question + 6 labels |
| GSM8K | HuggingFace `gsm8k` | Question + chain + answer |
| BBH (23 tasks) | HuggingFace `lukaemon/bbh` | Task-specific |
| SAMSum | HuggingFace `samsum` | Dialogue + summary |
| ASSET | HuggingFace `asset` | Source + simplifications |
| ETHOS | HuggingFace `ethos` | Text + binary label |
| CommonsenseQA | HuggingFace `commonsense_qa` | Question + 5 choices |
| MultiArith, SVAMP, AddSub, SingleEq, AQuA-RAT | Various (Chain-of-Thought Hub) | Question + answer |

---

## 7. Recommended Execution Order

1. **SST-2 on RoBERTa-large** — First experiment. Replicate BBT's setup exactly. If we can't match CMA-ES here, nothing else matters.
2. **SST-2 ablations** — Full-space vs. projected vs. minimal tokens. Token-aware vs. flat operators. Establish which genome config works.
3. **Expand to Tier 1 suite** — AG's News, DBPedia, SNLI, RTE. Validate the winning config generalizes.
4. **Switch model to Llama-3-8B** — Repeat SST-2 to establish modern model baseline.
5. **GSM8K** — First reasoning task. This is where soft-prompt evolution could either shine or fail spectacularly.
6. **BBH subset** — Pick 5 diverse BBH tasks. Test multi-step reasoning.
7. **ERP experiments** — Enable evolved reproduction on best-performing config.
8. **Transfer experiments** — Evolve on Llama-3-8B, evaluate decoded text on Mistral-7B.
