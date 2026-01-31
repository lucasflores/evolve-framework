"""
Evolvable Reproduction Protocols (ERP) module.

This module enables individuals to encode, evolve, and execute their own
reproductive compatibility logic and offspring construction strategies.

Key Components:
- ReproductionProtocol: The evolvable reproduction protocol genome (RPG)
- MatchabilityFunction: Determines mate acceptability
- ReproductionIntentPolicy: Governs when reproduction is attempted
- CrossoverProtocolSpec: Specifies offspring genome construction
- ERPEngine: Evolution engine with ERP support
"""

from evolve.reproduction.protocol import (
    CrossoverProtocolSpec,
    CrossoverType,
    IntentContext,
    MatchabilityFunction,
    MateContext,
    ReproductionEvent,
    ReproductionIntentPolicy,
    ReproductionProtocol,
)
from evolve.reproduction.sandbox import (
    StepCounter,
    StepLimitExceeded,
    sandboxed_execute,
    safe_execute,
)
from evolve.reproduction.matchability import (
    MatchabilityEvaluator,
    MatchabilityRegistry,
    evaluate_matchability,
    safe_evaluate_matchability,
    # Built-in evaluators
    AcceptAllMatchability,
    RejectAllMatchability,
    DistanceThresholdMatchability,
    SimilarityThresholdMatchability,
    FitnessRatioMatchability,
    DifferentNicheMatchability,
    ProbabilisticMatchability,
    DiversitySeekingMatchability,
)
from evolve.reproduction.crossover_protocol import (
    CrossoverExecutor,
    CrossoverRegistry,
    execute_crossover,
    safe_execute_crossover,
    inherit_protocol,
    validate_offspring,
    # Built-in executors
    SinglePointCrossoverExecutor,
    TwoPointCrossoverExecutor,
    UniformCrossoverExecutor,
    BlendCrossoverExecutor,
    CloneCrossoverExecutor,
)
from evolve.reproduction.intent import (
    IntentEvaluator,
    IntentRegistry,
    evaluate_intent,
    safe_evaluate_intent,
    # Built-in evaluators
    AlwaysWillingIntent,
    NeverWillingIntent,
    FitnessThresholdIntent,
    FitnessRankThresholdIntent,
    ResourceBudgetIntent,
    AgeDependentIntent,
    ProbabilisticIntent,
)
from evolve.reproduction.mutation import (
    MutationConfig,
    ProtocolMutator,
    mutate_matchability,
    mutate_intent,
    mutate_crossover,
    mutate_junk_data,
    promote_junk_to_param,
    demote_param_to_junk,
)
from evolve.reproduction.recovery import (
    RecoveryStrategy,
    ImmigrationRecovery,
    MutationBoostRecovery,
    RelaxedMatchingRecovery,
    CompositeRecovery,
)
# Note: Callbacks and ERPEngine are imported separately to avoid circular imports:
#   from evolve.reproduction.callbacks import ERPMetricsCallback, ERPLoggerCallback
#   from evolve.reproduction.engine import ERPConfig, ERPEngine

__all__ = [
    # Core protocol types
    "ReproductionProtocol",
    "MatchabilityFunction",
    "ReproductionIntentPolicy",
    "CrossoverProtocolSpec",
    "CrossoverType",
    # Context types
    "MateContext",
    "IntentContext",
    # Sandbox
    "StepCounter",
    "StepLimitExceeded",
    "sandboxed_execute",
    "safe_execute",
    # Events
    "ReproductionEvent",
    # Matchability
    "MatchabilityEvaluator",
    "MatchabilityRegistry",
    "evaluate_matchability",
    "safe_evaluate_matchability",
    "AcceptAllMatchability",
    "RejectAllMatchability",
    "DistanceThresholdMatchability",
    "SimilarityThresholdMatchability",
    "FitnessRatioMatchability",
    "DifferentNicheMatchability",
    "ProbabilisticMatchability",
    "DiversitySeekingMatchability",
    # Crossover
    "CrossoverExecutor",
    "CrossoverRegistry",
    "execute_crossover",
    "safe_execute_crossover",
    "inherit_protocol",
    "validate_offspring",
    "SinglePointCrossoverExecutor",
    "TwoPointCrossoverExecutor",
    "UniformCrossoverExecutor",
    "BlendCrossoverExecutor",
    "CloneCrossoverExecutor",
    # Intent
    "IntentEvaluator",
    "IntentRegistry",
    "evaluate_intent",
    "safe_evaluate_intent",
    "AlwaysWillingIntent",
    "NeverWillingIntent",
    "FitnessThresholdIntent",
    "FitnessRankThresholdIntent",
    "ResourceBudgetIntent",
    "AgeDependentIntent",
    "ProbabilisticIntent",
    # Mutation
    "MutationConfig",
    "ProtocolMutator",
    "mutate_matchability",
    "mutate_intent",
    "mutate_crossover",
    "mutate_junk_data",
    "promote_junk_to_param",
    "demote_param_to_junk",
    # Recovery
    "RecoveryStrategy",
    "ImmigrationRecovery",
    "MutationBoostRecovery",
    "RelaxedMatchingRecovery",
    "CompositeRecovery",
    # Callbacks - import separately: from evolve.reproduction.callbacks import ERPMetricsCallback
    # "ERPMetrics",
    # "ERPMetricsCallback",
    # "ERPLoggerCallback",
    # Engine - import separately: from evolve.reproduction.engine import ERPConfig, ERPEngine
    # "ERPConfig",
    # "ERPEngine",
]
