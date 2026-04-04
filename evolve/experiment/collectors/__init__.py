"""
Metric Collectors Package.

Provides specialized metric collectors for different evolution domains:
- ERPMetricCollector: ERP mating statistics
- MultiObjectiveMetricCollector: Pareto front quality metrics
- SpeciationMetricCollector: Species dynamics
- IslandsMetricCollector: Island model parallelism metrics
- NEATMetricCollector: Neuroevolution topology metrics
- FitnessMetadataCollector: Fitness.metadata extraction
- DerivedAnalyticsCollector: Computed analytical metrics

All collectors implement the MetricCollector protocol.
"""

from evolve.experiment.collectors.base import (
    CollectionContext,
    MatingStats,
    MetricCollector,
)
from evolve.experiment.collectors.derived import DerivedAnalyticsCollector
from evolve.experiment.collectors.erp import ERPMetricCollector
from evolve.experiment.collectors.islands import IslandsMetricCollector
from evolve.experiment.collectors.metadata import FitnessMetadataCollector
from evolve.experiment.collectors.multiobjective import MultiObjectiveMetricCollector
from evolve.experiment.collectors.neat import NEATMetricCollector
from evolve.experiment.collectors.speciation import SpeciationMetricCollector

__all__ = [
    "CollectionContext",
    "MetricCollector",
    "MatingStats",
    "DerivedAnalyticsCollector",
    "ERPMetricCollector",
    "FitnessMetadataCollector",
    "IslandsMetricCollector",
    "MultiObjectiveMetricCollector",
    "NEATMetricCollector",
    "SpeciationMetricCollector",
]
