# Experiment Management Interfaces Contract

**Module**: `evolve.experiment`  
**Purpose**: Define configuration, checkpointing, and metric tracking abstractions

---

## Configuration Protocol

```python
from typing import Protocol, Any, TypeVar, Self
from dataclasses import dataclass, field, asdict
import json
import hashlib
from pathlib import Path


@dataclass
class ExperimentConfig:
    """
    Complete configuration for an evolutionary experiment.
    
    Designed for:
    - Reproducibility (all parameters explicit)
    - Serialization (JSON-compatible)
    - Hashing (for deduplication)
    """
    # Identification
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    
    # Random seed
    seed: int = 42
    
    # Population
    population_size: int = 100
    n_generations: int = 100
    
    # Selection
    selection_method: str = "tournament"
    selection_params: dict[str, Any] = field(default_factory=dict)
    
    # Operators
    crossover_method: str = "uniform"
    crossover_rate: float = 0.9
    crossover_params: dict[str, Any] = field(default_factory=dict)
    
    mutation_method: str = "gaussian"
    mutation_rate: float = 0.1
    mutation_params: dict[str, Any] = field(default_factory=dict)
    
    # Representation
    genome_type: str = "vector"
    genome_params: dict[str, Any] = field(default_factory=dict)
    
    # Evaluation
    evaluator_type: str = "function"
    evaluator_params: dict[str, Any] = field(default_factory=dict)
    
    # Multi-objective (optional)
    multi_objective: bool = False
    n_objectives: int = 1
    
    # Island model (optional)
    islands: int = 1
    migration_rate: float = 0.1
    migration_interval: int = 10
    topology: str = "ring"
    
    # Diversity (optional)
    speciation: bool = False
    speciation_params: dict[str, Any] = field(default_factory=dict)
    
    # Callbacks
    callbacks: list[str] = field(default_factory=list)
    
    # Stopping criteria
    max_evaluations: int | None = None
    target_fitness: float | None = None
    stagnation_limit: int | None = None
    
    # Output
    output_dir: str = "./experiments"
    checkpoint_interval: int = 10
    log_level: str = "INFO"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dict."""
        return cls(**data)
    
    def to_json(self, path: Path | str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path | str) -> Self:
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
    
    def hash(self) -> str:
        """
        Deterministic hash of configuration.
        
        Useful for detecting duplicate experiments.
        """
        # Sort dict keys for deterministic serialization
        serialized = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    def validate(self) -> list[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if self.population_size < 2:
            errors.append("population_size must be >= 2")
        
        if self.n_generations < 1:
            errors.append("n_generations must be >= 1")
        
        if not 0 <= self.crossover_rate <= 1:
            errors.append("crossover_rate must be in [0, 1]")
        
        if not 0 <= self.mutation_rate <= 1:
            errors.append("mutation_rate must be in [0, 1]")
        
        if self.multi_objective and self.n_objectives < 2:
            errors.append("multi_objective requires n_objectives >= 2")
        
        if self.islands > 1 and not 0 < self.migration_rate <= 1:
            errors.append("migration_rate must be in (0, 1] for island model")
        
        return errors
```

---

## Checkpoint Protocol

```python
import pickle
from datetime import datetime


@dataclass
class Checkpoint:
    """
    Complete state for resuming an experiment.
    
    Includes everything needed to continue evolution
    from exactly this point.
    """
    # Identification
    experiment_name: str
    config_hash: str
    
    # State
    generation: int
    population: list['Individual']
    best_individual: 'Individual'
    
    # RNG state for reproducibility
    rng_state: dict[str, Any]
    
    # History
    fitness_history: list[dict[str, float]]  # Per-generation stats
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_evaluations: int = 0
    elapsed_time: float = 0.0
    
    # Optional state
    species: list['Species'] | None = None
    islands: list['Island'] | None = None
    novelty_archive: 'NoveltyArchive' | None = None
    
    def save(self, path: Path | str) -> None:
        """
        Save checkpoint to disk.
        
        Uses pickle for complex objects.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load checkpoint from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @classmethod
    def from_engine(
        cls,
        engine: 'EvolutionEngine',
        config: ExperimentConfig
    ) -> 'Checkpoint':
        """Create checkpoint from current engine state."""
        return cls(
            experiment_name=config.name,
            config_hash=config.hash(),
            generation=engine.generation,
            population=list(engine.population),
            best_individual=engine.best,
            rng_state=engine.get_rng_state(),
            fitness_history=engine.fitness_history,
            total_evaluations=engine.total_evaluations
        )


class CheckpointManager:
    """
    Manages checkpoint saving and loading.
    """
    
    def __init__(
        self,
        output_dir: Path | str,
        keep_last_n: int = 5,
        checkpoint_interval: int = 10
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoint_interval = checkpoint_interval
    
    def should_checkpoint(self, generation: int) -> bool:
        """Check if checkpoint should be saved."""
        return generation % self.checkpoint_interval == 0
    
    def save(self, checkpoint: Checkpoint) -> Path:
        """
        Save checkpoint and manage history.
        
        Prunes old checkpoints beyond keep_last_n.
        """
        filename = f"checkpoint_gen{checkpoint.generation:06d}.pkl"
        path = self.output_dir / filename
        checkpoint.save(path)
        
        # Prune old checkpoints
        self._prune_old_checkpoints()
        
        return path
    
    def load_latest(self) -> Checkpoint | None:
        """Load most recent checkpoint."""
        checkpoints = sorted(self.output_dir.glob("checkpoint_gen*.pkl"))
        if not checkpoints:
            return None
        return Checkpoint.load(checkpoints[-1])
    
    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """List all checkpoints with generation numbers."""
        checkpoints = []
        for path in self.output_dir.glob("checkpoint_gen*.pkl"):
            gen = int(path.stem.split('gen')[1])
            checkpoints.append((gen, path))
        return sorted(checkpoints)
    
    def _prune_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond keep_last_n."""
        checkpoints = sorted(self.output_dir.glob("checkpoint_gen*.pkl"))
        while len(checkpoints) > self.keep_last_n:
            checkpoints[0].unlink()
            checkpoints.pop(0)
```

---

## Metric Tracking Protocol

```python
class MetricTracker(Protocol):
    """
    Abstract interface for experiment tracking.
    
    Implementations may log to:
    - Local files (default)
    - MLflow
    - Weights & Biases
    - TensorBoard
    """
    
    def start_run(self, config: ExperimentConfig) -> None:
        """Start tracking a new experiment run."""
        ...
    
    def log_generation(
        self,
        generation: int,
        metrics: dict[str, float]
    ) -> None:
        """
        Log metrics for a generation.
        
        Standard metrics:
        - best_fitness
        - mean_fitness
        - std_fitness
        - diversity
        """
        ...
    
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        ...
    
    def log_artifact(self, path: Path, name: str | None = None) -> None:
        """Log file as artifact."""
        ...
    
    def end_run(self) -> None:
        """Finalize tracking."""
        ...


@dataclass
class LocalTracker:
    """
    Simple local file-based tracking.
    
    Creates CSV files and JSON logs.
    """
    output_dir: Path
    metrics_file: Path | None = None
    
    def start_run(self, config: ExperimentConfig) -> None:
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.to_json(self.output_dir / "config.json")
        
        # Initialize metrics CSV
        self.metrics_file = self.output_dir / "metrics.csv"
        with open(self.metrics_file, 'w') as f:
            f.write("generation,best_fitness,mean_fitness,std_fitness\n")
    
    def log_generation(
        self,
        generation: int,
        metrics: dict[str, float]
    ) -> None:
        with open(self.metrics_file, 'a') as f:
            f.write(f"{generation},{metrics.get('best_fitness', 0)},")
            f.write(f"{metrics.get('mean_fitness', 0)},")
            f.write(f"{metrics.get('std_fitness', 0)}\n")
    
    def log_params(self, params: dict[str, Any]) -> None:
        with open(self.output_dir / "params.json", 'w') as f:
            json.dump(params, f, indent=2)
    
    def log_artifact(self, path: Path, name: str | None = None) -> None:
        import shutil
        dest = self.output_dir / (name or path.name)
        shutil.copy(path, dest)
    
    def end_run(self) -> None:
        # Write summary
        summary = {"status": "completed", "timestamp": datetime.now().isoformat()}
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


class MLflowTracker:
    """
    MLflow tracking implementation.
    
    Requires: pip install mlflow
    """
    
    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri
        self.run_id: str | None = None
    
    def start_run(self, config: ExperimentConfig) -> None:
        import mlflow
        
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        mlflow.set_experiment(config.name)
        run = mlflow.start_run()
        self.run_id = run.info.run_id
        
        # Log all config as params
        mlflow.log_params(config.to_dict())
    
    def log_generation(
        self,
        generation: int,
        metrics: dict[str, float]
    ) -> None:
        import mlflow
        mlflow.log_metrics(metrics, step=generation)
    
    def log_params(self, params: dict[str, Any]) -> None:
        import mlflow
        mlflow.log_params(params)
    
    def log_artifact(self, path: Path, name: str | None = None) -> None:
        import mlflow
        mlflow.log_artifact(str(path))
    
    def end_run(self) -> None:
        import mlflow
        mlflow.end_run()
```

---

## Experiment Runner

```python
@dataclass
class ExperimentRunner:
    """
    Orchestrates experiment execution.
    
    Handles:
    - Configuration validation
    - Engine setup
    - Checkpointing
    - Metric tracking
    - Resume from checkpoint
    """
    config: ExperimentConfig
    tracker: MetricTracker | None = None
    checkpoint_manager: CheckpointManager | None = None
    
    def __post_init__(self):
        # Validate config
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid config: {errors}")
        
        # Setup output directory
        output_dir = Path(self.config.output_dir) / self.config.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default tracker
        if self.tracker is None:
            self.tracker = LocalTracker(output_dir)
        
        # Default checkpoint manager
        if self.checkpoint_manager is None:
            self.checkpoint_manager = CheckpointManager(
                output_dir / "checkpoints",
                checkpoint_interval=self.config.checkpoint_interval
            )
    
    def setup_engine(self) -> 'EvolutionEngine':
        """
        Build evolution engine from config.
        
        Factory method that creates all components.
        """
        from evolve.core import EvolutionEngine
        from evolve.operators import get_selection, get_crossover, get_mutation
        from evolve.representation import get_genome_factory
        from evolve.evaluation import get_evaluator
        
        # Create components from config
        selection = get_selection(
            self.config.selection_method,
            **self.config.selection_params
        )
        crossover = get_crossover(
            self.config.crossover_method,
            **self.config.crossover_params
        )
        mutation = get_mutation(
            self.config.mutation_method,
            **self.config.mutation_params
        )
        genome_factory = get_genome_factory(
            self.config.genome_type,
            **self.config.genome_params
        )
        evaluator = get_evaluator(
            self.config.evaluator_type,
            **self.config.evaluator_params
        )
        
        return EvolutionEngine(
            population_size=self.config.population_size,
            genome_factory=genome_factory,
            evaluator=evaluator,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate,
            seed=self.config.seed
        )
    
    def run(self, resume: bool = False) -> 'Individual':
        """
        Execute the experiment.
        
        Args:
            resume: Whether to resume from checkpoint
            
        Returns:
            Best individual found
        """
        # Try to resume
        start_generation = 0
        engine = self.setup_engine()
        
        if resume:
            checkpoint = self.checkpoint_manager.load_latest()
            if checkpoint:
                engine.restore_from_checkpoint(checkpoint)
                start_generation = checkpoint.generation + 1
                print(f"Resumed from generation {checkpoint.generation}")
        
        # Start tracking
        self.tracker.start_run(self.config)
        
        try:
            # Evolution loop
            for gen in range(start_generation, self.config.n_generations):
                engine.step()
                
                # Log metrics
                metrics = {
                    'best_fitness': engine.best.fitness.value,
                    'mean_fitness': engine.mean_fitness,
                    'std_fitness': engine.std_fitness,
                    'generation': gen
                }
                self.tracker.log_generation(gen, metrics)
                
                # Checkpoint
                if self.checkpoint_manager.should_checkpoint(gen):
                    checkpoint = Checkpoint.from_engine(engine, self.config)
                    self.checkpoint_manager.save(checkpoint)
                
                # Check stopping criteria
                if self._should_stop(engine):
                    print(f"Early stopping at generation {gen}")
                    break
            
            # Final checkpoint
            checkpoint = Checkpoint.from_engine(engine, self.config)
            self.checkpoint_manager.save(checkpoint)
            
            return engine.best
            
        finally:
            self.tracker.end_run()
    
    def _should_stop(self, engine: 'EvolutionEngine') -> bool:
        """Check stopping criteria."""
        if self.config.target_fitness is not None:
            if engine.best.fitness.value >= self.config.target_fitness:
                return True
        
        if self.config.max_evaluations is not None:
            if engine.total_evaluations >= self.config.max_evaluations:
                return True
        
        if self.config.stagnation_limit is not None:
            if engine.stagnation_counter >= self.config.stagnation_limit:
                return True
        
        return False
```

---

## Experiment Comparison

```python
@dataclass
class ExperimentComparison:
    """
    Compare results across multiple experiments.
    """
    experiments: dict[str, Path]  # name -> output_dir
    
    def load_metrics(self) -> dict[str, 'pd.DataFrame']:
        """Load metrics from all experiments."""
        import pandas as pd
        
        results = {}
        for name, path in self.experiments.items():
            metrics_file = path / "metrics.csv"
            if metrics_file.exists():
                results[name] = pd.read_csv(metrics_file)
        return results
    
    def summarize(self) -> 'pd.DataFrame':
        """Create summary table of all experiments."""
        import pandas as pd
        
        summaries = []
        for name, path in self.experiments.items():
            config = ExperimentConfig.from_json(path / "config.json")
            metrics = pd.read_csv(path / "metrics.csv")
            
            summaries.append({
                'name': name,
                'final_best': metrics['best_fitness'].iloc[-1],
                'final_mean': metrics['mean_fitness'].iloc[-1],
                'generations': len(metrics),
                'seed': config.seed,
                'population_size': config.population_size
            })
        
        return pd.DataFrame(summaries)
    
    def plot_convergence(self, metric: str = 'best_fitness') -> None:
        """Plot convergence curves for all experiments."""
        import matplotlib.pyplot as plt
        
        metrics = self.load_metrics()
        
        plt.figure(figsize=(10, 6))
        for name, df in metrics.items():
            plt.plot(df['generation'], df[metric], label=name)
        
        plt.xlabel('Generation')
        plt.ylabel(metric)
        plt.legend()
        plt.title('Convergence Comparison')
        plt.grid(True)
        plt.tight_layout()
```

---

## Hyperparameter Sweep

```python
from itertools import product


@dataclass
class SweepConfig:
    """
    Configuration for hyperparameter sweep.
    """
    base_config: ExperimentConfig
    param_grid: dict[str, list[Any]]  # param_name -> values to try
    n_seeds: int = 3  # Seeds per configuration
    
    def generate_configs(self) -> list[ExperimentConfig]:
        """Generate all configuration combinations."""
        configs = []
        
        # Get all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            
            # Generate for multiple seeds
            for seed_idx in range(self.n_seeds):
                config_dict = self.base_config.to_dict()
                config_dict.update(param_dict)
                config_dict['seed'] = self.base_config.seed + seed_idx
                config_dict['name'] = f"{self.base_config.name}_" + \
                                      "_".join(f"{k}={v}" for k, v in param_dict.items()) + \
                                      f"_seed{seed_idx}"
                
                configs.append(ExperimentConfig.from_dict(config_dict))
        
        return configs


class SweepRunner:
    """
    Run hyperparameter sweep.
    """
    
    def __init__(
        self,
        sweep_config: SweepConfig,
        parallel: bool = False,
        n_workers: int = 4
    ):
        self.sweep_config = sweep_config
        self.parallel = parallel
        self.n_workers = n_workers
    
    def run(self) -> list[tuple[ExperimentConfig, 'Individual']]:
        """
        Run all configurations.
        
        Returns:
            List of (config, best_individual) pairs
        """
        configs = self.sweep_config.generate_configs()
        results = []
        
        if self.parallel:
            from concurrent.futures import ProcessPoolExecutor
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [
                    executor.submit(self._run_single, config)
                    for config in configs
                ]
                for config, future in zip(configs, futures):
                    results.append((config, future.result()))
        else:
            for config in configs:
                best = self._run_single(config)
                results.append((config, best))
        
        return results
    
    def _run_single(self, config: ExperimentConfig) -> 'Individual':
        """Run single experiment."""
        runner = ExperimentRunner(config)
        return runner.run()
```
