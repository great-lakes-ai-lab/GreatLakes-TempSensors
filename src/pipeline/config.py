# src/pipeline/config.py
"""Load and validate pipeline configuration from YAML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import yaml
import shutil
import os


@dataclass
class PathsConfig:
    output_root: str

    # Derived paths
    run_dir: str = ""
    processed_dir: str = ""
    model_dir: str = ""
    data_processor_dir: str = ""
    seasonal_dir: str = ""
    active_learning_dir: str = ""

    def resolve(self, run_name: str):
        """Derive all output paths from output_root/run_name."""
        root = Path(self.output_root).expanduser()
        run_dir = root / run_name

        self.output_root = str(root)
        self.run_dir = str(run_dir)
        self.processed_dir = str(run_dir / "processed_data")
        self.model_dir = str(run_dir / "model")
        self.data_processor_dir = str(run_dir / "deepsensor_config" / "data_processor")
        self.seasonal_dir = str(run_dir / "seasonal_cycles")
        self.active_learning_dir = str(run_dir / "active_learning")

    def resolve_active_learning(self, al_name: str) -> Path:
        """Resolve the active learning experiment directory."""
        al_dir = Path(self.active_learning_dir) / al_name
        return al_dir


@dataclass
class DataSourceEntry:
    path: str
    format: str
    variable: Optional[str] = None
    variables: Optional[list] = None
    role: Union[str, list] = "context"                 # target, context, aux_at_targets, mask, or list
    use_anomalies: bool = False
    sampling: str = "all"                   # all, random_lake_points, or integer
    coarsen_factor: Optional[int] = None    # per-source coarsening


@dataclass
class PreprocessingConfig:
    fit_range: tuple = ("2019-01-01", "2019-12-31")
    force_reprocess: bool = False


@dataclass
class TrainingConfig:
    train_range: tuple = ("2019-01-01", "2020-12-31")
    val_range: tuple = ("2021-01-01", "2021-12-31")
    date_subsample_factor: int = 5
    n_epochs: int = 50
    lr: float = 5e-5
    internal_density: int = 250
    n_context_points: int = 50
    vary_n_context: bool = True
    min_n_context: int = 20
    max_n_context: int = 75
    patience: int = 0  # 0 = no early stopping
    include_bathy_as_context: bool = False
    bathy_context_sampling: str = "random_lake_points"  # "all", "random_lake_points", or integer (e.g. 1000)


@dataclass
class RunConfig:
    name: str = "unnamed_run"
    notes: str = ""
    display_plots: str = "auto"  # "auto", "show", or "save_only"


@dataclass
class PredictionConfig:
    dates: list = None              # Specific dates: ["2021-03-15", "2021-07-01"]
    n_random: int = None            # OR: pick n random dates from val range
    every_n_days: int = None        # OR: every n days across val range
    n_context: int = None           # Override training n_context for prediction (optional)
    seed: int = 42

    def get_dates(self, val_range: tuple) -> list:
        """Resolve prediction dates from config options."""
        import pandas as pd
        import numpy as np

        if self.dates:
            return pd.to_datetime(self.dates).normalize().tolist()

        val_start, val_end = val_range
        all_dates = pd.date_range(val_start, val_end, freq="D").normalize()

        if self.n_random:
            np.random.seed(self.seed)
            idx = np.random.choice(len(all_dates), size=min(self.n_random, len(all_dates)), replace=False)
            return sorted(all_dates[idx].tolist())

        if self.every_n_days:
            return all_dates[::self.every_n_days].tolist()

        # Default: 5 evenly spaced dates
        return pd.date_range(val_start, val_end, periods=5).normalize().tolist()


@dataclass
class ActiveLearningConfig:
    name: str = "default"  # creates subfolder: active_learning/<name>/

    eval_range: tuple = ("2021-01-01", "2021-12-31")
    eval_subsample_factor: int = 14

    n_new_sensors: int = 5

    # Supported acquistion function options:
    #
    # Sequential (non-parallel):
    #   mean_stddev, mean_variance, mean_marginal_entropy, joint_entropy,
    #   p_norm_stddev, oracle_rmse, oracle_mae, oracle_marginal_nll, oracle_joint_nll
    #
    # Parallel:
    #   stddev, context_dist, expected_improvement, random
    acquisition_function: str = "mean_stddev"

    # Parameters for specific acquisition functions
    acquisition_fn_p: float = 1.0  # p-norm exponent for p_norm_stddev
    acquisition_fn_seed: int = 42

    # Task context settings
    context_source: str = "random"  # "random", "geojson", or "buoy" (future)
    n_context: int = 50
    context_seed: int = 42
    context_geojson_path: str = None
    save_context_points: bool = True

    context_set_idx: int = 0
    target_set_idx: int = 0

    # DeepSensor GreedyAlgorithm settings
    model_infill_method: str = "mean"
    diff: bool = False
    progress_bar: bool = True

    # Grid controls
    candidate_coarsen_factor: int = 4
    target_coarsen_factor: int = 4

    # Placement constraints
    min_dist_between_sensors_km: float = 0.0

    # Future hook for existing buoy locations
    existing_sensors_path: str = None

    # Outputs
    save_acquisition_surface: bool = True
    plot_results: bool = True


@dataclass
class PipelineConfig:
    lake: str = "erie"
    environment: str = "local"  # "local" or "hpc"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data_sources: dict = field(default_factory=dict)  # str -> DataSourceEntry
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def validate(self):
        """Fail-fast validation of config before pipeline runs."""
        from lakes import LAKE_BOUNDS  # or whatever your dict is called

        # Check lake name
        if self.lake not in LAKE_BOUNDS:
            available = list(LAKE_BOUNDS.keys())
            raise ValueError(f"Unknown lake '{self.lake}'. Available: {available}")

        # Check data source paths exist
        for name, source in self.data_sources.items():
            p = Path(source.path)
            if not p.exists():
                raise FileNotFoundError(f"Data source '{name}' path not found: {p}")

        # Check date range consistency
        import pandas as pd
        fit_start, fit_end = pd.Timestamp(self.preprocessing.fit_range[0]), pd.Timestamp(
            self.preprocessing.fit_range[1])
        train_start, train_end = pd.Timestamp(self.training.train_range[0]), pd.Timestamp(self.training.train_range[1])
        val_start, val_end = pd.Timestamp(self.training.val_range[0]), pd.Timestamp(self.training.val_range[1])

        if fit_start > fit_end:
            raise ValueError(f"fit_range start {fit_start} is after end {fit_end}")
        if train_start > train_end:
            raise ValueError(f"train_range start {train_start} is after end {train_end}")
        if val_start > val_end:
            raise ValueError(f"val_range start {val_start} is after end {val_end}")
        if val_start <= train_end:
            import warnings
            warnings.warn(
                f"Validation range overlaps with training range (val starts {val_start}, train ends {train_end})")

        # Check output root is writable
        output_root = Path(self.paths.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        if not os.access(output_root, os.W_OK):
            raise PermissionError(f"Output root not writable: {output_root}")



def load_config(config_path: str) -> PipelineConfig:
    """Load a YAML config file and return a PipelineConfig dataclass."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Run config first, because paths depend on run.name
    run_cfg = RunConfig(**raw.get("run", {}))

    # Paths
    paths_raw = raw.get("paths", {})
    paths = PathsConfig(**paths_raw)
    paths.resolve(run_cfg.name)

    # Data sources
    sources_raw = raw.get("data_sources", {})
    data_sources = {}
    for name, entry in sources_raw.items():
        if "path" in entry:
            entry["path"] = str(Path(entry["path"]).expanduser())
        data_sources[name] = DataSourceEntry(**entry)

    preprocessing = PreprocessingConfig(**raw.get("preprocessing", {}))
    training = TrainingConfig(**raw.get("training", {}))
    prediction = PredictionConfig(**raw.get("prediction", {}))
    active_learning = ActiveLearningConfig(**raw.get("active_learning", {}))

    return PipelineConfig(
        lake=raw.get("lake", "erie"),
        environment=raw.get("environment", "local"),
        paths=paths,
        data_sources=data_sources,
        preprocessing=preprocessing,
        training=training,
        prediction=prediction,
        active_learning=active_learning,
        run=run_cfg,
    )

def copy_config_to_run_dir(config, config_path):
    run_dir = Path(config.paths.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(config_path, run_dir / "config_used.yaml")
    except shutil.SameFileError:
        pass


