# src/pipeline/config.py
"""Load and validate pipeline configuration from YAML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
import shutil


@dataclass
class PathsConfig:
    output_root: str

    # Derived paths
    run_dir: str = ""
    processed_dir: str = ""
    model_dir: str = ""
    data_processor_dir: str = ""
    seasonal_dir: str = ""

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


@dataclass
class DataSourceEntry:
    path: str
    format: str
    variable: Optional[str] = None
    variables: Optional[list] = None
    role: str = "context"                   # target, context, aux_at_targets, mask, or list
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
    include_bathy_as_context: bool = False
    bathy_context_sampling: str = "random_lake_points"  # "all", "random_lake_points", or integer (e.g. 1000)


@dataclass
class RunConfig:
    name: str = "unnamed_run"
    notes: str = ""


@dataclass
class PipelineConfig:
    lake: str = "erie"
    environment: str = "local"  # "local" or "hpc"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data_sources: dict = field(default_factory=dict)  # str -> DataSourceEntry
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    run: RunConfig = field(default_factory=RunConfig)


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

    return PipelineConfig(
        lake=raw.get("lake", "erie"),
        environment=raw.get("environment", "local"),
        paths=paths,
        data_sources=data_sources,
        preprocessing=preprocessing,
        training=training,
        run=run_cfg,
    )

def copy_config_to_run_dir(config, config_path):
    run_dir = Path(config.paths.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config_used.yaml")