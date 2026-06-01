# src/pipeline/config.py
"""Load and validate pipeline configuration from YAML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
import torch


@dataclass
class PathsConfig:
    data_cache: str
    raw_dir: str = ""
    processed_dir: str = ""
    model_dir: str = ""

    def __post_init__(self):
        base = Path(self.data_cache)
        if not self.raw_dir:
            self.raw_dir = str(base / "raw")
        if not self.processed_dir:
            self.processed_dir = str(base / "processed")
        if not self.model_dir:
            self.model_dir = str(base / "saved_models")


@dataclass
class DataSourceEntry:
    path: str
    format: str  # "netcdf" or "zarr"
    variable: Optional[str] = None  # optional rename hint


@dataclass
class PreprocessingConfig:
    static_coarsen_factor: int = 10
    mask_coarsen_factor: int = 20
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

    # Build PathsConfig (expand ~ to home dir)
    paths_raw = raw.get("paths", {})
    for key, val in paths_raw.items():
        if isinstance(val, str):
            paths_raw[key] = str(Path(val).expanduser())
    paths = PathsConfig(**paths_raw)

    # Build data_sources — expand ~ in each path
    sources_raw = raw.get("data_sources", {})
    data_sources = {}
    for name, entry in sources_raw.items():
        if "path" in entry:
            entry["path"] = str(Path(entry["path"]).expanduser())
        data_sources[name] = DataSourceEntry(**entry)


    # Build other sub-configs
    preprocessing = PreprocessingConfig(**raw.get("preprocessing", {}))
    training = TrainingConfig(**raw.get("training", {}))
    run_cfg = RunConfig(**raw.get("run", {}))

    return PipelineConfig(
        lake=raw.get("lake", "erie"),
        environment=raw.get("environment", "local"),
        paths=paths,
        data_sources=data_sources,
        preprocessing=preprocessing,
        training=training,
        run=run_cfg,
    )