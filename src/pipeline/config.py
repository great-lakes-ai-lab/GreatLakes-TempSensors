# src/pipeline/config.py
"""Load and validate pipeline configuration from YAML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import yaml
import shutil
import os

from utils.dates import dates_from_intervals


def _normalize_ranges(x) -> list:
    """
    Normalize a date-range spec into a canonical list of (start, end) tuples.

    Accepts:
        ["2021-01-01", "2021-12-31"]                      -> [("2021-01-01", "2021-12-31")]
        [["2018-01-01","2018-12-31"], ["2020-...","..."]] -> [("2018-01-01","2018-12-31"), (...)]
        [("2021-01-01", "2021-12-31")]                    -> [("2021-01-01", "2021-12-31")]

    Returns
    -------
    list[tuple[str, str]]
    """
    if x is None:
        return []

    # Nested form: first element is itself a list/tuple
    if len(x) > 0 and isinstance(x[0], (list, tuple)):
        intervals = [tuple(item) for item in x]
    else:
        # Flat form: a single [start, end]
        intervals = [tuple(x)]

    # Validate shape
    for iv in intervals:
        if len(iv) != 2:
            raise ValueError(
                f"Each date range interval must have exactly 2 elements "
                f"(start, end). Got: {iv}"
            )
    return intervals

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
    fill_values: Optional[list] = None  # NEW: None → default; [] → none; [..] → explicit


@dataclass
class PreprocessingConfig:
    fit_range: list = field(default_factory=lambda: [("2019-01-01", "2019-12-31")])
    force_reprocess: bool = False


@dataclass
class TrainingConfig:
    train_range: list = field(default_factory=lambda: [("2019-01-01", "2020-12-31")])
    val_range: list = field(default_factory=lambda: [("2021-01-01", "2021-12-31")])
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

    def get_dates(self, val_range) -> list:
        import pandas as pd
        import numpy as np

        if self.dates:
            return pd.to_datetime(self.dates).normalize().tolist()

        all_dates = dates_from_intervals(val_range, subsample_factor=1)

        if len(all_dates) == 0:
            return []

        if self.n_random:
            np.random.seed(self.seed)
            idx = np.random.choice(len(all_dates), size=min(self.n_random, len(all_dates)), replace=False)
            return sorted(all_dates[idx].tolist())

        if self.every_n_days:
            return all_dates[::self.every_n_days].tolist()

        # Default: 5 evenly spaced dates across the full span
        return pd.date_range(all_dates.min(), all_dates.max(), periods=5).normalize().tolist()


@dataclass
class ActiveLearningConfig:
    name: str = "default"  # creates subfolder: active_learning/<name>/
    notes: str = ""

    eval_range: list = field(default_factory=lambda: [("2021-01-01", "2021-12-31")])
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
        from lakes import LAKE_BOUNDS
        import pandas as pd
        import warnings

        # Defensive: ensure canonical list-of-intervals form
        self.preprocessing.fit_range = _normalize_ranges(self.preprocessing.fit_range)
        self.training.train_range = _normalize_ranges(self.training.train_range)
        self.training.val_range = _normalize_ranges(self.training.val_range)
        self.active_learning.eval_range = _normalize_ranges(self.active_learning.eval_range)

        # Lake name
        if self.lake not in LAKE_BOUNDS:
            raise ValueError(f"Unknown lake '{self.lake}'. Available: {list(LAKE_BOUNDS.keys())}")

        # Data source paths
        for name, source in self.data_sources.items():
            p = Path(source.path)
            if not p.exists():
                raise FileNotFoundError(f"Data source '{name}' path not found: {p}")

        # Per-interval date sanity
        def _check_intervals(intervals, label):
            spans = []
            for start_s, end_s in intervals:
                start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
                if start > end:
                    raise ValueError(f"{label} interval start {start} is after end {end}")
                spans.append((start, end))
            # Warn on overlapping intervals within the same set
            spans_sorted = sorted(spans)
            for (s1, e1), (s2, e2) in zip(spans_sorted, spans_sorted[1:]):
                if s2 <= e1:
                    warnings.warn(f"{label} has overlapping intervals: "
                                  f"[{s1.date()},{e1.date()}] and [{s2.date()},{e2.date()}]")
            return spans

        fit_spans = _check_intervals(self.preprocessing.fit_range, "fit_range")
        train_spans = _check_intervals(self.training.train_range, "train_range")
        val_spans = _check_intervals(self.training.val_range, "val_range")
        _check_intervals(self.active_learning.eval_range, "eval_range")

        # Train/val leakage check: warn if any val interval starts before max train end
        max_train_end = max(e for _, e in train_spans)
        min_val_start = min(s for s, _ in val_spans)
        if min_val_start <= max_train_end:
            warnings.warn(
                f"Validation may overlap training (earliest val start {min_val_start.date()} "
                f"<= latest train end {max_train_end.date()})"
            )

        # Output root writable
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
    preprocessing.fit_range = _normalize_ranges(preprocessing.fit_range)

    training = TrainingConfig(**raw.get("training", {}))
    training.train_range = _normalize_ranges(training.train_range)
    training.val_range = _normalize_ranges(training.val_range)

    prediction = PredictionConfig(**raw.get("prediction", {}))

    active_learning = ActiveLearningConfig(**raw.get("active_learning", {}))
    active_learning.eval_range = _normalize_ranges(active_learning.eval_range)

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


def load_al_config(al_config_path: str) -> PipelineConfig:
    """
    Load an Active Learning overlay config.

    Inherits data_sources / lake / preprocessing / training from the referenced
    trained-model run (run_ref), applies path-only overrides, overlays the
    active_learning section, and enforces that no structural field was changed.

    Guardrails:
      - Structural invariant: roles/variables/coarsening/anomalies/source names
        must match run_ref exactly (hard error otherwise).
      - force_reprocess is hard-set False (an AL run must never rebuild the
        training run's processed cache).
      - Model artifacts are verified to exist at the (re-resolved) paths.
    """
    al_path = Path(al_config_path).expanduser()
    with open(al_path) as f:
        al_raw = yaml.safe_load(f)

    if "run_ref" not in al_raw:
        raise ValueError("AL config must specify 'run_ref' (path to a trained run dir).")

    run_ref = Path(al_raw["run_ref"]).expanduser()
    base_cfg_path = run_ref / "config_used.yaml"
    if not base_cfg_path.exists():
        raise FileNotFoundError(
            f"run_ref config not found: {base_cfg_path}. "
            f"AL config must reference a completed training run "
            f"(one containing config_used.yaml + model/)."
        )

    # 1. Base config from the trained-model run (full provenance)
    config = load_config(str(base_cfg_path))

    # 2. Structural signature BEFORE overrides (for the invariant check)
    def _structural_sig(cfg):
        return {
            name: (
                tuple(s.role) if isinstance(s.role, list) else (s.role,),
                s.variable,
                tuple(s.variables) if s.variables else None,
                s.coarsen_factor,
                bool(s.use_anomalies),
            )
            for name, s in cfg.data_sources.items()
        }
    sig_before = _structural_sig(config)

    # 3. Path re-resolution via output_root (Decision 1: option a)
    paths_override = al_raw.get("paths_override") or {}
    new_root = paths_override.get("output_root")
    if new_root:
        config.paths.output_root = str(Path(new_root).expanduser())
        config.paths.resolve(config.run.name)  # re-derive all subpaths consistently

    # 4. Per-source PATH-ONLY overrides
    for name, ov in (al_raw.get("data_source_overrides") or {}).items():
        if name not in config.data_sources:
            raise ValueError(
                f"data_source_overrides references unknown source '{name}'. "
                f"Available: {list(config.data_sources.keys())}"
            )
        for field_name, value in ov.items():
            if field_name != "path":
                raise ValueError(
                    f"data_source_overrides['{name}'] may only override 'path', "
                    f"not '{field_name}' (structural fields are locked to run_ref)."
                )
            config.data_sources[name].path = str(Path(value).expanduser())

    # 5. Structural invariant (Decision 2: hard error)
    sig_after = _structural_sig(config)
    if sig_before != sig_after:
        # Report which sources changed, for a helpful message
        changed = {k for k in sig_before if sig_before[k] != sig_after.get(k)}
        changed |= set(sig_after) ^ set(sig_before)
        raise ValueError(
            f"AL overrides changed data-source structure for: {sorted(changed)}. "
            f"Roles/variables/coarsening/anomalies/source-set must match run_ref "
            f"exactly (only 'path' may be overridden). This would invalidate the "
            f"trained model."
        )

    # 6. Overlay the active_learning section (full replacement)
    config.active_learning = ActiveLearningConfig(**al_raw.get("active_learning", {}))
    config.active_learning.eval_range = _normalize_ranges(config.active_learning.eval_range)

    # 7. Guardrail: never reprocess the training run's cache (Decision 4)
    config.preprocessing.force_reprocess = False

    # 8. Verify model artifacts exist at (re-resolved) paths (Decision 3)
    model_dir = Path(config.paths.model_dir)
    required = [
        model_dir / "model.pt",
        model_dir / "model_config.json",
        model_dir / "data_processor",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Trained model artifacts not found under the resolved model_dir "
            f"({model_dir}). Missing: {missing}. "
            f"Check run_ref and paths_override.output_root."
        )

    # 9. Re-validate (paths exist at new locations, dates sane, writability)
    config.validate()

    # Stash the AL config source path so run_active_learning can archive it (Decision 5)
    config._al_config_source_path = str(al_path)

    return config

def copy_config_to_run_dir(config, config_path):
    run_dir = Path(config.paths.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(config_path, run_dir / "config_used.yaml")
    except shutil.SameFileError:
        pass


