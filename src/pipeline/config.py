# src/pipeline/config.py
"""Load and validate pipeline configuration from YAML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import yaml
import shutil
import os
import warnings
import pandas as pd

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


def _as_ts_pairs(intervals) -> list:
    """Convert canonical (start, end) string pairs to Timestamp pairs."""
    return [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in intervals]


def _merge_intervals(intervals) -> list:
    """Merge overlapping/adjacent intervals into a minimal sorted list."""
    if not intervals:
        return []
    ts = sorted(_as_ts_pairs(intervals))
    merged = [ts[0]]
    for s, e in ts[1:]:
        last_s, last_e = merged[-1]
        # Treat back-to-back days as contiguous
        if s <= last_e + pd.Timedelta(days=1):
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def _find_overlaps(a_intervals, b_intervals) -> list:
    """Return all (a, b) interval pairs that intersect."""
    out = []
    for s1, e1 in _as_ts_pairs(a_intervals):
        for s2, e2 in _as_ts_pairs(b_intervals):
            if s1 <= e2 and s2 <= e1:
                out.append(((s1, e1), (s2, e2)))
    return out


def _fmt_overlaps(pairs) -> str:
    return "; ".join(
        f"[{a[0].date()}→{a[1].date()}] ∩ [{b[0].date()}→{b[1].date()}]"
        for a, b in pairs
    )


def _is_contained(outer_merged, inner) -> bool:
    """True if `inner` (start, end) sits wholly inside one merged outer block."""
    s, e = pd.Timestamp(inner[0]), pd.Timestamp(inner[1])
    return any(os <= s and e <= oe for os, oe in outer_merged)

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
    fit_range: list = field(default_factory=list)   # empty -> inherit train_range
    force_reprocess: bool = False

    # Seasonal cycle / anomaly options
    climatology_method: str = "harmonic"   # "monthly" | "daily_doy" | "harmonic"
    n_harmonics: int = 3                   # method="harmonic"
    smooth_window: int = 15                # method="daily_doy", days; 0 = off

@dataclass
class TrainingConfig:
    train_range: list = field(default_factory=lambda: [("2019-01-01", "2020-12-31")])
    val_range: list = field(default_factory=lambda: [("2021-01-01", "2021-12-31")])
    test_range: list = field(default_factory=list)  # optional; required for `evaluate`

    train_date_mode: str = "random"  # "strided" or "random"
    train_date_stride: int = 5  # mode=strided: min gap in days
    train_date_fraction: float = 0.05  # mode=random: fraction of daily pool per epoch
    n_train_dates_per_epoch: int = None  # mode=random: absolute override of fraction
    val_date_stride: int = 5  # val dates are always deterministic

    n_epochs: int = 50
    lr: float = 5e-5
    internal_density: int = 250
    patience: int = 0  # 0 = no early stopping

    n_context_points: int = 50
    vary_n_context: bool = True
    min_n_context: int = 20
    max_n_context: int = 75

    resample_tasks_per_epoch: bool = True
    train_task_seed: int = 100 # per-epoch seed = train_task_seed + epoch
    include_bathy_as_context: bool = False
    bathy_context_sampling: str = "random_lake_points"  # "all", "random_lake_points", or integer (e.g. 1000)


@dataclass
class RunConfig:
    name: str = "unnamed_run"
    notes: str = ""
    display_plots: str = "auto"  # "auto", "show", or "save_only"


@dataclass
class PredictionConfig:
    split: str = "val"              # "train", "val", or "test"
    dates: list = None              # Specific dates: ["2021-03-15", "2021-07-01"]
    n_random: int = None            # OR: n random dates from the split range
    every_n_days: int = None        # OR: every n days across the split range
    n_context: int = None           # Override training n_context (optional)
    seed: int = 42

    VALID_SPLITS = ("train", "val", "test")

    def resolve_range(self, training) -> list:
        """Resolve `split` to the corresponding canonical interval list."""
        if self.split not in self.VALID_SPLITS:
            raise ValueError(
                f"prediction.split must be one of {self.VALID_SPLITS}, got '{self.split}'"
            )
        ranges = {
            "train": training.train_range,
            "val": training.val_range,
            "test": training.test_range,
        }
        rng = ranges[self.split]
        if not rng:
            raise ValueError(
                f"prediction.split='{self.split}' but training.{self.split}_range is empty."
            )
        return rng

    def get_dates(self, training) -> list:
        import numpy as np

        split_range = self.resolve_range(training)
        all_dates = dates_from_intervals(split_range, subsample_factor=1)

        if self.dates:
            requested = pd.to_datetime(self.dates).normalize()
            outside = [d for d in requested if d not in all_dates]
            if outside:
                warnings.warn(
                    f"prediction.dates contains {len(outside)} date(s) outside "
                    f"prediction.split='{self.split}' "
                    f"(e.g. {outside[0].date()}). Scores will not be split-clean."
                )
            return requested.tolist()

        if len(all_dates) == 0:
            return []

        if self.n_random:
            np.random.seed(self.seed)
            idx = np.random.choice(
                len(all_dates), size=min(self.n_random, len(all_dates)), replace=False
            )
            return sorted(all_dates[idx].tolist())

        if self.every_n_days:
            return all_dates[::self.every_n_days].tolist()

        # Default: 5 evenly spaced dates across the full span
        return pd.date_range(all_dates.min(), all_dates.max(), periods=5).normalize().tolist()


@dataclass
class EvaluationConfig:
    split: str = "test"                      # train, val, or test
    date_subsample_factor: int = 7           # stride over the split range
    n_context_sweep: list = field(
        default_factory=lambda: [5, 10, 25, 50, 100, 200]
    )
    seeds: list = field(default_factory=lambda: [0, 1, 2])

    VALID_SPLITS = ("train", "val", "test")


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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def validate(self):
        from lakes import LAKE_BOUNDS

        # PipelineConfig.validate(), after the _normalize_ranges block
        if not self.preprocessing.fit_range:
            self.preprocessing.fit_range = list(self.training.train_range)
            print(f"fit_range not set — inheriting train_range: {self.preprocessing.fit_range}")
        elif self.preprocessing.fit_range != self.training.train_range:
            warnings.warn(
                f"preprocessing.fit_range {self.preprocessing.fit_range} differs from "
                f"training.train_range {self.training.train_range}. Normalization and "
                f"climatology statistics will be fitted on a different period than the "
                f"model trains on. Andersson et al. use a single period for both."
            )

        # Defensive: ensure canonical list-of-intervals form
        self.preprocessing.fit_range = _normalize_ranges(self.preprocessing.fit_range)
        self.training.train_range = _normalize_ranges(self.training.train_range)
        self.training.val_range = _normalize_ranges(self.training.val_range)
        self.training.test_range = _normalize_ranges(self.training.test_range)
        self.active_learning.eval_range = _normalize_ranges(self.active_learning.eval_range)

        valid_methods = ("monthly", "daily_doy", "harmonic")
        if self.preprocessing.climatology_method not in valid_methods:
            raise ValueError(
                f"climatology_method must be one of {valid_methods}, "
                f"got '{self.preprocessing.climatology_method}'"
            )
        if self.preprocessing.n_harmonics < 1:
            raise ValueError("n_harmonics must be >= 1")


        # Lake name
        if self.lake not in LAKE_BOUNDS:
            raise ValueError(f"Unknown lake '{self.lake}'. Available: {list(LAKE_BOUNDS.keys())}")

        # Data source paths
        for name, source in self.data_sources.items():
            p = Path(source.path)
            if not p.exists():
                raise FileNotFoundError(f"Data source '{name}' path not found: {p}")

        # ------------------------------------------------------------------
        # Per-set interval sanity (start <= end; warn on self-overlap)
        # ------------------------------------------------------------------
        def _check_intervals(intervals, label):
            if not intervals:
                return []
            spans = []
            for start_s, end_s in intervals:
                start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
                if start > end:
                    raise ValueError(
                        f"{label} interval start {start.date()} is after end {end.date()}"
                    )
                spans.append((start, end))
            spans_sorted = sorted(spans)
            for (s1, e1), (s2, e2) in zip(spans_sorted, spans_sorted[1:]):
                if s2 <= e1:
                    warnings.warn(
                        f"{label} has overlapping intervals: "
                        f"[{s1.date()},{e1.date()}] and [{s2.date()},{e2.date()}]"
                    )
            return spans

        _check_intervals(self.preprocessing.fit_range, "fit_range")
        _check_intervals(self.training.train_range, "train_range")
        _check_intervals(self.training.val_range, "val_range")
        _check_intervals(self.training.test_range, "test_range")
        _check_intervals(self.active_learning.eval_range, "eval_range")

        if not self.training.train_range:
            raise ValueError("training.train_range must not be empty.")
        if not self.training.val_range:
            raise ValueError("training.val_range must not be empty.")

        # Date-selection sanity
        tc = self.training
        if tc.train_date_mode not in ("strided", "random"):
            raise ValueError(
                f"train_date_mode must be 'strided' or 'random', got '{tc.train_date_mode}'"
            )
        if tc.train_date_mode == "random":
            if not (0 < tc.train_date_fraction <= 1):
                raise ValueError(
                    f"train_date_fraction must be in (0, 1], got {tc.train_date_fraction}"
                )
            if not tc.resample_tasks_per_epoch:
                raise ValueError(
                    "train_date_mode='random' requires resample_tasks_per_epoch=True "
                    "(dates can only be redrawn if tasks are regenerated each epoch)."
                )
        if tc.train_date_stride < 1 or tc.val_date_stride < 1:
            raise ValueError("train_date_stride and val_date_stride must be >= 1")

        # ------------------------------------------------------------------
        # Cross-set leakage: train/val/test must be mutually disjoint
        # ------------------------------------------------------------------
        sets = {
            "train_range": self.training.train_range,
            "val_range": self.training.val_range,
            "test_range": self.training.test_range,
        }
        pairs = [
            ("train_range", "val_range"),
            ("train_range", "test_range"),
            ("val_range", "test_range"),
        ]
        for a, b in pairs:
            if not sets[a] or not sets[b]:
                continue
            overlaps = _find_overlaps(sets[a], sets[b])
            if overlaps:
                raise ValueError(
                    f"Data leakage: {a} overlaps {b} → {_fmt_overlaps(overlaps)}. "
                    f"train/val/test must be mutually disjoint."
                )

        # ------------------------------------------------------------------
        # fit_range must sit wholly inside train_range
        #   DataProcessor normalization stats and the seasonal climatology are
        #   fit on fit_range; if it touches val/test those statistics leak.
        # ------------------------------------------------------------------
        if not self.preprocessing.fit_range:
            raise ValueError("preprocessing.fit_range must not be empty.")

        train_merged = _merge_intervals(self.training.train_range)
        bad_fit = [
            iv for iv in self.preprocessing.fit_range
            if not _is_contained(train_merged, iv)
        ]
        if bad_fit:
            bad_str = "; ".join(
                f"[{pd.Timestamp(s).date()}→{pd.Timestamp(e).date()}]" for s, e in bad_fit
            )
            train_str = "; ".join(
                f"[{s.date()}→{e.date()}]" for s, e in train_merged
            )
            raise ValueError(
                f"fit_range interval(s) {bad_str} are not contained within "
                f"train_range {train_str}. Normalization statistics and the "
                f"seasonal climatology must be fit on training data only."
            )

        # ------------------------------------------------------------------
        # Full-year coverage advisory
        #   Committed project constraint: every split spans >= 1 full year so
        #   seasonal coverage is complete and month-stratified scores are valid.
        # ------------------------------------------------------------------
        for label, intervals in sets.items():
            if not intervals:
                continue
            total_days = sum(
                (pd.Timestamp(e) - pd.Timestamp(s)).days + 1 for s, e in intervals
            )
            if total_days < 365:
                warnings.warn(
                    f"{label} spans only {total_days} days (<1 year). "
                    f"Seasonal coverage is incomplete; RMSE will not be "
                    f"comparable across splits."
                )

        # ------------------------------------------------------------------
        # Prediction split must be populated
        # ------------------------------------------------------------------
        if self.prediction.split not in PredictionConfig.VALID_SPLITS:
            raise ValueError(
                f"prediction.split must be one of {PredictionConfig.VALID_SPLITS}, "
                f"got '{self.prediction.split}'"
            )
        if self.prediction.split == "test" and not self.training.test_range:
            raise ValueError(
                "prediction.split='test' but training.test_range is empty."
            )

        # Evaluation split must be populated
        if self.evaluation.split not in EvaluationConfig.VALID_SPLITS:
            raise ValueError(
                f"evaluation.split must be one of {EvaluationConfig.VALID_SPLITS}, "
                f"got '{self.evaluation.split}'"
            )
        if not sets.get(f"{self.evaluation.split}_range"):
            raise ValueError(
                f"evaluation.split='{self.evaluation.split}' but "
                f"training.{self.evaluation.split}_range is empty."
            )
        if not self.evaluation.n_context_sweep:
            raise ValueError("evaluation.n_context_sweep must not be empty.")
        if not self.evaluation.seeds:
            raise ValueError("evaluation.seeds must not be empty.")

        # ------------------------------------------------------------------
        # Output root writable
        # ------------------------------------------------------------------
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

    training_raw = dict(raw.get("training", {}))
    if "date_subsample_factor" in training_raw:
        old = training_raw.pop("date_subsample_factor")
        training_raw.setdefault("train_date_stride", old)
        training_raw.setdefault("val_date_stride", old)
        warnings.warn(
            f"'date_subsample_factor' is deprecated; mapped to "
            f"train_date_stride={old} and val_date_stride={old}. "
            f"Use the explicit keys instead.")
    if "resample_dates_per_epoch" in training_raw:
        old = training_raw.pop("resample_dates_per_epoch")
        training_raw.setdefault("train_date_mode", "random" if old else "strided")
        warnings.warn(
            f"'resample_dates_per_epoch' is deprecated; "
            f"train_date mode 'random' will shuffle per epoch. "
            f"train_date mode 'strided' will use same dates per epoch. "
        )
    training = TrainingConfig(**training_raw)


    training.train_range = _normalize_ranges(training.train_range)
    training.val_range = _normalize_ranges(training.val_range)
    training.test_range = _normalize_ranges(training.test_range)

    prediction = PredictionConfig(**raw.get("prediction", {}))
    evaluation = EvaluationConfig(**raw.get("evaluation", {}))

    active_learning = ActiveLearningConfig(**raw.get("active_learning", {}))
    active_learning.eval_range = _normalize_ranges(active_learning.eval_range)

    return PipelineConfig(
        lake=raw.get("lake", "erie"),
        environment=raw.get("environment", "local"),
        paths=paths,
        data_sources=data_sources,
        preprocessing=preprocessing,
        training=training,
        evaluation=evaluation,
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


