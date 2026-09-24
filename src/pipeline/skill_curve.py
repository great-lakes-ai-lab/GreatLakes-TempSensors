# src/pipeline/skill_curve.py
"""Skill-curve stage: model error vs. number of actively-placed sensors.

Reproduces the Andersson et al. (2023) Fig. 5 analysis. Reads one or more
completed active-learning experiments and, for each, walks the *prefix* of the
greedy recommendation list:

    context_k = baseline_context  U  recommendations[:k],   k = 0 ... N

scoring each k on a held-out split. Greedy curves are overlaid on a
random-placement envelope evaluated at matched total context size.

Design notes
------------
*   Greedy is NOT re-run. A single GreedyAlgorithm call already produces a
    nested ordering, so an open-loop sweep over prefixes is both correct and
    O(N) rather than O(N^2).
*   Y values at new sensor locations come from real data via TaskLoader.
    Model infill never enters the skill curve.
*   Scoring is delegated to utils.metrics.compute_weighted_scores -- the same
    estimator used during training and by evaluate.py -- so this y-axis is
    directly comparable to evaluation/<split>/skill_vs_ncontext.png.
*   The random baseline is nested too: one draw of (k_max) points per seed,
    then prefixes. This matches greedy's nesting instead of introducing extra
    variance from independent redraws at each k.

Produces:
    skill_curve/<name>/skill_curve.csv
    skill_curve/<name>/per_task_scores.csv
    skill_curve/<name>/skill_curve_metrics.json
    skill_curve/<name>/skill_vs_nsensors.png
"""

from __future__ import annotations

import copy
import json
import shutil
import warnings
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pipeline.model import load_trained_model
from pipeline.task_builder import gen_tasks
from pipeline.evaluate import _resolve_split_range
from pipeline.active_learning import get_x1_name, get_x2_name, get_pre_dp_target_grid, make_spatial_template, find_lat_lon_names
from utils.dates import dates_from_intervals
from utils.metrics import compute_weighted_scores


# Filenames run_active_learning may have written the recommendations to.
_REC_FILENAME_CANDIDATES = (
    "recommended_locations.csv",
    "recommended_sensors.csv",
    "X_new.csv",
)


# ---------------------------------------------------------------------
# Spatial skill-map accumulation
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Spatial skill-map accumulation
# ---------------------------------------------------------------------

class SkillMapAccumulator:
    """
    Bins point-level predictions onto the native target grid and accumulates
    time-sums, so per-cell sigma / RMSE / bias fall out for free from the
    predictions compute_weighted_scores already makes.

    One instance per (experiment, k). Nearest-cell binning via searchsorted
    against the grid's own 1-D coords -- the target points originate from that
    grid, so this is an exact inverse in practice, not an interpolation.
    """

    def __init__(self, template):
        self.lat_name, self.lon_name = find_lat_lon_names(template)
        self.lats = np.asarray(template[self.lat_name].values, dtype=float)
        self.lons = np.asarray(template[self.lon_name].values, dtype=float)
        self.template = template

        # searchsorted needs ascending coords; GLSEA lat may descend
        self._lat_desc = self.lats.size > 1 and self.lats[0] > self.lats[-1]
        self._lat_sorted = self.lats[::-1] if self._lat_desc else self.lats
        self._lon_desc = self.lons.size > 1 and self.lons[0] > self.lons[-1]
        self._lon_sorted = self.lons[::-1] if self._lon_desc else self.lons

        shape = (self.lats.size, self.lons.size)
        self.n = np.zeros(shape, dtype=np.int32)
        self.sum_std = np.zeros(shape, dtype=np.float64)
        self.sum_var = np.zeros(shape, dtype=np.float64)
        self.sum_err = np.zeros(shape, dtype=np.float64)
        self.sum_sq_err = np.zeros(shape, dtype=np.float64)
        self.sum_abs_err = np.zeros(shape, dtype=np.float64)
        self.sum_true = np.zeros(shape, dtype=np.float64)

    def _nearest_idx(self, values, sorted_coords, descending):
        idx = np.searchsorted(sorted_coords, values)
        idx = np.clip(idx, 1, sorted_coords.size - 1)
        left = sorted_coords[idx - 1]
        right = sorted_coords[idx]
        idx = np.where(np.abs(values - left) <= np.abs(right - values),
                       idx - 1, idx)
        if descending:
            idx = sorted_coords.size - 1 - idx
        return idx

    def __call__(self, task, lats, lons, mean, true, std, w):
        """point_sink signature; w is unused (maps are per-cell, unweighted)."""
        if lats.size == 0:
            return
        i = self._nearest_idx(lats, self._lat_sorted, self._lat_desc)
        j = self._nearest_idx(lons, self._lon_sorted, self._lon_desc)
        flat = i * self.lons.size + j

        err = mean - true
        for arr, vals in (
            (self.sum_std, std),
            (self.sum_var, std ** 2),
            (self.sum_err, err),
            (self.sum_sq_err, err ** 2),
            (self.sum_abs_err, np.abs(err)),
            (self.sum_true, true),
        ):
            np.add.at(arr.reshape(-1), flat, vals)
        np.add.at(self.n.reshape(-1), flat, 1)

    def to_dataset(self):
        import xarray as xr

        n = self.n.astype(float)
        valid = n > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            def _mean(s):
                return np.where(valid, s / np.where(valid, n, 1.0), np.nan)

            mean_std = _mean(self.sum_std)
            rmse = np.sqrt(_mean(self.sum_sq_err))
            bias = _mean(self.sum_err)
            mae = _mean(self.sum_abs_err)
            mean_true = _mean(self.sum_true)

        dims = (self.lat_name, self.lon_name)
        coords = {self.lat_name: self.lats, self.lon_name: self.lons}

        def _da(data, long_name, units):
            return xr.DataArray(
                data.astype(np.float32), dims=dims, coords=coords,
                attrs={"long_name": long_name, "units": units},
            )

        return xr.Dataset(
            {
                "mean_std": _da(mean_std, "time-mean predictive std dev",
                                "degC"),
                "rmse": _da(rmse, "per-cell RMSE over time", "degC"),
                "mae": _da(mae, "per-cell MAE over time", "degC"),
                "bias": _da(bias, "per-cell mean error (pred - truth)",
                            "degC"),
                "mean_truth": _da(mean_true, "time-mean observed anomaly",
                                  "degC"),
                "n_obs": xr.DataArray(
                    self.n, dims=dims, coords=coords,
                    attrs={"long_name": "tasks contributing to this cell"},
                ),
            },
            attrs={"note": "SST anomaly space (seasonal cycle removed)"},
        )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def run_skill_curve(config, bundle, tl_config):
    """Score greedy sensor-placement prefixes against a random envelope."""
    sc = config.skill_curve
    sc.validate()

    out_dir = Path(config.paths.resolve_skill_curve(sc.name))
    out_dir.mkdir(parents=True, exist_ok=True)

    # split_range = _resolve_split_range(config, sc.split)
    dates = dates_from_intervals(sc.eval_range, sc.date_subsample_factor)
    if len(dates) == 0:
        raise ValueError(
            f"No dates resolved from range={sc.eval_range} "
            f"with date_subsample_factor={sc.date_subsample_factor}"
        )

    experiments = _discover_experiments(config, sc)
    if not experiments:
        raise RuntimeError(
            f"No active-learning experiments found to score under "
            f"{config.paths.active_learning_dir}. Run the active_learning "
            f"stage first, or set skill_curve.al_experiments explicitly."
        )

    print(f"\n=== Skill curve '{sc.name}' ===")
    print(f"  Using  ({len(dates)} dates, "
          f"{dates.min().date()} to {dates.max().date()})")
    print(f"  Experiments: {[e['name'] for e in experiments]}")

    model = load_trained_model(config, bundle, tl_config.task_loader)

    rows = []
    per_task = []
    skill_map_stores = {}
    warnings_log = []
    baselines = {}  # label -> (X_base, k_max)

    # --- Greedy curves -------------------------------------------------
    for exp in experiments:
        exp_rows, exp_per_task, smaps, warns, X_base, k_max = _score_experiment(
            exp, config, bundle, tl_config, model, dates, sc,
        )
        rows.extend(exp_rows)
        per_task.extend(exp_per_task)
        warnings_log.extend(warns)
        if smaps:
            skill_map_stores[exp["name"]] = smaps

        # Dedupe: experiments sharing a baseline (e.g. several acquisition
        # functions off the same buoy network) need only one envelope.
        sig = (X_base.shape[1], float(np.sum(X_base)), float(np.sum(X_base ** 2)))
        prev = baselines.get(sig)
        if prev is None or k_max > prev[2]:
            baselines[sig] = (exp["name"], X_base, max(k_max, prev[2] if prev else 0))

    # --- Random placements -----------------------------------------------
    # Find the search mask
    search_mask = xr.open_dataset(list(experiments[0]['dir'].glob("search_mask.nc"))[0])
    if sc.random_baseline:
        for label, X_base, k_max in baselines.values():
            base_rows, base_per_task = _score_random_baseline(
                config, bundle, tl_config, model, dates, sc,
                X_base_actual=X_base, k_max=k_max, label=label,
                search_mask=search_mask
            )
            rows.extend(base_rows)
            per_task.extend(base_per_task)
    if not rows:
        raise RuntimeError("Skill curve produced no results.")

    curve_df = pd.DataFrame(rows)
    per_task_df = pd.DataFrame(per_task)

    if sc.random_baseline and sc.random_mode == "augment":
        k0 = curve_df[curve_df.k == 0]
        g = k0[k0.placement == "greedy"]["rmse"]
        r = k0[k0.placement == "random"]["rmse"]
        if len(g) and len(r) and not np.allclose(r, g.iloc[0], rtol=1e-6):
            w = ("k=0 RMSE differs between greedy and random-augment "
                 f"(greedy={g.iloc[0]:.6f}, random={sorted(set(r.round(6)))}). "
                 "In augment mode both use the identical baseline context, so "
                 "these must match -- suspect baseline reconstruction or "
                 "nondeterminism in task generation.")
            warnings.warn(w, UserWarning)
            warnings_log.append(w)

    curve_df.to_csv(out_dir / "skill_curve.csv", index=False)
    if not per_task_df.empty:
        per_task_df.to_csv(out_dir / "per_task_scores.csv", index=False)

    agg = _aggregate(curve_df)

    results = _write_metrics_json(
        config, sc, out_dir, dates, sc.eval_range,
        experiments, curve_df, agg, warnings_log,
    )

    _archive_config(config, out_dir)

    if skill_map_stores:
        _save_skill_maps(skill_map_stores, out_dir)

    _plot_skill_curve(config, curve_df, agg, sc, out_dir)

    print(f"\nSkill curve complete. Outputs: {out_dir}")
    return {"curve": curve_df, "aggregate": agg, "metrics": results}


# ---------------------------------------------------------------------
# Experiment discovery / loading
# ---------------------------------------------------------------------
def _discover_experiments(config, sc) -> list:
    """Resolve which AL experiment dirs to score, and load their metadata."""
    al_root = Path(config.paths.active_learning_dir)

    if sc.al_experiments:
        names = list(sc.al_experiments)
    else:
        if not al_root.exists():
            return []
        names = sorted(
            p.name for p in al_root.iterdir()
            if p.is_dir() and _find_recommendations_csv(p) is not None
        )
        if names:
            print(f"  Auto-discovered AL experiments: {names}")

    experiments = []
    for name in names:
        exp_dir = config.paths.resolve_active_learning(name)
        if not exp_dir.exists():
            raise FileNotFoundError(
                f"AL experiment directory not found: {exp_dir}"
            )
        experiments.append(_load_experiment(exp_dir, name))
    return experiments


def _find_recommendations_csv(exp_dir: Path):
    for fname in _REC_FILENAME_CANDIDATES:
        p = exp_dir / fname
        if p.exists():
            return p
    # Fall back to any single CSV that isn't a context-point dump
    csvs = [
        p for p in exp_dir.glob("*.csv")
        if "context" not in p.name.lower()
    ]
    return csvs[0] if len(csvs) == 1 else None


def _load_experiment(exp_dir: Path, name: str) -> dict:
    """Read al_config.json + recommendation CSV for one AL experiment."""
    cfg_path = exp_dir / "al_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing al_config.json in {exp_dir}. The skill curve needs it to "
            f"rebuild the exact baseline context that greedy conditioned on."
        )
    with open(cfg_path) as f:
        al_cfg = json.load(f)

    rec_path = _find_recommendations_csv(exp_dir)
    if rec_path is None:
        raise FileNotFoundError(
            f"No recommendations CSV found in {exp_dir}. "
            f"Looked for {_REC_FILENAME_CANDIDATES}."
        )
    rec_df = pd.read_csv(rec_path)

    return {
        "name": name,
        "dir": exp_dir,
        "al_config": al_cfg,
        "recommendations_path": rec_path,
        "recommendations": rec_df,
        "acquisition_function": al_cfg.get("acquisition_function", "unknown"),
        "n_context_baseline": int(al_cfg.get("n_context", 0)),
    }


def _extract_recommendation_coords(rec_df, model, exp_name) -> tuple:
    """
    Pull ordered (lat, lon) from a recommendations CSV.

    Returns
    -------
    raw : np.ndarray, shape (2, N), raw lat/lon in priority order
    greedy_iters : list[int] or None
        Original greedy iteration indices, if the CSV recorded them.
    """
    x1_name = get_x1_name(model)
    x2_name = get_x2_name(model)

    cols = {c.lower(): c for c in rec_df.columns}

    def _pick(candidates):
        for c in candidates:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    lat_col = _pick([x1_name, "lat", "latitude", "x1"])
    lon_col = _pick([x2_name, "lon", "longitude", "x2"])

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Could not find lat/lon columns in recommendations for "
            f"'{exp_name}'. Columns present: {list(rec_df.columns)}. "
            f"Expected something matching "
            f"('{x1_name}'/lat/latitude, '{x2_name}'/lon/longitude)."
        )

    df = rec_df
    # Preserve priority order. Prefer an explicit priority column if present.
    prio_col = _pick(["priority"])
    if prio_col is not None:
        df = df.sort_values(prio_col)

    raw = np.asarray(
        [df[lat_col].to_numpy(float), df[lon_col].to_numpy(float)]
    )

    iter_col = _pick(["greedy_iteration"])
    greedy_iters = (
        df[iter_col].astype(int).tolist() if iter_col is not None else None
    )

    return raw, greedy_iters


def _rebuild_baseline_context(config, bundle, al_cfg) -> np.ndarray:
    """
    Reconstruct the exact baseline context array that greedy conditioned on.

    Rebuilds an ActiveLearningConfig from the archived al_config.json and
    replays resolve_context_points against a shallow config copy, so the k=0
    point of the curve is genuinely the pre-placement skill.
    """
    from pipeline.config import ActiveLearningConfig
    from pipeline.active_learning import resolve_context_points

    valid = {f.name for f in dataclass_fields(ActiveLearningConfig)}
    kwargs = {k: v for k, v in al_cfg.items() if k in valid}

    shadow = copy.copy(config)
    shadow.active_learning = ActiveLearningConfig(**kwargs)

    return resolve_context_points(shadow, bundle)


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------

def _score_prefix_sweep(
    label_fields: dict,
    X_base: np.ndarray,
    X_extra: np.ndarray,
    k_values,
    config, bundle, tl_config, model, dates, sc,
    map_ks=None,
    map_template=None
):
    """
    Score context = X_base U X_extra[:, :k] for each k.

    Shared by the greedy and random-baseline paths so both use an identical
    task-construction and scoring code path.
    """
    rows, per_task_rows = [], []
    skill_maps = {}
    map_ks = map_ks or set()

    for k in k_values:
        if k > 0:
            X_k = np.concatenate([X_base, X_extra[:, :k]], axis=1)
        else:
            X_k = X_base

        tasks = gen_tasks(
            tl_config, dates, bundle, config,
            n_context=X_k.shape[1],
            vary_n_context=False,
            fixed_context_points=X_k,
            progress=False,
            verbose=False,
        )
        if not tasks:
            print(f"    k={k}: no tasks generated, skipping")
            continue

        acc = (
            SkillMapAccumulator(map_template)
            if (k in map_ks and map_template is not None) else None
        )

        scores = compute_weighted_scores(
            model, tasks, bundle, tl_config.task_loader,
            return_per_task=True,
            point_sink=acc,
        )

        if acc is not None:
            skill_maps[int(k)] = acc.to_dataset()

        row = {
            **label_fields,
            "k": int(k),
            "n_context_total": int(X_k.shape[1]),
            "n_tasks": scores["n_tasks"],
            "rmse": scores["rmse"],
            "mae": scores["mae"],
            "bias": scores["bias"],
            "nll": scores["nll"],
            "coverage_1sigma": scores["coverage_1sigma"],
            "coverage_2sigma": scores["coverage_2sigma"],
            "mean_std": scores["mean_std"],
        }

        if sc.compute_joint_nll:
            row["joint_nll"] = _joint_nll(model, tasks)

        rows.append(row)
        for r in scores["per_task"]:
            per_task_rows.append({**label_fields, "k": int(k), **r})

        msg = (f"    k={k:>3} n_ctx={X_k.shape[1]:>4} | "
               f"rmse={row['rmse']:.4f}  nll={row['nll']:.4f}  "
               f"mean_std={row['mean_std']:.4f}")
        if sc.compute_joint_nll:
            msg += f"  jnll={row['joint_nll']:.4f}"
        print(msg)


    return rows, per_task_rows, skill_maps


def _score_experiment(exp, config, bundle, tl_config, model, dates, sc):
    """Greedy prefix sweep for one AL experiment."""
    name = exp["name"]
    print(f"\n  [greedy] {name} "
          f"(acquisition={exp['acquisition_function']})")

    warns = []

    X_base = _rebuild_baseline_context(config, bundle, exp["al_config"])
    raw_new, greedy_iters = _extract_recommendation_coords(
        exp["recommendations"], model, name
    )

    dp = bundle["data_processor"]
    X_new = dp.map_coord_array(raw_new, unnorm=False)

    n_avail = X_new.shape[1]
    k_max = n_avail if sc.k_max is None else min(sc.k_max, n_avail)
    if sc.k_max is not None and sc.k_max > n_avail:
        w = (f"skill_curve.k_max={sc.k_max} exceeds the {n_avail} "
             f"recommendation(s) available in '{name}'; truncating to "
             f"{n_avail}.")
        warnings.warn(w, UserWarning)
        warns.append(w)

    # min-dist post-filtering in AL can break the nesting assumption
    if greedy_iters is not None:
        expected = list(range(len(greedy_iters)))
        if greedy_iters != expected:
            w = (f"Experiment '{name}' has non-contiguous greedy_iteration "
                 f"{greedy_iters}: min_dist_from_existing_km discarded "
                 f"picks post-hoc, so prefix k is 'sensors added', not "
                 f"'greedy's top-k'. Later picks were conditioned on sensors "
                 f"absent from this curve.")
            warnings.warn(w, UserWarning)
            warns.append(w)

    print(f"    baseline context: {X_base.shape[1]} points | "
          f"sweeping k = 0..{k_max}")

    map_ks = sc.resolve_map_ks(k_max)
    map_template = (
        make_spatial_template(get_pre_dp_target_grid(bundle, config))
        if map_ks else None
    )
    if map_ks:
        print(f"    skill maps at k = {sorted(map_ks)}")


    rows, per_task_rows, skill_maps = _score_prefix_sweep(
        label_fields={
            "experiment": name,
            "placement": "greedy",
            "acquisition_function": exp["acquisition_function"],
            "seed": -1,
        },
        X_base=X_base,
        X_extra=X_new,
        k_values=range(k_max + 1),
        config=config, bundle=bundle, tl_config=tl_config,
        model=model, dates=dates, sc=sc,
        map_ks=map_ks,
        map_template=map_template
        )


    # Placement coords travel with the maps so figures are self-contained
    if skill_maps:
        skill_maps["_placements"] = pd.DataFrame({
            "k": np.arange(1, X_new.shape[1] + 1),
            "lat": raw_new[0],
            "lon": raw_new[1],
        })
        # This is a temporary check
        acc_n = skill_maps[0]["n_obs"].values.sum()
        print(f"binned {acc_n} points; expected {len(dates)} × n_target")
        """
        Total binned points should equal the sum of n_target across per-task rows, and no cell should exceed len(dates). If cells exceed the date count, two target points collided into one cell — meaning map_coord_array(unnorm=True) isn't recovering the original grid coords exactly and you'd want method="nearest" reindexing instead.
Y_t_aux NaNs. Still outstanding from my previous message, and n_obs now gives you a free diagnostic: if n_obs is systematically zero or low along the shoreline, that's bathymetry NaNs silently deleting nearshore targets in every stage, including your existing evaluation_metrics.json
        """

    return rows, per_task_rows, skill_maps, warns, X_base, k_max


def _score_random_baseline(
    config, bundle, tl_config, model, dates, sc,
    X_base_actual: np.ndarray, k_max: int, label: str, search_mask,
):
    """
    Random-placement envelope, matched to a greedy curve's starting network.

    random_mode="augment" (primary): the AL experiment's real baseline context
    is held fixed and k *random* points are added. Directly comparable to the
    greedy curve -- identical network at k=0, so the two curves must coincide
    there -- and isolates placement quality from context-count effects.

    random_mode="replace": the whole (n_base + k) set is redrawn, answering the
    different question of whether an actively-designed network beats a random
    network of equal size. Flatters greedy, since the baseline loses the real
    buoy geometry. Secondary result only.
    """
    from utils.coordinates import generate_random_coordinates
    from utils.seeds import derive_rng

    n_base = X_base_actual.shape[1]
    print(f"\n  [random] envelope for '{label}', mode={sc.random_mode}, "
          f"n_base={n_base}, k = 0..{k_max}")

    rows, per_task_rows = [], []
    dp = bundle["data_processor"]
    if sc.random_mode == "augment":
        mask = search_mask
    else: # replace exisiting with random
        mask = bundle["lakemask_sampling"]

    for seed in sc.random_seeds:
        print(f"    seed={seed}")
        rng = derive_rng(seed, label, "random_placement")

        if sc.random_mode == "augment":
            X_base = X_base_actual
            X_extra = generate_random_coordinates(
                mask, N=max(k_max, 1), data_processor=dp, rng=rng,
            )
        else:  # "replace"
            # One draw, then split: guarantees the n_base and k sets are
            # disjoint (generate_random_coordinates is without-replacement
            # only *within* a call) and makes the prefixes properly nested.
            X_all = generate_random_coordinates(
                mask, N=n_base + max(k_max, 1), data_processor=dp, rng=rng,
            )
            X_base, X_extra = X_all[:, :n_base], X_all[:, n_base:]

        r, pt, _ = _score_prefix_sweep(
            label_fields={
                "experiment": f"random_{sc.random_mode}_{label}",
                "placement": "random",
                "acquisition_function": f"random_{sc.random_mode}",
                "seed": int(seed),
            },
            X_base=X_base,
            X_extra=X_extra,
            k_values=range(k_max + 1),
            config=config, bundle=bundle, tl_config=tl_config,
            model=model, dates=dates,
            sc=sc,
            map_ks=set()
        )
        rows.extend(r)
        per_task_rows.extend(pt)

    return rows, per_task_rows

# ---------------------------------------------------------------------
# Optional expensive metrics / dumps
# ---------------------------------------------------------------------

def _joint_nll(model, tasks) -> float:
    """
    Mean per-target-point joint NLL, -logpdf(task) / N_t.

    Unlike the marginal NLL from compute_weighted_scores, this credits the
    model for getting the *covariance* right -- which is what the
    joint_entropy acquisition function actually optimises. Covariance can be
    numerically non-positive-definite on large target sets, so failures are
    recorded as NaN rather than raised.
    """
    import torch

    vals = []
    n_failed = 0
    for task in tasks:
        try:
            with torch.no_grad():
                lp = float(model.logpdf(task))
            n_t = int(np.asarray(task["Y_t"][0]).size)
            vals.append(-lp / max(n_t, 1))
        except Exception:
            n_failed += 1
            vals.append(np.nan)

    if n_failed:
        print(f"      joint NLL: {n_failed}/{len(tasks)} tasks failed "
              f"(non-PD covariance), excluded from mean")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(vals)) if vals else float("nan")

# ---------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------

def _aggregate(curve_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse seeds within each (experiment, k).

    Greedy rows are deterministic (single seed=-1), so mean == value and std
    is 0; random rows get a genuine spread that becomes the envelope.
    """
    metric_cols = [
        "rmse", "nll", "mae", "bias",
        "coverage_1sigma", "coverage_2sigma", "mean_std",
    ]
    if "joint_nll" in curve_df.columns:
        metric_cols.append("joint_nll")

    agg_spec = {}
    for m in metric_cols:
        agg_spec[f"{m}_mean"] = (m, "mean")
        agg_spec[f"{m}_std"] = (m, "std")
        agg_spec[f"{m}_min"] = (m, "min")
        agg_spec[f"{m}_max"] = (m, "max")

    agg_spec.update({
        "n_context_total": ("n_context_total", "first"),
        "n_tasks": ("n_tasks", "first"),
        "n_seeds": ("seed", "nunique"),
    })

    return (
        curve_df
        .groupby(["experiment", "placement", "acquisition_function", "k"],
                 as_index=False)
        .agg(**agg_spec)
        .fillna(0.0)
        .sort_values(["placement", "experiment", "k"])
        .reset_index(drop=True)
    )

def _write_metrics_json(
        config, sc, out_dir, dates, split_range,
        experiments, curve_df, agg, warnings_log,
    ):
    train_meta = {}
    meta_path = Path(config.paths.model_dir) / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            tm = json.load(f)
        train_meta = {
            "best_val_rmse": tm.get("best_val_rmse"),
            "best_epoch": tm.get("best_epoch"),
            "total_epochs": tm.get("total_epochs"),
            "checkpoint_selected_on": tm.get("checkpoint_selected_on", "val"),
        }

    # Headline numbers: greedy improvement from k=0 to k=k_max, and where the
    # greedy curve sits relative to the random envelope.
    summary = {}
    for exp_name, grp in curve_df[curve_df.placement == "greedy"].groupby(
            "experiment"):
        grp = grp.sort_values("k")
        k0, kn = grp.iloc[0], grp.iloc[-1]
        rnd = curve_df[(curve_df.placement == "random") &
                       (curve_df.k == kn["k"])]
        summary[exp_name] = {
            "acquisition_function": kn["acquisition_function"],
            "k_max": int(kn["k"]),
            "rmse_k0": float(k0["rmse"]),
            "rmse_kmax": float(kn["rmse"]),
            "rmse_reduction": float(k0["rmse"] - kn["rmse"]),
            "rmse_reduction_pct": float(
                100.0 * (k0["rmse"] - kn["rmse"]) / k0["rmse"]
            ) if k0["rmse"] else None,
            "nll_k0": float(k0["nll"]),
            "nll_kmax": float(kn["nll"]),
            "random_rmse_mean_at_kmax": (
                float(rnd["rmse"].mean()) if not rnd.empty else None
            ),
            "beats_random_mean_at_kmax": (
                bool(kn["rmse"] < rnd["rmse"].mean())
                if not rnd.empty else None
            ),
        }

    results = {
        "run_name": config.run.name,
        "lake": config.lake,
        "skill_curve_name": sc.name,
        "notes": sc.notes,
        "split_range": [list(iv) for iv in split_range],
        "date_subsample_factor": sc.date_subsample_factor,
        "n_dates": int(len(dates)),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "metric": "area-weighted (cos-lat) RMSE, physical units "
                  "-- identical estimator to training and evaluate.py",
        "random_baseline": sc.random_baseline,
        "random_mode": sc.random_mode,
        "random_seeds": list(sc.random_seeds),
        "compute_joint_nll": sc.compute_joint_nll,
        "save_skill_maps": sc.save_skill_maps,
        "checkpoint": train_meta,
        "experiments": [
            {
                "name": e["name"],
                "dir": str(e["dir"]),
                "acquisition_function": e["acquisition_function"],
                "n_context_baseline": e["n_context_baseline"],
                "n_recommendations": int(len(e["recommendations"])),
                "al_config": e["al_config"],
            }
            for e in experiments
        ],
        "summary": summary,
        "curve": agg.to_dict(orient="records"),
        "curve_raw": curve_df.to_dict(orient="records"),
        "warnings": warnings_log,
        "evaluated_at": str(pd.Timestamp.now()),
    }

    with open(out_dir / "skill_curve_metrics.json", "w") as f:
        json.dump(results, f, indent=4, default=str)
    return results

def _archive_config(config, out_dir: Path):
    src = getattr(config, "_al_config_source_path", None)
    if not src:
        return
    try:
        shutil.copy2(src, out_dir / "skill_curve_config_used.yaml")
    except shutil.SameFileError:
        pass


def _save_skill_maps(skill_map_stores: dict, out_dir: Path):
    """
    Write per-experiment spatial skill maps: dims (k, lat, lon).

    Also writes a delta between the first and last stored k -- the sigma
    reduction map is the figure this whole stage exists to produce.
    """
    import xarray as xr

    for name, store in skill_map_stores.items():
        placements = store.pop("_placements", None)
        ks = sorted(store)
        if not ks:
            continue

        ds = xr.concat([store[k] for k in ks], dim="k").assign_coords(k=ks)

        if len(ks) >= 2:
            k_lo, k_hi = ks[0], ks[-1]
            for var in ("mean_std", "rmse", "mae"):
                ds[f"delta_{var}"] = (
                    ds[var].sel(k=k_hi) - ds[var].sel(k=k_lo)
                )
                ds[f"delta_{var}"].attrs = {
                    "long_name": f"{var} change, k={k_lo} -> k={k_hi} "
                                 f"(negative = improvement)",
                    "units": "degC",
                }

        enc = {
            v: {"zlib": True, "complevel": 4}
            for v in ds.data_vars
            if ds[v].dtype.kind == "f"
        }
        path = out_dir / f"skill_maps_{name}.nc"
        ds.to_netcdf(path, encoding=enc)
        print(f"  Saved skill maps: {path}")

        if placements is not None:
            p_path = out_dir / f"skill_maps_{name}_placements.csv"
            placements.to_csv(p_path, index=False)
            print(f"  Saved placements:  {p_path}")

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _plot_skill_curve(config, curve_df, agg, sc, out_dir):
    """RMSE / NLL / mean sigma vs number of added sensors."""
    import matplotlib.pyplot as plt
    from pipeline.plotting import _finish_plot

    panels = [
        ("rmse", "RMSE (physical units)", "RMSE vs sensors added"),
        ("nll", "Marginal NLL", "NLL vs sensors added"),
        ("mean_std", r"Mean predictive $\sigma$", "Uncertainty vs sensors added"),
    ]
    if sc.compute_joint_nll and "joint_nll" in curve_df.columns:
        panels.append(
            ("joint_nll", "Joint NLL / $N_t$", "Joint NLL vs sensors added")
        )

    fig, axs = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    axs = np.atleast_1d(axs)
    greedy = agg[agg.placement == "greedy"]
    random_ = agg[agg.placement == "random"]

    cmap = plt.get_cmap("tab10")
    greedy_names = list(greedy["experiment"].unique())
    colors = {n: cmap(i % 10) for i, n in enumerate(greedy_names)}

    for ax, (metric, ylabel, title) in zip(axs, panels):
        mean_col = f"{metric}_mean"
        if mean_col not in agg.columns:
            continue

        # Random envelope first, so greedy lines draw on top
        for name, grp in random_.groupby("experiment"):
            grp = grp.sort_values("k")
            ax.fill_between(
                grp["k"], grp[f"{metric}_min"], grp[f"{metric}_max"],
                color="grey", alpha=0.25, zorder=1,
                label=f"random (min-max, n={int(grp['n_seeds'].iloc[0])} seeds)",
            )
            ax.plot(
                grp["k"], grp[mean_col],
                color="grey", ls="--", lw=1.8, zorder=2,
                label="random (mean)",
            )

        for name, grp in greedy.groupby("experiment"):
            grp = grp.sort_values("k")
            acq = grp["acquisition_function"].iloc[0]
            ax.plot(
                grp["k"], grp[mean_col],
                marker="o", ms=4, lw=2.0, zorder=3,
                color=colors[name],
                label=f"{name} ({acq})",
            )

        ax.set_xlabel("Number of sensors added ($k$)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.xaxis.get_major_locator().set_params(integer=True)

    # Single deduplicated legend
    handles, labels = axs[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    axs[0].legend(h2, l2, fontsize=8, loc="best")

    fig.suptitle(
        f"{config.run.name} — {config.lake} — skill curve "
        f"('{sc.name}')",
        y=1.02,
    )
    fig.tight_layout()

    _finish_plot(config, out_dir / "skill_vs_nsensors.png")