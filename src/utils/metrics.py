# src/utils/metrics.py
"""Area-weighted, lake-only spatial metrics for skill-curve evaluation.

Canonical metric for this project
---------------------------------
Area-weighted RMSE, aggregated as the square root of the *mean over tasks* of
each task's area-weighted MSE:

    RMSE = sqrt( (1/T) * sum_t [ sum_i w_i e_i^2 / sum_i w_i ] )

Each task (date) contributes equally regardless of how many valid target
points it has, which keeps heavily-masked dates from silently dropping out of
the metric. When every task has identical coverage this is algebraically
identical to pooling all squared errors, so the definition is safe to adopt
now and remains correct if coverage becomes date-dependent later.
"""

import numpy as np
import torch
import xarray as xr


# -----------------------------------------------------------------------
# Masking and region (full lake or in target mask) helpers
# -----------------------------------------------------------------------
# Metrics computed per region; keep in sync with _region_metrics() keys.
REGION_METRIC_KEYS = (
    "mse", "rmse", "mae", "bias", "nll",
    "coverage_1sigma", "coverage_2sigma", "mean_std",
)


def mask_keep_from_coords(X_t, mask_da):
    """
    Boolean selector over off-grid target points.

    Parameters
    ----------
    X_t : np.ndarray, shape (2, N)
        Normalized x1/x2 target coords, i.e. task["X_t"][0].
    mask_da : xr.DataArray
        Mask on the *normalized* x1/x2 grid. Truthy / >0.5 = include.

    Returns
    -------
    np.ndarray of bool, shape (N,)
    """
    if isinstance(X_t, tuple):
        raise ValueError(
            "X_t is gridded (tuple of 1-D coords); target-mask selection "
            "expects off-grid (2, N) coords. Did remove_target_nans() run?"
        )
    X_t = np.asarray(X_t, dtype=float)
    x1 = xr.DataArray(X_t[0], dims="pt")
    x2 = xr.DataArray(X_t[1], dims="pt")
    keep = mask_da.sel(x1=x1, x2=x2, method="nearest").values
    keep = np.nan_to_num(np.asarray(keep, dtype=float), nan=0.0)
    return keep > 0.5


def _region_metrics(err, std, w):
    """
    Weighted scores over one region. `w` is raw (unnormalized) and is
    renormalized here so every region sums to 1 independently.

    Returns None if the region is empty / has no weight.
    """
    if err.size == 0:
        return None
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0:
        return None
    w = w / wsum

    z = np.abs(err) / std
    mse = float(np.sum(w * err ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.sum(w * np.abs(err))),
        "bias": float(np.sum(w * err)),
        "nll": float(np.sum(w * (0.5 * np.log(2 * np.pi * std ** 2)
                                 + 0.5 * (err / std) ** 2))),
        "coverage_1sigma": float(np.sum(w * (z <= 1.0))),
        "coverage_2sigma": float(np.sum(w * (z <= 2.0))),
        "mean_std": float(np.sum(w * std)),
        "n_target": int(err.size),
    }

# -----------------------------------------------------------------------
# Point-based weights (off-grid target arrays, used by the trainer)
# -----------------------------------------------------------------------

def point_area_weights(lats) -> np.ndarray:
    """
    cos(latitude) area weights for a set of scattered points.

    Regular lat/lon cells shrink toward the poles; weighting by cos(lat) makes
    a spatial mean area-correct rather than point-count-correct.

    Parameters
    ----------
    lats : array-like
        Latitudes in degrees, shape (N,).

    Returns
    -------
    np.ndarray
        Weights of shape (N,), strictly positive. Not normalized (the
        weighted-error functions normalize internally).
    """
    lats = np.asarray(lats, dtype=float).ravel()
    if lats.size == 0:
        raise ValueError("point_area_weights: received zero latitudes.")
    if np.any(np.abs(lats) > 90.0):
        raise ValueError(
            f"point_area_weights: latitudes outside [-90, 90] "
            f"(min={lats.min()}, max={lats.max()}). Are these normalized "
            f"coordinates? Unnormalize with data_processor.map_coord_array first."
        )
    return np.cos(np.deg2rad(lats))


def target_lats(task, data_processor) -> np.ndarray:
    """
    Physical latitudes of a task's target points.

    task["X_t"][0] is normalized. Returns shape (N,) in degrees.
    Handles both off-grid (2, N) arrays and gridded (x1, x2) tuples.
    """
    X_t = task["X_t"][0]

    if isinstance(X_t, (tuple, list)):
        # Gridded target: (x1, x2) coordinate vectors -> full outer product
        x1, x2 = np.asarray(X_t[0]).ravel(), np.asarray(X_t[1]).ravel()
        x1_grid = np.repeat(x1, x2.size)
        x2_grid = np.tile(x2, x1.size)
        coords = np.stack([x1_grid, x2_grid], axis=0)
    else:
        coords = np.asarray(X_t)

    return np.asarray(data_processor.map_coord_array(coords, unnorm=True)[0]).ravel()


# -----------------------------------------------------------------------
# Core weighted error functions
# -----------------------------------------------------------------------

def weighted_mse(pred, true, weights) -> float:
    """
    Area-weighted mean squared error for a single task.

    Parameters
    ----------
    pred, true : array-like
        Predicted and true values in physical units. Broadcast to shape (N,).
    weights : array-like
        Non-negative weights, shape (N,). Need not be normalized.

    Returns
    -------
    float
        sum(w * (pred - true)^2) / sum(w), ignoring NaN error terms.
    """
    pred = np.asarray(pred, dtype=float).ravel()
    true = np.asarray(true, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()

    if not (pred.shape == true.shape == w.shape):
        raise ValueError(
            f"weighted_mse: shape mismatch pred={pred.shape}, "
            f"true={true.shape}, weights={w.shape}"
        )
    if np.any(w < 0):
        raise ValueError("weighted_mse: negative weights.")

    sq_err = (pred - true) ** 2

    # Drop any NaN pairs (defensive; tasks should already be NaN-free)
    valid = np.isfinite(sq_err) & np.isfinite(w)
    if not valid.any():
        raise ValueError("weighted_mse: no valid (non-NaN) points.")
    sq_err, w = sq_err[valid], w[valid]

    w_sum = w.sum()
    if w_sum == 0.0:
        raise ValueError("weighted_mse: total weight is zero.")

    return float(np.sum(w * sq_err) / w_sum)


def weighted_rmse(pred, true, weights) -> float:
    """Square root of :func:`weighted_mse`. Single-task RMSE."""
    return float(np.sqrt(weighted_mse(pred, true, weights)))


def aggregate_rmse(per_task_mse) -> float:
    """
    Aggregate per-task weighted MSEs into the canonical overall RMSE.

    Uses sqrt(mean(MSE)) rather than mean(sqrt(MSE)) so that, under constant
    target coverage, this equals pooling all squared errors across tasks.

    Parameters
    ----------
    per_task_mse : sequence of float
        One area-weighted MSE per task.

    Returns
    -------
    float
    """
    arr = np.asarray(list(per_task_mse), dtype=float)
    if arr.size == 0:
        raise ValueError("aggregate_rmse: received zero tasks.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("aggregate_rmse: non-finite MSE encountered.")
    return float(np.sqrt(arr.mean()))


# -----------------------------------------------------------------------
# Gridded weights (for prediction-time / skill-curve evaluation on X_t grids)
# -----------------------------------------------------------------------

def lake_area_weights(
    da: xr.DataArray,
    lat_name: str = "lat",
    mask: xr.DataArray = None,
) -> xr.DataArray:
    """
    Build cos(latitude) area weights over the lake surface, normalized to sum to 1.

    Combines two effects:
      1. cos(lat) area correction — regular lat/lon cells shrink toward the
         poles; weighting by cos(lat) makes the mean area-correct rather than
         cell-count-correct.
      2. Lake masking — land/invalid cells receive zero weight.

    Parameters
    ----------
    da : xr.DataArray
        Reference field defining the spatial grid. May include a 'time' dim.
        Used to infer the lake mask (valid-at-any-time) unless `mask` is given.
    lat_name : str
        Name of the latitude coordinate. Default "lat".
    mask : xr.DataArray, optional
        Explicit boolean 2D lake mask (True = lake). If None, derived from `da`
        as "valid at any time".

    Returns
    -------
    xr.DataArray
        2D weights (lat, lon), zero over land, summing to 1 over the lake.
    """
    # 1. Derive a fixed lake mask if not supplied
    if mask is None:
        mask = da.notnull().any("time") if "time" in da.dims else da.notnull()
    mask = mask.astype(bool)

    # 2. cos(lat) weights, broadcast to the 2D spatial grid
    cos_lat = np.cos(np.deg2rad(da[lat_name]))
    spatial_template = da.isel(time=0, drop=True) if "time" in da.dims else da
    w = cos_lat.broadcast_like(spatial_template)

    # 3. Zero out land (multiply by mask -> land becomes 0, not NaN)
    w = w.where(mask, 0.0)

    # 4. Normalize to sum to 1 over the lake
    total = w.sum()
    if float(total) == 0.0:
        raise ValueError(
            "lake_area_weights: total weight is zero (empty lake mask?). "
            "Check the reference field / mask."
        )
    return w / total


def gridded_weighted_rmse(
    pred: xr.DataArray,
    true: xr.DataArray,
    weights: xr.DataArray,
) -> float:
    """
    Area-weighted RMSE between two gridded fields sharing a lat/lon grid.

    Parameters
    ----------
    pred, true : xr.DataArray
        Fields in physical units on the same (lat, lon) grid.
    weights : xr.DataArray
        2D weights from :func:`lake_area_weights`.

    Returns
    -------
    float
    """
    sq_err = (pred - true) ** 2
    w = weights.where(np.isfinite(sq_err), 0.0)
    w_sum = float(w.sum())
    if w_sum == 0.0:
        raise ValueError("gridded_weighted_rmse: total weight is zero.")
    return float(np.sqrt(float((w * sq_err).fillna(0.0).sum()) / w_sum))

def variable_scale(data_processor, var_ID: str) -> float:
    """
    Multiplicative scale for unnormalizing a *standard deviation*.

    Unlike map_array(..., unnorm=True), this applies no additive offset --
    a std must be rescaled, not shifted.
    """
    cfg = data_processor.config[var_ID]
    method = cfg.get("method", "mean_std")
    p = cfg.get("params", cfg)  # tolerate flat or nested
    if method == "mean_std":
        return float(p["std"])
    if method == "min_max":
        return float(p["max"] - p["min"]) / 2.0
    raise ValueError(f"Unsupported normalization method '{method}' for '{var_ID}'")


def compute_weighted_rmse(
    model,
    tasks: list,
    bundle: dict,
    task_loader,
    return_per_task: bool = False,
) -> dict:
    """
    Canonical project metric: area-weighted RMSE in physical units.

    Each task's area-weighted MSE is computed with cos(lat) weights, then
    aggregated as sqrt(mean(MSE)) over tasks so every date counts equally.
    Under constant target coverage this equals pooling all squared errors.

    Parameters
    ----------
    model : ConvNP
    tasks : list of Task
    bundle : dict
        Must contain 'data_processor'.
    task_loader : TaskLoader
        Used to resolve the target variable ID for unnormalization.
    return_per_task : bool
        If True, include per-task diagnostics in the result.

    Returns
    -------
    dict with keys:
        'rmse'     : float, overall area-weighted RMSE
        'per_task' : list of dicts (empty unless return_per_task=True)
    """
    data_processor = bundle["data_processor"]
    target_var_ID = task_loader.target_var_IDs[0][0]

    per_task_mse = []
    per_task_details = []

    for task in tasks:
        with torch.no_grad():
            mean = data_processor.map_array(
                model.mean(task), target_var_ID, unnorm=True
            )
            true = data_processor.map_array(
                task["Y_t"][0], target_var_ID, unnorm=True
            )

        lats = target_lats(task, data_processor)
        weights = point_area_weights(lats)

        task_mse = weighted_mse(mean, true, weights)
        per_task_mse.append(task_mse)

        if return_per_task:
            per_task_details.append({
                "date": str(task.get("time", "unknown")),
                "mse": task_mse,
                "rmse": float(np.sqrt(task_mse)),
                "n_context": task["X_c"][0].shape[1] if len(task["X_c"]) > 0 else 0,
                "n_target": int(weights.size),
            })

    return {
        "rmse": aggregate_rmse(per_task_mse),
        "per_task": per_task_details,
    }


def compute_weighted_scores(
    model,
    tasks: list,
    bundle: dict,
    task_loader,
    return_per_task: bool = True,
    point_sink=None,
    target_mask=None,
) -> dict:
    """
    Area-weighted (cos-lat) scores over the full target extent, and -- if
    `target_mask` is given -- additionally over the masked sub-region.

    Masked metrics are returned under the same keys with a `_masked` suffix.
    Full-extent keys are unchanged, so existing callers are unaffected.

    `target_mask` must be an xr.DataArray on the *normalized* x1/x2 grid.
    Masking is applied post-prediction: for a ConvNP the marginal predictive
    at a target point does not depend on which other target points were
    requested, so this is exactly equivalent to having asked for only the
    masked points -- EXCEPT for joint quantities (AR sampling, joint NLL),
    which must not be computed from a post-hoc subset.

    point_sink is called once per scorable task as
        point_sink(task, lats, lons, mean, true, std, w, in_mask=...)
    with all arrays 1-D and equal length, filtered to finite points.
    `in_mask` is a bool array (or None). Must not mutate its arguments.
    """
    dp = bundle["data_processor"]
    target_var_ID = task_loader.target_var_IDs[0][0]
    scale = variable_scale(dp, target_var_ID)

    per_task_mse = []
    per_task_mse_masked = []
    rows = []

    # Target coords are usually identical across tasks (same lake, same NaN
    # pattern) -- but NOT guaranteed if the target field has date-varying
    # NaNs (ice, missing swaths), so key the cache on the coords themselves.
    keep_cache = {}

    def _get_keep(X_t):
        key = (np.asarray(X_t[0]).tobytes(), np.asarray(X_t[1]).tobytes())
        if key not in keep_cache:
            keep_cache[key] = mask_keep_from_coords(X_t, target_mask)
        return keep_cache[key]

    for task in tasks:
        with torch.no_grad():
            mean = dp.map_array(model.mean(task), target_var_ID, unnorm=True)
            true = dp.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
            std = np.asarray(model.std(task)).ravel() * scale

        mean = np.asarray(mean).ravel()
        true = np.asarray(true).ravel()

        lats = target_lats(task, dp)
        w = point_area_weights(lats)

        in_mask = _get_keep(task["X_t"][0]) if target_mask is not None else None

        # Carry target coords alongside so a point_sink can place values on a grid
        if point_sink is not None:
            X_t = np.asarray(task["X_t"][0], dtype=float)
            latlon = dp.map_coord_array(X_t, unnorm=True)
            pt_lats = np.asarray(latlon[0]).ravel()
            pt_lons = np.asarray(latlon[1]).ravel()
        else:
            pt_lats = pt_lons = None

        # --- align every per-point array ---------------------------------
        # Truncation/masking must apply identically to all arrays.
        sizes = [mean.size, true.size, std.size, w.size]
        if pt_lats is not None:
            sizes += [pt_lats.size, pt_lons.size]
        if in_mask is not None:
            sizes.append(in_mask.size)
        n = min(sizes)

        mean, true, std, w = mean[:n], true[:n], std[:n], w[:n]
        if pt_lats is not None:
            pt_lats, pt_lons = pt_lats[:n], pt_lons[:n]
        if in_mask is not None:
            in_mask = in_mask[:n]

        ok = np.isfinite(mean) & np.isfinite(true) & np.isfinite(std) & (std > 0)
        mean, true, std, w = mean[ok], true[ok], std[ok], w[ok]
        if pt_lats is not None:
            pt_lats, pt_lons = pt_lats[ok], pt_lons[ok]
        if in_mask is not None:
            in_mask = in_mask[ok]

        if mean.size == 0 or np.sum(w) == 0:
            continue

        if point_sink is not None:
            point_sink(task, pt_lats, pt_lons, mean, true, std,
                       w / np.sum(w), in_mask=in_mask)

        err = mean - true

        full = _region_metrics(err, std, w)
        if full is None:
            continue
        per_task_mse.append(full["mse"])

        row = {
            "date": str(task.get("time", "unknown"))[:10],
            **full,
            "n_context": task["X_c"][0].shape[1] if len(task["X_c"]) > 0 else 0,
        }

        if in_mask is not None:
            masked = _region_metrics(err[in_mask], std[in_mask], w[in_mask])
            if masked is None:
                # Mask and this date's valid targets don't intersect (e.g. ice
                # wiped out the AOI). NaN, not 0 -- aggregate with nanmean.
                row.update({f"{k}_masked": np.nan for k in REGION_METRIC_KEYS})
                row["n_target_masked"] = 0
            else:
                row.update({f"{k}_masked": masked[k] for k in REGION_METRIC_KEYS})
                row["n_target_masked"] = masked["n_target"]
                per_task_mse_masked.append(masked["mse"])
            row["frac_target_in_mask"] = float(np.mean(in_mask))

        rows.append(row)

    if not rows:
        raise ValueError("compute_weighted_scores: no scorable tasks.")

    def _m(k):
        vals = np.asarray([r[k] for r in rows if k in r], dtype=float)
        if vals.size == 0 or np.all(np.isnan(vals)):
            return float("nan")
        return float(np.nanmean(vals))

    out = {
        "rmse": aggregate_rmse(per_task_mse),
        "mae": _m("mae"),
        "bias": _m("bias"),
        "nll": _m("nll"),
        "coverage_1sigma": _m("coverage_1sigma"),
        "coverage_2sigma": _m("coverage_2sigma"),
        "mean_std": _m("mean_std"),
        "n_tasks": len(rows),
        "n_target_mean": _m("n_target"),
        "per_task": rows if return_per_task else [],
    }

    if target_mask is not None:
        n_masked_tasks = int(sum(r.get("n_target_masked", 0) > 0 for r in rows))
        out.update({
            "rmse_masked": (aggregate_rmse(per_task_mse_masked)
                            if per_task_mse_masked else float("nan")),
            "mae_masked": _m("mae_masked"),
            "bias_masked": _m("bias_masked"),
            "nll_masked": _m("nll_masked"),
            "coverage_1sigma_masked": _m("coverage_1sigma_masked"),
            "coverage_2sigma_masked": _m("coverage_2sigma_masked"),
            "mean_std_masked": _m("mean_std_masked"),
            "n_tasks_masked": n_masked_tasks,
            "n_target_masked_mean": _m("n_target_masked"),
            "frac_target_in_mask": _m("frac_target_in_mask"),
        })
        if n_masked_tasks == 0:
            warnings.warn(
                "target_mask selected zero target points on every task. "
                "Check the mask is on the normalized x1/x2 grid and shares "
                "the target variable's coords.", UserWarning,
            )

    return out
# def compute_weighted_scores(
#     model,
#     tasks: list,
#     bundle: dict,
#     task_loader,
#     return_per_task: bool = True,
#     point_sink=None,
#     target_mask=None
# ) -> dict:
#     """
#     Superset of compute_weighted_rmse: adds MAE, bias, Gaussian NLL, and
#     calibration coverage. Uses identical cos(lat) weights and identical
#     per-task pooling, so 'rmse' here is numerically the same metric
#     reported during training.
#
#     Costs one extra forward pass component (model.std) per task, which is
#     why the epoch loop keeps using compute_weighted_rmse.
#     Point_sink: called once per scorable task as point_sink(task, lats, longs, means, true, std, w)
#     with all arrays 1-D of equal len, filtered to finite points. Lets callers accumulate spatial maps without
#     rerunning model. Must not mutate it's arguments
#     """
#     dp = bundle["data_processor"]
#     target_var_ID = task_loader.target_var_IDs[0][0]
#     scale = variable_scale(dp, target_var_ID)
#
#     per_task_mse = []
#     rows = []
#
#     for task in tasks:
#         with torch.no_grad():
#             mean = dp.map_array(model.mean(task), target_var_ID, unnorm=True)
#             true = dp.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
#             std = np.asarray(model.std(task)).ravel() * scale
#
#         mean = np.asarray(mean).ravel()
#         true = np.asarray(true).ravel()
#
#         lats = target_lats(task, dp)
#         w = point_area_weights(lats)
#
#         def build_target_mask_index(task, mask_da):
#             x1 = xr.DataArray(np.asarray(task['X_t'][0][0]), dims='pt')
#             x2 = xr.DataArray(np.asarray(task['X_t'][0][1]), dims='pt')
#             keep = mask_da.sel(x1=x1, x2=x2, method='nearest').values
#             return keep
#
#         _keep_cache = {}
#
#         def get_keep(task, mask_da):
#             key = (np.asarray(task['X_t'][0][0]).tobytes(), np.asarray(task['X_t'][0][1]).tobytes())
#             if key not in _keep_cache:
#                 _keep_cache[key] = build_target_mask_index(task, mask_da)
#             return _keep_cache[key]
#
#         # If the target mask is None, then the rest of the function works fine.
#         # What I'd like to have is that if target mask exists, then to compute both the masked
#         # and the full lake error metrics and have that added to the dataframe. Additionally,
#         # The skillmapAccumulator can then get the additional variables for the full lake and the
#         # target area.
#         if target_mask is not None:
#             keep = get_keep(task, target_mask)
#             target_mean = mean[keep]
#             target_true = true[keep]
#
#
#         # Carry target coords alongside so a point_sink can place values on a grid
#         # Truncation/masking must apply identically to all arrays
#         if point_sink is not None:
#             X_t = np.asarray(task["X_t"][0], dtype=float)
#             latlon = dp.map_coord_array(X_t, unnorm=True)
#             pt_lats = np.asarray(latlon[0].ravel())
#             pt_lons = np.asarray(latlon[1].ravel())
#         else:
#             pt_lats = pt_lons = None
#
#         sizes = [mean.size, true.size, std.size, w.size]
#         if pt_lats is not None:
#             sizes += [pt_lats.size, pt_lons.size]
#         n = min(sizes)
#         mean, true, std, w - mean[:n], true[:n], std[:n], w[:n]
#         if pt_lats is not None:
#             pt_lats, pt_lons = pt_lats[:n], pt_lons[:n]
#
#         ok = np.isfinite(mean) & np.isfinite(true) & np.isfinite(std) & (std > 0)
#         mean, true, std, w = mean[ok], true[ok], std[ok], w[ok]
#         if pt_lats is not None:
#             pt_lats, pt_lons = pt_lats[ok], pt_lons[ok]
#         if mean.size == 0 or w.sum() == 0:
#             continue
#         w = w / w.sum()
#
#         if point_sink is not None:
#             point_sink(task, pt_lats, pt_lons, mean, true, std, w)
#
#         err = mean - true
#         mse = float(np.sum(w * err ** 2))
#         per_task_mse.append(mse)
#
#         z = np.abs(err) / std
#         rows.append({
#             "date": str(task.get("time", "unknown"))[:10],
#             "mse": mse,
#             "rmse": float(np.sqrt(mse)),
#             "mae": float(np.sum(w * np.abs(err))),
#             "bias": float(np.sum(w * err)),
#             "nll": float(np.sum(w * (0.5 * np.log(2 * np.pi * std ** 2)
#                                      + 0.5 * (err / std) ** 2))),
#             "coverage_1sigma": float(np.sum(w * (z <= 1.0))),
#             "coverage_2sigma": float(np.sum(w * (z <= 2.0))),
#             "mean_std": float(np.sum(w * std)),
#             "n_context": task["X_c"][0].shape[1] if len(task["X_c"]) > 0 else 0,
#             "n_target": int(w.size),
#         })
#
#     if not rows:
#         raise ValueError("compute_weighted_scores: no scorable tasks.")
#
#     def _m(k):
#         return float(np.mean([r[k] for r in rows]))
#
#     return {
#         "rmse": aggregate_rmse(per_task_mse),
#         "mae": _m("mae"),
#         "bias": _m("bias"),
#         "nll": _m("nll"),
#         "coverage_1sigma": _m("coverage_1sigma"),
#         "coverage_2sigma": _m("coverage_2sigma"),
#         "mean_std": _m("mean_std"),
#         "n_tasks": len(rows),
#         "per_task": rows if return_per_task else [],
#     }
