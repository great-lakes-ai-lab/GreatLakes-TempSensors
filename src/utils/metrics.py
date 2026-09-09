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
) -> dict:
    """
    Superset of compute_weighted_rmse: adds MAE, bias, Gaussian NLL, and
    calibration coverage. Uses identical cos(lat) weights and identical
    per-task pooling, so 'rmse' here is numerically the same metric
    reported during training.

    Costs one extra forward pass component (model.std) per task, which is
    why the epoch loop keeps using compute_weighted_rmse.
    """
    dp = bundle["data_processor"]
    target_var_ID = task_loader.target_var_IDs[0][0]
    scale = variable_scale(dp, target_var_ID)

    per_task_mse = []
    rows = []

    for task in tasks:
        with torch.no_grad():
            mean = dp.map_array(model.mean(task), target_var_ID, unnorm=True)
            true = dp.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
            std = np.asarray(model.std(task)).ravel() * scale

        mean = np.asarray(mean).ravel()
        true = np.asarray(true).ravel()

        lats = target_lats(task, dp)
        w = point_area_weights(lats)

        n = min(mean.size, true.size, std.size, w.size)
        mean, true, std, w = mean[:n], true[:n], std[:n], w[:n]

        ok = np.isfinite(mean) & np.isfinite(true) & np.isfinite(std) & (std > 0)
        mean, true, std, w = mean[ok], true[ok], std[ok], w[ok]
        if mean.size == 0 or w.sum() == 0:
            continue
        w = w / w.sum()

        err = mean - true
        mse = float(np.sum(w * err ** 2))
        per_task_mse.append(mse)

        z = np.abs(err) / std
        rows.append({
            "date": str(task.get("time", "unknown"))[:10],
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.sum(w * np.abs(err))),
            "bias": float(np.sum(w * err)),
            "nll": float(np.sum(w * (0.5 * np.log(2 * np.pi * std ** 2)
                                     + 0.5 * (err / std) ** 2))),
            "coverage_1sigma": float(np.sum(w * (z <= 1.0))),
            "coverage_2sigma": float(np.sum(w * (z <= 2.0))),
            "mean_std": float(np.sum(w * std)),
            "n_context": task["X_c"][0].shape[1] if len(task["X_c"]) > 0 else 0,
            "n_target": int(w.size),
        })

    if not rows:
        raise ValueError("compute_weighted_scores: no scorable tasks.")

    def _m(k):
        return float(np.mean([r[k] for r in rows]))

    return {
        "rmse": aggregate_rmse(per_task_mse),
        "mae": _m("mae"),
        "bias": _m("bias"),
        "nll": _m("nll"),
        "coverage_1sigma": _m("coverage_1sigma"),
        "coverage_2sigma": _m("coverage_2sigma"),
        "mean_std": _m("mean_std"),
        "n_tasks": len(rows),
        "per_task": rows if return_per_task else [],
    }