# src/pipeline/plotting.py
"""Visualization functions for predictions and diagnostics."""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import os
import matplotlib


def _should_show_plots(config) -> bool:
    """Determine whether to call plt.show() or just save and close."""
    mode = config.run.display_plots

    if mode == "show":
        return True
    elif mode == "save_only":
        return False
    else:
        # Auto-detect: show if display is available
        # HPC batch jobs, SSH without X-forwarding, etc. won't have DISPLAY
        if os.environ.get("SLURM_JOB_ID"):
            return False
        if os.environ.get("PBS_JOBID"):
            return False
        if not os.environ.get("DISPLAY") and os.name != "nt":
            # No DISPLAY on Linux/Mac (but Windows doesn't use DISPLAY)
            # Also check if we're on macOS (always has a display framework)
            import sys
            if sys.platform == "darwin":
                return True
            return False
        return True


def _finish_plot(config, save_path=None):
    """Save and/or show a plot based on config."""
    import matplotlib.pyplot as plt

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if _should_show_plots(config):
        plt.show()
    else:
        plt.close()


def _get_actual_ds(bundle: dict, config) -> tuple:
    """
    Get the pre-DataProcessor actual target dataset and its variable name.

    Returns
    -------
    (actual_ds, actual_var) or (None, None) if not found.
    """
    for name, source in config.data_sources.items():
        roles = source.role if isinstance(source.role, list) else [source.role]
        if "target" in roles:
            if source.use_anomalies and f"{name}_anom_stand" in bundle:
                ds = bundle[f"{name}_anom_stand"]
                return ds, list(ds.data_vars)[0]
            elif f"{name}_stand" in bundle:
                ds = bundle[f"{name}_stand"]
                return ds, list(ds.data_vars)[0]
            break
    return None, None


def _get_target_var(prediction_result: dict, config) -> str:
    """Get the target variable name from the prediction dataset."""
    pred_ds = prediction_result["prediction"]
    # DeepSensor prediction keys match the target variable names
    for name, source in config.data_sources.items():
        roles = source.role if isinstance(source.role, list) else [source.role]
        if "target" in roles:
            if source.use_anomalies:
                candidate = f"{source.variable}_anom"
            else:
                candidate = source.variable
            if candidate in pred_ds:
                return candidate
            break
    # Fallback: first key in prediction dataset
    return list(pred_ds.data_vars)[0] if hasattr(pred_ds, 'data_vars') else list(pred_ds.keys())[0]


def plot_task(task, task_loader, config, title="", save_dir=None, date_str=None):
    """Plot a single task using DeepSensor's built-in plotting."""
    import deepsensor.plot

    fig = deepsensor.plot.task(task, task_loader)
    if title:
        plt.suptitle(title)
    plt.tight_layout()

    save_path = None
    if save_dir:
        save_dir = Path(save_dir)
        filename = f"task_{date_str}.png" if date_str else "task.png"
        save_path = save_dir / filename

    _finish_plot(config, save_path)
    return fig

def plot_prediction_summary(
    prediction_result: dict,
    bundle: dict,
    config,
    save_dir=None,
):
    """
    Four-panel plot: actual, predicted mean, uncertainty, and error.
    All panels masked to lake surface only.
    """
    pred_ds = prediction_result["prediction"]
    date = prediction_result["date"]

    # Generic target variable resolution
    target_var = _get_target_var(prediction_result, config)
    mean_da = pred_ds[target_var]["mean"]
    std_da = pred_ds[target_var]["std"]

    # Get actual values for this date
    actual_ds, actual_var = _get_actual_ds(bundle, config)

    # Build lake mask on the prediction grid
    # Use actual SST valid pixels as mask (most reliable)
    if actual_ds is not None:
        actual_var = list(actual_ds.data_vars)[0]
        actual = actual_ds[actual_var].sel(time=date, method="nearest")
        lake_mask = actual.notnull()
    else:
        lake_mask = None

    # Apply mask
    if lake_mask is not None:
        mean_masked = mean_da.where(lake_mask)
        std_masked = std_da.where(lake_mask)
        actual_masked = actual.where(lake_mask) if actual_ds else None
    else:
        mean_masked = mean_da
        std_masked = std_da
        actual_masked = actual if actual_ds else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Actual ---
    ax = axes[0, 0]
    if actual_masked is not None:
        actual_masked.plot(ax=ax, cmap="RdBu_r", add_colorbar=True)
        ax.set_title(f"Actual ({date})")
    else:
        ax.set_title("Actual (not available)")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

    # --- Panel 2: Predicted Mean ---
    ax = axes[0, 1]
    mean_masked.plot(ax=ax, cmap="RdBu_r", add_colorbar=True)
    ax.set_title(f"Predicted Mean ({date})")

    # --- Panel 3: Uncertainty (Std) ---
    ax = axes[1, 0]
    std_masked.plot(ax=ax, cmap="plasma", add_colorbar=True)
    ax.set_title(f"Predictive Uncertainty / Std ({date})")

    # --- Panel 4: Error (Predicted - Actual) ---
    ax = axes[1, 1]
    if actual_masked is not None:
        error = mean_masked - actual_masked.interp_like(mean_masked)
        max_err = float(np.nanmax(np.abs(error.values)))
        error.plot(
            ax=ax, cmap="RdBu_r",
            vmin=-max_err, vmax=max_err,
            add_colorbar=True,
        )
        ax.set_title(f"Error: Predicted - Actual ({date})")
    else:
        ax.set_title("Error (actual not available)")

    plt.suptitle(f"Prediction Summary: {config.lake.upper()} — {date}", fontsize=14)
    plt.tight_layout()

    save_path = None
    if save_dir:
        save_path = Path(save_dir) / f"prediction_{date}.png"
        print(f"Saved: {save_path}")

    _finish_plot(config, save_path)


def plot_uncertainty_vs_error(
    prediction_result: dict,
    bundle: dict,
    config,
    save_dir=None,
):
    """
    Scatter plot of predictive std vs absolute error at each grid point.
    """
    pred_ds = prediction_result["prediction"]
    date = prediction_result["date"]

    target_var = _get_target_var(prediction_result, config)
    mean_da = pred_ds[target_var]["mean"]
    std_da = pred_ds[target_var]["std"]

    actual_ds, actual_var = _get_actual_ds(bundle, config)
    if actual_ds is None:
        print("No actual data available for comparison.")
        return

    actual = actual_ds[actual_var].sel(time=date, method="nearest")

    # Lake mask from actual SST valid pixels
    lake_mask = actual.notnull()

    # Interpolate actual to prediction grid and apply mask
    actual_interp = actual.interp_like(mean_da)
    mask_interp = lake_mask.astype(float).interp_like(mean_da, method="nearest").fillna(0) > 0.5

    abs_error = np.abs((mean_da.where(mask_interp) - actual_interp.where(mask_interp)).values.ravel())
    uncertainty = std_da.where(mask_interp).values.ravel()

    # Remove NaN pairs
    valid = ~(np.isnan(abs_error) | np.isnan(uncertainty))
    abs_error = abs_error[valid]
    uncertainty = uncertainty[valid]

    # Temp add some prints about the error distributions
    print(f"\n  --- Calibration Stats ({date}) ---")
    print(f"  Lake points: {len(abs_error)}")
    print(f"  Abs Error:  mean={abs_error.mean():.4f}, median={np.median(abs_error):.4f}, "
          f"std={abs_error.std():.4f}, min={abs_error.min():.4f}, max={abs_error.max():.4f}")
    print(f"  Uncertainty: mean={uncertainty.mean():.4f}, median={np.median(uncertainty):.4f}, "
          f"std={uncertainty.std():.4f}, min={uncertainty.min():.4f}, max={uncertainty.max():.4f}")
    print(f"  Error/Std ratio: mean={np.mean(abs_error / uncertainty):.4f}, "
          f"median={np.median(abs_error / uncertainty):.4f}")
    print(f"  Correlation (std vs |error|): {np.corrcoef(uncertainty, abs_error)[0, 1]:.4f}")

    # What fraction of errors fall within 1σ and 2σ?
    within_1sigma = np.mean(abs_error <= uncertainty)
    within_2sigma = np.mean(abs_error <= 2 * uncertainty)
    print(f"  Within 1σ: {within_1sigma * 100:.1f}% (ideal: 68.3%)")
    print(f"  Within 2σ: {within_2sigma * 100:.1f}% (ideal: 95.4%)")
    print(f"  ---")


    if len(abs_error) == 0:
        print("No valid lake points for calibration plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(uncertainty, abs_error, alpha=0.1, s=5)

    # 1:1 line (perfect calibration)
    max_val = max(uncertainty.max(), abs_error.max())
    ax.plot([0, max_val], [0, max_val], "r--", label="1:1 (perfect calibration)")

    # Binned mean
    n_bins = 20
    bin_edges = np.linspace(0, uncertainty.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = []
    for i in range(n_bins):
        mask = (uncertainty >= bin_edges[i]) & (uncertainty < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(abs_error[mask].mean())
        else:
            bin_means.append(np.nan)
    ax.plot(bin_centers, bin_means, "g-o", markersize=4, label="Binned mean |error|")

    ax.set_xlabel("Predictive Std (uncertainty)")
    ax.set_ylabel("Absolute Error")
    ax.set_title(f"Calibration: Uncertainty vs Error ({date})\nLake points only")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = None
    if save_dir:
        save_path = Path(save_dir) / f"calibration_{date}.png"

    _finish_plot(config, save_path)


def plot_prediction_timeseries(
        model,
        task_loader,
        bundle: dict,
        config,
        dates: list,
        lat: float,
        lon: float,
        n_context: int = 50,
        save_dir=None,
):
    from .predict import predict_date

    means = []
    stds = []
    actuals = []

    actual_ds, actual_var = _get_actual_ds(bundle, config)

    for date in dates:
        result = predict_date(model, task_loader, bundle, config, date, n_context)
        pred = result["prediction"]
        target_var = _get_target_var(result, config)

        mean_val = float(pred[target_var]["mean"].sel(
            lat=lat, lon=lon, method="nearest").values)
        std_val = float(pred[target_var]["std"].sel(
            lat=lat, lon=lon, method="nearest").values)
        means.append(mean_val)
        stds.append(std_val)

        if actual_ds is not None:
            act_val = float(actual_ds[actual_var].sel(
                time=date, lat=lat, lon=lon, method="nearest").values)
            actuals.append(act_val)


    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(dates, means - 2 * stds, means + 2 * stds, alpha=0.2, label="±2σ")
    ax.plot(dates, means, "b-", label="Predicted mean")
    if actuals:
        ax.plot(dates, actuals, "r--", label="Actual")

    ax.set_xlabel("Date")
    ax.set_ylabel("SST Anomaly")
    ax.set_title(f"Time series at ({lat:.2f}, {lon:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = None
    if save_dir:
        save_path = Path(save_dir) / f"timeseries_{lat}_{lon}.png"

    _finish_plot(config, save_path)