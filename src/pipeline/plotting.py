# src/pipeline/plotting.py
"""Visualization functions for predictions and diagnostics."""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_task(task, task_loader, title=""):
    """Plot a single task using DeepSensor's built-in plotting."""
    import deepsensor.plot
    fig = deepsensor.plot.task(task, task_loader)
    if title:
        plt.suptitle(title)
    plt.show()
    return fig


def plot_prediction_summary(
    prediction_result: dict,
    bundle: dict,
    config,
    target_var: str = "sst_anom",
    save_dir=None,
):
    """
    Four-panel plot: actual, predicted mean, uncertainty, and error.
    All panels masked to lake surface only.
    """
    pred_ds = prediction_result["prediction"]
    date = prediction_result["date"]

    mean_da = pred_ds[target_var]["mean"]
    std_da = pred_ds[target_var]["std"]

    # Get actual values for this date
    if "sst_anom_stand" in bundle:
        actual_ds = bundle["sst_anom_stand"]
    elif "sst_stand" in bundle:
        actual_ds = bundle["sst_stand"]
    else:
        actual_ds = None

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

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"prediction_{date}.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir / f'prediction_{date}.png'}")

    plt.show()


def plot_uncertainty_vs_error(
    prediction_result: dict,
    bundle: dict,
    target_var: str = "sst_anom",
    save_dir=None,
):
    """
    Scatter plot of predictive std vs absolute error at each grid point.
    Only includes lake surface points.
    """
    pred_ds = prediction_result["prediction"]
    date = prediction_result["date"]

    mean_da = pred_ds[target_var]["mean"]
    std_da = pred_ds[target_var]["std"]

    if "sst_anom_stand" in bundle:
        actual_ds = bundle["sst_anom_stand"]
    elif "sst_stand" in bundle:
        actual_ds = bundle["sst_stand"]
    else:
        print("No actual data available for comparison.")
        return

    actual_var = list(actual_ds.data_vars)[0]
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

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"calibration_{date}.png", dpi=150, bbox_inches="tight")

    plt.show()


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
    """
    Plot predicted vs actual SST at a single point over time.
    Shows mean ± std band.
    """
    from .predict import predict_date

    means = []
    stds = []
    actuals = []

    if "sst_anom_stand" in bundle:
        actual_ds = bundle["sst_anom_stand"]
    elif "sst_stand" in bundle:
        actual_ds = bundle["sst_stand"]
    else:
        actual_ds = None

    actual_var = list(actual_ds.data_vars)[0] if actual_ds else None

    for date in dates:
        result = predict_date(model, task_loader, bundle, config, date, n_context)
        pred = result["prediction"]
        target_var = list(pred.data_vars)[0]

        mean_val = float(pred[target_var]["mean"].sel(
            lat=lat, lon=lon, method="nearest").values)
        std_val = float(pred[target_var]["std"].sel(
            lat=lat, lon=lon, method="nearest").values)
        means.append(mean_val)
        stds.append(std_val)

        if actual_ds:
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

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"timeseries_{lat}_{lon}.png", dpi=150, bbox_inches="tight")

    plt.show()