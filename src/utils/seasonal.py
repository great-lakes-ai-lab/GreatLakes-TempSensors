# src/utils/seasonal.py
"""Seasonal cycle computation and anomaly processing."""

import os
import json
import uuid
import warnings

import numpy as np
import pandas as pd
import xarray as xr





"""Seasonal cycle computation and anomaly processing."""

import os
import json

import numpy as np
import pandas as pd
import xarray as xr

_VALID_METHODS = ("monthly", "daily_doy", "harmonic")


class SeasonalCycleProcessor:
    """
    Estimate a seasonal cycle (climatology) on a fit period and use it to form
    anomalies over an arbitrary (typically longer) record.

    Methods
    -------
    monthly    : 12 monthly means.
    daily_doy  : 366 day-of-year means, optionally circularly smoothed.
    harmonic   : `n_harmonics` annual Fourier harmonics least-squares fitted
                 to the daily-DOY climatology. Smooth and low-variance.
    """

    def __init__(self, seasonal_cycle=None, metadata=None):
        self.seasonal_cycle = seasonal_cycle
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def calculate(
        self,
        ds,
        dim="time",
        fit_intervals=None,
        method="harmonic",
        n_harmonics=3,
        smooth_window=15,
    ):
        """
        Parameters
        ----------
        ds : xr.Dataset
            Full-record dataset (the cycle is *applied* to this later).
        dim : str
            Time dimension name.
        fit_intervals : list[tuple[str, str]] or None
            Estimate the cycle ONLY from these slices (the training period).
            None = full record (leaks val/test info; not recommended).
        method : {"monthly", "daily_doy", "harmonic"}
        n_harmonics : int
            Number of annual harmonics (method="harmonic"). 2-4 is typical;
            1 = pure annual sinusoid, 3 captures spring/fall asymmetry.
        smooth_window : int
            Circular rolling-mean width in days (method="daily_doy").
            0 = no smoothing.
        """
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got '{method}'")

        fit_ds = self._select_fit(ds, dim, fit_intervals)

        if method == "monthly":
            cycle = fit_ds.groupby(f"{dim}.month").mean(dim=dim)
            self._check_coverage(cycle, "month", 12)

        else:
            doy_clim = fit_ds.groupby(f"{dim}.dayofyear").mean(dim=dim)
            doy_clim = fit_ds.groupby(f"{dim}.dayofyear").mean(dim=dim).compute()
            doy_clim = self._complete_doy(doy_clim)

            if method == "daily_doy":
                cycle = (
                    _smooth_circular(doy_clim, smooth_window)
                    if smooth_window and smooth_window > 1
                    else doy_clim
                )
            else:  # harmonic
                cycle = _fit_harmonics(doy_clim, n_harmonics)

        self.seasonal_cycle = cycle

        counts = fit_ds[dim].to_index().month.value_counts().sort_index().to_dict()
        self.metadata.update({
            "method": method,
            "n_harmonics": int(n_harmonics) if method == "harmonic" else None,
            "smooth_window": int(smooth_window) if method == "daily_doy" else None,
            "calculation_date": str(pd.Timestamp.now()),
            "dataset_vars": list(ds.data_vars),
            "fit_intervals": [list(iv) for iv in (fit_intervals or [])],
            "fit_n_timesteps": int(fit_ds.sizes[dim]),
            "fit_time_min": str(fit_ds[dim].values.min())[:10],
            "fit_time_max": str(fit_ds[dim].values.max())[:10],
            "fit_counts_per_month": {int(k): int(v) for k, v in counts.items()},
            "cycle_dim": "month" if method == "monthly" else "dayofyear",
        })

        n_years = self.metadata["fit_n_timesteps"] / 365.25
        if method == "daily_doy" and n_years < 10 and not smooth_window:
            warnings.warn(
                f"daily_doy with only ~{n_years:.1f} years of fit data and no "
                f"smoothing gives a very noisy climatology (~"
                f"{n_years:.1f} samples per day-of-year). Use method='harmonic' "
                f"or set smooth_window > 1."
            )
        if method == "harmonic" and n_harmonics > 6:
            warnings.warn(
                f"n_harmonics={n_harmonics} ({2 * n_harmonics + 1} parameters) "
                f"may start fitting interannual noise rather than the seasonal cycle."
            )

        return self

        # ------------------------------------------------------------------
        # Application
        # ------------------------------------------------------------------

    def compute_anomalies(self, ds, dim="time"):
        """Subtract the fitted cycle from `ds`, matching on month or dayofyear."""
        if self.seasonal_cycle is None:
            raise ValueError("No seasonal cycle fitted. Call .calculate() first.")

        cycle_dim = self.metadata.get("cycle_dim")
        if cycle_dim is None:
            cycle_dim = "month" if "month" in self.seasonal_cycle.dims else "dayofyear"

        anom = ds.groupby(f"{dim}.{cycle_dim}") - self.seasonal_cycle

        # groupby arithmetic leaves the grouping coord behind
        if cycle_dim in anom.coords:
            anom = anom.drop_vars(cycle_dim)

        return anom

        # ------------------------------------------------------------------
        # Internals
        # ------------------------------------------------------------------

    @staticmethod
    def _select_fit(ds, dim, fit_intervals):
        if not fit_intervals:
            return ds
        slices = [ds.sel({dim: slice(s, e)}) for s, e in fit_intervals]
        slices = [s for s in slices if s.sizes.get(dim, 0) > 0]
        if not slices:
            raise ValueError(f"No data in climatology fit_intervals: {fit_intervals}")
        return xr.concat(slices, dim=dim) if len(slices) > 1 else slices[0]

    @staticmethod
    def _check_coverage(cycle, coord, expected):
        present = set(int(v) for v in cycle[coord].values)
        missing = sorted(set(range(1, expected + 1)) - present)
        if missing:
            raise ValueError(
                f"Climatology fit period is missing {coord} values {missing}. "
                f"Widen the fit intervals to cover a full annual cycle."
            )

    @staticmethod
    def _complete_doy(doy_clim):
        """
        Reindex to a full 1..366 dayofyear axis and circularly interpolate gaps.

        Short fit periods (or non-leap-only records) leave DOYs missing —
        notably 366. Without this, `groupby - cycle` silently drops those dates.
        """
        full = np.arange(1, 367)
        out = doy_clim.reindex(dayofyear=full)

        # Circular fill: tile [prev, self, next] so DOY 1 and 366 see each other
        tiled = xr.concat(
            [
                out.assign_coords(dayofyear=full - 366),
                out,
                out.assign_coords(dayofyear=full + 366),
            ],
            dim="dayofyear",
        )
        tiled = tiled.interpolate_na("dayofyear", method="linear")
        filled = tiled.sel(dayofyear=full)

        n_missing = int(doy_clim.sizes["dayofyear"])
        if n_missing < 366:
            print(f"    filled {366 - n_missing} missing day-of-year value(s) "
                  f"by circular interpolation")
        return filled

    def save(self, base_dir="seasonal_cycles", name="seasonal_cycle"):
        os.makedirs(base_dir, exist_ok=True)
        cycle_path = os.path.join(base_dir, f"{name}.nc")
        metadata_path = os.path.join(base_dir, f"{name}_metadata.json")
        self.seasonal_cycle.to_netcdf(cycle_path)
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)
        return {"seasonal_cycle_path": cycle_path, "metadata_path": metadata_path}

    @classmethod
    def load(cls, cycle_path, metadata_path=None):
        """
        Load a previously saved seasonal cycle.

        Parameters:
        -----------
        cycle_path : str
            Path to the seasonal cycle NetCDF file
        metadata_path : str, optional
            Path to the metadata JSON file

        Returns:
        --------
        SeasonalCycleProcessor
            Loaded seasonal cycle processor
        """
        # Load seasonal cycle
        seasonal_cycle = xr.load_dataset(cycle_path)

        # Load metadata if path provided
        metadata = None
        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        return cls(seasonal_cycle=seasonal_cycle, metadata=metadata)


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

def _smooth_circular(doy_clim, window):
    """
    Circular centred rolling mean over the dayofyear axis.

    Wraps by tiling ±366 so Dec 31 / Jan 1 are neighbours, avoiding the
    discontinuity a plain .rolling() would leave at the year boundary.
    """
    full = doy_clim["dayofyear"].values
    tiled = xr.concat(
        [
            doy_clim.assign_coords(dayofyear=full - 366),
            doy_clim,
            doy_clim.assign_coords(dayofyear=full + 366),
        ],
        dim="dayofyear",
    )
    smoothed = tiled.rolling(dayofyear=int(window), center=True, min_periods=1).mean()
    return smoothed.sel(dayofyear=full)


def _fit_harmonics(doy_clim, n_harmonics):
    """
    Least-squares fit of `n_harmonics` annual harmonics to a daily-DOY
    climatology, evaluated back onto the full 1..366 axis.

    Design matrix columns: [1, cos(2πkt/366), sin(2πkt/366)] for k=1..K.
    Solved per grid cell via lstsq; cells that are all-NaN (land) stay NaN.
    """
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")

    doy = doy_clim["dayofyear"].values.astype(float)
    t = 2.0 * np.pi * doy / 366.0

    cols = [np.ones_like(t)]
    for k in range(1, int(n_harmonics) + 1):
        cols.append(np.cos(k * t))
        cols.append(np.sin(k * t))
    A = np.stack(cols, axis=1)                       # (366, 2K+1)
    A_pinv = np.linalg.pinv(A)                       # (2K+1, 366)

    def _fit_var(da):
        # Move dayofyear to axis 0, flatten the spatial dims
        da = da.transpose("dayofyear", ...)
        vals = da.values
        shape_rest = vals.shape[1:]
        Y = vals.reshape(vals.shape[0], -1)          # (366, n_cells)

        # Zero-fill residual NaNs so lstsq is well-posed; restore land NaNs after.
        land = np.all(~np.isfinite(Y), axis=0)
        Yf = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        coeffs = A_pinv @ Yf                          # (2K+1, n_cells)
        fitted = A @ coeffs                           # (366, n_cells)
        fitted[:, land] = np.nan

        return xr.DataArray(
            fitted.reshape((A.shape[0],) + shape_rest),
            dims=da.dims,
            coords=da.coords,
            name=da.name,
        )

    if isinstance(doy_clim, xr.DataArray):
        return _fit_var(doy_clim)
    return xr.Dataset({v: _fit_var(doy_clim[v]) for v in doy_clim.data_vars})