
#!/usr/bin/env python
"""
check_glsea_nans.py

Determine whether the NaN/nodata mask of a GLSEA lake-surface-temperature Zarr
store is constant through time, or whether NaNs occasionally appear on water
cells.

Outputs:
  - console summary
  - nan_diagnostics.nc  (per-cell valid counts + mask classification)
  - per_date_nan_counts.csv
  - optional PNG figures
"""

import numpy as np
import pandas as pd
import xarray as xr
import dask

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
ZARR_PATH   = "/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/glsea3_sst.zarr"
VAR_NAME    = None        # None -> auto-detect the first 3-D (time, y, x) variable
TIME_DIM    = "time"
MAKE_PLOTS  = True
# Cells valid in at least this fraction of timesteps are treated as "core water"
CORE_WATER_FRAC = 0.95
# Physically plausible LST range (deg C) for a sanity check on sentinel values
VALID_RANGE = (-5.0, 40.0)


# ----------------------------------------------------------------------------
def open_data(path, var=None, time_dim="time"):
    ds = xr.open_zarr(path, consolidated=True, decode_cf=True, mask_and_scale=True)

    if var is None:
        candidates = [
            v for v in ds.data_vars
            if time_dim in ds[v].dims and ds[v].ndim == 3
        ]
        if not candidates:
            raise ValueError(f"No 3-D variable with dim '{time_dim}' found. "
                             f"Variables: {list(ds.data_vars)}")
        var = candidates[0]
        print(f"[info] auto-detected variable: {var}")

    da = ds[var]

    # Chunk so that time-reductions stream: full spatial tile, modest time slabs.
    ydim, xdim = [d for d in da.dims if d != time_dim]
    da = da.chunk({time_dim: 200, ydim: -1, xdim: -1})
    return ds, da, var, (ydim, xdim)


# ----------------------------------------------------------------------------
def main():
    ds, da, var, (ydim, xdim) = open_data(ZARR_PATH, VAR_NAME, TIME_DIM)

    n_time = da.sizes[TIME_DIM]
    n_cell = da.sizes[ydim] * da.sizes[xdim]
    print(f"[info] variable={var}  dtype={da.dtype}  shape={dict(da.sizes)}")
    print(f"[info] {n_time} timesteps, {n_cell} cells per timestep")

    # ---- 0. Sentinel-value sanity check on a small sample -------------------
    sample = da.isel({TIME_DIM: slice(0, min(5, n_time))}).compute()
    finite = sample.values[np.isfinite(sample.values)]
    if finite.size:
        print(f"[check] sample finite min/max = {finite.min():.3f} / {finite.max():.3f}")
        out_of_range = ((finite < VALID_RANGE[0]) | (finite > VALID_RANGE[1])).sum()
        if out_of_range:
            print(f"[WARN] {out_of_range} sampled values fall outside "
                  f"{VALID_RANGE} -- you may have undecoded fill values "
                  f"(e.g. -99, -999, 32767) masquerading as data.")
    else:
        print("[WARN] sample contained no finite values at all.")

    # ---- 1. Build the two reductions lazily, compute in ONE pass -----------
    isvalid = da.notnull()

    # per-cell number of valid timesteps
    valid_count = isvalid.sum(dim=TIME_DIM).astype("int32").rename("valid_count")

    # per-date number of valid cells (whole grid)
    valid_per_date = isvalid.sum(dim=(ydim, xdim)).astype("int64").rename("valid_cells")

    print("[info] computing reductions (single pass over the data)...")
    valid_count, valid_per_date = dask.compute(valid_count, valid_per_date)

    # ---- 2. Classify cells -------------------------------------------------
    vc = valid_count.values
    always_nan   = (vc == 0)
    always_valid = (vc == n_time)
    intermittent = (vc > 0) & (vc < n_time)

    ever_water = ~always_nan                       # valid at least once
    core_water = vc >= CORE_WATER_FRAC * n_time     # robust water definition

    print("\n" + "=" * 68)
    print("PER-CELL NaN MASK STABILITY")
    print("=" * 68)
    print(f"always NaN (land / outside domain) : {always_nan.sum():>10,} cells")
    print(f"always valid (stable water)        : {always_valid.sum():>10,} cells")
    print(f"INTERMITTENT NaN                   : {intermittent.sum():>10,} cells "
          f"({100*intermittent.sum()/max(ever_water.sum(),1):.2f}% of ever-water cells)")

    if intermittent.sum() == 0:
        print("\n>>> VERDICT: the nodata mask is IDENTICAL at every timestep. "
              "NaNs only ever occur on the fixed land mask.")
    else:
        print("\n>>> VERDICT: the nodata mask VARIES through time. Some water "
              "cells are NaN on some dates. See details below.")

        # distribution of how often intermittent cells are missing
        miss_frac = 1.0 - vc[intermittent] / n_time
        qs = np.percentile(miss_frac, [50, 90, 99, 100])
        print(f"    missing-fraction of intermittent cells: "
              f"median={qs[0]:.3f}  p90={qs[1]:.3f}  p99={qs[2]:.3f}  max={qs[3]:.3f}")
    # ---- 3. Interior holes vs shoreline flicker ---------------------------
    # Shoreline cells (adjacent to always-NaN) flickering is usually a
    # land-mask/regridding artefact. Interior holes are the real concern.
    interior_intermittent = None
    try:
        from scipy import ndimage

        # Distance (in cells) from each cell to the nearest always-NaN cell.
        # always_nan == True are the "sources"; EDT measures distance through
        # the non-land region.
        dist_to_land = ndimage.distance_transform_edt(~always_nan)

        SHORE_BUFFER = 2  # cells
        shoreline = intermittent & (dist_to_land <= SHORE_BUFFER)
        interior_intermittent = intermittent & (dist_to_land > SHORE_BUFFER)

        print("\n" + "-" * 68)
        print("INTERMITTENT CELL LOCATION")
        print("-" * 68)
        print(f"within {SHORE_BUFFER} cells of land (shoreline flicker) : "
              f"{shoreline.sum():>8,}")
        print(f"interior of the lake (genuine data holes)          : "
              f"{interior_intermittent.sum():>8,}")

        if interior_intermittent.sum() == 0 and intermittent.sum() > 0:
            print("    -> All variability is at the land/water boundary. Likely a "
                  "land-mask or regridding artefact rather than missing data.")
        elif interior_intermittent.sum() > 0:
            print("    -> Genuine gaps exist over open water. Treat the mask as "
                  "time-varying in any downstream analysis.")
    except ImportError:
        print("\n[info] scipy not available; skipping shoreline/interior split.")

    # ---- 4. Per-date diagnostics ------------------------------------------
    n_core = int(core_water.sum())
    n_ever = int(ever_water.sum())

    core_water_da = xr.DataArray(core_water, dims=(ydim, xdim),
                                 coords={d: da[d] for d in (ydim, xdim) if d in da.coords})

    # Recompute per-date NaN counts restricted to the core-water footprint.
    # This is a second pass, but only over water cells and it is what you
    # actually want to report on.
    print("\n[info] computing per-date NaN counts within the water mask...")
    nan_in_water = (da.isnull() & core_water_da).sum(dim=(ydim, xdim)).compute()

    times = pd.to_datetime(da[TIME_DIM].values)
    per_date = pd.DataFrame({
        "time": times,
        "valid_cells_grid": valid_per_date.values,
        "nan_cells_in_water": nan_in_water.values,
    })
    per_date["frac_water_missing"] = per_date["nan_cells_in_water"] / max(n_core, 1)
    per_date.to_csv("per_date_nan_counts.csv", index=False)

    print("\n" + "=" * 68)
    print("PER-DATE MISSINGNESS OVER WATER")
    print("=" * 68)
    print(f"core-water cells (valid >= {CORE_WATER_FRAC:.0%} of dates): {n_core:,}")
    print(f"ever-water cells (valid >= once)                     : {n_ever:,}")
    print(f"dates with ZERO missing water cells : "
          f"{(per_date.nan_cells_in_water == 0).sum():,} / {n_time:,}")
    print(f"dates with SOME missing water cells : "
          f"{(per_date.nan_cells_in_water > 0).sum():,} / {n_time:,}")

    worst = per_date.nlargest(15, "nan_cells_in_water")
    if worst.nan_cells_in_water.max() > 0:
        print("\nWorst 15 dates:")
        print(worst.to_string(
            index=False,
            formatters={"frac_water_missing": "{:.4f}".format},
        ))

        # Are the bad dates clustered? (e.g. a sensor outage or ice season)
        bad = per_date.loc[per_date.frac_water_missing > 0.01, "time"]
        if len(bad):
            print(f"\ndates with >1% of water missing: {len(bad)}")
            print("  month histogram:",
                  bad.dt.month.value_counts().sort_index().to_dict())
            print("  year histogram :",
                  bad.dt.year.value_counts().sort_index().to_dict())

    # ---- 5. Fully-empty or duplicated timesteps ---------------------------
    empty = per_date.loc[per_date.valid_cells_grid == 0, "time"]
    if len(empty):
        print(f"\n[WARN] {len(empty)} timesteps are entirely NaN, e.g. "
              f"{list(empty.head(10).astype(str))}")

    # ---- 6. Time-axis integrity (gaps, duplicates, non-monotonic) ---------
    print("\n" + "-" * 68)
    print("TIME AXIS")
    print("-" * 68)
    tser = pd.Series(times)
    print(f"range: {tser.min()} -> {tser.max()}")
    print(f"duplicated timestamps : {tser.duplicated().sum()}")
    print(f"monotonic increasing  : {tser.is_monotonic_increasing}")
    deltas = tser.diff().dropna()
    if len(deltas):
        print(f"step counts: {deltas.value_counts().head(5).to_dict()}")
        expected = deltas.mode().iloc[0]
        gaps = tser[1:][deltas.values > expected]
        if len(gaps):
            print(f"[WARN] {len(gaps)} gaps larger than the modal step "
                  f"({expected}); first few end at "
                  f"{list(gaps.head(5).astype(str))}")
            print("       NOTE: absent dates are a different problem from NaN "
                  "cells -- missing files simply may not be in the store.")

    # ---- 7. Save per-cell diagnostics ------------------------------------
    classification = np.zeros(vc.shape, dtype="int8")
    classification[always_nan] = 0      # land / never valid
    classification[always_valid] = 1    # always valid
    classification[intermittent] = 2    # intermittent
    if interior_intermittent is not None:
        classification[interior_intermittent] = 3   # intermittent, interior

    diag = xr.Dataset(
        {
            "valid_count": valid_count,
            "missing_count": (valid_count.dims, (n_time - vc).astype("int32")),
            "missing_frac": (valid_count.dims, ((n_time - vc) / n_time).astype("float32")),
            "classification": (valid_count.dims, classification),
        },
        coords=valid_count.coords,
    )
    diag["classification"].attrs["flag_values"] = [0, 1, 2, 3]
    diag["classification"].attrs["flag_meanings"] = (
        "always_nan always_valid intermittent_shoreline intermittent_interior"
    )
    diag.attrs["source"] = ZARR_PATH
    diag.attrs["variable"] = var
    diag.attrs["n_time"] = n_time
    diag.to_netcdf("nan_diagnostics.nc")
    print("\n[info] wrote nan_diagnostics.nc and per_date_nan_counts.csv")

    # ---- 8. Plots --------------------------------------------------------
    if MAKE_PLOTS:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap, BoundaryNorm

            fig, ax = plt.subplots(1, 2, figsize=(16, 6))

            cmap = ListedColormap(["#d9d9d9", "#2166ac", "#fdb863", "#b2182b"])
            norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)
            im = ax[0].imshow(classification, origin="lower", cmap=cmap, norm=norm,
                              interpolation="nearest")
            ax[0].set_title("Cell classification")
            cb = fig.colorbar(im, ax=ax[0], ticks=[0, 1, 2, 3])
            cb.ax.set_yticklabels(["always NaN", "always valid",
                                   "intermittent\n(shoreline)",
                                   "intermittent\n(interior)"])

            mf = np.where(always_nan, np.nan, (n_time - vc) / n_time)
            im2 = ax[1].imshow(mf, origin="lower", cmap="magma_r",
                               interpolation="nearest",
                               vmin=0, vmax=max(np.nanmax(mf), 1e-6))
            ax[1].set_title("Fraction of dates missing (water cells only)")
            fig.colorbar(im2, ax=ax[1])
            fig.tight_layout()
            fig.savefig("nan_maps.png", dpi=140)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(per_date.time, per_date.frac_water_missing, lw=0.7)
            ax.set_ylabel("fraction of water cells NaN")
            ax.set_title("Missing water cells per date")
            fig.tight_layout()
            fig.savefig("nan_timeseries.png", dpi=140)
            plt.close(fig)

            print("[info] wrote nan_maps.png and nan_timeseries.png")
        except ImportError:
            print("[info] matplotlib not available; skipping plots.")


if __name__ == "__main__":
    main()