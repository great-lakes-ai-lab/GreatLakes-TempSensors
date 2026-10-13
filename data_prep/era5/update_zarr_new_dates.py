"""
ERA5 Daily Mean Data — Append New Years to Existing Zarr
---------------------------------------------------------
Points at an existing daily-mean Zarr store (created by the extraction script),
determines which days are missing, pulls them from ArrayLake, and appends
along the time dimension.

Safe to re-run: if the store is already up to date it exits without writing.
"""

import gc
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from arraylake import Client

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Existing Zarr store to append to
STORE_PATH = Path('/Users/jagraha/Data/ciglr_globus/greatlakes_era5_1995_2025.zarr')

# Latest date you want in the store. The script will clamp this to whatever
# ERA5 actually has available (last complete 24-hour day).
APPEND_THROUGH = "2025-12-31"

# Variables to append — must match what's already in the store
VARIABLES = ["t2m", "ssr", "u10", "v10"]

# Set True to report what *would* happen without writing anything
DRY_RUN = True

# Make a copy of the store before writing (recommended; appends are in-place)
MAKE_BACKUP = True

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

# =============================================================================
# VARIABLE DEFINITIONS (source name -> output name, reduction, attrs)
# =============================================================================

VAR_SPEC = {
    "t2m": {
        "source": "t2",
        "output": "t2m_daily_mean",
        "how": "mean",
        "offset": -273.15,
        "attrs": {
            "units": "°C",
            "long_name": "Daily Mean 2m Temperature",
            "standard_name": "air_temperature",
        },
    },
    "ssr": {
        "source": "ssr",
        "output": "ssr_daily_sum",
        "how": "sum",
        "offset": 0.0,
        "attrs": {
            "units": "J/m²",
            "long_name": "Daily Sum Surface Net Shortwave Radiation",
            "standard_name": "surface_net_downward_shortwave_flux",
        },
    },
    "u10": {
        "source": "u10",
        "output": "u10_daily_mean",
        "how": "mean",
        "offset": 0.0,
        "attrs": {
            "units": "m/s",
            "long_name": "Daily Mean 10m Eastward Wind Component",
            "standard_name": "eastward_wind",
        },
    },
    "v10": {
        "source": "v10",
        "output": "v10_daily_mean",
        "how": "mean",
        "offset": 0.0,
        "attrs": {
            "units": "m/s",
            "long_name": "Daily Mean 10m Northward Wind Component",
            "standard_name": "northward_wind",
        },
    },
}

# =============================================================================
# INSPECT EXISTING STORE
# =============================================================================

if not STORE_PATH.exists():
    raise FileNotFoundError(f"Store not found: {STORE_PATH}")

print(f"Opening existing store: {STORE_PATH}")
ds_existing = xr.open_zarr(STORE_PATH)

existing_vars = list(ds_existing.data_vars)
existing_times = pd.to_datetime(ds_existing.time.values)
last_time = existing_times[-1]

# Exact grid from the existing store — we reuse these to guarantee alignment
target_lats = ds_existing.latitude.values
target_lons = ds_existing.longitude.values

# Chunk size along time, as stored (used only for reporting)
first_var = existing_vars[0]
time_chunk = ds_existing[first_var].encoding.get("chunks", (None,))[0]

has_time_label = "time_label" in ds_existing.coords
existing_attrs = dict(ds_existing.attrs)

print(f"  Variables:   {existing_vars}")
print(f"  Time range:  {existing_times[0].date()} to {last_time.date()} "
      f"({len(existing_times)} steps)")
print(f"  Grid:        {len(target_lats)} lat x {len(target_lons)} lon")
print(f"  Time chunk:  {time_chunk}")
print(f"  time_label:  {'present' if has_time_label else 'absent'}")
print()

# Sanity check: requested variables must already exist in the store
expected_outputs = [VAR_SPEC[v]["output"] for v in VARIABLES]
missing = [v for v in expected_outputs if v not in existing_vars]
extra = [v for v in existing_vars if v not in expected_outputs]
if missing or extra:
    raise ValueError(
        "Variable mismatch between store and VARIABLES config.\n"
        f"  In store but not requested: {extra}\n"
        f"  Requested but not in store: {missing}\n"
        "Appending a different variable set would corrupt the store."
    )

# Verify existing time axis is clean before we add to it
if not existing_times.is_monotonic_increasing:
    raise ValueError("Existing time axis is not monotonically increasing — fix before appending.")
if existing_times.duplicated().any():
    raise ValueError("Existing time axis contains duplicates — fix before appending.")

append_start = last_time + pd.Timedelta(days=1)
print(f"First day to append: {append_start.date()}")

# =============================================================================
# CONNECT TO SOURCE AND DETERMINE AVAILABLE RANGE
# =============================================================================

print("\nConnecting to ArrayLake...")
client = Client()
# repo = client.get_repo("earthmover-public/era5-surface-aws")
repo = client.get_repo("earthmover-public/era5")
session = repo.readonly_session("main")

ds_src = xr.open_dataset(
    session.store,
    engine="zarr",
    consolidated=False,
    zarr_format=3,
    chunks="auto",
    group="temporal",
)
print(f"  Source variables: {list(ds_src.data_vars)}")

# Last available hourly timestamp; clamp to last COMPLETE day so that
# daily sums/means are not computed from a partial day.
src_last = pd.Timestamp(ds_src.time.values[-1])
if src_last.hour == 23:
    last_complete_day = src_last.normalize()
else:
    last_complete_day = src_last.normalize() - pd.Timedelta(days=1)

requested_end = pd.Timestamp(APPEND_THROUGH).normalize()
append_end = min(requested_end, last_complete_day)

print(f"  Source last timestamp:  {src_last}")
print(f"  Last complete day:      {last_complete_day.date()}")
print(f"  Requested through:      {requested_end.date()}")
print(f"  Will append through:    {append_end.date()}")

if append_end < append_start:
    print("\nStore is already up to date with available ERA5 data. Nothing to do.")
    ds_existing.close()
    raise SystemExit(0)

n_days = (append_end - append_start).days + 1
print(f"  Days to append:         {n_days}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_daily(ds_src, var_key, t0, t1, lats, lons):
    """Extract one variable for [t0, t1] on the target grid, as a daily reduction."""
    spec = VAR_SPEC[var_key]

    da = ds_src[spec["source"]].sel(time=slice(t0, t1))

    # Snap to the existing store's exact grid, then re-stamp coordinates so
    # they match the store bit-for-bit (avoids float-comparison drift).
    da = da.sel(latitude=lats, longitude=lons, method="nearest")
    da = da.assign_coords(latitude=lats, longitude=lons)

    resampled = da.resample(time="1D")
    daily = resampled.sum() if spec["how"] == "sum" else resampled.mean()

    # Guard against partial days sneaking through
    counts = resampled.count(dim="time").isel(
        {d: 0 for d in da.dims if d != "time"}, missing_dims="ignore"
    )
    counts = counts.compute()
    incomplete = counts.where(counts != 24, drop=True)
    if incomplete.size:
        bad = pd.to_datetime(incomplete.time.values)
        raise RuntimeError(
            f"{var_key}: incomplete days found (hours != 24): "
            f"{[str(b.date()) for b in bad]}"
        )

    daily = (daily + spec["offset"]).compute()
    out = daily.to_dataset(name=spec["output"])
    out[spec["output"]].attrs = dict(spec["attrs"])
    return out


def process_window(ds_src, t0, t1, variables, lats, lons):
    """Build a merged in-memory daily dataset for the time window."""
    pieces = [extract_daily(ds_src, v, t0, t1, lats, lons) for v in variables]
    return xr.merge(pieces)


def process_with_retry(ds_src, t0, t1, variables, lats, lons):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return process_window(ds_src, t0, t1, variables, lats, lons)
        except Exception as e:
            print(f"    ⚠ Attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                print(f"    Waiting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)
                gc.collect()
            else:
                raise RuntimeError(f"Failed to process {t0} to {t1}") from e


def prepare_for_append(ds_new, add_time_label):
    """Strip encoding and add the auxiliary time label coordinate."""
    for name in list(ds_new.data_vars) + list(ds_new.coords):
        ds_new[name].encoding.clear()

    if add_time_label:
        labels = pd.to_datetime(ds_new.time.values).strftime("%Y-%m-%d")
        ds_new = ds_new.assign_coords(
            time_label=("time", np.asarray(labels, dtype="<U10"))
        )
    return ds_new


# =============================================================================
# BUILD YEARLY WINDOWS
# =============================================================================

windows = []
cursor = append_start
while cursor <= append_end:
    year_end = pd.Timestamp(year=cursor.year, month=12, day=31)
    stop = min(year_end, append_end)
    windows.append((cursor, stop))
    cursor = stop + pd.Timedelta(days=1)

print("\nWindows to process:")
for w0, w1 in windows:
    print(f"  {w0.date()} -> {w1.date()}  ({(w1 - w0).days + 1} days)")

if DRY_RUN:
    print("\nDRY_RUN is True — exiting without writing.")
    ds_existing.close()
    raise SystemExit(0)

# =============================================================================
# BACKUP
# =============================================================================

if MAKE_BACKUP:
    backup = STORE_PATH.with_name(
        STORE_PATH.stem + f".backup_{pd.Timestamp.now():%Y%m%dT%H%M%S}" + STORE_PATH.suffix
    )
    print(f"\nBacking up store to: {backup}")
    shutil.copytree(STORE_PATH, backup)
    print("  ✓ Backup complete.")

ds_existing.close()

# =============================================================================
# PROCESS AND APPEND
# =============================================================================

print()
appended_total = 0

for i, (w0, w1) in enumerate(windows, 1):
    print(f"[{i}/{len(windows)}] Processing {w0.date()} -> {w1.date()}...")
    t_start = time.time()

    ds_new = process_with_retry(
        ds_src,
        w0.strftime("%Y-%m-%d"),
        w1.strftime("%Y-%m-%d"),
        VARIABLES,
        target_lats,
        target_lons,
    )

    # Verify the window came back with exactly the days we expect
    got = pd.to_datetime(ds_new.time.values)
    want = pd.date_range(w0, w1, freq="D")
    if not got.equals(want):
        raise RuntimeError(
            f"Time axis mismatch for {w0.date()}-{w1.date()}: "
            f"got {len(got)} steps ({got[0].date()}..{got[-1].date()}), "
            f"expected {len(want)}"
        )

    ds_new = prepare_for_append(ds_new, add_time_label=has_time_label)

    # Confirm nothing is Dask-backed, so the append is a serial write and it is
    # safe to relax chunk checks when filling the store's partial final chunk.
    for name, var in ds_new.variables.items():
        if var.chunks is not None:
            ds_new[name] = ds_new[name].load()

    print(f"  Writing {ds_new.sizes['time']} days...")
    ds_new.to_zarr(STORE_PATH, append_dim="time", safe_chunks=False)

    appended_total += ds_new.sizes["time"]
    print(f"  ✓ Done in {time.time() - t_start:.1f}s")

    del ds_new
    gc.collect()

# =============================================================================
# UPDATE GLOBAL ATTRIBUTES
# =============================================================================

print("\nUpdating global attributes...")
root = zarr.open_group(str(STORE_PATH), mode="r+")

old_range = existing_attrs.get("time_range", "")
start_label = old_range.split(" to ")[0] if " to " in old_range else str(existing_times[0].date())

root.attrs.update(
    {
        **existing_attrs,
        "time_range": f"{start_label} to {append_end.date()}",
        "history": (
            existing_attrs.get("history", "")
            + f" | Appended {append_start.date()}..{append_end.date()} "
              f"on {pd.Timestamp.now():%Y-%m-%d}"
        ),
    }
)
print("  ✓ Attributes updated.")

# =============================================================================
# VERIFY
# =============================================================================

print("\nVerifying store...")
ds_check = xr.open_zarr(STORE_PATH)
t = pd.to_datetime(ds_check.time.values)

checks = {
    "monotonic increasing": bool(t.is_monotonic_increasing),
    "no duplicate times": not bool(t.duplicated().any()),
    "no missing days": len(t) == (t[-1] - t[0]).days + 1,
    "grid unchanged": (
        np.array_equal(ds_check.latitude.values, target_lats)
        and np.array_equal(ds_check.longitude.values, target_lons)
    ),
}
if has_time_label:
    checks["time_label in sync"] = bool(
        (ds_check.time_label.values == t.strftime("%Y-%m-%d")).all()
    )

for label, ok in checks.items():
    print(f"  {'✓' if ok else '✗'} {label}")

print(f"\n{'=' * 60}")
print(f"Appended {appended_total} days. Store now covers "
      f"{t[0].date()} to {t[-1].date()} ({len(t)} steps).")
print(f"{'=' * 60}\n")

print(ds_check)
ds_check.close()

if not all(checks.values()):
    raise RuntimeError("Post-append verification failed — inspect the backup copy.")