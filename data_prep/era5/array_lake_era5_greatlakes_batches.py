"""
ERA5 Daily Mean Data Extraction from ArrayLake (Batched)
--------------------------------------------------------
Extracts user-specified variables in yearly batches, computes daily means,
saves intermediate results to a temp directory, then merges into a unified Zarr store.

Variables available:
  - t2m: 2m Temperature (converted to °C)
  - ssr: Surface Net Shortwave Radiation (daily sum, J/m²)
  - u10: 10m Eastward (U) Wind Component (daily mean, m/s)
  - v10: 10m Northward (V) Wind Component (daily mean, m/s)

Extent (WGS84 from QGIS):
  -92.4269584928940304, 38.8679794360660651 : -75.8746325293894728, 50.6129829563175306
"""

import numpy as np
import pandas as pd
import xarray as xr
from arraylake import Client
from pathlib import Path
import shutil
import time
import gc

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Output format: "netcdf" or "zarr"
OUTPUT_FORMAT = "zarr"

# Output file path (extension will be added automatically)
OUTPUT_PATH = Path('/Users/jagraha/Data/ciglr_globus/greatlakes_era5_1995_2025')

# Temporary directory for intermediate batch files
TEMP_DIR = OUTPUT_PATH.parent / "_temp_era5_batches"

# Variables to include: any combination of ["t2m", "ssr", "u10", "v10"]
VARIABLES = ["t2m", "ssr", "u10", "v10"]

# Time range
TIME_START = "1995-01-01"
TIME_END = "2024-12-31"

# Spatial extent in WGS84 (from QGIS)
LON_MIN_WGS84 = -92.4269584928940304
LON_MAX_WGS84 = -75.8746325293894728
LAT_MIN_WGS84 = 38.8679794360660651
LAT_MAX_WGS84 = 50.6129829563175306

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

# =============================================================================
# COORDINATE CONVERSION
# =============================================================================

def wgs84_lon_to_era5(lon):
    """Convert WGS84 longitude (-180 to 180) to ERA5 (0 to 360)."""
    return lon % 360


# Convert longitude to ERA5's 0-360 range
lon_min_era5 = wgs84_lon_to_era5(LON_MIN_WGS84)
lon_max_era5 = wgs84_lon_to_era5(LON_MAX_WGS84)

# ERA5 latitude is descending (90 to -90), so slice high to low
lat_max_era5 = LAT_MAX_WGS84
lat_min_era5 = LAT_MIN_WGS84

print(f"Spatial extent (ERA5 coordinates):")
print(f"  Longitude: {lon_min_era5:.4f}° to {lon_max_era5:.4f}°")
print(f"  Latitude:  {lat_max_era5:.4f}° to {lat_min_era5:.4f}° (descending)")
print(f"Time range: {TIME_START} to {TIME_END}")
print(f"Variables: {VARIABLES}")
print(f"Output format: {OUTPUT_FORMAT}")
print()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def select_region(da, time_start, time_end, lat_max, lat_min, lon_min, lon_max):
    """Select spatial and temporal subset from a DataArray."""
    return da.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
        time=slice(time_start, time_end),
    )


def process_year_batch(ds, year_start, year_end, variables, lat_max, lat_min, lon_min, lon_max):
    """Process a single time batch and return a merged dataset."""
    time_start = f"{year_start}-01-01"
    time_end = f"{year_end}-12-31"

    datasets = []

    # --- 2m Temperature ---
    if "t2m" in variables:
        t2 = select_region(ds["t2"], time_start, time_end,
                           lat_max, lat_min, lon_min, lon_max)
        t2m_daily = (t2.resample(time="1D").mean() - 273.15).compute()
        t2m_ds = t2m_daily.to_dataset(name="t2m_daily_mean")
        t2m_ds["t2m_daily_mean"].attrs = {
            "units": "°C",
            "long_name": "Daily Mean 2m Temperature",
            "standard_name": "air_temperature",
        }
        datasets.append(t2m_ds)

    # --- Surface Net Shortwave Radiation ---
    if "ssr" in variables:
        ssr = select_region(ds["ssr"], time_start, time_end,
                            lat_max, lat_min, lon_min, lon_max)
        ssr_daily = ssr.resample(time="1D").sum().compute()
        ssr_ds = ssr_daily.to_dataset(name="ssr_daily_sum")
        ssr_ds["ssr_daily_sum"].attrs = {
            "units": "J/m²",
            "long_name": "Daily Sum Surface Net Shortwave Radiation",
            "standard_name": "surface_net_downward_shortwave_flux",
        }
        datasets.append(ssr_ds)

    # --- 10m U Wind Component ---
    if "u10" in variables:
        u10 = select_region(ds["u10"], time_start, time_end,
                            lat_max, lat_min, lon_min, lon_max)
        u10_daily = u10.resample(time="1D").mean().compute()
        u10_ds = u10_daily.to_dataset(name="u10_daily_mean")
        u10_ds["u10_daily_mean"].attrs = {
            "units": "m/s",
            "long_name": "Daily Mean 10m Eastward Wind Component",
            "standard_name": "eastward_wind",
        }
        datasets.append(u10_ds)

    # --- 10m V Wind Component ---
    if "v10" in variables:
        v10 = select_region(ds["v10"], time_start, time_end,
                            lat_max, lat_min, lon_min, lon_max)
        v10_daily = v10.resample(time="1D").mean().compute()
        v10_ds = v10_daily.to_dataset(name="v10_daily_mean")
        v10_ds["v10_daily_mean"].attrs = {
            "units": "m/s",
            "long_name": "Daily Mean 10m Northward Wind Component",
            "standard_name": "northward_wind",
        }
        datasets.append(v10_ds)

    return xr.merge(datasets)


def process_with_retry(ds, year_start, year_end, variables, lat_max, lat_min, lon_min, lon_max):
    """Process a batch with retries on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return process_year_batch(
                ds, year_start, year_end, variables,
                lat_max, lat_min, lon_min, lon_max
            )
        except Exception as e:
            print(f"    ⚠ Attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                print(f"    Waiting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)
                gc.collect()
            else:
                raise RuntimeError(
                    f"Failed to process {year_start}-{year_end} after {MAX_RETRIES} attempts"
                ) from e


# =============================================================================
# GENERATE YEAR BATCHES
# =============================================================================

start_year = int(TIME_START[:4])
end_year = int(TIME_END[:4])
year_batches = [(y, y) for y in range(start_year, end_year + 1)]

print(f"Will process {len(year_batches)} yearly batches: {start_year} to {end_year}")
print()

# =============================================================================
# CONNECT TO ARRAYLAKE
# =============================================================================

print("Connecting to ArrayLake...")
client = Client()
repo = client.get_repo("earthmover-public/era5-surface-aws")
session = repo.readonly_session("main")

# Open dataset with dask chunks for parallel computation
ds = xr.open_dataset(
    session.store,
    engine="zarr",
    consolidated=False,
    zarr_format=3,
    chunks="auto",
    group="temporal",
)

print(f"Dataset loaded. Available variables: {list(ds.data_vars)}")
print()

# =============================================================================
# PROCESS IN BATCHES
# =============================================================================

# Create temp directory
TEMP_DIR.mkdir(parents=True, exist_ok=True)
print(f"Temporary directory: {TEMP_DIR}")
print()

batch_paths = []

for i, (y_start, y_end) in enumerate(year_batches, 1):
    batch_file = TEMP_DIR / f"batch_{y_start}_{y_end}.zarr"

    # Skip if batch already exists (allows resuming interrupted runs)
    if batch_file.exists():
        print(f"[{i}/{len(year_batches)}] Batch {y_start}-{y_end} already exists, skipping.")
        batch_paths.append(batch_file)
        continue

    print(f"[{i}/{len(year_batches)}] Processing {y_start}-{y_end}...")
    t_start = time.time()

    ds_batch = process_with_retry(
        ds, y_start, y_end, VARIABLES,
        lat_max_era5, lat_min_era5, lon_min_era5, lon_max_era5
    )

    # Save batch to temp zarr
    ds_batch.to_zarr(batch_file, mode="w")
    batch_paths.append(batch_file)

    elapsed = time.time() - t_start
    print(f"  ✓ Done in {elapsed:.1f}s — saved to {batch_file.name}")

    # Free memory
    del ds_batch
    gc.collect()

print()
print(f"All {len(batch_paths)} batches processed successfully.")
print()

# =============================================================================
# MERGE BATCHES INTO UNIFIED ZARR
# =============================================================================

print("Merging batches into unified output...")

# Open all batch datasets lazily
batch_datasets = [
    xr.open_zarr(p, chunks="auto") for p in sorted(batch_paths)
]

# Concatenate along time dimension
ds_out = xr.concat(batch_datasets, dim="time")

# Sort by time to ensure correct ordering
ds_out = ds_out.sortby("time")

# Rechunk for uniform chunk structure in the final output
ds_out = ds_out.chunk({"time": 365, "latitude": -1, "longitude": -1})

# Clear any inherited encoding from the batch files
for var in ds_out.data_vars:
    ds_out[var].encoding.clear()
for coord in ds_out.coords:
    ds_out[coord].encoding.clear()

# Add time labels as auxiliary coordinate
date_strings = pd.to_datetime(ds_out.time.values).strftime("%Y-%m-%d")
ds_out = ds_out.assign_coords(time_label=("time", date_strings))

# Add global attributes
ds_out.attrs = {
    "source": "earthmover-public/era5-surface-aws via ArrayLake",
    "history": "Processed with daily mean resampling (yearly batches)",
    "time_range": f"{TIME_START} to {TIME_END}",
    "spatial_extent_wgs84": (
        f"lon: [{LON_MIN_WGS84}, {LON_MAX_WGS84}], "
        f"lat: [{LAT_MIN_WGS84}, {LAT_MAX_WGS84}]"
    ),
    "spatial_extent_era5": (
        f"lon: [{lon_min_era5:.4f}, {lon_max_era5:.4f}], "
        f"lat: [{lat_min_era5:.4f}, {lat_max_era5:.4f}]"
    ),
    "variables_included": ", ".join(VARIABLES),
    "conventions": "CF-1.8",
}

# Save final output
if OUTPUT_FORMAT.lower() == "netcdf":
    output_file = OUTPUT_PATH.with_suffix(".nc")
    print(f"Saving to NetCDF: {output_file}")
    ds_out.to_netcdf(output_file)
elif OUTPUT_FORMAT.lower() == "zarr":
    output_file = OUTPUT_PATH.with_suffix(".zarr")
    print(f"Saving to Zarr: {output_file}")
    ds_out.to_zarr(output_file, mode="w")
else:
    raise ValueError(f"Unknown output format: {OUTPUT_FORMAT}. Use 'netcdf' or 'zarr'.")

# Close lazy datasets
for batch_ds in batch_datasets:
    batch_ds.close()

print(f"\n{'='*60}")
print(f"Output saved successfully: {output_file}")
print(f"{'='*60}")

# =============================================================================
# CLEANUP TEMP DIRECTORY
# =============================================================================

print(f"\nCleaning up temporary directory: {TEMP_DIR}")
shutil.rmtree(TEMP_DIR)
print("  ✓ Temp directory removed.")

print(f"\nDataset summary:")
# Re-open to print summary without loading into memory
ds_final = xr.open_zarr(output_file) if OUTPUT_FORMAT == "zarr" else xr.open_dataset(output_file)
print(ds_final)
ds_final.close()