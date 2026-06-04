"""
ERA5 Daily Mean Data Extraction from ArrayLake
----------------------------------------------
Extracts user-specified variables, computes daily means, and saves to NetCDF or Zarr.

Variables available:
  - t2m: 2m Temperature (converted to °C)
  - ssr: Surface Net Shortwave Radiation (daily mean, J/m²)
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

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Output format: "netcdf" or "zarr"
OUTPUT_FORMAT = "zarr"

# Output file path (extension will be added automatically)
OUTPUT_PATH = Path("/Users/jagraha/dev/deepsensor_projects/data/greatlakes_raw_temporal_ds_inputs")


# Variables to include: any combination of ["t2m", "ssr", "u10", "v10"]
VARIABLES = ["t2m", "ssr", "u10", "v10"]

# Time range
TIME_START = "2018-01-01"
# TIME_START = "2022-12-01"
TIME_END = "2022-12-31"

# Spatial extent in WGS84 (from QGIS)
LON_MIN_WGS84 = -92.4269584928940304
LON_MAX_WGS84 = -75.8746325293894728
LAT_MIN_WGS84 = 38.8679794360660651
LAT_MAX_WGS84 = 50.6129829563175306

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
# HELPER FUNCTIONS
# =============================================================================

def select_region(da, time_start, time_end, lat_max, lat_min, lon_min, lon_max):
    """Select spatial and temporal subset from a DataArray."""
    return da.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
        time=slice(time_start, time_end),
    )

# =============================================================================
# PROCESS VARIABLES
# =============================================================================

datasets = []

# --- 2m Temperature ---
if "t2m" in VARIABLES:
    print("Processing: 2m Temperature (t2m)...")
    t2 = select_region(ds["t2"], TIME_START, TIME_END,
                       lat_max_era5, lat_min_era5, lon_min_era5, lon_max_era5)

    # Compute daily mean and convert K to °C
    t2m_daily = (t2.resample(time="1D").mean() - 273.15).compute()

    t2m_ds = t2m_daily.to_dataset(name="t2m_daily_mean")
    t2m_ds["t2m_daily_mean"].attrs = {
        "units": "°C",
        "long_name": "Daily Mean 2m Temperature",
        "standard_name": "air_temperature",
    }
    datasets.append(t2m_ds)
    print("  ✓ Done")

# --- Surface Net Shortwave Radiation ---
if "ssr" in VARIABLES:
    print("Processing: Surface Net Shortwave Radiation (ssr)...")
    ssr = select_region(ds["ssr"], TIME_START, TIME_END,
                        lat_max_era5, lat_min_era5, lon_min_era5, lon_max_era5)

    # Compute daily mean
    ssr_daily = ssr.resample(time="1D").sum().compute()

    ssr_ds = ssr_daily.to_dataset(name="ssr_daily_sum")
    ssr_ds["ssr_daily_sum"].attrs = {
        "units": "J/m²",
        "long_name": "Daily Sum Surface Net Shortwave Radiation",
        "standard_name": "surface_net_downward_shortwave_flux",
    }
    datasets.append(ssr_ds)
    print("  ✓ Done")

# --- 10m U Wind Component ---
if "u10" in VARIABLES:
    print("Processing: 10m U Wind Component (u10)...")
    u10 = select_region(ds["u10"], TIME_START, TIME_END,
                        lat_max_era5, lat_min_era5, lon_min_era5, lon_max_era5)

    # Compute daily mean
    u10_daily = u10.resample(time="1D").mean().compute()

    u10_ds = u10_daily.to_dataset(name="u10_daily_mean")
    u10_ds["u10_daily_mean"].attrs = {
        "units": "m/s",
        "long_name": "Daily Mean 10m Eastward Wind Component",
        "standard_name": "eastward_wind",
    }
    datasets.append(u10_ds)
    print("  ✓ Done")

# --- 10m V Wind Component ---
if "v10" in VARIABLES:
    print("Processing: 10m V Wind Component (v10)...")
    v10 = select_region(ds["v10"], TIME_START, TIME_END,
                        lat_max_era5, lat_min_era5, lon_min_era5, lon_max_era5)

    # Compute daily mean
    v10_daily = v10.resample(time="1D").mean().compute()

    v10_ds = v10_daily.to_dataset(name="v10_daily_mean")
    v10_ds["v10_daily_mean"].attrs = {
        "units": "m/s",
        "long_name": "Daily Mean 10m Northward Wind Component",
        "standard_name": "northward_wind",
    }
    datasets.append(v10_ds)
    print("  ✓ Done")

# =============================================================================
# MERGE AND SAVE
# =============================================================================

print("\nMerging datasets...")
ds_out = xr.merge(datasets)

# Add time labels as auxiliary coordinate
date_strings = pd.to_datetime(ds_out.time.values).strftime("%Y-%m-%d")
ds_out = ds_out.assign_coords(time_label=("time", date_strings))

# Add global attributes
ds_out.attrs = {
    "source": "earthmover-public/era5-surface-aws via ArrayLake",
    "history": "Processed with daily mean resampling",
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

# Save output
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


print(f"\n{'='*60}")
print(f"Output saved successfully: {output_file}")
print(f"{'='*60}")
print(f"\nDataset summary:")
print(ds_out)

