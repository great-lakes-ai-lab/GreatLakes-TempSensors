# src/pipeline/data_loader.py
"""Load raw datasets from netcdf or zarr, with optional spatial subsetting."""

import xarray as xr
from pipeline.config import PipelineConfig, DataSourceEntry
from lakes import LAKE_BOUNDS


def load_raw_datasets(config: PipelineConfig) -> dict:
    """Load all datasets specified in config, clipped to lake bounding box."""
    bbox = LAKE_BOUNDS.get(config.lake)
    datasets = {}

    for name, source in config.data_sources.items():
        print(f"  Loading: {name}")
        ds = _load_single(source, bbox)
        datasets[name] = ds

    return datasets


def _load_single(source: DataSourceEntry, bbox: dict) -> xr.Dataset:
    """Load one dataset and clip to lake bounding box."""

    if source.format == "netcdf":
        ds = xr.open_dataset(source.path)
    elif source.format == "zarr":
        ds = xr.open_zarr(source.path)
    else:
        raise ValueError(f"Unsupported format: {source.format}")

    # Subset to requested variables
    if source.variables:
        available = [v for v in source.variables if v in ds.data_vars]
        missing = [v for v in source.variables if v not in ds.data_vars]
        if missing:
            print(f"  Warning: variables {missing} not found in {source.path}")
        ds = ds[available]

    # Always clip to lake bounding box
    if bbox is not None:
        ds = _clip_to_bbox(ds, bbox)

    return ds

def _clip_to_bbox(ds: xr.Dataset, bbox: dict) -> xr.Dataset:
    """
    Clip dataset to lat/lon bounding box.

    bbox format: {"lat": [min, max], "lon": [min, max]} (in -180 to 180 convention)

    Handles:
        - 0-360 longitude convention (converts to -180 to 180 first)
        - Descending lat/lon coordinate order
    """
    lat_name = _find_coord(ds, ["lat", "latitude", "y"])
    lon_name = _find_coord(ds, ["lon", "longitude", "x"])

    if lat_name is None or lon_name is None:
        return ds

    # Convert 0-360 longitude to -180 to 180 if needed
    lon_vals = ds[lon_name].values
    if float(lon_vals.max()) > 180:
        ds = ds.assign_coords({lon_name: ((ds[lon_name] + 180) % 360) - 180})
        ds = ds.sortby(lon_name)

    lat_min, lat_max = bbox["lat"]
    lon_min, lon_max = bbox["lon"]

    # Handle descending latitude
    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    # Handle descending longitude
    lon_vals = ds[lon_name].values
    if lon_vals[0] > lon_vals[-1]:
        lon_slice = slice(lon_max, lon_min)
    else:
        lon_slice = slice(lon_min, lon_max)

    ds = ds.sel({
        lat_name: lat_slice,
        lon_name: lon_slice,
    })

    return ds



def _find_coord(ds: xr.Dataset, candidates: list) -> str | None:
    """Find which coordinate name is present in the dataset."""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None