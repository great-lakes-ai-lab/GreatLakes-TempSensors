# src/pipeline/data_loader.py
"""Load raw datasets from netcdf or zarr, with optional spatial subsetting."""

import xarray as xr
from pipeline.config import PipelineConfig, DataSourceEntry
from lakes import LAKE_BOUNDS


def load_raw_datasets(config: PipelineConfig) -> dict:
    """
    Load all datasets specified in config.data_sources.

    Returns a dict of {name: xr.Dataset}.
    Handles netcdf vs zarr formats and applies spatial bounding box
    for zarr files on HPC (where data covers all Great Lakes).
    """
    bbox = LAKE_BOUNDS.get(config.lake)
    datasets = {}

    for name, source in config.data_sources.items():
        ds = _load_single(source, bbox, clip_spatial=(config.environment == "hpc"))
        datasets[name] = ds

    return datasets


def _load_single(
    source: DataSourceEntry,
    bbox: dict,
    clip_spatial: bool = False,
) -> xr.Dataset:
    """Load one dataset from disk/zarr and optionally clip to bounding box."""

    if source.format == "netcdf":
        ds = xr.open_dataset(source.path)
    elif source.format == "zarr":
        ds = xr.open_zarr(source.path)
    else:
        raise ValueError(f"Unsupported format: {source.format}")

    # Optional variable rename (e.g., "z" -> "bathymetry")
    if source.variable and source.variable in ds.data_vars:
        # Check if there's a rename mapping needed
        # For now, just a marker — renaming handled in preprocessor
        pass

    # Spatial subsetting for HPC zarr stores covering all lakes
    if clip_spatial and bbox is not None:
        ds = _clip_to_bbox(ds, bbox)

    return ds


def _clip_to_bbox(ds: xr.Dataset, bbox: dict) -> xr.Dataset:
    """
    Clip dataset to lat/lon bounding box.

    bbox format: {"lat": [min, max], "lon": [min, max]}

    Handles descending lat/lon by checking coordinate order before slicing.
    """
    lat_name = _find_coord(ds, ["lat", "latitude", "y"])
    lon_name = _find_coord(ds, ["lon", "longitude", "x"])

    if lat_name is None or lon_name is None:
        return ds

    lat_min, lat_max = bbox["lat"]
    lon_min, lon_max = bbox["lon"]

    # Handle descending latitude (slice needs to match coordinate order)
    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    # Handle descending longitude (less common but possible)
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