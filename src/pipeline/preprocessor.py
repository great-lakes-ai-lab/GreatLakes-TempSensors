# src/pipeline/preprocessor.py
"""Standardize, compute anomalies, coarsen, fit DataProcessor, build bundle."""

from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
import xarray as xr

from deepsensor.data import DataProcessor, construct_circ_time_ds
from deepsensor_greatlakes.utils import standardize_dates, standardize_coords
from deepsensor_greatlakes.preprocessor import SeasonalCycleProcessor

from pipeline.config import PipelineConfig, DataSourceEntry


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def preprocess_all(config: PipelineConfig, raw_datasets: dict) -> dict:
    """
    Full preprocessing pipeline. Returns a dict of processed products
    ready for task loading.

    Steps:
        1. Standardize coords and dates
        2. Handle variable renaming / cleaning
        3. Compute anomalies
        4. Coarsen static datasets
        5. Fit or load DataProcessor
        6. Process all datasets through DataProcessor
        7. Build temporal encodings + mask_time_ds
        8. Save processed outputs
        9. Return processed bundle dict

    Parameters
    ----------
    config : PipelineConfig
    raw_datasets : dict
        Output of data_loader.load_raw_datasets()

    Returns
    -------
    dict with keys like 'sst', 'sst_anom', 'bathy', 'lakemask', 'data_processor', etc.
    """
    processed_dir = Path(config.paths.processed_dir)
    dp_dir = Path(config.paths.data_cache) / "deepsensor_config" / "data_processor"

    # Check cache first
    if not config.preprocessing.force_reprocess and _cache_exists(processed_dir, dp_dir):
        print("Loading from processed cache...")
        return load_processed_cache(config)

    print("Running preprocessing pipeline...")

    # 1. Standardize
    standardized = _standardize_all(raw_datasets, config)

    # 2. Compute lake mask from bathymetry
    lake_mask_stand = _derive_lake_mask(standardized["bathy"])

    # 3. Compute anomalies (if SST present)
    seasonal_processor = None
    sst_anom_stand = None
    if "sst" in standardized:
        seasonal_dir = Path(config.paths.data_cache) / "seasonal_cycles"
        sst_anom_stand, seasonal_processor = _compute_anomalies(
            standardized["sst"], seasonal_dir
        )

    # 4. Coarsen static datasets
    bathy_coarse, lake_mask_coarse = _coarsen_statics(
        standardized["bathy"],
        lake_mask_stand,
        bathy_factor=config.preprocessing.static_coarsen_factor,
        mask_factor=config.preprocessing.mask_coarsen_factor,
    )

    # Keep coarse mask for sampling (pre-DataProcessor, lat/lon coords)
    lakemask_sampling = lake_mask_coarse.copy()

    # 5. Fit DataProcessor
    data_processor, processed_datasets = _fit_and_process(
        config=config,
        standardized=standardized,
        sst_anom_stand=sst_anom_stand,
        bathy_coarse=bathy_coarse,
        lake_mask_coarse=lake_mask_coarse,
    )

    # 6. Build temporal features
    sst_ds = processed_datasets.get("sst") or processed_datasets.get("sst_anom")
    cosD, sinD = _make_time_features(sst_ds)

    # 7. Build mask + time context dataset
    mask_time_ds = xr.Dataset({
        "mask": processed_datasets["lakemask"]["mask"],
        "cos_D": cosD,
        "sin_D": sinD,
    })

    # 8. Assemble output bundle
    bundle = {
        "data_processor": data_processor,
        "lakemask_sampling": lakemask_sampling,
        "mask_time_ds": mask_time_ds,
        "cosD": cosD,
        "sinD": sinD,
        "seasonal_processor": seasonal_processor,
    }

    # Add all processed datasets
    bundle.update(processed_datasets)

    # Keep pre-DataProcessor versions for model.predict(X_t=...)
    bundle["sst_stand"] = standardized.get("sst")
    bundle["sst_anom_stand"] = sst_anom_stand

    # 9. Save
    _save_cache(config, bundle)

    return bundle


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _standardize_all(raw_datasets: dict, config: PipelineConfig) -> dict:
    """Standardize coords and dates for all raw datasets."""
    standardized = {}

    # Variable rename map (from config hints)
    rename_map = {
        "z": "bathymetry",
        "Band1": "bathymetry",
    }

    for name, ds in raw_datasets.items():
        source_entry = config.data_sources[name]

        # Rename time if needed (e.g., "date" -> "time")
        if "date" in ds.dims or "date" in ds.coords:
            ds = ds.rename({"date": "time"})

        # Ensure time is datetime
        if "time" in ds.coords:
            ds = ds.assign_coords(time=pd.to_datetime(ds.time.values))

        # Drop CRS if present
        if "crs" in ds.data_vars:
            ds = ds.drop_vars("crs")

        # Rename variables based on hints
        if source_entry.variable and source_entry.variable in ds.data_vars:
            target_name = rename_map.get(source_entry.variable, source_entry.variable)
            if target_name != source_entry.variable:
                ds = ds.rename({source_entry.variable: target_name})

        # Standardize spatial coords (lat/lon naming, ascending, -180 to 180)
        ds = standardize_coords(ds)

        # Standardize time to date-only
        if "time" in ds.coords:
            ds = standardize_dates(ds)

        # Replace sentinel values
        ds = ds.where(ds != -1, np.nan)
        ds = ds.where(ds != -99999, np.nan)

        standardized[name] = ds

    return standardized


def _derive_lake_mask(bathy_stand: xr.Dataset) -> xr.Dataset:
    """Derive binary lake mask from bathymetry (water where depth <= 0)."""
    bathy_var = list(bathy_stand.data_vars)[0]
    mask = xr.where(bathy_stand[bathy_var] <= 0, 1, 0)
    return mask.to_dataset(name="mask")


def _compute_anomalies(
    sst_stand: xr.Dataset,
    seasonal_dir: Optional[Path] = None,
) -> tuple:
    """Compute SST anomalies by removing monthly climatology."""
    seasonal_processor = SeasonalCycleProcessor()
    seasonal_processor.calculate(sst_stand)

    if seasonal_dir is not None:
        seasonal_dir = Path(seasonal_dir)
        seasonal_dir.mkdir(parents=True, exist_ok=True)
        seasonal_processor.save(str(seasonal_dir))

    sst_anom = seasonal_processor.compute_anomalies(sst_stand)

    # Rename variable for clarity
    if "sst" in sst_anom.data_vars:
        sst_anom = sst_anom.rename({"sst": "sst_anom"})

    return sst_anom, seasonal_processor


def _coarsen_statics(
    bathy_stand: xr.Dataset,
    lake_mask_stand: xr.Dataset,
    bathy_factor: int = 10,
    mask_factor: int = 20,
) -> tuple:
    """Coarsen bathymetry and lake mask at different resolutions."""
    bathy_coarse = (
        bathy_stand
        .coarsen(lat=bathy_factor, lon=bathy_factor, boundary="trim")
        .mean()
        .compute()
    )
    bathy_coarse = bathy_coarse.fillna(0)

    # Mask: fraction -> binary threshold
    lake_mask_frac = (
        lake_mask_stand
        .coarsen(lat=mask_factor, lon=mask_factor, boundary="trim")
        .mean()
        .compute()
    )
    lake_mask_binary = xr.where(lake_mask_frac >= 0.5, 1, 0)

    if isinstance(lake_mask_binary, xr.DataArray):
        lake_mask_binary = lake_mask_binary.to_dataset(name="mask")

    return bathy_coarse, lake_mask_binary


def _fit_and_process(
    config: PipelineConfig,
    standardized: dict,
    sst_anom_stand: Optional[xr.Dataset],
    bathy_coarse: xr.Dataset,
    lake_mask_coarse: xr.Dataset,
) -> tuple:
    """
    Fit DataProcessor on fit_range, then process all datasets.

    IMPORTANT: The target variable (sst) is processed first, since it
    defines the normalized spatial coordinate bounds for all subsequent datasets.

    Returns (data_processor, processed_datasets_dict).
    """
    data_processor = DataProcessor(x1_name="lat", x2_name="lon")
    fit_start, fit_end = config.preprocessing.fit_range
    processed = {}

    # --- 1. Target variable FIRST (defines spatial normalization) ---
    target_name = "sst"  # Could make this configurable later
    if target_name in standardized and "time" in standardized[target_name].dims:
        ds = standardized[target_name]
        _ = data_processor(ds.sel(time=slice(fit_start, fit_end)))
        processed[target_name] = data_processor(ds)

    # --- 2. SST anomalies (derived from target, same spatial grid) ---
    if sst_anom_stand is not None:
        _ = data_processor(sst_anom_stand.sel(time=slice(fit_start, fit_end)))
        processed["sst_anom"] = data_processor(sst_anom_stand)

    # --- 3. Remaining temporal datasets ---
    temporal_names = [
        name for name, ds in standardized.items()
        if "time" in ds.dims and name != target_name
    ]

    for name in temporal_names:
        ds = standardized[name]
        _ = data_processor(ds.sel(time=slice(fit_start, fit_end)))
        processed[name] = data_processor(ds)

    # --- 4. Static datasets last (min_max) ---
    bathy, lakemask = data_processor(
        [bathy_coarse, lake_mask_coarse],
        method="min_max",
    )
    processed["bathy"] = bathy
    processed["lakemask"] = lakemask

    return data_processor, processed

def _make_time_features(reference_ds: xr.Dataset) -> tuple:
    """Build circular day-of-year features matching a reference dataset's time range."""
    dates = pd.date_range(
        pd.to_datetime(reference_ds.time.values.min()),
        pd.to_datetime(reference_ds.time.values.max()),
        freq="D",
    )
    doy_ds = construct_circ_time_ds(dates, freq="D")
    cosD = standardize_dates(doy_ds["cos_D"])
    sinD = standardize_dates(doy_ds["sin_D"])
    return cosD, sinD


# -----------------------------------------------------------------------
# Cache save / load
# -----------------------------------------------------------------------

def _cache_exists(processed_dir: Path, dp_dir: Path) -> bool:
    """Check if processed cache has minimum required files."""
    required = [
        processed_dir / "sst.nc",
        processed_dir / "lakemask.nc",
        processed_dir / "bathy.nc",
        processed_dir / "mask_time_ds.nc",
        processed_dir / "lakemask_sampling.nc",
        dp_dir,
    ]
    return all(p.exists() for p in required)


def _clean_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Remove conflicting encoding attrs before saving."""
    for var in ds.data_vars:
        for attr in ["_FillValue", "missing_value"]:
            ds[var].attrs.pop(attr, None)
            ds[var].encoding.pop(attr, None)
    return ds


def _save_cache(config: PipelineConfig, bundle: dict):
    """Save processed datasets and DataProcessor to disk."""
    processed_dir = Path(config.paths.processed_dir)
    dp_dir = Path(config.paths.data_cache) / "deepsensor_config" / "data_processor"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dp_dir.mkdir(parents=True, exist_ok=True)

    # Save DataProcessor
    bundle["data_processor"].save(str(dp_dir))

    # Save any xr.Dataset or xr.DataArray in the bundle
    saved_datasets = []
    for name, obj in bundle.items():
        if isinstance(obj, xr.DataArray):
            _clean_encoding(obj.to_dataset()).to_netcdf(processed_dir / f"{name}.nc")
            saved_datasets.append(name)
        elif isinstance(obj, xr.Dataset):
            _clean_encoding(obj).to_netcdf(processed_dir / f"{name}.nc")
            saved_datasets.append(name)

    # Metadata
    metadata = {
        "lake": config.lake,
        "environment": config.environment,
        "fit_range": list(config.preprocessing.fit_range),
        "saved_datasets": saved_datasets,
        "saved_at": str(pd.Timestamp.now()),
    }
    with open(processed_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def load_processed_cache(config: PipelineConfig) -> dict:
    """Load previously saved processed bundle from disk."""
    processed_dir = Path(config.paths.processed_dir)
    dp_dir = Path(config.paths.data_cache) / "deepsensor_config" / "data_processor"
    seasonal_dir = Path(config.paths.data_cache) / "seasonal_cycles"

    data_processor = DataProcessor(str(dp_dir))
    bundle = {"data_processor": data_processor}

    # Load metadata to know what was saved
    meta_path = processed_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        dataset_names = metadata.get("saved_datasets", [])
    else:
        # Fallback: load whatever .nc files exist
        dataset_names = [p.stem for p in processed_dir.glob("*.nc")]

    for name in dataset_names:
        path = processed_dir / f"{name}.nc"
        if path.exists():
            bundle[name] = xr.open_dataset(path)

    # Extract temporal features from mask_time_ds
    if "mask_time_ds" in bundle:
        bundle["cosD"] = bundle["mask_time_ds"]["cos_D"]
        bundle["sinD"] = bundle["mask_time_ds"]["sin_D"]

    # Load seasonal processor if available
    bundle["seasonal_processor"] = None
    if seasonal_dir.exists():
        nc_files = list(seasonal_dir.glob("*_seasonal_cycle.nc"))
        meta_files = list(seasonal_dir.glob("*_metadata.json"))
        if nc_files and meta_files:
            bundle["seasonal_processor"] = SeasonalCycleProcessor.load(
                str(nc_files[-1]), str(meta_files[-1])
            )

    return bundle