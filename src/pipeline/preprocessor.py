
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

from pipeline.config import PipelineConfig


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def preprocess_all(config: PipelineConfig, raw_datasets: dict) -> dict:
    """
    Full preprocessing pipeline. Returns a dict of processed products
    ready for task loading.

    Steps:
        1. Standardize coords and dates
        2. Apply per-source coarsening
        3. Compute anomalies (if requested)
        4. Extract lakemask for sampling (pre-DataProcessor)
        5. Fit DataProcessor (target first) and process all datasets
        6. Build temporal encodings + mask_time_ds
        7. Save processed outputs
        8. Return bundle dict

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
    dp_dir = Path(config.paths.data_processor_dir)

    # Check cache first
    if not config.preprocessing.force_reprocess and _cache_exists(processed_dir, dp_dir):
        print("Loading from processed cache...")
        return load_processed_cache(config)

    print("Running preprocessing pipeline...")

    # 1. Standardize coords and dates
    standardized = _standardize_all(raw_datasets, config)

    # 2. Apply per-source coarsening
    standardized = _coarsen_sources(standardized, config)

    # 3. Compute anomalies for any source with use_anomalies=True
    seasonal_processor = None
    for name, source in config.data_sources.items():
        if source.use_anomalies and name in standardized:
            seasonal_dir = Path(config.paths.seasonal_dir)
            anom_ds, seasonal_processor = _compute_anomalies(
                standardized[name], seasonal_dir
            )
            # Store anomalies alongside original
            anom_key = f"{name}_anom"
            standardized[anom_key] = anom_ds

    # 4. Extract lakemask for sampling (pre-DataProcessor, lat/lon coords)
    lakemask_sampling = _get_lakemask_sampling(standardized, config)

    # 5. Keep pre-DataProcessor versions for model.predict(X_t=...)
    pre_dp_datasets = {}
    for name, source in config.data_sources.items():
        roles = source.role if isinstance(source.role, list) else [source.role]
        if "target" in roles:
            pre_dp_datasets[f"{name}_stand"] = standardized[name].copy()
            anom_key = f"{name}_anom"
            if anom_key in standardized:
                pre_dp_datasets[f"{anom_key}_stand"] = standardized[anom_key].copy()

    # 6. Fit DataProcessor and process all datasets
    data_processor, processed = _fit_and_process(config, standardized)

    # 7. Build temporal features and mask_time_ds
    reference_ds = _get_reference_temporal_ds(processed, config)
    cosD, sinD = _make_time_features(reference_ds)
    mask_time_ds = _make_mask_time_ds(processed, config, cosD, sinD)

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
    bundle.update(processed)

    # Add pre-DataProcessor datasets
    bundle.update(pre_dp_datasets)

    # 9. Save
    _save_cache(config, bundle)

    return bundle


# -----------------------------------------------------------------------
# Standardization
# -----------------------------------------------------------------------

def _standardize_all(raw_datasets: dict, config: PipelineConfig) -> dict:
    """Standardize coords and dates for all raw datasets."""
    standardized = {}

    # Variable rename map (common raw names -> clean names)
    rename_map = {
        "z": "bathymetry",
        "Band1": "bathymetry",
    }

    for name, ds in raw_datasets.items():
        source_entry = config.data_sources[name]
        print(f'standardizing {name}')
        # Rename time if needed (e.g., "date" -> "time")
        if "date" in ds.dims or "date" in ds.coords:
            ds = ds.rename({"date": "time"})

        # Ensure time is datetime
        if "time" in ds.coords:
            ds = ds.assign_coords(time=pd.to_datetime(ds.time.values))

        # Drop CRS if present
        if "crs" in ds.data_vars:
            ds = ds.drop_vars("crs")

        # Rename variables based on hint
        if source_entry.variable and source_entry.variable in ds.data_vars:
            target_name = rename_map.get(source_entry.variable, source_entry.variable)
            if target_name != source_entry.variable:
                ds = ds.rename({source_entry.variable: target_name})

        # Standardize spatial coords (lat/lon naming, ascending, -180 to 180)
        ds = standardize_coords(ds)

        # Standardize time to date-only
        if "time" in ds.coords:
            ds = standardize_dates(ds)

        # Replace common sentinel values with NaN
        ds = ds.where(ds != -1, np.nan)
        ds = ds.where(ds != -99999, np.nan)

        standardized[name] = ds

    return standardized


# -----------------------------------------------------------------------
# Coarsening
# -----------------------------------------------------------------------

def _coarsen_sources(raw_datasets: dict, config: PipelineConfig) -> dict:
    """
    Apply per-source coarsening based on coarsen_factor in config.

    Only coarsens sources that have a coarsen_factor set.
    Returns the dict with coarsened versions replacing originals.
    """
    coarsened = {}

    for name, ds in raw_datasets.items():
        source = config.data_sources.get(name)

        # Skip derived datasets (like sst_anom) that aren't in data_sources
        if source is None:
            coarsened[name] = ds
            continue

        if source.coarsen_factor and source.coarsen_factor > 1:
            factor = source.coarsen_factor

            if "lat" in ds.dims and "lon" in ds.dims:
                ds_coarse = (
                    ds
                    .coarsen(lat=factor, lon=factor, boundary="trim")
                    .mean()
                    .compute()
                )

                # For mask: threshold back to binary
                roles = source.role if isinstance(source.role, list) else [source.role]
                if "mask" in roles:
                    for var in ds_coarse.data_vars:
                        ds_coarse[var] = xr.where(ds_coarse[var] >= 0.5, 1, 0)
                else:
                    # Fill NaN for non-mask static fields (land = 0)
                    ds_coarse = ds_coarse.fillna(0)

                print(f"  Coarsened {name}: factor={factor}, "
                      f"shape {dict(ds.sizes)} -> {dict(ds_coarse.sizes)}")
                coarsened[name] = ds_coarse
            else:
                print(f"  Warning: {name} has no lat/lon dims, skipping coarsen")
                coarsened[name] = ds
        else:
            coarsened[name] = ds

    return coarsened


# -----------------------------------------------------------------------
# Anomaly computation
# -----------------------------------------------------------------------

def _compute_anomalies(
    sst_stand: xr.Dataset,
    seasonal_dir: Optional[Path] = None,
) -> tuple:
    """Compute anomalies by removing monthly climatology."""
    seasonal_processor = SeasonalCycleProcessor()
    seasonal_processor.calculate(sst_stand)

    if seasonal_dir is not None:
        seasonal_dir = Path(seasonal_dir)
        seasonal_dir.mkdir(parents=True, exist_ok=True)
        seasonal_processor.save(str(seasonal_dir))

    sst_anom = seasonal_processor.compute_anomalies(sst_stand)

    # Rename variable for clarity (e.g., "sst" -> "sst_anom")
    for var in list(sst_anom.data_vars):
        if not var.endswith("_anom"):
            sst_anom = sst_anom.rename({var: f"{var}_anom"})

    return sst_anom, seasonal_processor


# -----------------------------------------------------------------------
# Lakemask extraction
# -----------------------------------------------------------------------

def _get_lakemask_sampling(standardized: dict, config: PipelineConfig) -> xr.Dataset:
    """
    Get the lakemask dataset for coordinate sampling.
    This is pre-DataProcessor (lat/lon coords, binary 0/1).
    """
    for name, source in config.data_sources.items():
        roles = source.role if isinstance(source.role, list) else [source.role]
        if "mask" in roles and name in standardized:
            return standardized[name].copy()

    raise ValueError("No data source with role='mask' found in config.")


# -----------------------------------------------------------------------
# DataProcessor fitting
# -----------------------------------------------------------------------

def _fit_and_process(config: PipelineConfig, standardized: dict) -> tuple:
    """
    Fit DataProcessor and process all datasets, ordered by role.

    Order:
        1. Target first (defines spatial normalization bounds)
        2. Target anomalies (if present)
        3. Temporal context datasets
        4. Static datasets (min_max)

    Returns (data_processor, processed_datasets_dict).
    """
    data_processor = DataProcessor(x1_name="lat", x2_name="lon")
    fit_start, fit_end = config.preprocessing.fit_range
    processed = {}

    # --- 1. Target first (defines spatial bounds) ---
    for name, source in config.data_sources.items():
        roles = source.role if isinstance(source.role, list) else [source.role]
        if "target" not in roles:
            continue

        ds = standardized[name]
        if "time" in ds.dims:
            _ = data_processor(ds.sel(time=slice(fit_start, fit_end)))
            processed[name] = data_processor(ds)

            # --- 2. Anomalies (if computed) ---
            anom_key = f"{name}_anom"
            if source.use_anomalies and anom_key in standardized:
                _ = data_processor(
                    standardized[anom_key].sel(time=slice(fit_start, fit_end))
                )
                processed[anom_key] = data_processor(standardized[anom_key])

    # --- 3. Temporal context datasets ---
    for name, source in config.data_sources.items():
        if name in processed:
            continue

        roles = source.role if isinstance(source.role, list) else [source.role]

        # Skip mask and pure aux_at_targets (they're static)
        if "context" not in roles and "target" not in roles:
        # if not any(r in roles for r in ["context", "target", "aux_at_targets"]):  # This was noticed in a new iteration. I will need to verify
            continue

        ds = standardized[name]
        if "time" in ds.dims:
            _ = data_processor(ds.sel(time=slice(fit_start, fit_end)))
            processed[name] = data_processor(ds)

    # --- 4. Static datasets (min_max) ---
    for name, source in config.data_sources.items():
        if name in processed:
            continue

        ds = standardized[name]
        if "time" not in ds.dims:
            _ = data_processor(ds, method="min_max")
            processed[name] = data_processor(ds, method="min_max")

    return data_processor, processed


# -----------------------------------------------------------------------
# Temporal features
# -----------------------------------------------------------------------

def _get_reference_temporal_ds(processed: dict, config: PipelineConfig) -> xr.Dataset:
    """Get a reference temporal dataset (target) for building time features."""
    for name, source in config.data_sources.items():
        roles = source.role if isinstance(source.role, list) else [source.role]
        if "target" in roles:
            # Prefer anomaly version if it exists
            anom_key = f"{name}_anom"
            if source.use_anomalies and anom_key in processed:
                return processed[anom_key]
            if name in processed:
                return processed[name]

    # Fallback: first temporal dataset
    for name, ds in processed.items():
        if hasattr(ds, "dims") and "time" in ds.dims:
            return ds

    raise ValueError("No temporal dataset found for building time features.")


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


def _make_mask_time_ds(
    processed: dict,
    config: PipelineConfig,
    cosD: xr.DataArray,
    sinD: xr.DataArray,
) -> xr.Dataset:
    """Build the mask + temporal encoding context dataset."""
    # Find the processed mask dataset
    mask_ds = None
    for name, source in config.data_sources.items():
        roles = source.role if isinstance(source.role, list) else [source.role]
        if "mask" in roles and name in processed:
            mask_ds = processed[name]
            break

    if mask_ds is None:
        raise ValueError("No processed mask dataset found for mask_time_ds.")

    # Get the mask variable (first data_var in the mask dataset)
    mask_var = list(mask_ds.data_vars)[0]

    mask_time_ds = xr.Dataset({
        "mask": mask_ds[mask_var],
        "cos_D": cosD,
        "sin_D": sinD,
    })

    return mask_time_ds


# -----------------------------------------------------------------------
# Cache save / load
# -----------------------------------------------------------------------

def _cache_exists(processed_dir: Path, dp_dir: Path) -> bool:
    """Check if processed cache has minimum required files."""
    if not processed_dir.exists():
        return False
    if not dp_dir.exists():
        return False

    meta_path = processed_dir / "metadata.json"
    return meta_path.exists()


def _clean_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Remove conflicting encoding attrs and problematic coordinates before saving."""
    ds = ds.copy()

    # Compute any lazy arrays (dask -> numpy)
    ds = ds.compute()

    # Drop problematic coordinates that aren't needed
    coords_to_drop = []
    for coord in ds.coords:
        if coord in ds.dims:
            continue  # keep dimension coordinates
        if ds[coord].dtype == object:
            coords_to_drop.append(coord)

    if coords_to_drop:
        ds = ds.drop_vars(coords_to_drop)

    # Clean encoding attrs on data variables
    for var in ds.data_vars:
        for attr in ["_FillValue", "missing_value"]:
            ds[var].attrs.pop(attr, None)
            ds[var].encoding.pop(attr, None)

    # Clean encoding on coordinate variables too
    for coord in ds.coords:
        for attr in ["_FillValue", "missing_value"]:
            if hasattr(ds[coord], "attrs"):
                ds[coord].attrs.pop(attr, None)
            if hasattr(ds[coord], "encoding"):
                ds[coord].encoding.pop(attr, None)

    return ds


def _save_cache(config: PipelineConfig, bundle: dict):
    """Save processed datasets and DataProcessor to disk."""
    processed_dir = Path(config.paths.processed_dir)
    dp_dir = Path(config.paths.data_processor_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    dp_dir.mkdir(parents=True, exist_ok=True)

    # Save DataProcessor
    bundle["data_processor"].save(str(dp_dir))

    # Save any xr.Dataset or xr.DataArray in the bundle
    saved_datasets = []
    for name, obj in bundle.items():
        try:
            if isinstance(obj, xr.DataArray):
                _clean_encoding(obj.to_dataset()).to_netcdf(processed_dir / f"{name}.nc")
                saved_datasets.append(name)
            elif isinstance(obj, xr.Dataset):
                _clean_encoding(obj).to_netcdf(processed_dir / f"{name}.nc")
                saved_datasets.append(name)
        except Exception as e:
            print(f"Error saving {name}: {e}")

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

    print(f"Processed cache saved to: {processed_dir}")


def load_processed_cache(config: PipelineConfig) -> dict:
    """Load previously saved processed bundle from disk."""
    processed_dir = Path(config.paths.processed_dir)
    dp_dir = Path(config.paths.data_processor_dir)
    seasonal_dir = Path(config.paths.seasonal_dir)

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
