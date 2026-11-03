# src/pipeline/active_learning.py
"""Active learning / sensor placement for Great Lakes DeepSensor pipeline."""

from pathlib import Path
from typing import Union
import math

import numpy as np
import pandas as pd
import xarray as xr
from deepsensor.active_learning import AcquisitionFunction, AcquisitionFunctionParallel

from utils.dates import dates_from_intervals
from pipeline.model import load_trained_model
from pipeline.task_builder import gen_tasks
from pipeline.plotting import _finish_plot


def run_active_learning(config, bundle, tl_config):
    """
    Run DeepSensor greedy active learning to recommend new buoy locations.
    """
    from deepsensor.active_learning.algorithms import GreedyAlgorithm

    al_cfg = config.active_learning

    # Resolve output directory using AL experiment name
    output_dir = config.paths.resolve_active_learning(al_cfg.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the active learning config for this experiment
    _save_al_config(config, output_dir)

    print("\n=== Active Learning ===")
    print(f"Experiment name: {al_cfg.name}")
    print(f"Output directory: {output_dir}")
    print(f"Acquisition function: {al_cfg.acquisition_function}")
    print(f"New sensors requested: {al_cfg.n_new_sensors}")

    # 1. Load trained model
    model = load_trained_model(config, bundle, tl_config.task_loader)

    # 2. Resolve context points
    fixed_context_points = resolve_context_points(config, bundle)

    # Save context points if random (for reproducibility/visualization)
    if al_cfg.save_context_points:
        save_context_points(fixed_context_points, bundle, config, output_dir)

    # 3. Build active learning dates
    al_dates = make_active_learning_dates(config)
    print(f"Active learning eval dates: {len(al_dates)}")

    # 4. Generate tasks with fixed context
    tasks = gen_tasks(
        tl_config,
        al_dates,
        bundle,
        config,
        n_context=al_cfg.n_context,
        vary_n_context=False,
        seed=al_cfg.context_seed,
        fixed_context_points=fixed_context_points,
        progress=True,
    )

    if len(tasks) == 0:
        raise ValueError("No active learning tasks were generated.")

    print(f"Generated {len(tasks)} active learning tasks")

    # 5. Build acquisition function
    acquisition_fn = build_acquisition_function(
        al_cfg.acquisition_function,
        model,
        p=al_cfg.acquisition_fn_p,
        seed=al_cfg.acquisition_fn_seed,
        context_set_idx=al_cfg.context_set_idx,
        target_set_idx=al_cfg.target_set_idx,
    )

    effective_diff = al_cfg.diff
    if isinstance(acquisition_fn, AcquisitionFunctionParallel):
        if al_cfg.diff:
            import warnings
            warnings.warn(
                f"diff=True has no effect for parallel acquisition function "
                f"'{al_cfg.acquisition_function}' (parallel functions score all "
                f"candidates in one pass with no per-point baseline). "
                f"Proceeding with diff=False.",
                UserWarning,
                stacklevel=2,
            )
        effective_diff = False


    # 6. Build grids and masks
    grids = build_grids_and_masks(bundle, config, acquisition_fn)

    search_grid = grids["search_grid"]
    search_mask = grids["search_mask"]
    target_grid = grids["target_grid"]
    target_mask = grids["target_mask"]
    target_mask_norm = grids["target_mask_norm"]

    print(f"  Search grid shape: {search_grid.shape}")
    print(f"  Target grid shape: {target_grid.shape}")

    # Save the search mask to get used in skill_curve.py
    search_mask_path = output_dir / "search_mask.nc"
    search_mask.to_netcdf(search_mask_path)

    target_mask_path = output_dir / "target_mask.nc"
    target_mask.to_netcdf(target_mask_path)

    target_mask_norm_path = output_dir / "target_mask_norm.nc"
    target_mask_norm.to_netcdf(target_mask_norm_path)

    # TDOD: Save out a version of target mask that is regridded to the target variable size (eg. bundle['sst_anom']) then put through the
    #   data processor to be in normalized data coords. Then it can get picked up in compute weighted scores via skill curve to mask only
    #   to the areas identified in the target mask (Optional)

    n_valid_candidates = int(search_mask.sum())
    print(f"Search grid: {n_valid_candidates} cells to be evaluated")
    if isinstance(acquisition_fn, AcquisitionFunctionParallel):
        print(f"Will compute {al_cfg.n_new_sensors * len(al_dates)} tasks")
    else:
        print(f"Will compute {al_cfg.n_new_sensors * len(al_dates) * n_valid_candidates} tasks")
    assert al_cfg.n_new_sensors < n_valid_candidates, (
        f"n_new_sensors ({al_cfg.n_new_sensors}) >= valid masked candidates "
        f"({n_valid_candidates}). Reduce n_new_sensors or use a finer search grid."
    )

    # 8. Run GreedyAlgorithm
    greedy = GreedyAlgorithm(
        model=model,
        X_s=search_grid,
        X_t=target_grid,
        X_s_mask=search_mask,
        X_t_mask=target_mask,
        N_new_context=al_cfg.n_new_sensors,
        X_normalised=False,
        model_infill_method=al_cfg.model_infill_method,
        context_set_idx=al_cfg.context_set_idx,
        target_set_idx=al_cfg.target_set_idx,
        progress_bar=al_cfg.progress_bar,
        task_loader=tl_config.task_loader,
        verbose=True,
    )

    X_new_df, acquisition_fn_ds = greedy(
        acquisition_fn,
        tasks,
        diff=effective_diff,
    )

    # 9. Post-processing: enforce minimum distance. DISCOURAGED FROM USING THIS. IT BREAKS THE SEQUENTIAL
    # SLATED TO MOVE OUT. N
    # if al_cfg.min_dist_from_existing_km > 0:
    #     X_new_df = enforce_min_distance(
    #         X_new_df,
    #         min_dist_km=al_cfg.min_dist_from_existing_km,
    #         model=model,
    #     )

    # 10. Save outputs
    save_recommended_locations(X_new_df, model, output_dir, config)

    if al_cfg.save_acquisition_surface:
        nc_path = output_dir / "acquisition_surfaces.nc"
        nc_mean_path = output_dir / "acquisition_surface_mean.nc"
        try:
            acquisition_fn_ds.to_netcdf(nc_path)
            print(f"Saved acquisition surfaces: {nc_path}")
        except Exception as e:
            print(f"Warning: Could not save acquisition surface as NetCDF: {e}")
            np.save(output_dir / "acquisition_surfaces.npy", acquisition_fn_ds.values)
        try:
            # Take the mean of the acquistion surface and save it 
            mean_surface = acquisition_fn_ds.mean(["time"],skipna=True)
            mean_surface.to_netcdf(nc_mean_path)
            print(f"Saved acquisition surfaces: {nc_mean_path}")
        except Exception as e:
            print(f"Warning: Could not save acquisition surface as NetCDF: {e}")
            pass

    if al_cfg.plot_results:
        overlays = build_plot_overlays(
            context_latlon=context_points_to_latlon(fixed_context_points, bundle),
            search_exclusion_points=grids.get("search_exclusion_points"),
        )
        plot_active_learning_results(
            X_new_df,
            acquisition_fn_ds,
            config,
            output_dir,
            acquisition_name=al_cfg.acquisition_function,
            x1_name=get_x1_name(model),
            x2_name=get_x2_name(model),
            overlays=overlays,
            search_mask=search_mask,
        )

    print("\nActive learning complete.")
    print(X_new_df)

    return {
        "recommended_locations": X_new_df,
        "acquisition_surfaces": acquisition_fn_ds,
    }

def _save_al_config(config, output_dir: Path):
    """
    Save the active learning configuration for this experiment.

    Saves both:
      - The full pipeline config (for reference to which model was used)
      - The active learning section specifically
    """
    import json
    from dataclasses import asdict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    # Archive the verbatim AL YAML alongside the parsed al_config.json
    src_path = getattr(config, "_al_config_source_path", None)
    if src_path:
        try:
            shutil.copy2(src_path, output_dir / "al_config_used.yaml")
            print(f"  AL config (verbatim) archived: {output_dir / 'al_config_used.yaml'}")
        except shutil.SameFileError:
            pass

    # Save AL-specific config as JSON
    al_cfg = config.active_learning
    al_dict = {
        "name": al_cfg.name,
        "acquisition_function": al_cfg.acquisition_function,
        "acquisition_fn_p": al_cfg.acquisition_fn_p,
        "acquisition_fn_seed": al_cfg.acquisition_fn_seed,
        "eval_range": list(al_cfg.eval_range),
        "eval_subsample_factor": al_cfg.eval_subsample_factor,
        "n_new_sensors": al_cfg.n_new_sensors,
        "context_source": al_cfg.context_source,
        "n_context": al_cfg.n_context,
        "context_seed": al_cfg.context_seed,
        "context_geojson_path": al_cfg.context_geojson_path,
        "context_set_idx": al_cfg.context_set_idx,
        "target_set_idx": al_cfg.target_set_idx,
        "model_infill_method": al_cfg.model_infill_method,
        "diff": al_cfg.diff,
        "candidate_coarsen_factor": al_cfg.candidate_coarsen_factor,
        "target_coarsen_factor": al_cfg.target_coarsen_factor,
        "min_dist_from_existing_km_search": al_cfg.min_dist_from_existing_km_search,
        "exclusion_points_path_search": al_cfg.exclusion_points_path_search,
        "min_dist_from_exclusion_km_search": al_cfg.min_dist_from_exclusion_km_search,
        "min_shore_dist_search": al_cfg.min_shore_dist_search,
        "min_depth_search": al_cfg.min_depth_search,
        "min_dist_from_existing_km_target": al_cfg.min_dist_from_existing_km_target,
        "exclusion_points_path_target": al_cfg.exclusion_points_path_target,
        "min_dist_from_exclusion_km_target": al_cfg.min_dist_from_exclusion_km_target,
        "min_shore_dist_target": al_cfg.min_shore_dist_target,
        "min_depth_target": al_cfg.min_depth_target,
        "run_name": config.run.name,
        "lake": config.lake,
        "model_dir": config.paths.model_dir,
    }

    config_path = output_dir / "al_config.json"
    with open(config_path, "w") as f:
        json.dump(al_dict, f, indent=4)

    print(f"  AL config saved: {config_path}")

# ---------------------------------------------------------------------
# Coordinate I/O utilities
# ---------------------------------------------------------------------

def load_points_from_geojson(path) -> tuple:
    """
    Load Point features from a GeoJSON file.

    Validates WGS84 coordinate ranges.

    Parameters
    ----------
    path : str or Path
        Path to GeoJSON file with Point features.

    Returns
    -------
    tuple (lats, lons) as numpy arrays
    """
    import json

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {path}")

    with open(path) as f:
        geojson = json.load(f)

    ## Since we're not using geopandas the crs isn't readily available. This should be
    ## Caught on the bounds check later
    # # Check CRS if specified
    # crs = geojson.get("crs", None)
    # if crs is not None:
    #     crs_name = str(crs.get("properties", {}).get("name", "")).lower()
    #     if crs_name and "4326" not in crs_name and "wgs84" not in crs_name:
    #         raise ValueError(
    #             f"GeoJSON CRS appears to be '{crs_name}', not WGS84/EPSG:4326. "
    #             f"Please reproject to WGS84 before use."
    #         )

    features = geojson.get("features", [])
    if not features:
        raise ValueError(f"No features found in GeoJSON: {path}")

    lats = []
    lons = []
    skipped = 0

    for feature in features:
        geom = feature.get("geometry", {})
        if geom.get("type") != "Point":
            skipped += 1
            continue

        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            skipped += 1
            continue

        # GeoJSON coordinate order is [longitude, latitude]
        lons.append(float(coords[0]))
        lats.append(float(coords[1]))

    if not lats:
        raise ValueError(
            f"No valid Point features found in GeoJSON: {path}. "
            f"Skipped {skipped} non-Point features."
        )

    if skipped > 0:
        print(f"  Warning: Skipped {skipped} non-Point features in GeoJSON")

    lats = np.array(lats)
    lons = np.array(lons)

    _validate_wgs84_arrays(lats, lons, str(path))

    return lats, lons


def load_points_from_csv(path) -> tuple:
    """
    Load points from a CSV with lat/lon columns.

    Validates WGS84 coordinate ranges.

    Parameters
    ----------
    path : str or Path
        Path to CSV file with 'lat' and 'lon' columns.

    Returns
    -------
    tuple (lats, lons) as numpy arrays
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns 'lat' and 'lon'. "
            f"Found columns: {list(df.columns)}"
        )

    lats = df["lat"].astype(float).values
    lons = df["lon"].astype(float).values

    _validate_wgs84_arrays(lats, lons, str(path))

    print(f"  Loaded {len(lats)} points from CSV: {path}")
    return lats, lons


def save_points_to_geojson(
    lats: np.ndarray,
    lons: np.ndarray,
    path,
    properties: dict = None,
):
    """
    Save points as a GeoJSON FeatureCollection.

    Parameters
    ----------
    lats, lons : np.ndarray
        Coordinate arrays in WGS84.
    path : str or Path
        Output path.
    properties : dict, optional
        Additional properties to attach to each feature.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    features = []
    for i in range(len(lats)):
        props = {"id": i}
        if properties:
            props.update(properties)

        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Point",
                "coordinates": [float(lons[i]), float(lats[i])],
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"  Saved {len(lats)} points to GeoJSON: {path}")


def _validate_wgs84_arrays(lats: np.ndarray, lons: np.ndarray, source: str):
    """
    Validate that coordinates are in WGS84 range.

    Raises on invalid coordinates, warns if outside Great Lakes bounds.
    """
    # Hard validation: valid WGS84 ranges
    if np.any(lats < -90) or np.any(lats > 90):
        bad_idx = np.where((lats < -90) | (lats > 90))[0][0]
        raise ValueError(
            f"Point {bad_idx} in '{source}' has latitude={lats[bad_idx]:.4f}, "
            f"outside valid WGS84 range [-90, 90]. "
            f"Check coordinate order (GeoJSON uses [lon, lat])."
        )

    if np.any(lons < -180) or np.any(lons > 180):
        bad_idx = np.where((lons < -180) | (lons > 180))[0][0]
        raise ValueError(
            f"Point {bad_idx} in '{source}' has longitude={lons[bad_idx]:.4f}, "
            f"outside valid WGS84 range [-180, 180]. "
            f"Data may not be in WGS84."
        )

    # Soft validation: Great Lakes bounding box
    GL_LAT_MIN, GL_LAT_MAX = 40.5, 49.5
    GL_LON_MIN, GL_LON_MAX = -92.5, -75.5

    outside = (
        (lats < GL_LAT_MIN) | (lats > GL_LAT_MAX) |
        (lons < GL_LON_MIN) | (lons > GL_LON_MAX)
    )

    if np.any(outside):
        n_outside = int(outside.sum())
        import warnings
        warnings.warn(
            f"{n_outside} point(s) in '{source}' fall outside the Great Lakes "
            f"bounding box (lat: [{GL_LAT_MIN}, {GL_LAT_MAX}], "
            f"lon: [{GL_LON_MIN}, {GL_LON_MAX}]).",
            UserWarning,
            stacklevel=3,
        )


def save_recommended_locations(X_new_df, model, output_dir, config):
    """
    Save recommended sensor locations as CSV and GeoJSON.

    Parameters
    ----------
    X_new_df : pd.DataFrame
        DataFrame with columns for x1/x2 (lat/lon) coordinates.
        Index represents priority (0 = highest priority).
    model : ConvNP
        Used to get coordinate names from data_processor.
    output_dir : Path
        Directory to save outputs.
    config : PipelineConfig
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x1_name = get_x1_name(model)
    x2_name = get_x2_name(model)

    lats = X_new_df[x1_name].values
    lons = X_new_df[x2_name].values

    # Save CSV
    csv_path = output_dir / "recommended_sensor_locations.csv"
    csv_df = pd.DataFrame({
        "priority": range(1, len(lats) + 1),
        "lat": lats,
        "lon": lons,
    })
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved recommended locations CSV: {csv_path}")

    # Save GeoJSON with priority field
    geojson_path = output_dir / "recommended_sensor_locations.geojson"

    features = []
    for i in range(len(lats)):
        features.append({
            "type": "Feature",
            "properties": {
                "priority": int(i + 1),
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "lake": config.lake,
                "acquisition_function": config.active_learning.acquisition_function,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(lons[i]), float(lats[i])],
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "properties": {
            "description": "Recommended sensor locations from DeepSensor active learning",
            "lake": config.lake,
            "run_name": config.run.name,
            "acquisition_function": config.active_learning.acquisition_function,
            "n_sensors": len(lats),
        },
        "features": features,
    }

    import json
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"  Saved recommended locations GeoJSON: {geojson_path}")

    return csv_path, geojson_path

# ---------------------------------------------------------------------
# Context point resolution
# ---------------------------------------------------------------------

def resolve_context_points(config, bundle) -> np.ndarray:
    """
    Resolve fixed context points for active learning based on config.

    Returns
    -------
    np.ndarray of shape (2, N) in normalized coordinates [x1, x2].
    """
    al_cfg = config.active_learning
    source = al_cfg.context_source.lower().strip()

    if source == "random":
        points = _generate_random_context(al_cfg, bundle)
    elif source == "geojson":
        points = _load_geojson_context(al_cfg, bundle)
    elif source == "buoy":
        raise NotImplementedError(
            "Buoy-based context points not yet implemented. "
            "Use 'random' or 'geojson' for now."
        )
    else:
        raise ValueError(
            f"Unknown context_source='{source}'. "
            f"Options: 'random', 'geojson', 'buoy'"
        )

    print(f"  Context source: {source}")
    print(f"  Number of context points: {points.shape[1]}")

    return points


def _generate_random_context(al_cfg, bundle) -> np.ndarray:
    """Generate random lake points and optionally save them."""
    from utils.coordinates import generate_random_coordinates

    points = generate_random_coordinates(
        bundle["lakemask_sampling"],
        N=al_cfg.n_context,
        data_processor=bundle["data_processor"],
        rng=al_cfg.context_seed
    )

    return points


# def _load_geojson_context(al_cfg, bundle) -> np.ndarray:
#     """
#     Load context points from a GeoJSON file and normalize to model coordinates.
#
#     Expects Point features in WGS84.
#     """
#     lats, lons = load_points_from_geojson(al_cfg.context_geojson_path)
#
#     # Stack as (2, N) in raw coordinates [lat, lon]
#     raw_points = np.array([lats, lons])
#
#     # Normalize to model coordinate space
#     data_processor = bundle["data_processor"]
#     normalized_points = data_processor.map_coord_array(raw_points, unnorm=False)
#
#     return normalized_points

def _load_geojson_context(al_cfg, bundle) -> np.ndarray:
    """Backwards-compatible wrapper: pulls the path off an active-learning config."""
    return load_context_points(al_cfg.context_geojson_path, bundle)


def load_context_points(geojson_path, bundle) -> np.ndarray:
    """
    Load context points from a GeoJSON file and normalize to model coordinates.

    Expects Point features in WGS84.

    Parameters
    ----------
    geojson_path : str or Path
    bundle : dict
        Processed bundle (needs 'data_processor')

    Returns
    -------
    np.ndarray, shape (2, N)
        Row 0 = x1 (lat), row 1 = x2 (lon), normalized.
    """
    lats, lons = load_points_from_geojson(geojson_path)
    raw_points = np.array([lats, lons])
    return bundle["data_processor"].map_coord_array(raw_points, unnorm=False)


def save_context_points(
    context_points: np.ndarray,
    bundle: dict,
    config,
    output_dir: Path,
):
    """
    Save context points as GeoJSON and plot.

    Parameters
    ----------
    context_points : np.ndarray
        Shape (2, N) in normalized coordinates.
    bundle : dict
    config : PipelineConfig
    output_dir : Path
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unnormalize to lat/lon
    lats, lons = context_points_to_latlon(context_points, bundle)

    # Save as GeoJSON
    geojson_path = output_dir / "context_points.geojson"
    save_points_to_geojson(
        lats, lons, geojson_path,
        properties={"source": config.active_learning.context_source},
    )

    # Plot context points on lake mask
    fig, ax = plt.subplots(figsize=(8, 6))

    if "lakemask_sampling" in bundle:
        mask_ds = bundle["lakemask_sampling"]
        mask_var = list(mask_ds.data_vars)[0]
        mask_ds[mask_var].plot(
            ax=ax, cmap="Blues", alpha=0.3,
            add_colorbar=False,
        )

    ax.scatter(
        lons, lats,
        facecolors="none",
        edgecolors="red",
        linewidths=1.2,
        s=40,
        zorder=5,
        label=f"Context points (n={len(lats)})",
    )

    ax.set_title(
        f"Active Learning Context Points\n"
        f"{config.lake.upper()} — Source: {config.active_learning.context_source}"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best")

    save_path = output_dir / "context_points_map.png"
    _finish_plot(config, save_path)

    return np.array([lats, lons])


def context_points_to_latlon(context_points: np.ndarray, bundle: dict) -> tuple:
    """
    Convert normalized (2, N) context points to (lats, lons) in WGS84.

    Parameters
    ----------
    context_points : np.ndarray
        Shape (2, N); row 0 = x1 (normalized lat), row 1 = x2 (normalized lon).
    bundle : dict
        Processed bundle (needs 'data_processor').

    Returns
    -------
    tuple (lats, lons) as float numpy arrays.
    """
    pts = bundle["data_processor"].map_coord_array(context_points, unnorm=True)
    return np.asarray(pts[0], dtype=float), np.asarray(pts[1], dtype=float)

# ---------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------

def make_active_learning_dates(config):
    al_cfg = config.active_learning
    return dates_from_intervals(al_cfg.eval_range, al_cfg.eval_subsample_factor, months_to_drop=al_cfg.al_months_drop)


# ---------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------

def build_acquisition_function(name: str, model, **kwargs):
    """
    Instantiate a DeepSensor acquisition function from config name.

    Parameters
    ----------
    name : str
        Name of the acquisition function (case-insensitive, underscores/hyphens flexible).
    model : DeepSensor model
        The trained model to use for predictions.
    **kwargs : dict
        Additional keyword arguments passed to specific acquisition functions:
        - p : float (for pNormStddev, default=1)
        - seed : int (for Random, default=42)
        - context_set_idx : int (default=0)
        - target_set_idx : int (default=0)

    Returns
    -------
    AcquisitionFunction
        Instantiated acquisition function.

    Notes
    -----
    Sequential (non-parallel) acquisition functions evaluate the model over
    the full target set for each candidate point hypothetically added as context.
    These are more principled but slower:
        - MeanStddev, MeanVariance, MeanMarginalEntropy, JointEntropy,
          pNormStddev, OracleRMSE, OracleMAE, OracleMarginalNLL, OracleJointNLL

    Parallel acquisition functions score all candidate points in one forward pass.
    These are faster but less fully greedy/informative:
        - Stddev, ContextDist, ExpectedImprovement, Random
    """
    from deepsensor.active_learning.acquisition_fns import (
        AcquisitionFunctionParallel,
        AcquisitionFunctionOracle,
        MeanStddev,
        Stddev,
        MeanVariance,
        MeanMarginalEntropy,
        JointEntropy,
        pNormStddev,
        ContextDist,
        ExpectedImprovement,
        Random,
        OracleRMSE,
        OracleMAE,
        OracleMarginalNLL,
        OracleJointNLL,
    )

    # Extract shared parameters with defaults
    context_set_idx = kwargs.pop("context_set_idx", 0)
    target_set_idx = kwargs.pop("target_set_idx", 0)

    # Extract function-specific parameters
    p = kwargs.pop("p", 1)
    seed = kwargs.pop("seed", 42)

    # Normalize name: lowercase, replace hyphens with underscores
    name_clean = name.lower().strip().replace("-", "_")

    ACQUISITION_MAPPING = {
        # Sequential
        "mean_stddev": MeanStddev,
        "mean_std": MeanStddev,
        "mean_variance": MeanVariance,
        "mean_var": MeanVariance,
        "mean_marginal_entropy": MeanMarginalEntropy,
        "marginal_entropy": MeanMarginalEntropy,
        "joint_entropy": JointEntropy,
        "p_norm_stddev": pNormStddev,
        "pnorm_stddev": pNormStddev,
        "pnorm": pNormStddev,
        "p_norm": pNormStddev,
        # Oracle (sequential; require true target values)
        "oracle_rmse": OracleRMSE,
        "oracle_mae": OracleMAE,
        "oracle_marginal_nll": OracleMarginalNLL,
        "oracle_joint_nll": OracleJointNLL,
        # Parallel
        "stddev": Stddev,
        "std": Stddev,
        "context_dist": ContextDist,
        "context_distance": ContextDist,
        "expected_improvement": ExpectedImprovement,
        "ei": ExpectedImprovement,
        "random": Random,
    }

    if name_clean not in ACQUISITION_MAPPING:
        available = sorted(ACQUISITION_MAPPING.keys())
        raise ValueError(
            f"Unsupported acquisition_function='{name}'.\n"
            f"  Available (incl. aliases): {available}"
        )

    cls = ACQUISITION_MAPPING[name_clean]

    # ──────────────────────────────────────────────────────────────────────
    # Instantiate with appropriate arguments
    # ──────────────────────────────────────────────────────────────────────
    shared_kwargs = dict(
        model=model,
        context_set_idx=context_set_idx,
        target_set_idx=target_set_idx,
    )

    if cls is pNormStddev:
        acq_fn = cls(**shared_kwargs, p=p)
        print(f"Using acquisition function: {cls.__name__} (p={p})")
    elif cls is Random:
        acq_fn = cls(**shared_kwargs, seed=seed)
        print(f"Using acquisition function: {cls.__name__} (seed={seed})")
    else:
        acq_fn = cls(**shared_kwargs)
        print(f"Using acquisition function: {cls.__name__}")

    # ──────────────────────────────────────────────────────────────────────
    # Derive classification structurally (authoritative)
    # ──────────────────────────────────────────────────────────────────────
    is_parallel = isinstance(acq_fn, AcquisitionFunctionParallel)
    is_oracle = isinstance(acq_fn, AcquisitionFunctionOracle)

    if is_oracle:
        import warnings
        warnings.warn(
            f"Oracle acquisition function '{cls.__name__}' requires true target values. "
            f"This is intended for retrospective evaluation, not operational deployment.",
            UserWarning,
            stacklevel=2,
        )

    parallel_str = "parallel" if is_parallel else "sequential (non-parallel)"
    print(f"  Type: {parallel_str}")
    print(f"  Optimization direction: {acq_fn.min_or_max}")

    if kwargs:
        import warnings
        warnings.warn(
            f"Unused keyword arguments passed to build_acquisition_function: {kwargs}",
            UserWarning,
            stacklevel=2,
        )

    return acq_fn

# ---------------------------------------------------------------------
# Target/search grid helpers
# ---------------------------------------------------------------------

def get_pre_dp_target_grid(bundle: dict, config):
    """
    Get the pre-DataProcessor target grid used for spatial prediction.

    Prefers anomaly target if configured and available.
    """
    X_t = None

    for name, source in config.data_sources.items():
        roles = source.role if isinstance(source.role, list) else [source.role]

        if "target" in roles:
            if source.use_anomalies and f"{name}_anom_stand" in bundle:
                X_t = bundle[f"{name}_anom_stand"]
            elif f"{name}_stand" in bundle:
                X_t = bundle[f"{name}_stand"]
            break

    if X_t is None:
        raise ValueError(
            "No pre-DataProcessor target grid found in bundle. "
            "Expected '<target>_stand' or '<target>_anom_stand'."
        )

    return X_t


def make_spatial_template(ds_or_da):
    """
    Convert a possibly time-varying xarray object into a 2D spatial template.

    GreedyAlgorithm only needs coordinates and shape. Values are not important.
    """
    if isinstance(ds_or_da, xr.Dataset):
        var = list(ds_or_da.data_vars)[0]
        da = ds_or_da[var]
    else:
        da = ds_or_da

    if "time" in da.dims:
        da = da.isel(time=0, drop=True)

    return da


def build_bathy_dist_mask(
    base_grid: Union[xr.DataArray, xr.Dataset],
    bathy: xr.DataArray,
    dist: xr.DataArray,
    depth_tsh: float,
    dist_tsh: float,
) -> xr.DataArray:
    """Construct a unified boolean mask from bathymetry and distance thresholds,
    regridded onto base_grid's coordinates and shape.

    Args:
        base_grid (xr.DataArray | xr.Dataset):
            Reference grid defining the target shape/coordinates for the
            output mask.
        bathy (xr.DataArray):
            Bathymetry (depth) values. Must share the same shape and
            coordinates as ``dist``.
        dist (xr.DataArray):
            Distance values. Must share the same shape and coordinates as
            ``bathy``.
        depth_tsh (float):
            Depth threshold. Values <= depth_tsh become False, else True.
        dist_tsh (float):
            Distance threshold. Values <= dist_tsh become False, else True.

    Returns:
        xr.DataArray:
            Boolean mask with the same shape and coordinates as ``base_grid``.
    """
    if bathy.shape != dist.shape:
        raise ValueError(
            f"`bathy` and `dist` must have the same shape, got "
            f"{bathy.shape} and {dist.shape}"
        )

    bathy_mask = bathy > depth_tsh
    dist_mask = dist > dist_tsh
    unified_mask = bathy_mask.squeeze() & dist_mask.squeeze()
    unified_mask_on_base = unified_mask.astype("int8").interp_like(base_grid, method="nearest", kwargs={"fill_value": 0})
    unified_mask_on_base = unified_mask_on_base.astype(bool)

    return unified_mask_on_base


def make_lake_mask_from_target(ds_or_da):
    """
    Build a boolean lake mask from valid target pixels.

    If time is present, a pixel is considered lake if it is valid at any time.
    """
    if isinstance(ds_or_da, xr.Dataset):
        var = list(ds_or_da.data_vars)[0]
        da = ds_or_da[var]
    else:
        da = ds_or_da

    if "time" in da.dims:
        mask = da.notnull().any("time")
    else:
        mask = da.notnull()

    return mask


def coarsen_spatial(xobj, factor: int):
    """Coarsen a spatial xarray Dataset/DataArray."""
    lat_name, lon_name = find_lat_lon_names(xobj)

    return (
        xobj
        .coarsen({lat_name: factor, lon_name: factor}, boundary="trim")
        .mean()
    )


def coarsen_mask(mask, factor: int):
    """Coarsen a boolean mask, preserving cells where any fine cell is valid."""
    lat_name, lon_name = find_lat_lon_names(mask)

    mask_float = mask.astype(float)
    coarse = (
        mask_float
        .coarsen({lat_name: factor, lon_name: factor}, boundary="trim")
        .min()
    )

    return coarse > 0.5


def find_lat_lon_names(xobj):
    """Find spatial coordinate names."""
    coords_and_dims = set(list(xobj.coords) + list(xobj.dims))

    lat_candidates = ["lat", "latitude", "y", "x1"]
    lon_candidates = ["lon", "longitude", "x", "x2"]

    lat_name = next((c for c in lat_candidates if c in coords_and_dims), None)
    lon_name = next((c for c in lon_candidates if c in coords_and_dims), None)

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not identify lat/lon coordinates. "
            f"Available dims={list(xobj.dims)}, coords={list(xobj.coords)}"
        )

    return lat_name, lon_name


def build_grids_and_masks(bundle, config, acq_fn):
    """
    Build search and target grids for active learning.

    For parallel acquisition functions, X_s == X_t (same grid).
    For sequential acquisition functions, X_s can be coarser than X_t.
    """
    # TODO: This could use a cleaner flow considering that if parallel then search == target and there's no point loading/running a few lines but it's innocent and low priority
    print("\nNow building search/target grids and masks...")
    al_cfg = config.active_learning

    # Start from the full pre-DP target grid
    base_grid_full = get_pre_dp_target_grid(bundle, config)
    base_grid = make_spatial_template(base_grid_full)
    bathy = bundle['bathy_pre_dp'].to_dataarray()  # Going to assume bathy is always there

    if 'dist_to_land_pre_dp' in bundle:     # TODO: at some point maybe this can draw from the dist path and spatial slice for if dist to land wasn't chosen as context input
        dist = bundle['dist_to_land_pre_dp'].to_dataarray()
        search_mask = build_bathy_dist_mask(base_grid=base_grid, bathy=bathy, dist=dist, depth_tsh=al_cfg.min_depth_search, dist_tsh=al_cfg.min_shore_dist_search)
        target_mask = build_bathy_dist_mask(base_grid=base_grid, bathy=bathy, dist=dist, depth_tsh=al_cfg.min_depth_target, dist_tsh=al_cfg.min_shore_dist_target)
    else:
        search_mask = build_bathy_dist_mask(base_grid=base_grid, bathy=bathy, dist=bathy, depth_tsh=al_cfg.min_depth_search, dist_tsh=al_cfg.min_depth_search)
        target_mask = build_bathy_dist_mask(base_grid=base_grid, bathy=bathy, dist=bathy, depth_tsh=al_cfg.min_depth_target, dist_tsh=al_cfg.min_depth_target) # Just use bathy threshold twice. Hack for now

    search_exclusion_points = load_points_from_file(al_cfg.exclusion_points_path_search)
    fixed_context_points = load_points_from_file(al_cfg.context_geojson_path)
    search_mask = mask_near_points(mask=search_mask, points=search_exclusion_points,
                                   min_dist_km=al_cfg.min_dist_from_exclusion_km_search,
                                   points_2=fixed_context_points, min_dist_km_2=al_cfg.min_dist_from_existing_km_search)

    target_exclusion_points = load_points_from_file(al_cfg.exclusion_points_path_target)
    target_mask = mask_near_points(mask=target_mask, points=target_exclusion_points,
                                   min_dist_km=al_cfg.min_dist_from_exclusion_km_target,
                                   points_2=fixed_context_points, min_dist_km_2=al_cfg.min_dist_from_existing_km_target)

    points = {
        "search_exclusion_points": search_exclusion_points,
        "target_exclusion_points": target_exclusion_points,
        "fixed_context_points_latlon": fixed_context_points,
    }

    is_parallel = isinstance(acq_fn, AcquisitionFunctionParallel)

    target_mask_norm = bundle['data_processor'].map_coords(target_mask, unnorm=False)

    if is_parallel:
        candidate = al_cfg.candidate_coarsen_factor or 1
        target = al_cfg.target_coarsen_factor or 1
        if target != candidate:
            import warnings
            warnings.warn(
                f"Parallel acquisition function requires X_s == X_t. "
                f"Ignoring target_coarsen_factor={target}; using "
                f"candidate_coarsen_factor={candidate} for both search and target grids.",
                UserWarning,
                stacklevel=2,
            )
        coarsen = candidate
        if coarsen > 1:
            base_grid = coarsen_spatial(base_grid, coarsen)
            search_mask = coarsen_mask(search_mask, coarsen)
            # target_mask = coarsen_mask(target_mask, coarsen)

        return {
            "search_grid": base_grid, "search_mask": search_mask,
            "target_grid": base_grid, "target_mask": search_mask,
            "target_mask_norm": target_mask_norm,
            **points
        }
    else:
        # Sequential: X_s can be coarser than X_t
        target_coarsen = al_cfg.target_coarsen_factor or 1
        candidate_coarsen = al_cfg.candidate_coarsen_factor or 1

        if target_coarsen > 1:
            target_grid = coarsen_spatial(base_grid, target_coarsen)
            target_mask = coarsen_mask(target_mask, target_coarsen)
        else:
            target_grid = base_grid
            target_mask = target_mask

        if candidate_coarsen > 1:
            search_grid = coarsen_spatial(base_grid, candidate_coarsen)
            search_mask = coarsen_mask(search_mask, candidate_coarsen)
        else:
            search_grid = base_grid
            search_mask = search_mask

        return {
            "search_grid": search_grid,
            "search_mask": search_mask,
            "target_grid": target_grid,
            "target_mask": target_mask,
            "target_mask_norm": target_mask_norm,
            **points
        }


# ---------------------------------------------------------------------
# Distance masking
# ---------------------------------------------------------------------

def load_points_from_file(path) -> list:
    """
    Load existing sensor locations from CSV or GeoJSON.

    Returns list of (lat, lon) tuples, or None if path is None.
    """
    if path is None:
        return None

    path = Path(path).expanduser()
    suffix = path.suffix.lower()

    if suffix == ".csv":
        lats, lons = load_points_from_csv(path)
    elif suffix in [".geojson", ".json"]:
        lats, lons = load_points_from_geojson(path)
    else:
        raise ValueError(
            f"Unsupported format '{suffix}' for exclusion_points_path. "
            f"Supported: .csv, .geojson, .json"
        )

    return list(zip(lats.tolist(), lons.tolist()))


def mask_near_points(
    mask: xr.DataArray,
    points,
    min_dist_km: float,
    points_2=None,
    min_dist_km_2: float | None = None,
):
    """
    Set mask=False near one or two sets of points.

    Parameters
    ----------
    mask : xr.DataArray
        Boolean 2D mask where True means the candidate is allowed.
    points : iterable[tuple[float, float]]
        First collection of (latitude, longitude) points.
    min_dist_km : float
        Exclusion distance, in kilometers, for `points`.
    points_2 : iterable[tuple[float, float]] or None, optional
        Second collection of (latitude, longitude) points.
    min_dist_km_2 : float or None, optional
        Exclusion distance, in kilometers, for `points_2`.
        Required when `points_2` is provided and nonempty.

    Returns
    -------
    xr.DataArray
        A copy of `mask`, with candidates inside either exclusion distance
        set to False.
    """
    point_groups = []

    if points and min_dist_km > 0:
        point_groups.append((points, min_dist_km))

    if points_2:
        if min_dist_km_2 is None:
            raise ValueError(
                "min_dist_km_2 must be provided when points_2 is nonempty."
            )
        if min_dist_km_2 > 0:
            point_groups.append((points_2, min_dist_km_2))

    if not point_groups:
        return mask

    lat_name, lon_name = find_lat_lon_names(mask)

    # Create 2D coordinate grids aligned with mask dimensions.
    lats_1d = mask[lat_name].values
    lons_1d = mask[lon_name].values
    lon_2d, lat_2d = np.meshgrid(lons_1d, lats_1d)

    updated = mask.values.astype(bool, copy=True)

    for group_points, group_min_dist_km in point_groups:
        for lat0, lon0 in group_points:
            dist = approx_distance_km(lat_2d, lon_2d, lat0, lon0)
            updated &= dist >= group_min_dist_km

    result = mask.copy(data=updated)

    n_before = int(mask.sum().item())
    n_after = int(result.sum().item())

    print(
        f"  Masked points: {n_before} -> {n_after} valid candidates "
        f"({n_before - n_after} removed)"
    )

    if n_after == 0:
        distances = ", ".join(
            f"{distance:g} km" for _, distance in point_groups
        )
        raise ValueError(
            "All candidate locations were masked out! "
            f"Exclusion distances used: {distances}. "
            "Try reducing the exclusion distances or using a finer candidate "
            "grid (lower candidate_coarsen_factor)."
        )

    return result

def approx_distance_km(lat, lon, lat0, lon0):
    """
    Approximate distance in km using equirectangular projection.
    Works with scalars or numpy arrays.
    """
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(float(lat0)))

    dy = (lat - lat0) * km_per_deg_lat
    dx = (lon - lon0) * km_per_deg_lon

    return np.sqrt(dx ** 2 + dy ** 2)

def enforce_min_distance(X_new_df, min_dist_km, model):
    """
    Post-hoc filter: remove recommended sensors that are too close together.

    Keeps earlier (higher priority) placements and drops later ones that violate
    the minimum distance constraint.
    """
    x1_name = get_x1_name(model)
    x2_name = get_x2_name(model)

    keep = []
    for idx, row in X_new_df.iterrows():
        lat = row[x1_name]
        lon = row[x2_name]

        too_close = False
        for kept_row in keep:
            dist = float(approx_distance_km(lat, lon, kept_row[x1_name], kept_row[x2_name]))
            if dist < min_dist_km:
                too_close = True
                break

        if not too_close:
            keep.append(row)

    filtered = pd.DataFrame(keep)
    filtered["greedy_iteration"] = filtered.index  # keep original acq-surface iteration
    filtered = filtered.reset_index(drop=True)  # contiguous 0..k-1
    filtered.index.name = "priority"
    if len(filtered) < len(X_new_df):
        print(f"  Min distance filter: kept {len(filtered)}/{len(X_new_df)} sensors")

    return filtered


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

# def plot_active_learning_results(
#     recommended_df: pd.DataFrame,
#     acquisition_ds: xr.DataArray,
#     config,
#     output_dir: Path,
#     acquisition_name: str,
#     x1_name="lat",
#     x2_name="lon",
# ):
#     """
#     Plot acquisition surfaces and selected sensor locations.
#
#     Produces:
#       - one plot per greedy iteration
#       - a final map of recommended locations
#     """
#     import matplotlib.pyplot as plt
#
#     output_dir = Path(output_dir)
#     plots_dir = output_dir / "plots"
#     plots_dir.mkdir(parents=True, exist_ok=True)
#
#     # One acquisition map per iteration
#     # TODO: Make this so that you get a multipane plot (1x3, 2x2, 2x3, etc) in addition to the individual pngs
#     for iteration in acquisition_ds.iteration.values:
#         da_iter = acquisition_ds.sel(iteration=iteration).mean("time", skipna=True)
#
#         fig, ax = plt.subplots(figsize=(8, 6))
#         da_iter.plot(ax=ax, cmap="viridis", add_colorbar=True)
#
#         # Plot all recommended points up to this iteration
#         subset = recommended_df.loc[recommended_df.index <= iteration]
#
#         ax.scatter(
#             subset[x2_name],
#             subset[x1_name],
#             facecolors="none",
#             edgecolors="red",
#             linewidths=1.5,
#             s=80,
#             zorder=5,
#             label="Recommended sensors",
#         )
#
#         for idx, row in subset.iterrows():
#             ax.text(
#                 row[x2_name],
#                 row[x1_name],
#                 str(idx + 1),
#                 color="red",
#                 fontsize=9,
#                 ha="center",
#                 va="center",
#                 zorder=6,
#             )
#
#         ax.set_title(
#             f"Active Learning Iteration {int(iteration) + 1}\n"
#             f"Acquisition: {acquisition_name}"
#         )
#         ax.legend(loc="best")
#
#         save_path = plots_dir / f"acquisition_iteration_{int(iteration) + 1}.png"
#         _finish_plot(config, save_path)
#
#     # Final recommended locations map
#     fig, ax = plt.subplots(figsize=(8, 6))
#
#     final_surface = acquisition_ds.mean(["iteration", "time"], skipna=True)
#     final_surface.plot(ax=ax, cmap="viridis", add_colorbar=True)
#
#     ax.scatter(
#         recommended_df[x2_name],
#         recommended_df[x1_name],
#         facecolors="none",
#         edgecolors="red",
#         linewidths=1.8,
#         s=90,
#         zorder=5,
#         label="Recommended sensors",
#     )
#
#     for idx, row in recommended_df.iterrows():
#         ax.text(
#             row[x2_name],
#             row[x1_name],
#             str(idx + 1),
#             color="red",
#             fontsize=9,
#             ha="center",
#             va="center",
#             zorder=6,
#         )
#
#     ax.set_title(
#         f"Recommended Sensor Locations\n"
#         f"{config.lake.upper()} — Acquisition: {acquisition_name}"
#     )
#     ax.legend(loc="best")
#
#     save_path = plots_dir / "recommended_sensor_locations.png"
#     _finish_plot(config, save_path)
#
#     print(f"Active learning plots saved to: {plots_dir}")


def get_x1_name(model):
    return model.data_processor.config["coords"]["x1"]["name"]


def get_x2_name(model):
    return model.data_processor.config["coords"]["x2"]["name"]

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


_PREFERRED_LAYOUTS = {
    1: (1, 1),
    2: (1, 2),
    3: (1, 3),
    4: (2, 2),
    5: (2, 3),
    6: (2, 3),
    7: (2, 4),
    8: (2, 4),
    9: (3, 3),
    10: (3, 4),
    11: (3, 4),
    12: (3, 4),
}


def panel_grid_shape(n_panels: int, max_cols: int = 4) -> tuple:
    """
    Choose a (nrows, ncols) layout for `n_panels` subplots.

    Uses hand-picked layouts for small counts (1x3, 2x2, 2x3, ...) and falls
    back to a near-square grid capped at `max_cols` columns.
    """
    if n_panels < 1:
        raise ValueError(f"n_panels must be >= 1, got {n_panels}")

    if n_panels in _PREFERRED_LAYOUTS:
        return _PREFERRED_LAYOUTS[n_panels]

    ncols = min(max_cols, int(math.ceil(math.sqrt(n_panels))))
    nrows = int(math.ceil(n_panels / ncols))
    return nrows, ncols


def _select_panel_iterations(iterations, max_panels: int):
    """
    Subsample iterations for the multipane figure if there are too many.

    Always keeps the first and last iteration.
    """
    iterations = list(iterations)
    if max_panels is None or len(iterations) <= max_panels:
        return iterations

    idx = np.unique(
        np.linspace(0, len(iterations) - 1, max_panels).round().astype(int)
    )
    return [iterations[i] for i in idx]


def _iteration_surface(acquisition_ds: xr.DataArray, iteration) -> xr.DataArray:
    """Mean-over-time 2D acquisition surface for a single greedy iteration."""
    da = acquisition_ds.sel(iteration=iteration)
    if "time" in da.dims:
        da = da.mean("time", skipna=True)
    return da.squeeze(drop=True)


def _robust_limits(das, q=(2.0, 98.0)):
    """Robust (vmin, vmax) across one or more DataArrays; (None, None) if empty."""
    if isinstance(das, xr.DataArray):
        das = [das]

    vals = [np.ravel(np.asarray(d.values, dtype=float)) for d in das]
    vals = np.concatenate(vals) if vals else np.array([])
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return None, None

    vmin, vmax = np.nanpercentile(vals, q)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None, None

    return float(vmin), float(vmax)


def _points_up_to_iteration(recommended_df: pd.DataFrame, iteration) -> pd.DataFrame:
    """
    Rows of `recommended_df` placed at or before `iteration`.

    Honours the `greedy_iteration` column if present (set by
    `enforce_min_distance`), otherwise falls back to the index.
    """
    if "greedy_iteration" in recommended_df.columns:
        return recommended_df.loc[recommended_df["greedy_iteration"] <= iteration]
    return recommended_df.loc[recommended_df.index <= iteration]


def _point_labels(df: pd.DataFrame) -> list:
    """1-based priority labels for annotation."""
    if "priority" in df.columns:
        return [str(int(v)) for v in df["priority"].values]
    try:
        return [str(int(i) + 1) for i in df.index.values]
    except (TypeError, ValueError):
        return [str(i + 1) for i in range(len(df))]


# def _save_fig(config, fig, save_path: Path):
#     """
#     Save a specific figure via `_finish_plot`.
#
#     `_finish_plot` operates on the current pyplot figure, so make `fig` current
#     before handing off. Needed because the multipane figure stays open while
#     the per-iteration figures are created.
#     """
#     import matplotlib.pyplot as plt
#
#     plt.figure(fig.number)
#     _finish_plot(config, save_path)

def _save_fig(config, fig, save_path: Path):
    """Save a specific figure via `_finish_plot`, tolerating either signature."""
    import inspect
    import matplotlib.pyplot as plt

    try:
        accepts_fig = "fig" in inspect.signature(_finish_plot).parameters
    except (TypeError, ValueError):
        accepts_fig = False

    if accepts_fig:
        _finish_plot(config, save_path, fig=fig)
    else:
        plt.figure(fig.number)
        _finish_plot(config, save_path)


def _draw_acquisition_panel(
    ax,
    da: xr.DataArray,
    points_df: pd.DataFrame,
    x1_name: str,
    x2_name: str,
    *,
    vmin=None,
    vmax=None,
    cmap="viridis",
    add_colorbar=False,
    highlight_last=False,
    annotate=True,
    title=None,
    show_legend=False,
    axis_labels=False,
    overlays=None,
    search_mask=None
):
    """
    Draw a single acquisition-surface panel with sensor placements.

    This is the single rendering path used by both the per-iteration PNGs and
    the multipane figure -- extend here (cartopy features, existing buoys,
    bathymetry contours, ...) and both outputs inherit the change.

    Returns
    -------
    The mappable artist (for building a shared colorbar).
    """
    import matplotlib.patheffects as pe

    mappable = da.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=add_colorbar,
        add_labels=False,
    )

    if search_mask is not None:
        _draw_excluded_region(ax, search_mask)

    _draw_overlays(ax, overlays)

    if points_df is not None and len(points_df) > 0:
        labels = _point_labels(points_df)

        if highlight_last and len(points_df) > 1:
            prev_df = points_df.iloc[:-1]
            new_df = points_df.iloc[-1:]
            prev_labels = labels[:-1]
            new_labels = labels[-1:]
        elif highlight_last:
            prev_df = points_df.iloc[:0]
            new_df = points_df
            prev_labels = []
            new_labels = labels
        else:
            prev_df = points_df
            new_df = points_df.iloc[:0]
            prev_labels = labels
            new_labels = []

        if len(prev_df) > 0:
            ax.scatter(
                prev_df[x2_name],
                prev_df[x1_name],
                facecolors="none",
                edgecolors="white",
                linewidths=1.4,
                s=70,
                zorder=5,
                label="Previously placed",
            )

        if len(new_df) > 0:
            ax.scatter(
                new_df[x2_name],
                new_df[x1_name],
                facecolors="none",
                edgecolors="red",
                linewidths=1.8,
                s=90,
                zorder=6,
                label="New this iteration" if highlight_last else "Recommended sensors",
            )

        if annotate:
            stroke = [pe.withStroke(linewidth=2.0, foreground="black")]
            for (_, row), label, color in [
                *[(r, l, "white") for r, l in zip(prev_df.iterrows(), prev_labels)],
                *[(r, l, "red") for r, l in zip(new_df.iterrows(), new_labels)],
            ]:
                ax.text(
                    row[x2_name],
                    row[x1_name],
                    label,
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=7,
                    path_effects=stroke,
                )

    if title:
        ax.set_title(title, fontsize=10)
    if axis_labels:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if show_legend:
        ax.legend(loc="best", fontsize=8, framealpha=0.8)

    return mappable


def build_plot_overlays(
    context_latlon=None,
    search_exclusion_points=None,
    extra_layers=None,
) -> list:
    """
    Assemble overlay layer specs for the acquisition panels.

    Each layer is a dict understood by `_draw_overlays`:
        {lats, lons, label, marker, facecolor, edgecolor, s, linewidth, alpha}

    Parameters
    ----------
    context_latlon : tuple(np.ndarray, np.ndarray) or None
        (lats, lons) of the fixed context sensors actually fed to the model.
    search_exclusion_points : list[tuple[float, float]] or None
        (lat, lon) pairs used to carve exclusion buffers out of the search grid.
    extra_layers : list[dict] or None
        Pre-built layer dicts appended as-is.
    """
    layers = []

    if context_latlon is not None:
        lats, lons = context_latlon
        if len(lats):
            layers.append({
                "lats": lats, "lons": lons,
                "label": f"Active buoys / context (n={len(lats)})",
                "marker": "^", "facecolor": "none", "edgecolor": "white",
                "s": 55, "linewidth": 1.3,
            })

    if search_exclusion_points:
        ex = np.asarray(search_exclusion_points, dtype=float)
        layers.append({
            "lats": ex[:, 0], "lons": ex[:, 1],
            "label": f"Excluded sites (n={len(ex)})",
            "marker": "x", "facecolor": "none", "edgecolor": "0.75",
            "s": 40, "linewidth": 1.1, "alpha": 0.8,
        })

    if extra_layers:
        layers.extend(extra_layers)

    return layers


def _draw_overlays(ax, overlays, zorder=4):
    """Scatter each overlay layer onto `ax`."""
    for layer in overlays or []:
        lats = np.asarray(layer.get("lats", []), dtype=float)
        lons = np.asarray(layer.get("lons", []), dtype=float)
        if lats.size == 0:
            continue
        ax.scatter(
            lons, lats,
            marker=layer.get("marker", "^"),
            facecolors=layer.get("facecolor", "none"),
            edgecolors=layer.get("edgecolor", "white"),
            s=layer.get("s", 50),
            linewidths=layer.get("linewidth", 1.2),
            alpha=layer.get("alpha", 0.95),
            zorder=zorder,
            label=layer.get("label"),
        )


def _draw_excluded_region(ax, search_mask, color="0.55", alpha=0.35):
    """Shade cells where the search mask is False (land, shallows, buffers)."""
    from matplotlib.colors import ListedColormap

    excluded = ~search_mask.astype(bool)
    excluded = excluded.where(excluded)  # True -> 1.0, False -> NaN
    excluded.plot(
        ax=ax,
        cmap=ListedColormap([color]),
        vmin=0, vmax=1,
        alpha=alpha,
        add_colorbar=False,
        add_labels=False,
        zorder=2,
    )

def plot_active_learning_results(
    recommended_df: pd.DataFrame,
    acquisition_ds: xr.DataArray,
    config,
    output_dir: Path,
    acquisition_name: str,
    x1_name="lat",
    x2_name="lon",
    *,
    overlays=None,
    search_mask=None,
    shade_excluded=True,
    shade_excluded_multipane=False,
    cmap="viridis",
    shared_color_scale=True,
    max_panel_cols=4,
    max_panels=12,
    plot_individual=True,
    plot_multipane=True,
):
    """
    Plot acquisition surfaces and selected sensor locations.

    Produces:
      - one PNG per greedy iteration (`acquisition_iteration_<k>.png`)
      - a multipane summary of all/most iterations (`acquisition_iterations_grid.png`)
      - a final map of recommended locations (`recommended_sensor_locations.png`)

    Parameters
    ----------
    shared_color_scale : bool
        Use one robust (2-98th percentile) color range across all iterations so
        panels are directly comparable. If False, each panel autoscales.
    max_panel_cols : int
        Column cap for the auto-chosen grid layout.
    max_panels : int or None
        Max panels in the multipane figure; extra iterations are evenly
        subsampled (first and last always kept). None = include all.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mask_full = search_mask if shade_excluded else None
    mask_panel = search_mask if (shade_excluded and shade_excluded_multipane) else None

    # Tolerate a surface without an explicit iteration dim (single placement)
    if "iteration" not in acquisition_ds.dims:
        acquisition_ds = acquisition_ds.expand_dims("iteration")

    iterations = list(acquisition_ds.iteration.values)

    # Pre-compute the time-mean surface per iteration (reused by both outputs)
    surfaces = {int(it): _iteration_surface(acquisition_ds, it) for it in iterations}

    if shared_color_scale:
        vmin, vmax = _robust_limits(list(surfaces.values()))
    else:
        vmin, vmax = None, None

    # -----------------------------------------------------------------
    # 1. Individual per-iteration PNGs
    # -----------------------------------------------------------------
    if plot_individual:
        for it in iterations:
            it_int = int(it)
            subset = _points_up_to_iteration(recommended_df, it)

            fig, ax = plt.subplots(figsize=(8, 6))
            _draw_acquisition_panel(
                ax,
                surfaces[it_int],
                subset,
                x1_name,
                x2_name,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                add_colorbar=True,
                highlight_last=True,
                title=(
                    f"Active Learning Iteration {it_int + 1}\n"
                    f"Acquisition: {acquisition_name}"
                ),
                show_legend=True,
                axis_labels=True,
                overlays=overlays,
                search_mask=mask_full
            )

            _save_fig(
                config, fig, plots_dir / f"acquisition_iteration_{it_int + 1}.png"
            )

    # -----------------------------------------------------------------
    # 2. Multipane summary across iterations
    # -----------------------------------------------------------------
    if plot_multipane:
        panel_its = _select_panel_iterations(iterations, max_panels)
        nrows, ncols = panel_grid_shape(len(panel_its), max_cols=max_panel_cols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.8 * nrows),
            sharex=True,
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )
        flat_axes = axes.ravel()
        mappable = None

        for panel_idx, it in enumerate(panel_its):
            it_int = int(it)
            ax = flat_axes[panel_idx]
            subset = _points_up_to_iteration(recommended_df, it)

            mappable = _draw_acquisition_panel(
                ax,
                surfaces[it_int],
                subset,
                x1_name,
                x2_name,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                add_colorbar=not shared_color_scale,
                highlight_last=True,
                title=f"Iteration {it_int + 1}",
                show_legend=(panel_idx == 0),
                axis_labels=False,
                overlays=overlays,
                search_mask=mask_panel
            )

        # Blank out unused cells
        for ax in flat_axes[len(panel_its):]:
            ax.set_visible(False)

        # Outer axis labels only
        for r in range(nrows):
            axes[r, 0].set_ylabel("Latitude")
        for c in range(ncols):
            bottom_r = nrows - 1
            while bottom_r > 0 and not axes[bottom_r, c].get_visible():
                bottom_r -= 1
            axes[bottom_r, c].set_xlabel("Longitude")

        if shared_color_scale and mappable is not None:
            cbar = fig.colorbar(
                mappable,
                ax=axes.ravel().tolist(),
                shrink=0.85,
                pad=0.02,
            )
            cbar.set_label(f"{acquisition_name} acquisition value")

        suptitle = (
            f"{config.lake.upper()} — Greedy Sensor Placement\n"
            f"Acquisition: {acquisition_name}"
        )
        if len(panel_its) < len(iterations):
            suptitle += (
                f"  (showing {len(panel_its)} of {len(iterations)} iterations)"
            )
        fig.suptitle(suptitle, fontsize=13)

        _save_fig(config, fig, plots_dir / "acquisition_iterations_grid.png")

    # -----------------------------------------------------------------
    # 3. Final recommended locations map (mean surface)
    # -----------------------------------------------------------------
    final_surface = acquisition_ds.mean(
        [d for d in ("iteration", "time") if d in acquisition_ds.dims], skipna=True
    ).squeeze(drop=True)

    f_vmin, f_vmax = _robust_limits(final_surface)

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_acquisition_panel(
        ax,
        final_surface,
        recommended_df,
        x1_name,
        x2_name,
        vmin=f_vmin,
        vmax=f_vmax,
        cmap=cmap,
        add_colorbar=True,
        highlight_last=False,
        title=(
            f"Recommended Sensor Locations\n"
            f"{config.lake.upper()} — Acquisition: {acquisition_name}"
        ),
        show_legend=True,
        axis_labels=True,
        overlays=overlays,
        search_mask=mask_full
    )

    _save_fig(config, fig, plots_dir / "recommended_sensor_locations.png")

    print(f"Active learning plots saved to: {plots_dir}")