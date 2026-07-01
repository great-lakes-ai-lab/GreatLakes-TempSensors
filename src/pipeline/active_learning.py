# src/pipeline/active_learning.py
"""Active learning / sensor placement for Great Lakes DeepSensor pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


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

    # 6. Build grids and masks
    grids = build_grids_and_masks(bundle, config, acquisition_fn)

    search_grid = grids["search_grid"]
    search_mask = grids["search_mask"]
    target_grid = grids["target_grid"]
    target_mask = grids["target_mask"]

    print(f"  Search grid shape: {search_grid.shape}")
    print(f"  Target grid shape: {target_grid.shape}")

    # 7. Optionally mask existing sensor locations
    existing_points = load_existing_sensor_points(al_cfg.existing_sensors_path)
    if existing_points is not None and al_cfg.min_dist_between_sensors_km > 0:
        search_mask = mask_near_points(
            search_mask,
            existing_points,
            min_dist_km=al_cfg.min_dist_between_sensors_km,
        )

    # For parallel acquisition functions, enforce X_s == X_t constraint
    if acquisition_fn._is_parallel:
        target_grid = search_grid
        target_mask = search_mask

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
        diff=al_cfg.diff,
    )

    # 9. Post-processing: enforce minimum distance
    if al_cfg.min_dist_between_sensors_km > 0:
        X_new_df = enforce_min_distance(
            X_new_df,
            min_dist_km=al_cfg.min_dist_between_sensors_km,
            model=model,
        )

    # 10. Save outputs
    save_recommended_locations(X_new_df, model, output_dir, config)

    if al_cfg.save_acquisition_surface:
        nc_path = output_dir / "acquisition_surfaces.nc"
        try:
            acquisition_fn_ds.to_netcdf(nc_path)
            print(f"Saved acquisition surfaces: {nc_path}")
        except Exception as e:
            print(f"Warning: Could not save acquisition surface as NetCDF: {e}")
            np.save(output_dir / "acquisition_surfaces.npy", acquisition_fn_ds.values)

    if al_cfg.plot_results:
        plot_active_learning_results(
            X_new_df,
            acquisition_fn_ds,
            config,
            output_dir,
            acquisition_name=al_cfg.acquisition_function,
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
        "min_dist_between_sensors_km": al_cfg.min_dist_between_sensors_km,
        "existing_sensors_path": al_cfg.existing_sensors_path,
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

    print(f"  Loaded {len(lats)} points from GeoJSON: {path}")
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
    from deepsensor_greatlakes.utils import generate_random_coordinates

    np.random.seed(al_cfg.context_seed)

    points = generate_random_coordinates(
        bundle["lakemask_sampling"],
        N=al_cfg.n_context,
        data_processor=bundle["data_processor"],
    )

    return points


def _load_geojson_context(al_cfg, bundle) -> np.ndarray:
    """
    Load context points from a GeoJSON file and normalize to model coordinates.

    Expects Point features in WGS84.
    """
    lats, lons = load_points_from_geojson(al_cfg.context_geojson_path)

    # Stack as (2, N) in raw coordinates [lat, lon]
    raw_points = np.array([lats, lons])

    # Normalize to model coordinate space
    data_processor = bundle["data_processor"]
    normalized_points = data_processor.map_coord_array(raw_points, unnorm=False)

    return normalized_points

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
    data_processor = bundle["data_processor"]
    points_latlon = data_processor.map_coord_array(context_points, unnorm=True)
    lats = points_latlon[0]
    lons = points_latlon[1]

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

    return points_latlon


# ---------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------

def make_active_learning_dates(config):
    """Generate active learning eval dates from config."""
    al_cfg = config.active_learning
    dates = pd.date_range(al_cfg.eval_range[0], al_cfg.eval_range[1], freq="D")
    dates = dates[::al_cfg.eval_subsample_factor]
    return pd.to_datetime(dates).normalize()


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

    # ──────────────────────────────────────────────────────────────────────
    # Mapping: config name → (class, requires_special_args)
    # ──────────────────────────────────────────────────────────────────────

    # Non-parallel (sequential) acquisition functions
    SEQUENTIAL_MAPPING = {
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
        # Oracle functions (require true target values)
        "oracle_rmse": OracleRMSE,
        "oracle_mae": OracleMAE,
        "oracle_marginal_nll": OracleMarginalNLL,
        "oracle_joint_nll": OracleJointNLL,
    }

    # Parallel acquisition functions
    PARALLEL_MAPPING = {
        "stddev": Stddev,
        "std": Stddev,
        "context_dist": ContextDist,
        "context_distance": ContextDist,
        "expected_improvement": ExpectedImprovement,
        "ei": ExpectedImprovement,
        "random": Random,
    }

    # Oracle functions (subset of sequential, flagged for user warnings)
    ORACLE_NAMES = {"oracle_rmse", "oracle_mae", "oracle_marginal_nll", "oracle_joint_nll"}

    # ──────────────────────────────────────────────────────────────────────
    # Look up the class
    # ──────────────────────────────────────────────────────────────────────

    all_mapping = {**SEQUENTIAL_MAPPING, **PARALLEL_MAPPING}

    if name_clean not in all_mapping:
        available_sequential = sorted(set(SEQUENTIAL_MAPPING.keys()))
        available_parallel = sorted(set(PARALLEL_MAPPING.keys()))
        raise ValueError(
            f"Unsupported acquisition_function='{name}'.\n"
            f"  Sequential (non-parallel): {available_sequential}\n"
            f"  Parallel: {available_parallel}"
        )

    cls = all_mapping[name_clean]
    is_parallel = name_clean in PARALLEL_MAPPING
    is_oracle = name_clean in ORACLE_NAMES

    # ──────────────────────────────────────────────────────────────────────
    # Warnings
    # ──────────────────────────────────────────────────────────────────────

    if is_oracle:
        import warnings
        warnings.warn(
            f"Oracle acquisition function '{cls.__name__}' requires true target values. "
            f"This is intended for retrospective evaluation, not operational deployment.",
            UserWarning,
            stacklevel=2,
        )

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
    # Attach metadata for downstream use
    # ──────────────────────────────────────────────────────────────────────

    acq_fn._is_parallel = is_parallel
    acq_fn._is_oracle = is_oracle
    acq_fn._config_name = name

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
        .max()
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
    al_cfg = config.active_learning

    # Start from the full pre-DP target grid
    target_grid_full = get_pre_dp_target_grid(bundle, config)
    base_grid = make_spatial_template(target_grid_full)
    base_mask = make_lake_mask_from_target(target_grid_full)

    # is_parallel = al_cfg.acquisition_function.lower() in [
    #     "stddev", "std", "mean_variance", "mean_var",
    # ]
    is_parallel = acq_fn._is_parallel

    if is_parallel:
        # Parallel: X_s == X_t, use candidate_coarsen_factor for both
        coarsen = al_cfg.candidate_coarsen_factor or 1
        if coarsen > 1:
            grid = coarsen_spatial(base_grid, coarsen)
            mask = coarsen_mask(base_mask, coarsen)
        else:
            grid = base_grid
            mask = base_mask

        return {
            "search_grid": grid,
            "search_mask": mask,
            "target_grid": grid,
            "target_mask": mask,
        }
    else:
        # Sequential: X_s can be coarser than X_t
        target_coarsen = al_cfg.target_coarsen_factor or 1
        candidate_coarsen = al_cfg.candidate_coarsen_factor or 1

        if target_coarsen > 1:
            target_grid = coarsen_spatial(base_grid, target_coarsen)
            target_mask = coarsen_mask(base_mask, target_coarsen)
        else:
            target_grid = base_grid
            target_mask = base_mask

        if candidate_coarsen > 1:
            search_grid = coarsen_spatial(base_grid, candidate_coarsen)
            search_mask = coarsen_mask(base_mask, candidate_coarsen)
        else:
            search_grid = base_grid
            search_mask = base_mask

        return {
            "search_grid": search_grid,
            "search_mask": search_mask,
            "target_grid": target_grid,
            "target_mask": target_mask,
        }


# ---------------------------------------------------------------------
# Distance masking
# ---------------------------------------------------------------------

def load_existing_sensor_points(path) -> list:
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
            f"Unsupported format '{suffix}' for existing_sensors_path. "
            f"Supported: .csv, .geojson, .json"
        )

    return list(zip(lats.tolist(), lons.tolist()))

def mask_near_points(mask: xr.DataArray, points, min_dist_km: float):
    """
    Set mask=False near selected/existing points.

    Parameters
    ----------
    mask : xr.DataArray
        Boolean mask (2D) where True means candidate is allowed.
    points : list[(lat, lon)]
    min_dist_km : float
    """
    if not points or min_dist_km <= 0:
        return mask

    lat_name, lon_name = find_lat_lon_names(mask)

    # Create 2D coordinate grids aligned with mask dimensions
    lats_1d = mask[lat_name].values
    lons_1d = mask[lon_name].values

    # meshgrid with indexing that matches (lat, lon) dimension order
    lon_2d, lat_2d = np.meshgrid(lons_1d, lats_1d)

    updated = mask.values.copy().astype(bool)

    for lat0, lon0 in points:
        dist = approx_distance_km(lat_2d, lon_2d, lat0, lon0)
        updated = updated & (dist >= min_dist_km)

    result = mask.copy(data=updated)

    n_before = int(mask.sum())
    n_after = int(result.sum())
    print(f"  Masked existing sensors: {n_before} -> {n_after} valid candidates "
          f"({n_before - n_after} removed)")

    if n_after == 0:
        raise ValueError(
            f"All candidate locations were masked out! "
            f"Try reducing min_dist_between_sensors_km (currently {min_dist_km} km) "
            f"or using a finer candidate grid (lower candidate_coarsen_factor)."
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
    filtered.index.name = "priority"  # Recommended by Claude. Not sure what it does. Need to explore
    if len(filtered) < len(X_new_df):
        print(f"  Min distance filter: kept {len(filtered)}/{len(X_new_df)} sensors")

    return filtered


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_active_learning_results(
    recommended_df: pd.DataFrame,
    acquisition_ds: xr.DataArray,
    config,
    output_dir: Path,
    acquisition_name: str,
):
    """
    Plot acquisition surfaces and selected sensor locations.

    Produces:
      - one plot per greedy iteration
      - a final map of recommended locations
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    x1_name = recommended_df.columns[0]
    x2_name = recommended_df.columns[1]

    # One acquisition map per iteration
    for iteration in acquisition_ds.iteration.values:
        da_iter = acquisition_ds.sel(iteration=iteration).mean("time", skipna=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        da_iter.plot(ax=ax, cmap="viridis", add_colorbar=True)

        # Plot all recommended points up to this iteration
        subset = recommended_df.loc[recommended_df.index <= iteration]

        ax.scatter(
            subset[x2_name],
            subset[x1_name],
            facecolors="none",
            edgecolors="red",
            linewidths=1.5,
            s=80,
            zorder=5,
            label="Recommended sensors",
        )

        for idx, row in subset.iterrows():
            ax.text(
                row[x2_name],
                row[x1_name],
                str(idx + 1),
                color="red",
                fontsize=9,
                ha="center",
                va="center",
                zorder=6,
            )

        ax.set_title(
            f"Active Learning Iteration {int(iteration) + 1}\n"
            f"Acquisition: {acquisition_name}"
        )
        ax.legend(loc="best")

        save_path = plots_dir / f"acquisition_iteration_{int(iteration) + 1}.png"
        _finish_plot(config, save_path)

    # Final recommended locations map
    fig, ax = plt.subplots(figsize=(8, 6))

    final_surface = acquisition_ds.mean(["iteration", "time"], skipna=True)
    final_surface.plot(ax=ax, cmap="viridis", add_colorbar=True)

    ax.scatter(
        recommended_df[x2_name],
        recommended_df[x1_name],
        facecolors="none",
        edgecolors="red",
        linewidths=1.8,
        s=90,
        zorder=5,
        label="Recommended sensors",
    )

    for idx, row in recommended_df.iterrows():
        ax.text(
            row[x2_name],
            row[x1_name],
            str(idx + 1),
            color="red",
            fontsize=9,
            ha="center",
            va="center",
            zorder=6,
        )

    ax.set_title(
        f"Recommended Sensor Locations\n"
        f"{config.lake.upper()} — Acquisition: {acquisition_name}"
    )
    ax.legend(loc="best")

    save_path = plots_dir / "recommended_sensor_locations.png"
    _finish_plot(config, save_path)

    print(f"Active learning plots saved to: {plots_dir}")


def get_x1_name(model):
    return model.data_processor.config["coords"]["x1"]["name"]


def get_x2_name(model):
    return model.data_processor.config["coords"]["x2"]["name"]