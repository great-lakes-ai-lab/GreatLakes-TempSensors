#!/usr/bin/env python3
"""
Generate derived layers from Great Lakes bathymetry data.

Outputs:
1. Clean bathymetry (depth as "z")
2. Land mask (water=1, land=0)
3. Distance to land (in km)
4. Core depth metric (Gaussian-smoothed depth × normalized distance to land)
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate derived layers from Great Lakes bathymetry."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input bathymetry NetCDF file.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where output NetCDF files will be written.",
    )
    parser.add_argument(
        "--sigma-km",
        type=float,
        default=15.0,
        help="Gaussian smoothing radius in km for core depth metric (default: 15.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def compute_pixel_spacing_km(lat: np.ndarray, lon: np.ndarray):
    """
    Compute approximate pixel spacing in km for lat/lon grid.

    Returns (lat_spacing_km, lon_spacing_km) at the center latitude.
    """
    # Approximate degrees to km
    center_lat = np.mean(lat)
    lat_spacing_deg = np.abs(np.mean(np.diff(lat)))
    lon_spacing_deg = np.abs(np.mean(np.diff(lon)))

    # 1 degree latitude ≈ 111.32 km
    lat_spacing_km = lat_spacing_deg * 111.32

    # 1 degree longitude ≈ 111.32 * cos(lat) km
    lon_spacing_km = lon_spacing_deg * 111.32 * np.cos(np.radians(center_lat))

    return lat_spacing_km, lon_spacing_km


def generate_clean_bathymetry(depth: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """
    Generate clean bathymetry dataset with depth as 'z'.
    """
    ds = xr.Dataset(
        {
            "z": (["lat", "lon"], depth.astype(np.float32)),
        },
        coords={
            "lat": lat,
            "lon": lon,
        },
    )
    ds["z"].attrs = {
        "long_name": "bathymetric depth",
        "units": "m",
        "positive": "down",
        "description": "Depth below water surface. 0 indicates land.",
    }
    ds.attrs = {
        "title": "Great Lakes Bathymetry (3 arc-second)",
        "Conventions": "CF-1.8",
        "source": "Derived from Great Lakes bathymetry grid",
    }
    return ds


def generate_land_mask(depth: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """
    Generate land mask: water=1, land=0.
    """
    mask = (depth > 0).astype(np.int8)

    ds = xr.Dataset(
        {
            "mask": (["lat", "lon"], mask),
        },
        coords={
            "lat": lat,
            "lon": lon,
        },
    )
    ds["mask"].attrs = {
        "long_name": "land-water mask",
        "flag_values": "0, 1",
        "flag_meanings": "land water",
        "description": "Binary mask where 1 = water, 0 = land.",
    }
    ds.attrs = {
        "title": "Great Lakes Land Mask (3 arc-second)",
        "Conventions": "CF-1.8",
        "source": "Derived from Great Lakes bathymetry grid",
    }
    return ds


def generate_distance_to_land(
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_spacing_km: float,
    lon_spacing_km: float,
) -> xr.Dataset:
    """
    Generate distance-to-land field using Euclidean distance transform.

    Distance is computed in km, accounting for pixel spacing.
    Land pixels get a value of 0.
    """
    # Water mask: True where water
    water_mask = depth > 0

    # distance_transform_edt computes distance from False (land) pixels
    # sampling parameter accounts for non-square pixels
    dist_pixels = ndimage.distance_transform_edt(
        water_mask,
        sampling=[lat_spacing_km, lon_spacing_km],
    )

    # Result is in km (since sampling is in km)
    dist_km = dist_pixels.astype(np.float32)

    ds = xr.Dataset(
        {
            "dist_to_land": (["lat", "lon"], dist_km),
        },
        coords={
            "lat": lat,
            "lon": lon,
        },
    )
    ds["dist_to_land"].attrs = {
        "long_name": "distance to nearest land",
        "units": "km",
        "description": (
            "Euclidean distance from each water pixel to the nearest land pixel. "
            "Land pixels have a value of 0. Accounts for latitude-dependent "
            "pixel spacing."
        ),
    }
    ds.attrs = {
        "title": "Great Lakes Distance to Land (3 arc-second)",
        "Conventions": "CF-1.8",
        "source": "Derived from Great Lakes bathymetry grid",
    }
    return ds


def generate_core_depth(
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_spacing_km: float,
    lon_spacing_km: float,
    sigma_km: float,
) -> xr.Dataset:
    """
    Generate core depth metric.

    core_depth = gaussian_smoothed_depth × normalized_distance_to_land

    This captures:
    - How deep a point is (local + neighborhood depth)
    - How far from land it is
    - How surrounded by deep water it is (Gaussian smoothing)

    High values = deep basin interiors (e.g., center of Lake Superior)
    Low values = nearshore, shallow areas, or isolated deep spots near land
    """
    # Compute distance to land in km
    water_mask = depth > 0
    dist_km = ndimage.distance_transform_edt(
        water_mask,
        sampling=[lat_spacing_km, lon_spacing_km],
    ).astype(np.float32)

    # Create depth array with land as 0 (already the case, but ensure)
    depth_clean = np.where(water_mask, depth, 0).astype(np.float32)

    # Gaussian smooth the depth field
    # Convert sigma from km to pixels for each axis
    sigma_lat_px = sigma_km / lat_spacing_km
    sigma_lon_px = sigma_km / lon_spacing_km

    print(f"  Gaussian sigma: {sigma_km} km = ({sigma_lat_px:.1f}, {sigma_lon_px:.1f}) pixels")

    # Apply Gaussian filter
    # We smooth the depth field — land (0) acts as boundary pulling values down near shore
    smoothed_depth = ndimage.gaussian_filter(
        depth_clean,
        sigma=[sigma_lat_px, sigma_lon_px],
        mode="constant",
        cval=0.0,
    )

    # Normalize distance to land to [0, 1] range
    max_dist = np.max(dist_km)
    if max_dist > 0:
        dist_normalized = dist_km / max_dist
    else:
        dist_normalized = np.zeros_like(dist_km)

    # Compute core depth metric
    core = smoothed_depth * dist_normalized

    # Mask land as 0
    core = np.where(water_mask, core, 0).astype(np.float32)

    # Also provide a normalized version [0, 1] for convenience
    # max_core = np.max(core)
    # if max_core > 0:
    #     core_normalized = (core / max_core).astype(np.float32)
    # else:
    #     core_normalized = np.zeros_like(core)

    ds = xr.Dataset(
        {
            "core_depth": (["lat", "lon"], core),
            # "core_depth_normalized": (["lat", "lon"], core_normalized),
        },
        coords={
            "lat": lat,
            "lon": lon,
        },
    )
    ds["core_depth"].attrs = {
        "long_name": "core depth metric",
        "units": "m·km (depth × normalized distance)",
        "description": (
            "Gaussian-smoothed bathymetric depth multiplied by normalized distance "
            "to land. High values indicate deep, interior basin locations surrounded "
            "by other deep water. Low values indicate nearshore, shallow, or isolated "
            "deep locations."
        ),
        "gaussian_sigma_km": sigma_km,
    }
    # ds["core_depth_normalized"].attrs = {
    #     "long_name": "normalized core depth metric",
    #     "units": "dimensionless",
    #     "valid_range": [0.0, 1.0],
    #     "description": (
    #         "Core depth metric normalized to [0, 1] range. "
    #         "1.0 = deepest, most interior basin location."
    #     ),
    #     "gaussian_sigma_km": sigma_km,
    # }
    ds.attrs = {
        "title": "Great Lakes Core Depth Metric (3 arc-second)",
        "Conventions": "CF-1.8",
        "source": "Derived from Great Lakes bathymetry grid",
        "methodology": (
            "core_depth = gaussian_filter(depth, sigma) × (dist_to_land / max_dist_to_land). "
            f"Gaussian sigma = {sigma_km} km."
        ),
    }
    return ds


def write_dataset(ds: xr.Dataset, output_path: Path, overwrite: bool = False):
    """Write dataset to NetCDF with compression."""
    if output_path.exists() and not overwrite:
        print(f"  [SKIP] {output_path.name} already exists. Use --overwrite to replace.")
        return False

    # Set up compression encoding for all data variables
    encoding = {}
    for var_name in ds.data_vars:
        encoding[var_name] = {
            "zlib": True,
            "complevel": 4,
            "dtype": ds[var_name].dtype,
        }

    ds.to_netcdf(output_path, encoding=encoding)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  [SAVED] {output_path.name} ({size_mb:.1f} MB)")
    return True


def main():
    args = parse_args()

    input_file = args.input_file.resolve()
    output_dir = args.output_dir.resolve()

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input file:   {input_file}")
    print(f"Output dir:   {output_dir}")
    print(f"Sigma (km):   {args.sigma_km}")
    print()

    # Load bathymetry
    print("Loading bathymetry...")
    src = xr.open_dataset(input_file)

    # Extract depth array and coordinates
    depth = src["Band1"].values
    lat = src["lat"].values
    lon = src["lon"].values

    print(f"  Grid shape: {depth.shape} (lat={len(lat)}, lon={len(lon)})")
    print(f"  Lat range:  {lat.min():.4f} to {lat.max():.4f}")
    print(f"  Lon range:  {lon.min():.4f} to {lon.max():.4f}")
    print(f"  Depth range: {depth[depth > 0].min():.2f} to {depth.max():.2f} m (water only)")
    print(f"  Water pixels: {np.sum(depth > 0):,} / {depth.size:,} "
          f"({100 * np.sum(depth > 0) / depth.size:.1f}%)")
    print()

    # Compute pixel spacing
    lat_spacing_km, lon_spacing_km = compute_pixel_spacing_km(lat, lon)
    print(f"  Pixel spacing: {lat_spacing_km*1000:.1f} m (lat) × {lon_spacing_km*1000:.1f} m (lon)")
    print()

    # 1. Clean bathymetry
    print("1. Generating clean bathymetry...")
    ds_bathy = generate_clean_bathymetry(depth, lat, lon)
    write_dataset(
        ds_bathy,
        output_dir / "gl_bathy_depth_3arcsec_clean.nc",
        overwrite=args.overwrite,
    )
    print()

    # 2. Land mask
    print("2. Generating land mask...")
    ds_mask = generate_land_mask(depth, lat, lon)
    write_dataset(
        ds_mask,
        output_dir / "gl_landmask_3arcsec.nc",
        overwrite=args.overwrite,
    )
    print()

    # 3. Distance to land
    print("3. Generating distance to land...")
    ds_dist = generate_distance_to_land(depth, lat, lon, lat_spacing_km, lon_spacing_km)
    print(f"  Max distance to land: {ds_dist['dist_to_land'].values.max():.2f} km")
    write_dataset(
        ds_dist,
        output_dir / "gl_dist_to_land_3arcsec.nc",
        overwrite=args.overwrite,
    )
    print()

    # 4. Core depth metric
    print("4. Generating core depth metric...")
    ds_core = generate_core_depth(
        depth, lat, lon, lat_spacing_km, lon_spacing_km, args.sigma_km
    )
    core_vals = ds_core["core_depth"].values
    print(f"  Core depth range (water): {core_vals[core_vals > 0].min():.2f} "
          f"to {core_vals.max():.2f}")
    # print(f"  Top locations (by normalized core depth):")
    #
    # # Find and report top 5 locations
    # core_norm = ds_core["core_depth_normalized"].values
    # top_indices = np.unravel_index(
    #     np.argsort(core_norm, axis=None)[-5:][::-1],
    #     core_norm.shape,
    # )
    # for i in range(5):
    #     lat_idx, lon_idx = top_indices[0][i], top_indices[1][i]
    #     print(f"    #{i+1}: lat={lat[lat_idx]:.3f}, lon={lon[lon_idx]:.3f}, "
    #           f"depth={depth[lat_idx, lon_idx]:.1f} m, "
    #           f"core_norm={core_norm[lat_idx, lon_idx]:.4f}")

    write_dataset(
        ds_core,
        output_dir / "gl_core_depth_3arcsec.nc",
        overwrite=args.overwrite,
    )
    print()

    src.close()
    print("All done!")


if __name__ == "__main__":
    main()