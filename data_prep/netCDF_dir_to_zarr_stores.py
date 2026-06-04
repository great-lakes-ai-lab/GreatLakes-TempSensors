#!/usr/bin/env python3
"""
Combine yearly NetCDF files into consolidated Zarr stores.

Groups NetCDF files by dataset name (stripping the year suffix),
concatenates them along the time dimension, updates attributes
to reflect the full time range, and writes each group to a Zarr store.
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import xarray as xr
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine yearly NetCDF files into Zarr stores."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the yearly .nc files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the output .zarr stores will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Zarr stores if they exist.",
    )
    parser.add_argument(
        "--time-chunk-size",
        type=int,
        default=90,
        help="Chunk size along the time dimension (default: 365).",
    )
    parser.add_argument(
        "--spatial-chunk-size",
        type=int,
        default=128,
        help=(
            "Chunk size along latitude and longitude dimensions (default: 128). "
            "Set to -1 to keep spatial dims as single chunks (no spatial chunking)."
        ),
    )
    return parser.parse_args()


def group_nc_files(input_dir: Path) -> dict[str, list[Path]]:
    """
    Group .nc files by dataset name, stripping the trailing _YYYY.nc.

    For example:
        GL_Ice_Concentration_GCS_2018.nc -> GL_Ice_Concentration_GCS
        GLSEA_GCS_SST_2019.nc           -> GLSEA_GCS_SST
        oisst_2020.nc                    -> oisst
    """
    pattern = re.compile(r"^(.+)_(\d{4})\.nc$")
    groups = defaultdict(list)

    for f in sorted(input_dir.glob("*.nc")):
        # Skip auxiliary xml files
        if f.suffixes != [".nc"]:
            continue
        match = pattern.match(f.name)
        if match:
            dataset_name = match.group(1)
            groups[dataset_name].append(f)

    return dict(groups)


def determine_chunk_sizes(
    combined: xr.Dataset,
    time_chunk_size: int,
    spatial_chunk_size: int,
) -> dict[str, int]:
    """
    Determine the final chunk sizes, respecting dimension sizes.

    If spatial_chunk_size is -1, spatial dims remain unchunked (full extent).
    Chunk sizes are clamped to the actual dimension size to avoid
    chunks larger than the dimension.
    """
    total_time = combined.sizes["time"]
    lat_size = combined.sizes["latitude"]
    lon_size = combined.sizes["longitude"]

    # Clamp time chunk to dimension size
    time_chunk = min(time_chunk_size, total_time)

    # Spatial chunking
    if spatial_chunk_size == -1:
        lat_chunk = lat_size
        lon_chunk = lon_size
    else:
        lat_chunk = min(spatial_chunk_size, lat_size)
        lon_chunk = min(spatial_chunk_size, lon_size)

    return {
        "time": time_chunk,
        "latitude": lat_chunk,
        "longitude": lon_chunk,
    }


def print_chunk_summary(combined: xr.Dataset, chunk_sizes: dict[str, int]):
    """Print a summary of chunking decisions."""
    total_time = combined.sizes["time"]
    lat_size = combined.sizes["latitude"]
    lon_size = combined.sizes["longitude"]

    time_chunk = chunk_sizes["time"]
    lat_chunk = chunk_sizes["latitude"]
    lon_chunk = chunk_sizes["longitude"]

    n_time_chunks = (total_time + time_chunk - 1) // time_chunk
    n_lat_chunks = (lat_size + lat_chunk - 1) // lat_chunk
    n_lon_chunks = (lon_size + lon_chunk - 1) // lon_chunk
    total_chunks = n_time_chunks * n_lat_chunks * n_lon_chunks

    # Estimate single chunk size (for first data variable)
    first_var = list(combined.data_vars)[0]
    dtype_size = combined[first_var].dtype.itemsize
    chunk_bytes = time_chunk * lat_chunk * lon_chunk * dtype_size
    chunk_mb = chunk_bytes / (1024 * 1024)

    print(f"  Chunk layout:")
    print(f"    time:      {time_chunk:>5} / {total_time:>5}  ({n_time_chunks} chunks)")
    print(f"    latitude:  {lat_chunk:>5} / {lat_size:>5}  ({n_lat_chunks} chunks)")
    print(f"    longitude: {lon_chunk:>5} / {lon_size:>5}  ({n_lon_chunks} chunks)")
    print(f"    Total chunks per variable: {total_chunks}")
    print(f"    Approx chunk size: {chunk_mb:.1f} MB")

    # Performance estimates
    print(f"  Access patterns:")

    # Full year, full spatial
    year_chunks = n_lat_chunks * n_lon_chunks
    print(f"    Full year, full extent:   {year_chunks} chunks "
          f"({year_chunks * chunk_mb:.0f} MB)")

    # Full year, single spatial tile
    print(f"    Full year, one tile:      1 chunk ({chunk_mb:.1f} MB)")

    # Single day, full spatial
    day_chunks = n_lat_chunks * n_lon_chunks
    day_mb = day_chunks * chunk_mb
    print(f"    Single day, full extent:  {day_chunks} chunks "
          f"({day_mb:.0f} MB loaded, ~{lat_size * lon_size * dtype_size / 1024 / 1024:.1f} MB used)")

    # Single day, single tile
    print(f"    Single day, one tile:     1 chunk "
          f"({chunk_mb:.1f} MB loaded, ~{lat_chunk * lon_chunk * dtype_size / 1024:.0f} KB used)")


def validate_zarr(
    zarr_path: Path,
    expected_time_start,
    expected_time_end,
    expected_time_count: int,
    expected_chunk_sizes: dict[str, int],
):
    """
    Validate the written Zarr store by checking time range, count,
    data integrity, and chunk structure.
    """
    print(f"  Validating {zarr_path}...")
    ds = xr.open_zarr(zarr_path)

    # Check time dimension count
    actual_time_count = ds.sizes["time"]
    assert actual_time_count == expected_time_count, (
        f"Time count mismatch: expected {expected_time_count}, got {actual_time_count}"
    )

    # Check time range
    actual_start = pd.Timestamp(ds.time.values[0])
    actual_end = pd.Timestamp(ds.time.values[-1])
    assert actual_start == pd.Timestamp(expected_time_start), (
        f"Start time mismatch: expected {expected_time_start}, got {actual_start}"
    )
    assert actual_end == pd.Timestamp(expected_time_end), (
        f"End time mismatch: expected {expected_time_end}, got {actual_end}"
    )

    # Check time is monotonically increasing with no duplicates
    time_series = pd.Series(ds.time.values)
    assert time_series.is_monotonic_increasing, "Time is not monotonically increasing!"
    assert not time_series.duplicated().any(), "Duplicate time values found!"

    # Check that data variables have no all-NaN time slices
    # (spot check first, middle, last time steps)
    check_indices = [0, actual_time_count // 2, actual_time_count - 1]
    for var_name in ds.data_vars:
        for idx in check_indices:
            slice_data = ds[var_name].isel(time=idx).values
            if np.all(np.isnan(slice_data)):
                print(f"  [WARNING] Variable '{var_name}' at time index {idx} "
                      f"({ds.time.values[idx]}) is entirely NaN.")

    # Check attributes
    assert "time_coverage_start" in ds.attrs, "Missing time_coverage_start attribute"
    assert "time_coverage_end" in ds.attrs, "Missing time_coverage_end attribute"

    # Check chunk uniformity per dimension
    for var_name in ds.data_vars:
        if ds[var_name].chunks is not None:
            dim_names = ["time", "latitude", "longitude"]
            for i, dim_name in enumerate(dim_names):
                chunks = ds[var_name].chunks[i]
                if len(chunks) > 1:
                    main_chunks = chunks[:-1]
                    expected = expected_chunk_sizes[dim_name]
                    if len(set(main_chunks)) > 1:
                        print(f"  [ERROR] Non-uniform {dim_name} chunks in "
                              f"'{var_name}': {chunks}")
                    elif main_chunks[0] != expected:
                        print(f"  [WARNING] {dim_name} chunk size {main_chunks[0]} "
                              f"differs from expected {expected}")
                    else:
                        print(f"  ✓ '{var_name}' {dim_name}: "
                              f"{chunks[0]} × {len(chunks)} chunks "
                              f"(last: {chunks[-1]})")
                else:
                    print(f"  ✓ '{var_name}' {dim_name}: single chunk ({chunks[0]})")

    ds.close()
    print(f"  [VALID] All checks passed.")
    return True


def combine_and_write_zarr(
    dataset_name: str,
    file_list: list[Path],
    output_dir: Path,
    overwrite: bool = False,
    time_chunk_size: int = 365,
    spatial_chunk_size: int = 128,
):
    """
    Open all files for a dataset, concatenate along time, rechunk uniformly,
    update attrs, and write to a Zarr store.
    """
    zarr_path = output_dir / f"{dataset_name}.zarr"

    if zarr_path.exists() and not overwrite:
        print(f"  [SKIP] {zarr_path} already exists. Use --overwrite to replace.")
        return

    print(f"  Opening {len(file_list)} files for '{dataset_name}'...")
    for f in file_list:
        print(f"    - {f.name}")

    # Open each file with chunking that aligns with our target
    # This helps dask plan the rechunking more efficiently
    datasets = []
    for f in file_list:
        ds = xr.open_dataset(
            f,
            chunks={
                "time": time_chunk_size,
                "latitude": spatial_chunk_size if spatial_chunk_size > 0 else -1,
                "longitude": spatial_chunk_size if spatial_chunk_size > 0 else -1,
            },
        )
        datasets.append(ds)

    # Sort datasets by their first time coordinate value
    datasets.sort(key=lambda ds: ds.time.values[0])

    # Concatenate along time dimension
    print(f"  Concatenating along time dimension...")
    combined = xr.concat(datasets, dim="time")

    # Verify time is monotonically increasing
    time_vals = combined.time.values
    if not (pd.Series(time_vals).is_monotonic_increasing):
        print(f"  [WARNING] Time is not monotonically increasing. Sorting...")
        combined = combined.sortby("time")
        time_vals = combined.time.values

    # Determine and apply chunk sizes
    chunk_sizes = determine_chunk_sizes(combined, time_chunk_size, spatial_chunk_size)

    print(f"  Rechunking to uniform chunks...")
    print_chunk_summary(combined, chunk_sizes)

    combined = combined.chunk(chunk_sizes)

    # Verify chunks are now uniform (quick check)
    for var_name in combined.data_vars:
        chunks = combined[var_name].data.chunks
        print(f"  Final dask chunks for '{var_name}':")
        print(f"    time: {chunks[0]}")
        print(f"    lat:  {chunks[1]}")
        print(f"    lon:  {chunks[2]}")

    # Update time-related attributes
    time_start = pd.Timestamp(time_vals[0]).isoformat() + "Z"
    time_end = pd.Timestamp(time_vals[-1]).isoformat() + "Z"

    combined.attrs["time_coverage_start"] = time_start
    combined.attrs["time_coverage_end"] = time_end

    # Update history attribute
    source_files = ", ".join(f.name for f in file_list)
    history_entry = f"Combined from yearly files: {source_files}"
    existing_history = combined.attrs.get("history", "")
    if existing_history:
        combined.attrs["history"] = f"{history_entry}; {existing_history}"
    else:
        combined.attrs["history"] = history_entry

    # Remove the Zarr store if overwriting
    if zarr_path.exists() and overwrite:
        import shutil
        shutil.rmtree(zarr_path)
        print(f"  Removed existing {zarr_path}")

    # Write to Zarr with encoding for better compression
    print(f"  Writing to {zarr_path}...")

    # Set up encoding for each data variable
    encoding = {}
    for var_name in combined.data_vars:
        encoding[var_name] = {
            "chunks": (
                chunk_sizes["time"],
                chunk_sizes["latitude"],
                chunk_sizes["longitude"],
            ),
        }

    combined.to_zarr(
        zarr_path,
        mode="w",
        consolidated=True,
        encoding=encoding,
    )

    # Close source datasets
    for ds in datasets:
        ds.close()

    # Validate the output
    expected_time_count = len(time_vals)
    validate_zarr(
        zarr_path,
        time_vals[0],
        time_vals[-1],
        expected_time_count,
        chunk_sizes,
    )

    print()


def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory:    {input_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Time chunk size:    {args.time_chunk_size}")
    print(f"Spatial chunk size: {args.spatial_chunk_size}")
    print()

    # Group files by dataset name
    groups = group_nc_files(input_dir)

    if not groups:
        print("No .nc files matching the pattern *_YYYY.nc found.")
        return

    print(f"Found {len(groups)} dataset group(s):")
    for name, files in groups.items():
        years = [f.stem.split("_")[-1] for f in files]
        print(f"  {name}: {len(files)} files ({', '.join(years)})")
    print()

    # Process each group
    for dataset_name, file_list in groups.items():
        print(f"{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")
        combine_and_write_zarr(
            dataset_name=dataset_name,
            file_list=file_list,
            output_dir=output_dir,
            overwrite=args.overwrite,
            time_chunk_size=args.time_chunk_size,
            spatial_chunk_size=args.spatial_chunk_size,
        )

    print("All done!")


if __name__ == "__main__":
    main()
