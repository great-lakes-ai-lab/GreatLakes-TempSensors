import xarray as xr
import numpy as np
import pandas as pd
import zarr
from pathlib import Path


def fill_missing_dates_with_zero_ice(zarr_path, time_dim='time',
                                     data_var=None, nodata_value=None,
                                     dry_run=True):
    """
    Fill missing dates in an ice concentration Zarr store with arrays
    containing 0 (valid ice-free) where data could exist, and the nodata
    value where land is.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the Zarr store directory.
    time_dim : str
        Name of the time dimension (default: 'time').
    data_var : str, optional
        Name of the ice concentration variable. If None, auto-detects
        the first non-coordinate data variable.
    nodata_value : float or int, optional
        The nodata/fill value for land pixels. If None, attempts to
        detect from _FillValue, missing_value attributes, or the most
        common non-valid value.
    dry_run : bool
        If True, only reports what would be done without writing.
        Set to False to actually modify the Zarr store.

    Returns
    -------
    missing_dates : list
        List of dates that were (or would be) filled.
    """
    zarr_path = Path(zarr_path)

    print("=" * 70)
    print("FILL MISSING DATES — ICE CONCENTRATION ZARR")
    print("=" * 70)
    print(f"Zarr path: {zarr_path}")
    print(f"Dry run:   {dry_run}")

    # ================================================================
    # 1. Open the dataset
    # ================================================================
    ds = xr.open_zarr(zarr_path)

    # Auto-detect data variable
    if data_var is None:
        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            data_var = data_vars[0]
        else:
            print(f"  Multiple data variables found: {data_vars}")
            print(f"  Please specify data_var parameter.")
            return []

    print(f"Data var:  {data_var}")
    print(f"Time dim:  {time_dim}")
    print(f"Shape:     {ds[data_var].shape}")
    print(f"Dtype:     {ds[data_var].dtype}")

    # ================================================================
    # 2. Detect nodata value
    # ================================================================
    if nodata_value is None:
        # Try common attribute names
        var_attrs = ds[data_var].attrs
        for attr_name in ['_FillValue', 'missing_value', 'nodata', 'fill_value']:
            if attr_name in var_attrs:
                nodata_value = var_attrs[attr_name]
                print(f"Nodata:    {nodata_value} (from attribute '{attr_name}')")
                break

        if nodata_value is None:
            # Try encoding
            if '_FillValue' in ds[data_var].encoding:
                nodata_value = ds[data_var].encoding['_FillValue']
                print(f"Nodata:    {nodata_value} (from encoding '_FillValue')")

    if nodata_value is None:
        print("\n  WARNING: Could not auto-detect nodata value.")
        print("  Please specify nodata_value parameter.")
        print("  Common values: -1, 255, -9999, np.nan")
        return []
    else:
        print(f"Nodata:    {nodata_value}")

    # ================================================================
    # 3. Create the land mask from an existing valid slice
    # ================================================================
    print("\n" + "-" * 70)
    print("BUILDING LAND MASK")
    print("-" * 70)

    # Load a single time slice to determine the land mask
    # Use the first available time step
    sample_slice = ds[data_var].isel({time_dim: 0}).load()

    if np.isnan(nodata_value):
        land_mask = np.isnan(sample_slice.values)
    else:
        land_mask = (sample_slice.values == nodata_value)

    # Verify with a second slice to ensure consistency
    if ds.dims[time_dim] > 1:
        sample_slice_2 = ds[data_var].isel({time_dim: -1}).load()
        if np.isnan(nodata_value):
            land_mask_2 = np.isnan(sample_slice_2.values)
        else:
            land_mask_2 = (sample_slice_2.values == nodata_value)

        if np.array_equal(land_mask, land_mask_2):
            print("  ✓ Land mask verified (consistent between first and last time step)")
        else:
            # Use the intersection (pixels that are ALWAYS nodata = land)
            land_mask = land_mask & land_mask_2
            # Check a few more slices to be safe
            check_indices = np.linspace(0, ds.dims[time_dim] - 1,
                                        min(10, ds.dims[time_dim]), dtype=int)
            for idx in check_indices:
                sl = ds[data_var].isel({time_dim: int(idx)}).load()
                if np.isnan(nodata_value):
                    land_mask = land_mask & np.isnan(sl.values)
                else:
                    land_mask = land_mask & (sl.values == nodata_value)
            print(f"  ⚠ Land mask built from intersection of {len(check_indices)} slices")

    n_land = land_mask.sum()
    n_water = (~land_mask).sum()
    n_total = land_mask.size
    print(f"  Land pixels:   {n_land:,} ({100 * n_land / n_total:.1f}%)")
    print(f"  Water pixels:  {n_water:,} ({100 * n_water / n_total:.1f}%)")

    # ================================================================
    # 4. Build the fill array (0 on water, nodata on land)
    # ================================================================
    spatial_dims = [d for d in ds[data_var].dims if d != time_dim]
    spatial_shape = tuple(ds.dims[d] for d in spatial_dims)

    fill_array = np.zeros(spatial_shape, dtype=ds[data_var].dtype)
    fill_array[land_mask] = nodata_value

    print(f"\n  Fill array constructed: shape={spatial_shape}, dtype={fill_array.dtype}")
    print(f"  Water pixels → 0")
    print(f"  Land pixels  → {nodata_value}")

    # ================================================================
    # 5. Identify missing dates
    # ================================================================
    print("\n" + "-" * 70)
    print("IDENTIFYING MISSING DATES")
    print("-" * 70)

    existing_times = pd.DatetimeIndex(ds[time_dim].values)
    existing_dates = existing_times.normalize().unique().sort_values()

    # Full date range
    date_start = existing_dates.min()
    date_end = existing_dates.max()
    full_range = pd.date_range(start=date_start, end=date_end, freq='D')

    missing_dates = full_range.difference(existing_dates)

    print(f"  Date range:      {date_start.date()} to {date_end.date()}")
    print(f"  Expected dates:  {len(full_range):,}")
    print(f"  Present dates:   {len(existing_dates):,}")
    print(f"  Missing dates:   {len(missing_dates):,}")

    if len(missing_dates) == 0:
        print("\n  ✓ No missing dates! Dataset is temporally complete.")
        ds.close()
        return []

    # Show missing by year
    missing_series = pd.Series(missing_dates)
    missing_by_year = missing_series.groupby(missing_series.dt.year).count()
    print(f"\n  Missing dates by year:")
    for year, count in missing_by_year.items():
        days_in_year = 366 if pd.Timestamp(year=year, month=1, day=1).is_leap_year else 365
        print(f"    {year}: {count:>4} missing out of {days_in_year}")

    # ================================================================
    # 6. Fill missing dates
    # ================================================================
    if dry_run:
        print("\n" + "-" * 70)
        print("DRY RUN — No data written")
        print("-" * 70)
        print(f"  Would fill {len(missing_dates)} dates with zero-ice arrays.")
        print(f"  Set dry_run=False to write to the Zarr store.")

        # Show first few dates that would be filled
        n_show = min(20, len(missing_dates))
        print(f"\n  First {n_show} dates that would be filled:")
        for d in missing_dates[:n_show]:
            print(f"    {d.date()}")
        if len(missing_dates) > n_show:
            print(f"    ... and {len(missing_dates) - n_show} more")

        ds.close()
        return missing_dates.tolist()

    # Actually write the data
    print("\n" + "-" * 70)
    print("WRITING MISSING DATES TO ZARR")
    print("-" * 70)

    ds.close()  # Close the read handle

    # Build a new dataset with all missing dates
    # Process in batches to manage memory
    batch_size = 100
    n_batches = int(np.ceil(len(missing_dates) / batch_size))

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(missing_dates))
        batch_dates = missing_dates[batch_start:batch_end]

        # Create array for this batch: (n_times, *spatial_dims)
        batch_shape = (len(batch_dates),) + spatial_shape
        batch_data = np.broadcast_to(fill_array, batch_shape).copy()

        # Build xarray dataset for this batch
        coords = {time_dim: batch_dates.values}

        # Reconstruct spatial coordinates
        orig_ds = xr.open_zarr(zarr_path)
        for dim in spatial_dims:
            if dim in orig_ds.coords:
                coords[dim] = orig_ds.coords[dim].values
        orig_encoding = orig_ds[data_var].encoding.copy()
        orig_attrs = orig_ds[data_var].attrs.copy()
        orig_ds.close()

        dims = [time_dim] + spatial_dims
        fill_da = xr.DataArray(
            data=batch_data,
            dims=dims,
            coords=coords,
            attrs=orig_attrs
        )
        fill_ds = xr.Dataset({data_var: fill_da})

        # Append to Zarr
        fill_ds.to_zarr(
            zarr_path,
            mode='a',
            append_dim=time_dim
        )

        print(f"  Batch {batch_idx + 1}/{n_batches}: "
              f"wrote {len(batch_dates)} dates "
              f"({batch_dates[0].date()} to {batch_dates[-1].date()})")

    # ================================================================
    # 7. Re-sort the time dimension
    # ================================================================
    print("\n  Sorting time dimension...")
    ds_final = xr.open_zarr(zarr_path)

    # Check if time is already sorted
    times = ds_final[time_dim].values
    if not np.all(times[:-1] <= times[1:]):
        print("  Time is out of order — re-sorting and rewriting...")
        ds_sorted = ds_final.sortby(time_dim)

        # Write sorted dataset to a temporary zarr, then replace
        temp_path = zarr_path.parent / (zarr_path.name + "_sorted_temp")

        # Preserve chunking
        chunks = ds_final.chunks
        if chunks:
            ds_sorted = ds_sorted.chunk(chunks)

        ds_sorted.to_zarr(temp_path, mode='w')
        ds_final.close()
        ds_sorted.close()

        # Replace original with sorted version
        import shutil
        shutil.rmtree(zarr_path)
        temp_path.rename(zarr_path)
        print("  ✓ Zarr store re-sorted by time.")
    else:
        print("  ✓ Time is already sorted (appended dates are after existing).")
        ds_final.close()

    # ================================================================
    # 8. Verify
    # ================================================================
    print("\n" + "-" * 70)
    print("VERIFICATION")
    print("-" * 70)
    ds_check = xr.open_zarr(zarr_path)
    new_times = pd.DatetimeIndex(ds_check[time_dim].values)
    new_dates = new_times.normalize().unique().sort_values()
    still_missing = full_range.difference(new_dates)

    print(f"  Dates now present: {len(new_dates):,}")
    print(f"  Still missing:     {len(still_missing):,}")

    if len(still_missing) == 0:
        print("  ✓ Dataset is now temporally complete!")
    else:
        print(f"  ⚠ {len(still_missing)} dates still missing")

    ds_check.close()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return missing_dates.tolist()


# ====================================================================
# USAGE
# ====================================================================
if __name__ == "__main__":
    zarr_path = "/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/ice_concentration.zarr"

    # First, do a dry run to see what would happen
    missing = fill_missing_dates_with_zero_ice(
        zarr_path,
        time_dim='time',  # adjust if your time dim has a different name
        data_var=None,  # auto-detect, or specify e.g. 'ice_concentration'
        nodata_value=None,  # auto-detect, or specify e.g. -1, 255, np.nan
        dry_run=False  # SET TO FALSE TO ACTUALLY WRITE
    )

    # When ready, run for real:
    # missing = fill_missing_dates_with_zero_ice(
    #     zarr_path,
    #     dry_run=False
    # )