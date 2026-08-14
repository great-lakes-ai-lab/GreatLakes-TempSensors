# src/utils/metrics.py
"""Area-weighted, lake-only spatial metrics for skill-curve evaluation."""

import numpy as np
import xarray as xr


def lake_area_weights(
    da: xr.DataArray,
    lat_name: str = "lat",
    mask: xr.DataArray = None,
) -> xr.DataArray:
    """
    Build cos(latitude) area weights over the lake surface, normalized to sum to 1.

    Combines two effects:
      1. cos(lat) area correction — regular lat/lon cells shrink toward the poles;
         weighting by cos(lat) makes the mean area-correct rather than cell-count.
      2. Lake masking — land/invalid cells receive zero weight.

    Parameters
    ----------
    da : xr.DataArray
        Reference field defining the spatial grid. May include a 'time' dim.
        Used to infer the lake mask (valid-at-any-time) unless `mask` is given.
    lat_name : str
        Name of the latitude coordinate. Default "lat".
    mask : xr.DataArray, optional
        Explicit boolean 2D lake mask (True = lake). If None, derived from `da`
        as "valid at any time" (matches make_lake_mask_from_target).

    Returns
    -------
    xr.DataArray
        2D weights (lat, lon), zero over land, summing to 1 over the lake.
    """
    # 1. Derive a fixed lake mask if not supplied
    if mask is None:
        if "time" in da.dims:
            mask = da.notnull().any("time")
        else:
            mask = da.notnull()
    mask = mask.astype(bool)

    # 2. cos(lat) weights, broadcast to the 2D spatial grid
    cos_lat = np.cos(np.deg2rad(da[lat_name]))
    # Broadcast against the 2D spatial template (drop time if present)
    spatial_template = da.isel(time=0, drop=True) if "time" in da.dims else da
    w = cos_lat.broadcast_like(spatial_template)

    # 3. Zero out land (multiply by mask -> land becomes 0, not NaN)
    w = w.where(mask, 0.0)

    # 4. Normalize to sum to 1 over the lake
    total = w.sum()
    if float(total) == 0.0:
        raise ValueError(
            "lake_area_weights: total weight is zero (empty lake mask?). "
            "Check the reference field / mask."
        )
    w = w / total

    # Drop any lingering non-spatial coords for cleanliness
    return w