# src/utils/coordinates.py
import numpy as np
import pandas as pd
import xarray as xr


def standardize_dates(ds):
    """
    Convert the 'time' dimension in an xarray dataset to date-only precision
    with datetime64[D].

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset whose 'time' dimension you wish to modify.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Modified dataset with time as datetime64[D].
    """
    if 'time' in ds.coords:
        ds['time'] = ds['time'].dt.floor('D').values.astype('datetime64[D]')
    return ds


def standardize_coords(ds):
    """
    Standardize spatial coordinates to:
    - names: lat/lon
    - longitude convention: -180 to 180
    - latitude ascending

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with spatial coordinates.

    Returns
    -------
    xarray.Dataset
        Dataset with standardized coordinate names and ordering.
    """
    lat_names = ["latitude", "y", "lat"]
    lon_names = ["longitude", "x", "lon"]

    rename_dict = {}

    for name in lat_names:
        if name in ds.dims or name in ds.coords:
            if name != "lat":
                rename_dict[name] = "lat"
            break

    for name in lon_names:
        if name in ds.dims or name in ds.coords:
            if name != "lon":
                rename_dict[name] = "lon"
            break

    if rename_dict:
        ds = ds.rename(rename_dict)

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError(
            f"Dataset must have lat/lon coordinates after standardization. "
            f"Found coords: {list(ds.coords)}"
        )

    # Convert 0-360 lon to -180 to 180
    if float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=(((ds["lon"] + 180) % 360) - 180))
        ds = ds.sortby("lon")

    # Ensure lat ascending
    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")

    return ds


def generate_random_coordinates(mask_da, N, data_processor=None, rng=None):
    """
    Draw N random lake points from a 1/0 sampling mask.

    Parameters
    ----------
    mask_da : xr.Dataset
        Must contain a 'mask' variable with dims (lat, lon), 1 = valid.
    N : int
        Number of points to draw, without replacement.
    data_processor : DataProcessor, optional
        If given, returned coords are normalised to x1/x2.
    rng : np.random.Generator | int | None
        Local random stream. None -> nondeterministic. An int seeds a fresh
        Generator; an existing Generator is used in place (and advanced), so
        callers can thread one stream through many calls.

    Returns
    -------
    np.ndarray, shape (2, N)
        Row 0 = x1 (lat), row 1 = x2 (lon). Normalised iff data_processor given.
    """

    rng = np.random.default_rng(rng)

    mask = mask_da['mask'].values
    valid_indices = np.argwhere(mask == 1)

    n_valid = valid_indices.shape[0]

    if N > n_valid:
        raise ValueError(
            f"generate_random_coordinates: requested N={N} points but the "
            f"sampling mask has only {n_valid} valid cells. Reduce N "
            f"(training.max_n_context / active_learning.n_context) or lower "
            f"the lake_mask coarsen_factor."
        )

    random_indices = valid_indices[rng.choice(n_valid, N, replace=False)]

    latitudes = mask_da['lat'].values[random_indices[:, 0]]
    longitudes = mask_da['lon'].values[random_indices[:, 1]]

    raw = np.stack([latitudes, longitudes])  # (2, N)

    if data_processor is not None:
        return data_processor.map_coord_array(raw, unnorm=False)

    return raw