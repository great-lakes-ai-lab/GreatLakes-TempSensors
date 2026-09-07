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


def generate_random_coordinates(mask_da, N, data_processor=None):
    """
    Generate N random coordinates (lat, lon) from a mask with values 1 inside
    the lake area, and normalize them using the DataProcessor if provided.

    Parameters
    ----------
    mask_da : xarray.Dataset
        Dataset containing a 'mask' variable (1 for valid, 0 for invalid areas).
    N : int
        Number of random points to generate.
    data_processor : DataProcessor, optional
        DataProcessor object for normalization.

    Returns
    -------
    numpy.ndarray
        Array of shape (2, N) with [lat, lon] coordinates.
    """
    mask = mask_da['mask'].values
    valid_indices = np.argwhere(mask == 1)

    random_indices = valid_indices[np.random.choice(valid_indices.shape[0], N, replace=False)]

    latitudes = mask_da['lat'].values[random_indices[:, 0]]
    longitudes = mask_da['lon'].values[random_indices[:, 1]]

    dummy_variable = np.random.rand(N)

    random_coords_df = pd.DataFrame({
        'lat': latitudes,
        'lon': longitudes,
        'dummy': dummy_variable,
    }).set_index(['lat', 'lon'])

    if data_processor:
        normalized_coords_df = data_processor(random_coords_df, method="min_max")
        return normalized_coords_df.index.to_frame(index=False).values.T
    else:
        return np.vstack((latitudes, longitudes))
