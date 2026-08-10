import fsspec
import xarray as xr

# 1. Define your geographic bounding box (Area of Interest)
LAT_MIN, LAT_MAX = 39.0, 41.0
LON_MIN, LON_MAX = -106.0, -104.0

# Define a tight time-slice to speed up the local download pipeline
START_DATE, END_DATE = "2020-01-01", "2020-01-07"

# 2. S3 Endpoint Storage Configuration (Public Anonymous Access)
storage_options = {
    "anon": True,
    "requester_pays": False,
    "client_kwargs": {"endpoint_url": "https://usgs.osn.mghpcc.org/"},
}

# 3. Establish direct paths to the target Zarr stores
# The BA daily store holds your adjusted T2
zarr_ba_url = "s3://hytest/conus404-biasadjusted/conus404-biasadjusted_daily.zarr"
# The Diagnostic store holds SWDOWN, GLW, U10, V10, Q2
zarr_diag_url = "s3://hytest/conus404/conus404_daily.zarr"

print("Streaming cloud metadata from USGS storage networks...")
store_ba = fsspec.get_mapper(zarr_ba_url, **storage_options)
store_diag = fsspec.get_mapper(zarr_diag_url, **storage_options)

# 4. Open the Zarr datasets as lazy Dask arrays without downloading yet
ds_ba = xr.open_zarr(store_ba, consolidated=True)
ds_diag = xr.open_zarr(store_diag, consolidated=True)

# 5. Extract variables and apply spatial masks
# Generate the mask based on the dataset's curvilinear coordinates
spatial_mask = (
    (ds_ba.lat >= LAT_MIN)
    & (ds_ba.lat <= LAT_MAX)
    & (ds_ba.lon >= LON_MIN)
    & (ds_ba.lon <= LON_MAX)
)

print(f"Slicing area and variables for {START_DATE} to {END_DATE}...")

# Pull T2 from the Bias-Adjusted Dataset
subset_ba = (
    ds_ba[["T2"]].where(spatial_mask, drop=True).sel(time=slice(START_DATE, END_DATE))
)

# Pull SWDOWN, GLW, U10, V10, Q2 from the Diagnostic Dataset
subset_diag = (
    ds_diag[["SWDOWN", "GLW", "U10", "V10", "Q2"]]
    .where(spatial_mask, drop=True)
    .sel(time=slice(START_DATE, END_DATE))
)

# 6. Merge the variables into one cohesive xarray Dataset
combined_ds = xr.merge([subset_ba, subset_diag])

print("Downloading and compiling local NetCDF file...")
combined_ds.to_netcdf("conus404_combined_forcing.nc")
print("Process finished successfully!")