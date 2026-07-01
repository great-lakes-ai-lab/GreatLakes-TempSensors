# This directory contains scripts needed to obtain the data for running the greatlakes SST pipeline. 

/era5
If you want to use ERA5 variables you can download them from the arraylake public access. You'll still 
need to make an arraylake account and setup tokens. This dataset is optional but seems to be helpful. 
Will come natively as a zarr

/glsea_download
In here are three scripts to download raw .nc for GLSEA sst, GLSEA3 sst, and GLSEA ice concentrations. It will 
download a data_YYYY.nc. They come year by year to bypass issues with data volume and requests timeouts.

netCDF_dir_to_zarr_stores.py
Use this script to convert the individual years to an efficient zarr store for each of the datasets


/fill_ice_concentration_zarr_missing_dates.py
Because the ice_concentration data doesn't create data when there is no ice cover the data doesn't have the temporal 
density it should. This script will create the datatime indexes with a dummy values (0 for ice_concentration on the lakes,
NaN for on land). NOTE: uses the NaN values from the first temporal slice, which is consistent through the time series
TO USE: update `zarr_path` in the main entry point part
