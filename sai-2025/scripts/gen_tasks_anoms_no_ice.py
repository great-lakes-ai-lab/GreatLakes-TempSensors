import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da
import gcsfs
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import shutil

import deepsensor.torch
from deepsensor.data import DataProcessor, TaskLoader, construct_circ_time_ds
from deepsensor.data.sources import get_era5_reanalysis_data, get_earthenv_auxiliary_data, \
    get_gldas_land_mask
from deepsensor.model import ConvNP
from deepsensor.train import Trainer, set_gpu_default_device

# Local package utilities
from deepsensor_greatlakes.utils import standardize_dates, generate_random_coordinates, apply_mask_to_prediction
from deepsensor_greatlakes.preprocessor import SeasonalCycleProcessor, list_saved_seasonal_cycles
from deepsensor_greatlakes.model import save_model, load_convnp_model

#set_gpu_default_device()

#Training/data config (adapted for Great Lakes)
data_range = ("2009-01-01", "2022-12-31")
#train_range = ("2009-01-01", "2021-12-31")
#val_range = ("2022-01-01", "2022-12-31")

train_range = ("2009-01-01", "2018-12-15")
val_range = ("2019-01-01", "2020-12-15")
test_range = ("2021-01-01", "2022-12-31")
date_subsample_factor = 10


# Path to the files on U-M HPC
bathymetry_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/bathymetry/interpolated_bathymetry.nc'
mask_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/masks/lakemask.nc'
#ice_concentration_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/ice_concentration_processed.zarr'
glsea_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/glsea_anom_processed.zarr'
glsea_raw_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA3_combined.zarr'

# Paths to saved configurations
deepsensor_folder = '../deepsensor_config/'

data_processor = DataProcessor(os.path.join(deepsensor_folder, "data_processor/no_ice/anoms"))


# Open the Zarr stores
#ice_concentration = xr.open_zarr(ice_concentration_path)
glsea = xr.open_zarr(glsea_path)
glsea_raw = xr.open_zarr(glsea_raw_path)

# Replace -1 (land value) with NaN
#ice_concentration = ice_concentration.where(ice_concentration != -1, float('nan'))

# Convert all times to date-only format, removing the time component
#ice_concentration = standardize_dates(ice_concentration)
glsea = standardize_dates(glsea)
glsea_raw = standardize_dates(glsea_raw)
#glsea3 = standardize_dates(glsea3


# Open the NetCDF files using xarray 
bathymetry_raw = xr.open_dataset(bathymetry_path)
lakemask_raw = xr.open_dataset(mask_path)


# process the bathymetry and lake
bathymetry, lakemask = data_processor([bathymetry_raw, lakemask_raw], method="min_max")


dates = pd.date_range(glsea.time.values.min(), glsea.time.values.max(), freq="D")
dates = pd.to_datetime(dates).normalize()  # This will set all times to 00:00:00

doy_ds = construct_circ_time_ds(dates, freq="D")
cos_D = standardize_dates(doy_ds["cos_D"])
sin_D = standardize_dates(doy_ds["sin_D"])


# Make auxiliary context dataset
aux_ds = xr.Dataset({
    "lakemask": lakemask["mask"],
    "bathymetry": bathymetry["bathymetry"],
    "cos_D": cos_D,
    "sin_D": sin_D,
})

# Initialize task loader
task_loader = TaskLoader(
    context = [glsea, aux_ds],
    target = glsea,
)

from tqdm import tqdm

# Function to generate tasks
def gen_tasks(dates, N=100, progress=True, lakemask_raw=None, data_processor=None):
    if lakemask_raw is None or data_processor is None:
        raise ValueError("You must pass both `lakemask_raw` and `data_processor`.")

    tasks = []
    for date in tqdm(dates, disable=not progress):
        # Generate a fresh set of random lake points for each date
        random_points = generate_random_coordinates(lakemask_raw, N, data_processor)
        
        # Sample the task
        task = task_loader(date, context_sampling=random_points, target_sampling="all")
        
        # Remove NaNs from the target
        task = task.remove_target_nans()
        
        tasks.append(task)
    
    return tasks

#save task to file
def save_task(task, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(task, f)

#write saved tasks to folder for access during training
def cache_tasks_to_disk(
    dates,
    outdir: str,
    N: int = 100,
    lakemask_raw=None,
    data_processor=None
):
    if lakemask_raw is None or data_processor is None:
        raise ValueError("You must pass both `lakemask_raw` and `data_processor`.")

    outdir = Path(outdir)
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for date in tqdm(dates, desc="Caching tasks to disk"):
        # build task
        #print(date)
        random_points = generate_random_coordinates(lakemask_raw, N, data_processor)
        task = task_loader(date, context_sampling=random_points, target_sampling="all")
        task = task.remove_target_nans()
        
        # make filename safe / deterministic
        fname = outdir / f"task_{date.strftime('%Y-%m-%d')}.pkl"
        save_task(task, fname)

# Generate training and validation dates
train_dates = pd.date_range(train_range[0], train_range[1])[::date_subsample_factor]
val_dates = pd.date_range(val_range[0], val_range[1])[::date_subsample_factor]

# Normalize to datetime64[D]
train_dates = pd.to_datetime(train_dates).normalize()
val_dates = pd.to_datetime(val_dates).normalize()

SCRATCH_DIR = "../../../../scratch/dannes_root/dannes0/saiavu/"
cache_tasks_to_disk(
    train_dates,
    outdir= SCRATCH_DIR + "anoms/no_ice/train_tasks",
    N=250,
    lakemask_raw=lakemask_raw,
    data_processor=data_processor,
)

cache_tasks_to_disk(
    val_dates,
    outdir= SCRATCH_DIR + "anoms/no_ice/val_tasks",
    N=250,
    lakemask_raw=lakemask_raw,
    data_processor=data_processor,
)


#task_loader.save(SCRATCH_DIR)






