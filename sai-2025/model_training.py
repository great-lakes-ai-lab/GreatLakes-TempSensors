import deepsensor.torch
from deepsensor.data import DataProcessor, TaskLoader
from deepsensor.model import ConvNP
from deepsensor.train import Trainer

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm import notebook
from deepsensor.train import set_gpu_default_device
from xarray.groupers import TimeResampler

import torch

import subprocess
import time
import wandb

def compute_val_rmse(model, val_tasks):
    errors = []
    target_var_ID = task_loader.target_var_IDs[0][0]
    for task in val_tasks:
        #mean = torch.tensor(data_processor.map_array(model.mean(task), target_var_ID, unnorm=True), device=device)
        #true = torch.tensor(data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True), device=device)
        #errors.extend(torch.abs(mean - true).cpu().numpy())  # Convert back to NumPy if needed
        mean = data_processor.map_array(model.mean(task), target_var_ID, unnorm=True)
        true = data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
        errors.extend(np.abs(mean - true))
    return np.sqrt(np.mean(np.concatenate(errors) ** 2))

    
def gen_tasks(dates, progress=True):
    tasks = []
    for date in notebook.tqdm(dates, disable=not progress):
        N_c = np.random.randint(0, 500)
        task = task_loader(date, context_sampling=[N_c, "all", "all", N_c], target_sampling="all")
        tasks.append(task)
    return tasks

def transform_ice(da):
    da = xr.DataArray(da)  # Ensure input is always an xarray.DataArray
    nan_mask = da.isnull()  # This correctly creates a mask in xarray
    transformed = xr.where(da > 0.2, 0, 1)  # Apply thresholding
    transformed = transformed.where(~nan_mask, np.nan)  # Preserve NaNs
    return transformed

set_gpu_default_device()
scratch_path = "/scratch/dannes_root/dannes0/saiavu"

start_year = 2007
end_year = 2023
start_dt = str(start_year) + "-01-01T12:00:00.000000000"
end_dt = str(end_year) + "-12-31T12:00:00.000000000"

bathymetry_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/bathymetry/interpolated_bathymetry.nc'
mask_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/masks/lakemask.nc'

mask = xr.open_dataset(mask_path)
bath = xr.open_dataset(bathymetry_path)


fpath = [f'/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA3_NETCDF/GLSEA3_{date}.nc' for date in range(start_year, end_year+1)]

# Load dataset using xarray with chunking
glsea3_raw = xr.open_mfdataset(
    fpath, combine="by_coords", parallel=True, chunks={"time": 3, "lat": 512, "lon": 512}
).drop_vars('crs', errors="ignore")

# Apply transform_ice function while keeping it as xarray
ice_mask = xr.apply_ufunc(
    transform_ice,
    glsea3_raw["sst"],  # Ensure function applies only to 'sst' DataArray
    dask="allowed",  # Ensures correct Dask processing
    output_dtypes=[glsea3_raw["sst"].dtype],
    keep_attrs=True,  # Preserve metadata
)

ice_mask = ice_mask.rename("binary_ice_indicator")

#glsea3_new = xr.apply_ufunc(transform_nan, glsea3_raw, dask="allowed", output_dtypes=[glsea3_raw["sst"].dtype])
glsea3_new = glsea3_raw.where(np.isnan(glsea3_raw.sst) == False, -0.009) 

climatology = glsea3_new.groupby("time.dayofyear").mean("time")

anomalies = glsea3_new.groupby("time.dayofyear") - climatology

anomalies = anomalies.chunk({"time": 3, "lat": 512, "lon": 512})
ice_mask = ice_mask.chunk({"time": 3, "lat": 512, "lon": 512})



data_processor = DataProcessor(x1_name="lat", x2_name="lon")

mask_ds = data_processor(mask)

bath_ds = data_processor(bath)

_ = data_processor(anomalies.sel(time = slice("2007-01-01T12:00:00.000000000", "2007-04-01T12:00:00.000000000")))
anom_ds = data_processor(anomalies)

_ = data_processor(ice_mask.sel(time = slice("2007-01-01T12:00:00.000000000", "2007-04-01T12:00:00.000000000")))
ice_ds = data_processor(ice_mask)

data_processor.save(scratch_path + "/deepsensor_config/")


task_loader = TaskLoader(
    context = [anom_ds, mask_ds, bath_ds, ice_ds],
    target = anom_ds, 
)

set_gpu_default_device()


val_dates = pd.date_range(str(end_year-1) + "-01-01T12:00:00.000000000", str(end_year) + "-12-31T12:00:00.000000000")[::5]
val_tasks = gen_tasks(val_dates)
model = ConvNP(data_processor, task_loader)

losses = []
val_rmses = []
train_range = pd.date_range(str(start_year) + "-01-01T12:00:00.000000000", str(end_year-2) + "-12-31T12:00:00.000000000")

 

val_rmse_best = np.inf
trainer = Trainer(model, lr=2e-5)

epochs = 25

run = wandb.init(
  # Set the project where this run will be logged
  project="deepsensor-greatlakes",
  # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
  name=f"full_training_run1",
  # Track hyperparameters and run metadata
  config={
  "contexts": "sst anomalies, land mask, bathymetry mask, ice mask",
  "sampling": "anomalies and ice randomly [0,500]",
  "years": "2007-2023",
  "epochs": "25",
  "sampling": "15"
  })

for epoch in range(epochs):
    train_tasks = gen_tasks(train_range[::15], progress=True)

    batch_losses = trainer(train_tasks, progress_bar = True)
    mean_loss = np.mean(batch_losses)
    losses.append(mean_loss)
    try:
        run.log({"loss": mean_loss})
    except:
        print("error logging loss")

    val_rmse = compute_val_rmse(model, val_tasks)
    val_rmses.append(val_rmse)
    run.log({"val_rmse": val_rmse})
    if val_rmses[-1] < val_rmse_best:
        val_rmse_best = val_rmses[-1]
        model.save(scratch_path + "/model")
        try:
            run.log_artifact(scratch_path + "/model", name="trained-model", type="model")
        except:
            print("error logging model")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses)
axes[1].plot(val_rmses)
_ = axes[0].set_xlabel("Epoch")
_ = axes[1].set_xlabel("Epoch")
_ = axes[0].set_title("Training Cost")
_ = axes[1].set_title("Validation RMSE")
plt.savefig(scratch_path + 'training_20.png')
try:
    wandb.log({"Training/Validation Cost": plt})
except:
    print("error logging plot")

run.finish()
