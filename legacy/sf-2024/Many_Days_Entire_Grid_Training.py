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

def compute_val_rmse(model, val_tasks):
    errors = []
    target_var_ID = task_loader.target_var_IDs[0][0]  # assume 1st target set and 1D
    for task in val_tasks:
#         print("im in for loop")
        mean = data_processor.map_array(model.mean(task), target_var_ID, unnorm=True)
#         print("mean calc")
        true = data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
#         print("true calc")
        errors.extend(np.abs(mean - true))
    return np.sqrt(np.mean(np.concatenate(errors) ** 2))
def gen_tasks(dates, progress=True):
    tasks = []
    for date in notebook.tqdm(dates, disable=not progress):
#         N_c = np.random.randint(0, 500)
        N_c = np.random.randint(0, 500)
        task = task_loader(date, context_sampling=[N_c, "all", "all"], target_sampling="all")
        tasks.append(task)
    return tasks

fpath = ['/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA3_NETCDF/GLSEA3_' + str(date) + '.nc' for date in range(2007,2022)]
set_gpu_default_device()
mask = xr.open_dataset('/home/erinredd/CIGLRProj/lakemask2.nc')
bath = xr.open_dataset('/home/erinredd/interpolated_bathymetry.nc')
dat = xr.open_mfdataset(fpath, concat_dim='time', combine='nested')
mdat = dat.where(np.isnan(dat.sst) == False, -0.009)
climatology = mdat.groupby('time.dayofyear').mean('time')
anomalies = mdat.groupby('time.dayofyear') - climatology
anomalies = anomalies.sel(time = slice('2007-01-01T12:00:00.000000000', '2016-12-31T12:00:00.000000000'))

data_processor = DataProcessor(x1_name="lat", x2_name="lon")
mask_ds = data_processor(mask)
anom_ds = data_processor(anomalies)
bath_ds = data_processor(bath)
task_loader = TaskLoader(
    context = [anom_ds, mask_ds, bath_ds],
    target = anom_ds, 
)

val_dates = pd.date_range('2015-01-01T12:00:00.000000000', '2016-12-31T12:00:00.000000000')[::5]
val_tasks = gen_tasks(val_dates)
model = ConvNP(data_processor, task_loader)

losses = []
val_rmses = []
train_range = pd.date_range('2007-01-01T12:00:00.000000000', '2014-12-31T12:00:00.000000000')



val_rmse_best = np.inf
trainer = Trainer(model, lr=2e-5)
deepsensor_folder = "/home/erinredd/deepsensor_config1/"


for epoch in range(25):
#     print("step1")
    train_tasks = gen_tasks(train_range[::5], progress=False)

    batch_losses = trainer(train_tasks)
    losses.append(np.mean(batch_losses))

    val_rmses.append(compute_val_rmse(model, val_tasks))
    if val_rmses[-1] < val_rmse_best:
        val_rmse_best = val_rmses[-1]
        model.save(deepsensor_folder)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses)
axes[1].plot(val_rmses)
_ = axes[0].set_xlabel("Epoch")
_ = axes[1].set_xlabel("Epoch")
_ = axes[0].set_title("Training Cost")
_ = axes[1].set_title("Validation RMSE")
plt.savefig('training.png')


# pred_val = model.predict(val_tasks, X_t = mask) 
# predxr = pred_val['sst']
# predxr = predxr.rename({'mean':'mn','std': 'sd'})
# predxr.to_netcdf("wholepredictions.nc")