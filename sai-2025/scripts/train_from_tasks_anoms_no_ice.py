import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da
import gcsfs
import os
from pathlib import Path
import wandb
import pickle
import torch
from typing import Optional

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


try:
    set_gpu_default_device()
    device = "cuda"
except:
    print("No GPU runtime")
    device = "cpu"


SCRATCH_DIR = "../../../../scratch/dannes_root/dannes0/saiavu/"
output_dir = "../../models/"

#Load Task Loader from saved file path
#task_loader = TaskLoader(SCRATCH_DIR)

training_run_name = "anoms_no_ice_11_12"


# Paths to saved configurations
deepsensor_folder = '../deepsensor_config/'

data_processor = DataProcessor(os.path.join(deepsensor_folder, "data_processor"))


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
#glsea_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA_combined.zarr'
glsea_raw_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA3_combined.zarr'


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


from tqdm.notebook import tqdm 
# ---------- IO helpers ----------

def load_task(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def list_task_files(outdir: str):
    # deterministic order; we shuffle per-epoch
    return sorted(Path(outdir).glob("task_*.pkl"))


# ---------- Validation (GPU-aware, with notebook progress) ----------

def compute_val_rmse(
    model,
    val_tasks,
    data_processor,
    task_loader,
    device: str | torch.device = None,
    show_progress: bool = True,
) -> float:
    """
    GPU-aware RMSE for DeepSensor-style tasks.
    Shows a tqdm notebook progress bar over validation tasks.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    target_var_ID = task_loader.target_var_IDs[0][0]
    sq_errors = []

    try:
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.eval()
    except Exception as e:
        print(f"Warning: could not set model to eval mode ({e})")
    with torch.no_grad():
        iterator = tqdm(val_tasks, desc="Validating", leave=False) if show_progress else val_tasks
        for task in iterator:
            # Predict
            pred = model.mean(task)

            print(pred)

            # Map/unnormalize → may return torch.Tensor or np.ndarray
            pred_mapped = data_processor.map_array(pred, target_var_ID, unnorm=True)
            true_mapped = data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)

            # Ensure torch tensors on the selected device
            if isinstance(pred_mapped, np.ndarray):
                pred_t = torch.from_numpy(pred_mapped).to(device=device, dtype=torch.float32)
            else:
                pred_t = pred_mapped.to(device=device, dtype=torch.float32)

            if isinstance(true_mapped, np.ndarray):
                true_t = torch.from_numpy(true_mapped).to(device=device, dtype=torch.float32)
            else:
                true_t = true_mapped.to(device=device, dtype=torch.float32)

            diff2 = (pred_t - true_t).pow(2).flatten()
            sq_errors.append(diff2)

        all_sq = torch.cat(sq_errors, dim=0)
        rmse = torch.sqrt(all_sq.mean())

    return float(rmse.item())


# ---------- Training from disk using Trainer (with notebook tqdm + batching) ----------

def train_from_disk(
    model,
    train_task_dir: str,
    val_task_dir: str,
    trainer,                           # instance of your Trainer class
    data_processor,
    task_loader,
    device: str | torch.device,
    *,
    epochs: int = 50,
    load_chunk_files: int = 128,        # how many task files to load from disk at a time
    trainer_batch_size: Optional[int] = 16,  # batch size passed into Trainer (None = no batching)
    run=None,                          # e.g., wandb-like logger (optional)
    output_dir: str = ".",
    model_name: str = "model",
):
    """
    Stream tasks from disk in chunks; delegate batching + backprop to Trainer.
    Shows notebook progress bars for epochs and validation.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)

    train_files = list_task_files(train_task_dir)
    val_files = list_task_files(val_task_dir)

    # Load validation tasks once (kept small ideally)
    val_tasks = [load_task(p) for p in val_files]

    losses = []
    val_rmses = []
    val_rmse_best = np.inf

    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_losses = []

        # Shuffle file order each epoch
        np.random.shuffle(train_files)

        # Stream train tasks in chunks from disk
        for i in tqdm(range(0, len(train_files), load_chunk_files), desc="Batches", leave=False):
            file_chunk = train_files[i : i + load_chunk_files]
            train_tasks = [load_task(p) for p in file_chunk]

            # Optional: shuffle within the chunk so Trainer sees randomized mini-batches
            np.random.shuffle(train_tasks)

            # Delegate to your Trainer (it will shuffle again, batch via concat_tasks, and backprop)
            batch_losses = trainer(
                train_tasks,
                batch_size=None,   # batching handled inside train_epoch
                progress_bar=True,               # show per-batch bar inside Trainer
                tqdm_notebook=True,              # use notebook widget
            )

            mean_batch_loss = float(np.mean(batch_losses)) if len(batch_losses) > 0 else float("nan")
            epoch_losses.append(mean_batch_loss)

            if run is not None:
                run.log({"batch_loss": mean_batch_loss})

            # Free memory
            del train_tasks

        # Epoch-level loss
        epoch_loss = float(np.nanmean(epoch_losses)) if len(epoch_losses) > 0 else float("nan")
        losses.append(epoch_loss)

        # Validation (GPU-aware) with its own small progress bar
        print("here")
        val_rmse = compute_val_rmse(
            model,
            val_tasks,
            data_processor=data_processor,
            task_loader=task_loader,
            device=device,
            show_progress=True,
        )
        val_rmses.append(val_rmse)

        if run is not None:
            run.log({"epoch_loss": epoch_loss, "val_rmse": val_rmse})

        # Save best model
        if val_rmse < val_rmse_best:
            val_rmse_best = val_rmse
            # DeepSensor model save (path without extension is fine if your .save handles it)
            model.save(model_path)
            if run is not None:
                run.log_artifact(model_path, name=training_run_name, type="model")


    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses)
    axes[1].plot(val_rmses)
    _ = axes[0].set_xlabel("Epoch")
    _ = axes[1].set_xlabel("Epoch")
    _ = axes[0].set_title("Training Cost")
    _ = axes[1].set_title("Validation RMSE")
    plt.savefig(output_dir + training_run_name + "_plot")
    try:
        run.log({"Training/Validation Cost": plt})
    except:
        print("error logging plot")

    return losses, val_rmses


model = ConvNP(data_processor, task_loader)

learning_rate = 5e-5
epochs = 50
batch_size = 16

trainer = Trainer(model, lr=learning_rate)

run = wandb.init(
  # Set the project where this run will be logged
  project="deepsensor-greatlakes",
  name= training_run_name,
  # Track hyperparameters and run metadata
  config={
  "contexts": "sst anomalies, land mask, bathymetry mask, ice mask",
  "sampling": "anomalies and ice randomly [0,500]",
  "train_years": "2009-2018",
  "val_years": "2019-2020",   
  "epochs": str(epochs),
  "task_sampling": str(10),    
  "batch_size" : str(batch_size),
  "learning_rate" : str(learning_rate),
  "cross_val": "No"
  })

train_from_disk(model, SCRATCH_DIR + "anoms/no_ice/train_tasks", SCRATCH_DIR + "anoms/no_ice/val_tasks",
                trainer, data_processor, task_loader, device, epochs = epochs, trainer_batch_size = batch_size,
                run = run, output_dir =  output_dir, model_name =  training_run_name + "_model")

run.finish()




