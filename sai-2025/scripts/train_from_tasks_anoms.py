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

training_run_name = "anoms_ice_11_13"


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
ice_concentration_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/ice_concentration_processed.zarr'
glsea_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/glsea_anom_processed.zarr'
#glsea_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA_combined.zarr'
glsea_raw_path = '/nfs/turbo/seas-dannes/SST-sensor-placement-input/GLSEA3_combined.zarr'


# Open the Zarr stores
ice_concentration = xr.open_zarr(ice_concentration_path)
glsea = xr.open_zarr(glsea_path)
glsea_raw = xr.open_zarr(glsea_raw_path)

# Replace -1 (land value) with NaN
ice_concentration = ice_concentration.where(ice_concentration != -1, float('nan'))

# Convert all times to date-only format, removing the time component
ice_concentration = standardize_dates(ice_concentration)
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
    context = [glsea, ice_concentration, aux_ds],
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




def compute_val_nll(
    model,
    val_tasks,
    data_processor,
    task_loader,
    device: str | torch.device = None,
    show_progress: bool = True,
) -> float:
    """
    Compute the joint Gaussian Negative Log Likelihood (NLL) over validation tasks.
    Compatible with DeepSensor-style models that return predictive mean and std.

    NLL = 0.5 * sum(log(2πσ²) + ((y - μ)² / σ²))

    Parameters
    ----------
    model : deepsensor.model.DeepSensorModel
        Trained DeepSensor model with a .predict or .mean_std method.
    val_tasks : list
        List of DeepSensor tasks for validation.
    data_processor : deepsensor.data.DataProcessor
        Handles normalization / unnormalization.
    task_loader : deepsensor.data.TaskLoader
        Provides variable IDs and metadata.
    device : str | torch.device, optional
        Torch device for GPU or CPU evaluation.
    show_progress : bool
        Whether to show tqdm progress bar.

    Returns
    -------
    float
        Average Gaussian NLL over all validation tasks.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    target_var_ID = task_loader.target_var_IDs[0][0]
    nll_terms = []

    try:
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.eval()
    except Exception as e:
        print(f"Warning: could not set model to eval mode ({e})")

    with torch.no_grad():
        iterator = tqdm(val_tasks, desc="Validating (NLL)", leave=False) if show_progress else val_tasks

        for task in iterator:
            # Predictive mean and std from model
            pred = model.predict(task)
            mu = pred.mean[target_var_ID]
            sigma = pred.std[target_var_ID]

            # Unnormalize (map back to physical units)
            mu_mapped = data_processor.map_array(mu, target_var_ID, unnorm=True)
            sigma_mapped = data_processor.map_array(sigma, target_var_ID, unnorm=True)
            true_mapped = data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)

            # Convert to torch tensors
            def to_torch(x):
                return torch.from_numpy(x).to(device=device, dtype=torch.float32) if isinstance(x, np.ndarray) else x.to(device=device, dtype=torch.float32)

            mu_t = to_torch(mu_mapped)
            sigma_t = to_torch(sigma_mapped)
            true_t = to_torch(true_mapped)

            # Stabilize σ (avoid log(0))
            var_t = torch.clamp(sigma_t.pow(2), min=1e-6)

            # Per-point NLL: 0.5 * [ log(2πσ²) + ((y - μ)² / σ²) ]
            nll = 0.5 * (torch.log(2 * torch.pi * var_t) + (true_t - mu_t).pow(2) / var_t)

            nll_terms.append(nll.flatten())

        # Combine all validation points
        all_nll = torch.cat(nll_terms, dim=0)

        # Compute joint (sum) or mean NLL
        mean_nll = all_nll.mean()

    return float(mean_nll.item())


# assumes you already have:
# - list_task_files, load_task
# - compute_val_rmse (your function)
# - compute_val_nll  (the NLL function we just wrote)

def train_from_disk(
    model,
    train_task_dir: str,
    val_task_dir: str,
    trainer,                                # instance of your Trainer class
    data_processor,
    task_loader,
    device: str | torch.device,
    *,
    epochs: int = 50,
    load_chunk_files: int = 128,            # how many task files to load from disk at a time
    trainer_batch_size: int | None = 16,    # batch size passed into Trainer (None = no batching)
    run=None,                               # e.g., wandb-like logger (optional)
    output_dir: str = ".",
    model_name: str = "model",
    patience: int = 5,                      # stop if no RMSE or NLL improvement for this many epochs
    plot_name: str = "training_curves"      # filename stem for saved plot
):
    """
    Stream tasks from disk in chunks; delegate batching + backprop to Trainer.
    Logs val_RMSE and val_NLL; early-stops when neither improves for `patience` epochs.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)

    train_files = list_task_files(train_task_dir)
    val_files = list_task_files(val_task_dir)

    # Load validation tasks once (keep this reasonably small)
    val_tasks = [load_task(p) for p in val_files]

    losses = []
    val_rmses = []
    val_nlls = []

    best_rmse = np.inf
    best_nll = np.inf
    epochs_since_improve = 0

    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_losses = []

        # Shuffle training files each epoch
        np.random.shuffle(train_files)

        # ===== Train over streamed chunks =====
        for i in tqdm(range(0, len(train_files), load_chunk_files), desc="Batches", leave=False):
            file_chunk = train_files[i : i + load_chunk_files]
            train_tasks = [load_task(p) for p in file_chunk]
            np.random.shuffle(train_tasks)

            # Delegate to your Trainer (handles batching + backprop internally)
            batch_losses = trainer(
                train_tasks,
                batch_size=trainer_batch_size,   # was None; now respects the function arg
                progress_bar=True,
                tqdm_notebook=True,
            )

            mean_batch_loss = float(np.mean(batch_losses)) if len(batch_losses) > 0 else float("nan")
            epoch_losses.append(mean_batch_loss)

            if run is not None:
                run.log({"batch_loss": mean_batch_loss})

            # Free memory
            del train_tasks

        # Epoch-level (training) loss
        epoch_loss = float(np.nanmean(epoch_losses)) if len(epoch_losses) > 0 else float("nan")
        losses.append(epoch_loss)

        # ===== Validation (RMSE + NLL) =====
        val_rmse = compute_val_rmse(
            model,
            val_tasks,
            data_processor=data_processor,
            task_loader=task_loader,
            device=device,
            show_progress=True,
        )
        val_rmses.append(val_rmse)

        val_nll = compute_val_nll(
            model,
            val_tasks,
            data_processor=data_processor,
            task_loader=task_loader,
            device=device,
            show_progress=True,
        )
        val_nlls.append(val_nll)

        if run is not None:
            run.log({
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "val_rmse": val_rmse,
                "val_nll": val_nll
            })

        # ===== Check improvements & save =====
        improved = False
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            improved = True
        if val_nll < best_nll:
            best_nll = val_nll
            improved = True

        if improved:
            epochs_since_improve = 0
            # Save best-so-far model (based on either metric improving)
            try:
                model.save(model_path)
            except Exception as e:
                print(f"Warning: model save failed at epoch {epoch}: {e}")
        else:
            epochs_since_improve += 1

        # ===== Early stopping =====
        if epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch}: no RMSE or NLL improvement for {patience} epochs.")
            break

        # Optional: free CUDA memory between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ===== Plot & save curves =====
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(losses)
    axes[1].plot(val_rmses)
    axes[2].plot(val_nlls)

    axes[0].set_xlabel("Epoch"); axes[0].set_title("Training Cost")
    axes[1].set_xlabel("Epoch"); axes[1].set_title("Validation RMSE")
    axes[2].set_xlabel("Epoch"); axes[2].set_title("Validation NLL")

    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_plots.png")
    plt.savefig(plot_path, dpi=150)

    if run is not None:
        try:
            run.log({"training_plot": plot_path})
        except Exception as e:
            print(f"Warning: failed to log plot: {e}")

    return losses, val_rmses, val_nlls, {"best_rmse": best_rmse, "best_nll": best_nll}


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

train_from_disk(model, SCRATCH_DIR + "anoms/train_tasks", SCRATCH_DIR + "anoms/val_tasks",
                trainer, data_processor, task_loader, device, epochs = epochs, trainer_batch_size = batch_size,
                run = run, output_dir =  output_dir, model_name =  training_run_name + "_model")

run.finish()




