# src/pipeline/trainer.py
"""Training loop with validation, checkpointing, and diagnostics."""

from pathlib import Path
import json
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from deepsensor.train import Trainer

from pipeline.config import PipelineConfig
from pipeline.model import save_trained_model


def train_model(
    model,
    task_loader,
    train_tasks: list,
    val_tasks: list,
    bundle: dict,
    config: PipelineConfig,
) -> dict:
    tc = config.training
    trainer = Trainer(model, lr=tc.lr)

    losses = []
    val_rmses = []
    best_val_rmse = np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    start_time = time.time()

    print(f"Training for {tc.n_epochs} epochs | lr={tc.lr} | "
          f"{len(train_tasks)} train tasks | {len(val_tasks)} val tasks")
    if tc.patience > 0:
        print(f"Early stopping enabled: patience={tc.patience}")

    for epoch in tqdm(range(1, tc.n_epochs + 1), desc="Training"):
        # Train
        batch_losses = trainer(train_tasks)
        epoch_loss = float(np.mean(batch_losses))
        losses.append(epoch_loss)

        # Validate
        val_rmse = compute_val_rmse(model, val_tasks, bundle, task_loader)
        val_rmses.append(val_rmse)

        # Checkpoint best
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            save_trained_model(model, config)
        else:
            epochs_without_improvement += 1

        # Log periodically
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={epoch_loss:.4f}, "
                  f"val_rmse={val_rmse:.4f}, best={best_val_rmse:.4f} "
                  f"(patience: {epochs_without_improvement}/{tc.patience if tc.patience > 0 else '∞'})")

        # Early stopping check
        if tc.patience > 0 and epochs_without_improvement >= tc.patience:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"No improvement for {tc.patience} epochs.")
            break

        # Free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60.0

    print(f"\nTraining complete in {elapsed_min:.1f} minutes. "
          f"Best val RMSE: {best_val_rmse:.4f} (epoch {best_epoch})")

    results = {
        "losses": losses,
        "val_rmses": val_rmses,
        "best_val_rmse": float(best_val_rmse),
        "best_epoch": best_epoch,
        "total_epochs": len(losses),
        "early_stopped": tc.patience > 0 and epochs_without_improvement >= tc.patience,
        "training_time_seconds": float(elapsed),
        "training_time_minutes": float(elapsed_min),
    }

    # Save training metadata and plots
    _save_training_metadata(config, results)
    _save_training_plots(config, results)

    return results


def compute_val_rmse(model, val_tasks: list, bundle: dict, task_loader) -> float:
    """
    Compute RMSE over validation tasks in physical (unnormalized) units.

    Parameters
    ----------
    model : ConvNP
    val_tasks : list
    bundle : dict
        Must contain 'data_processor'
    task_loader : TaskLoader

    Returns
    -------
    float : RMSE value
    """
    data_processor = bundle["data_processor"]
    target_var_ID = task_loader.target_var_IDs[0][0]

    errors = []

    for task in val_tasks:
        with torch.no_grad():
            mean = data_processor.map_array(
                model.mean(task), target_var_ID, unnorm=True
            )
            true = data_processor.map_array(
                task["Y_t"][0], target_var_ID, unnorm=True
            )

        errors.extend((mean - true) ** 2)

    rmse = float(np.sqrt(np.mean(np.concatenate(errors))))
    return rmse


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _save_training_metadata(config: PipelineConfig, results: dict):
    """Save training run metadata to JSON."""
    model_dir = Path(config.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tc = config.training
    metadata = {
        "run_name": config.run.name,
        "run_notes": config.run.notes,
        "lake": config.lake,
        "environment": config.environment,
        "train_range": list(tc.train_range),
        "val_range": list(tc.val_range),
        "date_subsample_factor": tc.date_subsample_factor,
        "n_epochs": tc.n_epochs,
        "lr": tc.lr,
        "patience": tc.patience,
        "total_epochs": results["total_epochs"],
        "early_stopped": results["early_stopped"],
        "internal_density": tc.internal_density,
        "n_context_points": tc.n_context_points,
        "vary_n_context": tc.vary_n_context,
        "min_n_context": tc.min_n_context,
        "max_n_context": tc.max_n_context,
        "include_bathy_as_context": tc.include_bathy_as_context,
        "bathy_context_sampling": tc.bathy_context_sampling,
        "best_val_rmse": results["best_val_rmse"],
        "best_epoch": results["best_epoch"],
        "final_train_loss": results["losses"][-1],
        "losses": [float(l) for l in results["losses"]],
        "val_rmses": [float(r) for r in results["val_rmses"]],
        "training_time_seconds": results["training_time_seconds"],
        "training_time_minutes": results["training_time_minutes"],
    }

    meta_path = model_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Training metadata saved to: {meta_path}")


def _save_training_plots(config: PipelineConfig, results: dict):
    """Save training curves plot."""
    import matplotlib
    import matplotlib.pyplot as plt
    from pipeline.plotting import _finish_plot

    model_dir = Path(config.paths.model_dir)
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(results["losses"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, results["losses"], "b-")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, results["val_rmses"], "r-")
    axes[1].axhline(
        y=results["best_val_rmse"], color="g", linestyle="--", alpha=0.5,
        label=f"Best: {results['best_val_rmse']:.4f}",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Validation RMSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _finish_plot(config, plots_dir / "training_curves.png")
    print(f"Training plots saved to: {plots_dir}")