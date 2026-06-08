# src/pipeline/predict.py
"""Generate predictions from a trained model."""

import numpy as np
import pandas as pd
import torch

from deepsensor_greatlakes.utils import generate_random_coordinates
from pipeline.model import load_trained_model


def predict_date(
    model,
    task_loader,
    bundle: dict,
    config,
    date: str,
    n_context: int = 50,
    seed: int = 42,
) -> dict:
    """
    Make a prediction for a single date.

    Returns dict with:
        - 'task': the generated task
        - 'prediction': xr.Dataset with mean/std
        - 'date': the prediction date
    """
    np.random.seed(seed)

    random_lake_points = generate_random_coordinates(
        bundle["lakemask_sampling"],
        N=n_context,
        data_processor=bundle["data_processor"],
    )

    # Build context sampling from task_loader's map
    context_sampling = []
    for strategy in task_loader._context_sampling_map:
        if strategy == "random_lake_points":
            context_sampling.append(random_lake_points)
        elif strategy == "all":
            context_sampling.append("all")
        else:
            context_sampling.append(int(strategy))

    task = task_loader(
        date,
        context_sampling=context_sampling,
        target_sampling="all",
    )
    task = task.remove_context_nans()
    task = task.remove_target_nans()

    # Fix aux NaNs if present
    if task["Y_t_aux"] is not None:
        task["Y_t_aux"] = np.nan_to_num(task["Y_t_aux"], nan=0.0)

    # Predict on the target grid (pre-DataProcessor version)
    # Use anomaly grid if available, otherwise raw sst
    if "sst_anom_stand" in bundle:
        X_t = bundle["sst_anom_stand"]
    elif "sst_stand" in bundle:
        X_t = bundle["sst_stand"]
    else:
        raise ValueError("No pre-DataProcessor target grid found in bundle.")

    with torch.no_grad():
        prediction_ds = model.predict(task, X_t=X_t)

    return {
        "task": task,
        "prediction": prediction_ds,
        "date": date,
        "context_points": random_lake_points,
    }


def run_predictions(config, bundle, task_loader):
    """Run predictions for configured dates and generate plots."""
    from pathlib import Path
    from .plotting import plot_prediction_summary, plot_task, plot_uncertainty_vs_error
    from .model import load_trained_model

    model = load_trained_model(config, bundle, task_loader)

    # Resolve prediction dates
    pred_dates = config.prediction.get_dates(config.training.val_range)
    n_context = config.prediction.n_context or config.training.n_context_points

    save_dir = Path(config.paths.run_dir) / "predictions"

    print(f"Running predictions for {len(pred_dates)} dates, n_context={n_context}")

    for date in pred_dates:
        date_str = str(date.date()) if hasattr(date, "date") else str(date)
        print(f"\nPredicting: {date_str}")

        result = predict_date(
            model, task_loader, bundle, config,
            date=date_str,
            n_context=n_context,
            seed=config.prediction.seed,
        )

        plot_task(result["task"], task_loader, title=f"Task: {date_str}")
        plot_prediction_summary(result, bundle, config, save_dir=save_dir)
        plot_uncertainty_vs_error(result, bundle, save_dir=save_dir)

    print(f"\nPredictions complete. Saved to: {save_dir}")