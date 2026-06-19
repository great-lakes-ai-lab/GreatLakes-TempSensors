# src/pipeline/predict.py
"""Generate predictions from a trained model."""

import numpy as np
import pandas as pd
import torch

from deepsensor_greatlakes.utils import generate_random_coordinates
from pipeline.model import load_trained_model


def predict_date(
    model,
    tl_config,  # was: task_loader
    bundle: dict,
    config,
    date: str,
    n_context: int = 50,
    seed: int = 42,
) -> dict:
    np.random.seed(seed)

    random_lake_points = generate_random_coordinates(
        bundle["lakemask_sampling"],
        N=n_context,
        data_processor=bundle["data_processor"],
    )

    context_sampling = []
    for strategy in tl_config.context_sampling_map:
        if strategy == "random_lake_points":
            context_sampling.append(random_lake_points)
        elif strategy == "all":
            context_sampling.append("all")
        else:
            context_sampling.append(int(strategy))

    task = tl_config.task_loader(
        date,
        context_sampling=context_sampling,
        target_sampling="all",
    )
    task = task.remove_context_nans()
    task = task.remove_target_nans()

    if task["Y_t_aux"] is not None:
        task["Y_t_aux"] = np.nan_to_num(task["Y_t_aux"], nan=0.0)

    # Resolve X_t
    X_t = None
    for name, source in config.data_sources.items():
        roles = source.role if isinstance(source.role, list) else [source.role]
        if "target" in roles:
            if source.use_anomalies and f"{name}_anom_stand" in bundle:
                X_t = bundle[f"{name}_anom_stand"]
            elif f"{name}_stand" in bundle:
                X_t = bundle[f"{name}_stand"]
            break

    if X_t is None:
        raise ValueError(
            "No pre-DataProcessor target grid found in bundle. "
            "Expected keys like '<target_name>_stand' or '<target_name>_anom_stand'."
        )

    with torch.no_grad():
        prediction_ds = model.predict(task, X_t=X_t)

    return {
        "task": task,
        "prediction": prediction_ds,
        "date": date,
        "context_points": random_lake_points,
    }

def run_predictions(config, bundle, tl_config):
    """Run predictions for configured dates and generate plots."""
    from pathlib import Path
    from .plotting import plot_prediction_summary, plot_task, plot_uncertainty_vs_error
    from .model import load_trained_model

    model = load_trained_model(config, bundle, tl_config.task_loader)

    pred_dates = config.prediction.get_dates(config.training.val_range)
    n_context = config.prediction.n_context or config.training.n_context_points

    save_dir = Path(config.paths.run_dir) / "predictions"

    print(f"Running predictions for {len(pred_dates)} dates, n_context={n_context}")

    for date in pred_dates:
        date_str = str(date.date()) if hasattr(date, "date") else str(date)
        print(f"\nPredicting: {date_str}")

        result = predict_date(
            model, tl_config, bundle, config,
            date=date_str,
            n_context=n_context,
            seed=config.prediction.seed,
        )

        plot_task(result["task"], tl_config.task_loader, config, title=f"Task: {date_str}", save_dir=save_dir, date_str=date_str)
        plot_prediction_summary(result, bundle, config, save_dir=save_dir)
        plot_uncertainty_vs_error(result, bundle, config, save_dir=save_dir)

    print(f"\nPredictions complete. Saved to: {save_dir}")