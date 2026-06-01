# src/pipeline/task_builder.py
"""TaskLoader setup and task generation."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from deepsensor.data import TaskLoader
from deepsensor_greatlakes.utils import generate_random_coordinates

from pipeline.config import PipelineConfig


def build_task_loader(config: PipelineConfig, bundle: dict) -> TaskLoader:
    """
    Construct a TaskLoader from the processed bundle and config.

    Context sets are built dynamically based on what's available in the bundle.
    Order:
        1. Target variable (sst or sst_anom) — sampled at random lake points
        2. Additional temporal contexts (t2m, ssr, ice_concentration, etc.) — "all"
        3. mask_time_ds (mask + cos_D + sin_D) — "all"
        4. (Optional) bathy as context — configurable sampling

    Target: the target variable (sst_anom by default)
    Aux at targets: bathymetry
    """
    # Determine target dataset
    if "sst_anom" in bundle:
        target_ds = bundle["sst_anom"]
        target_name = "sst_anom"
    elif "sst" in bundle:
        target_ds = bundle["sst"]
        target_name = "sst"
    else:
        raise ValueError("No target variable (sst or sst_anom) found in bundle.")

    # Build context list
    # First context is always the target variable (will be sampled at random points)
    context = [target_ds]

    # Additional temporal datasets as context (sampled "all")
    # These are anything temporal that isn't the target and isn't mask_time_ds
    skip_names = {target_name, "sst", "sst_anom", "mask_time_ds", "bathy", "lakemask",
                  "lakemask_sampling", "sst_stand", "sst_anom_stand",
                  "cosD", "sinD", "data_processor", "seasonal_processor"}
    temporal_context_names = []

    for name, ds in bundle.items():
        if name in skip_names:
            continue
        if hasattr(ds, "dims") and "time" in getattr(ds, "dims", {}):
            context.append(ds)
            temporal_context_names.append(name)

    # mask_time_ds (static mask + temporal encoding)
    if "mask_time_ds" in bundle:
        context.append(bundle["mask_time_ds"])

    # Optional: bathymetry as context
    if config.training.include_bathy_as_context and "bathy" in bundle:
        context.append(bundle["bathy"])

    # Aux at targets
    aux_at_targets = bundle.get("bathy")

    task_loader = TaskLoader(
        context=context,
        target=target_ds,
        aux_at_targets=aux_at_targets,
    )

    # Store context layout info for gen_tasks to reference
    task_loader._context_layout = {
        "target_idx": 0,
        "temporal_context_names": temporal_context_names,
        "temporal_context_start_idx": 1,
        "mask_time_idx": 1 + len(temporal_context_names) if "mask_time_ds" in bundle else None,
        "bathy_idx": len(context) - 1 if config.training.include_bathy_as_context else None,
        "n_contexts": len(context),
    }

    print(f"TaskLoader built: {len(context)} context sets, target='{target_name}'")
    print(f"  Context[0]: {target_name} (random lake points)")
    for i, name in enumerate(temporal_context_names, start=1):
        print(f"  Context[{i}]: {name} (all)")
    if "mask_time_ds" in bundle:
        print(f"  Context[{1 + len(temporal_context_names)}]: mask_time_ds (all)")
    if config.training.include_bathy_as_context:
        print(f"  Context[{len(context) - 1}]: bathy ({config.training.bathy_context_sampling})")
    print(f"  Aux at targets: bathy")

    return task_loader


def gen_tasks(
    task_loader: TaskLoader,
    dates,
    bundle: dict,
    config: PipelineConfig,
    n_context: int = None,
    vary_n_context: bool = None,
    min_n: int = None,
    max_n: int = None,
    seed: int = None,
    progress: bool = True,
) -> list:
    """
    Generate tasks for a list of dates.

    Parameters
    ----------
    task_loader : TaskLoader
    dates : array-like of datetime
    bundle : dict
        Processed bundle (needs 'lakemask_sampling' and 'data_processor')
    config : PipelineConfig
        Falls back to config.training values if kwargs not provided
    n_context : int, optional
        Fixed number of context points. Overridden by vary_n_context.
    vary_n_context : bool, optional
    min_n, max_n : int, optional
    seed : int, optional
    progress : bool

    Returns
    -------
    list of Task objects
    """
    # Resolve parameters: explicit kwargs > config values
    tc = config.training
    n_context = n_context if n_context is not None else tc.n_context_points
    vary_n_context = vary_n_context if vary_n_context is not None else tc.vary_n_context
    min_n = min_n if min_n is not None else tc.min_n_context
    max_n = max_n if max_n is not None else tc.max_n_context

    if seed is not None:
        np.random.seed(seed)

    # Build context_sampling list based on layout
    layout = task_loader._context_layout

    tasks = []
    skipped = []

    for date in tqdm(dates, disable=not progress, desc="Generating tasks"):
        # Determine N for this task
        if vary_n_context:
            N = np.random.randint(min_n, max_n)
        else:
            N = n_context

        # Generate random lake points for target variable context
        random_lake_points = generate_random_coordinates(
            bundle["lakemask_sampling"],
            N=N,
            data_processor=bundle["data_processor"],
        )

        # Build context_sampling: random points for target, "all" for everything else
        context_sampling = []
        for i in range(layout["n_contexts"]):
            if i == layout["target_idx"]:
                context_sampling.append(random_lake_points)
            elif i == layout["bathy_idx"]:
                raw_val = tc.bathy_context_sampling
                if raw_val == "all":
                    context_sampling.append("all")
                elif raw_val == "random_lake_points":
                    context_sampling.append(random_lake_points)
                elif isinstance(raw_val, int):
                    context_sampling.append(raw_val)
                else:
                    context_sampling.append(int(raw_val))
            else:
                context_sampling.append("all")

        try:
            task = task_loader(
                date,
                context_sampling=context_sampling,
                target_sampling="all",
            )
        except (KeyError, pd.errors.InvalidIndexError) as e:
            skipped.append((date, str(e)))
            continue

        task = task.remove_context_nans()
        task = task.remove_target_nans()
        tasks.append(task)

    if skipped:
        print(f"Skipped {len(skipped)} dates due to errors.")

    return tasks


def make_train_val_dates(config: PipelineConfig) -> tuple:
    """
    Generate train and validation date arrays from config.

    Returns (train_dates, val_dates) as normalized pandas DatetimeIndex.
    """
    tc = config.training

    train_dates = pd.date_range(tc.train_range[0], tc.train_range[1])[::tc.date_subsample_factor]
    val_dates = pd.date_range(tc.val_range[0], tc.val_range[1])[::tc.date_subsample_factor]

    train_dates = pd.to_datetime(train_dates).normalize()
    val_dates = pd.to_datetime(val_dates).normalize()

    return train_dates, val_dates