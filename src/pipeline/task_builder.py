# src/pipeline/task_builder.py
"""TaskLoader setup and task generation."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import xarray as xr
from dataclasses import dataclass, field

from deepsensor.data import TaskLoader
from utils.coordinates import generate_random_coordinates

from pipeline.config import PipelineConfig


@dataclass
class TaskLoaderConfig:
    """Bundles a TaskLoader with its context sampling strategy."""
    task_loader: TaskLoader
    context_sampling_map: list = field(default_factory=list)


def build_task_loader(config: PipelineConfig, bundle: dict) -> TaskLoaderConfig:
    context = []
    context_sampling_map = []
    aux_at_targets_list = []
    target_ds = None

    for name, source in config.data_sources.items():
        roles = source.role if isinstance(source.role, list) else [source.role]

        if "target" in roles:
            if source.use_anomalies and f"{name}_anom" in bundle:
                target_ds = bundle[f"{name}_anom"]
            else:
                target_ds = bundle[name]
            context.append(target_ds)
            context_sampling_map.append(source.sampling)

        if "context" in roles and "target" not in roles:
            context.append(bundle[name])
            context_sampling_map.append(source.sampling)

        if "aux_at_targets" in roles:
            aux_at_targets_list.append(bundle[name])

        if "mask" in roles:
            context.append(bundle["mask_time_ds"])
            context_sampling_map.append("all")

    if aux_at_targets_list:
        aux_at_targets = xr.merge(aux_at_targets_list)
    else:
        aux_at_targets = None

    task_loader = TaskLoader(
        context=context,
        target=target_ds,
        aux_at_targets=aux_at_targets,
    )

    return TaskLoaderConfig(
        task_loader=task_loader,
        context_sampling_map=context_sampling_map,
    )


def gen_tasks(
    tl_config,
    dates,
    bundle: dict,
    config,
    n_context: int = None,
    vary_n_context: bool = None,
    min_n: int = None,
    max_n: int = None,
    seed: int = None,
    fixed_context_points: np.ndarray = None,
    progress: bool = True,
) -> list:
    """
    Generate tasks for a list of dates.

    Parameters
    ----------
    tl_config : TaskLoaderConfig
    dates : array-like of datetime
    bundle : dict
        Processed bundle (needs 'lakemask_sampling' and 'data_processor')
    config : PipelineConfig
    n_context : int, optional
    vary_n_context : bool, optional
    min_n, max_n : int, optional
    seed : int, optional
    fixed_context_points : np.ndarray, optional
        Array of shape (2, N) with normalized [lat, lon] coordinates.
        If provided, these exact points are used as context for every task.
        Overrides random generation and vary_n_context.
    progress : bool

    Returns
    -------
    list of Task objects
    """
    tc = config.training
    n_context = n_context if n_context is not None else tc.n_context_points
    vary_n_context = vary_n_context if vary_n_context is not None else tc.vary_n_context
    min_n = min_n if min_n is not None else tc.min_n_context
    max_n = max_n if max_n is not None else tc.max_n_context

    if seed is not None:
        np.random.seed(seed)

    tasks = []
    skipped = []

    for date in tqdm(dates, disable=not progress, desc="Generating tasks"):
        # Determine context points for this task
        if fixed_context_points is not None:
            random_lake_points = fixed_context_points
        else:
            if vary_n_context:
                N = np.random.randint(min_n, max_n)
            else:
                N = n_context

            random_lake_points = generate_random_coordinates(
                bundle["lakemask_sampling"],
                N=N,
                data_processor=bundle["data_processor"],
            )

        # Build context_sampling list
        context_sampling = []
        for strategy in tl_config.context_sampling_map:
            if strategy == "random_lake_points":
                context_sampling.append(random_lake_points)
            elif strategy == "all":
                context_sampling.append("all")
            else:
                context_sampling.append(int(strategy))

        try:
            task = tl_config.task_loader(
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