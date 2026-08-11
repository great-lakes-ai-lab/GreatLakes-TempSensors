# src/pipeline/task_builder.py
"""TaskLoader setup and task generation."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import xarray as xr
from dataclasses import dataclass, field

from deepsensor.data import TaskLoader
from utils.coordinates import generate_random_coordinates
from utils.dates import dates_from_intervals
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
        skipped_dates = pd.to_datetime([d for d, _ in skipped])
        print(f"Skipped {len(skipped)} dates due to errors.")
        print(f"    Skip date range: {skipped_dates.min().date()} to {skipped_dates.max().date()}")
        # Optional: show first few
        for d, msg in skipped[:5]:
            print(f"    {d}: {msg}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    return tasks


def make_train_val_dates(config: PipelineConfig) -> tuple:
    tc = config.training
    train_dates = dates_from_intervals(tc.train_range, tc.date_subsample_factor)
    val_dates   = dates_from_intervals(tc.val_range,   tc.date_subsample_factor)
    return train_dates, val_dates


def dates_from_intervals(intervals, subsample_factor: int = 1, per_interval_stride: bool = True):
    """
    Build a normalized DatetimeIndex from a list of (start, end) intervals.

    Parameters
    ----------
    intervals : list[tuple[str, str]]
    subsample_factor : int
        Stride applied to daily dates.
    per_interval_stride : bool
        If True, stride is applied *within each interval* (each block represented,
        avoids phase artifacts). If False, intervals are concatenated then strided.

    Returns
    -------
    pd.DatetimeIndex (normalized, sorted, de-duplicated)
    """
    import pandas as pd

    if not intervals:
        return pd.DatetimeIndex([])

    factor = max(1, int(subsample_factor))
    pieces = []
    for start, end in intervals:
        block = pd.date_range(start, end, freq="D")
        if per_interval_stride:
            block = block[::factor]
        pieces.append(block)

    all_dates = pieces[0]
    for b in pieces[1:]:
        all_dates = all_dates.union(b)  # union sorts + dedupes

    if not per_interval_stride:
        all_dates = all_dates[::factor]

    return pd.to_datetime(all_dates).normalize()