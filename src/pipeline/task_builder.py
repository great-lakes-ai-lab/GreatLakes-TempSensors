# src/pipeline/task_builder.py
"""TaskLoader setup and task generation."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import xarray as xr
from dataclasses import dataclass, field
from collections import Counter

from deepsensor.data import TaskLoader
from utils.coordinates import generate_random_coordinates
from utils.dates import dates_from_intervals
from utils.seeds import derive_rng
from pipeline.config import PipelineConfig


TASK_MODES = (
    "fully_random",
    "active_plus_random",
    "subset_active_plus_random",
    "active_only",
)

_ACTIVE_MODES = ("active_plus_random", "subset_active_plus_random", "active_only")


@dataclass
class TaskLoaderConfig:
    """Bundles a TaskLoader with its context sampling strategy."""
    task_loader: TaskLoader
    context_sampling_map: list = field(default_factory=list)


def build_task_loader(config: PipelineConfig, bundle: dict) -> TaskLoaderConfig:
    print("\nNow building TaskLoader...")
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

    print(f"  Context sets ({len(context)}): sampling={context_sampling_map}")
    print(f"  Target: {list(target_ds.data_vars) if target_ds is not None else None}")
    print(f"  Aux-at-targets: {list(aux_at_targets.data_vars) if aux_at_targets is not None else None}")

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
    task_ratios: dict = None,
    tasks_per_date: float = None,
    progress: bool = True,
    verbose: bool = True
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
    seed : int | np.random.Generator | optional
    fixed_context_points : np.ndarray, optional
        Array of shape (2, N) with normalized [lat, lon] coordinates.
        If provided, these exact points are used as context for every task.
        Overrides random generation, vary_n_context, and task_ratios.
    task_ratios : dict, optional
        Mixing ratios over the four context-sampling modes, keyed by
        TASK_MODES. Assumed to sum to 1.0. When None (default), falls back to
        the legacy single-mode behavior (fully random context points).
    tasks_per_date : float, optional
        Average tasks generated per date. Only used when task_ratios is given.
        Defaults to 1.0. Values > 1.0 let a date receive tasks from more than
        one mode.
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

    rng = np.random.default_rng(seed)
    dates = list(dates)

    # ---- Plan the (date, mode) schedule -------------------------------------
    use_modes = task_ratios is not None and fixed_context_points is None
    active_points = None

    if use_modes:
        from pipeline.active_learning import load_context_points
        ratios = _validate_task_ratios(task_ratios)
        tasks_per_date = 1.0 if tasks_per_date is None else float(tasks_per_date)

        date_slots = _build_date_slots(len(dates), tasks_per_date, rng)
        mode_counts = _largest_remainder_counts(len(date_slots), ratios)

        mode_labels = [m for m in TASK_MODES for _ in range(mode_counts[m])]
        mode_labels = [mode_labels[k] for k in rng.permutation(len(mode_labels))]
        mode_labels = _repair_active_only_duplicates(
            date_slots, mode_labels, rng, verbose=verbose
        )

        # Active buoy points are static: load once.
        if any(mode_counts[m] > 0 for m in _ACTIVE_MODES):
            active_points = load_context_points(tc.active_buoy_points_path, bundle)
            n_active = active_points.shape[1]
            if verbose and not vary_n_context and n_active > n_context:
                print(f"  Note: {n_active} active points exceeds n_context="
                      f"{n_context}; active modes will emit {n_active} points.")

        schedule = list(zip(date_slots, mode_labels))
    else:
        schedule = [(i, None) for i in range(len(dates))]

    if verbose:
        seed_label = seed if isinstance(seed, (int, np.integer)) else "derived"
        print(f"\nNow generating {len(schedule)} tasks from {len(dates)} dates "
              f"(varying N context points={vary_n_context}, seed={seed_label})...")
        if use_modes:
            for m in TASK_MODES:
                if mode_counts[m]:
                    print(f"    {m}: {mode_counts[m]}")

    tasks = []
    skipped = []
    mode_used = Counter()

    for date_i, mode in tqdm(schedule, disable=not progress, desc="Generating tasks"):
        date = dates[date_i]

        # Determine context points for this task
        if fixed_context_points is not None:
            context_points = fixed_context_points
        else:
            if vary_n_context:
                N = int(rng.integers(min_n, max_n))
            else:
                N = n_context

            # Potential sampling mask on a per-date basis, dropping where
            # sst == 0.2 (ice).
            sst_mask = valid_mask_for_date(bundle['sst_anom_stand'], date)

            if mode is None:
                context_points = generate_random_coordinates(
                    sst_mask,
                    N=N,
                    data_processor=bundle["data_processor"],
                    rng=rng
                )
            else:
                context_points = _context_points_for_mode(
                    mode,
                    N=N,
                    active_points=active_points,
                    sst_mask=sst_mask,
                    data_processor=bundle["data_processor"],
                    rng=rng
                )

        # Build context_sampling list
        context_sampling = []
        for strategy in tl_config.context_sampling_map:
            if strategy == "random_lake_points":
                context_sampling.append(context_points)
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
        if mode is not None:
            mode_used[mode] += 1

    if skipped and verbose:
        skipped_dates = pd.to_datetime([d for d, _ in skipped])
        print(f"Skipped {len(skipped)} dates due to errors.")
        print(f"    Skip date range: {skipped_dates.min().date()} to {skipped_dates.max().date()}")
        for d, msg in skipped[:5]:
            print(f"    {d}: {msg}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    if use_modes and verbose:
        print(f"Kept {len(tasks)} tasks: "
              + ", ".join(f"{m}={mode_used[m]}" for m in TASK_MODES if mode_counts[m]))

    return tasks

# def gen_tasks(
#     tl_config,
#     dates,
#     bundle: dict,
#     config,
#     n_context: int = None,
#     vary_n_context: bool = None,
#     min_n: int = None,
#     max_n: int = None,
#     seed: int = None,
#     fixed_context_points: np.ndarray = None,
#     progress: bool = True,
#     verbose: bool = True
# ) -> list:
#     """
#     Generate tasks for a list of dates.
#
#     Parameters
#     ----------
#     tl_config : TaskLoaderConfig
#     dates : array-like of datetime
#     bundle : dict
#         Processed bundle (needs 'lakemask_sampling' and 'data_processor')
#     config : PipelineConfig
#     n_context : int, optional
#     vary_n_context : bool, optional
#     min_n, max_n : int, optional
#     seed : int | np.random.Generator |optional
#     fixed_context_points : np.ndarray, optional
#         Array of shape (2, N) with normalized [lat, lon] coordinates.
#         If provided, these exact points are used as context for every task.
#         Overrides random generation and vary_n_context.
#     progress : bool
#
#     Returns
#     -------
#     list of Task objects
#     """
#
#     tc = config.training
#     n_context = n_context if n_context is not None else tc.n_context_points
#     vary_n_context = vary_n_context if vary_n_context is not None else tc.vary_n_context
#     min_n = min_n if min_n is not None else tc.min_n_context
#     max_n = max_n if max_n is not None else tc.max_n_context
#
#     rng = np.random.default_rng(seed)
#
#     if verbose:
#         seed_label = seed if isinstance(seed, (int, np.integer)) else "derived"
#         print(f"\nNow generating {len(dates)} tasks "
#               f"(varying N context points={vary_n_context}, seed={seed_label})...")
#
#     tasks = []
#     skipped = []
#
#     for date in tqdm(dates, disable=not progress, desc="Generating tasks"):
#         # Determine context points for this task
#         # TODO: Wire in train_tasks_random, plus, active, task ratios from config and update so that context points per
#         #   task are generating using the active geojson as requested
#         if fixed_context_points is not None:
#             context_points = fixed_context_points
#         else:
#             if vary_n_context:
#                 N = int(rng.integers(min_n, max_n))
#             else:
#                 N = n_context
#
#
#             # Now implemented to create the potential sampling mask on a per date basis dropping where sst==0.2(ice)
#             sst_mask = valid_mask_for_date(bundle['sst_anom_stand'], date)
#
#             context_points = generate_random_coordinates(
#                 # bundle["lakemask_sampling"],
#                 sst_mask,
#                 N=N,
#                 data_processor=bundle["data_processor"],
#                 rng=rng
#             )
#
#         # Build context_sampling list
#         context_sampling = []
#         for strategy in tl_config.context_sampling_map:
#             if strategy == "random_lake_points":
#                 context_sampling.append(context_points)
#             elif strategy == "all":
#                 context_sampling.append("all")
#             else:
#                 context_sampling.append(int(strategy))
#
#         try:
#             task = tl_config.task_loader(
#                 date,
#                 context_sampling=context_sampling,
#                 target_sampling="all",
#             )
#         except (KeyError, pd.errors.InvalidIndexError) as e:
#             skipped.append((date, str(e)))
#             continue
#
#         task = task.remove_context_nans()
#         task = task.remove_target_nans()
#         tasks.append(task)
#
#     if skipped and verbose:
#         skipped_dates = pd.to_datetime([d for d, _ in skipped])
#         print(f"Skipped {len(skipped)} dates due to errors.")
#         print(f"    Skip date range: {skipped_dates.min().date()} to {skipped_dates.max().date()}")
#         # Optional: show first few
#         for d, msg in skipped[:5]:
#             print(f"    {d}: {msg}")
#         if len(skipped) > 5:
#             print(f"    ... and {len(skipped) - 5} more")
#
#     return tasks


def make_train_val_dates(config: PipelineConfig) -> tuple:
    """
    Resolve deterministic train/val date sets.

    Val dates always use val_date_stride. Train dates use train_date_stride —
    these are the actual training dates when train_date_mode='strided', and
    serve only as a fallback/reference when mode='random'.
    """
    tc = config.training
    train_dates = dates_from_intervals(tc.train_range, tc.train_date_stride, months_to_drop=tc.train_months_drop)
    val_dates = dates_from_intervals(tc.val_range, tc.val_date_stride, months_to_drop=tc.train_months_drop)

    # print(f"\nNow resolving train/val dates...")
    # if not tc.train_date_mode == 'random':
    #     print(f"  train_range {tc.train_range} | mode={tc.train_date_mode}, "
    #           f"stride={tc.train_date_stride} → {len(train_dates)} strided dates")
    # print(f"  training sampling is random. Dates chosen later per epoch")
    # print(f"  val_range   {tc.val_range} | stride={tc.val_date_stride} "
    #       f"→ {len(val_dates)} dates")
    return train_dates, val_dates


def make_train_date_sampler(config: PipelineConfig):
    """
    Build a callable(epoch) -> DatetimeIndex drawing random dates from the
    full daily train_range pool. Only used when train_date_mode='random'.

    Returns (sampler, pool, n_per_epoch).
    """
    tc = config.training
    pool = dates_from_intervals(tc.train_range, subsample_factor=1, months_to_drop=tc.train_months_drop)

    if len(pool) == 0:
        raise ValueError(f"train_range produced no dates: {tc.train_range}")

    if tc.n_train_dates_per_epoch is not None:
        n = int(tc.n_train_dates_per_epoch)
        basis = f"n_train_dates_per_epoch={n}"
    else:
        n = max(1, round(tc.train_date_fraction * len(pool)))
        basis = f"fraction={tc.train_date_fraction}"

    n = min(n, len(pool))

    print(f"  train date sampler: pool={len(pool)} daily dates, "
          f"drawing {n}/epoch ({basis}, {100 * n / len(pool):.1f}% coverage/epoch)")

    def sample_dates(epoch: int):
        rng = derive_rng(tc.train_task_seed, epoch, "dates")
        idx = rng.choice(len(pool), size=n, replace=False)
        return pool[np.sort(idx)]

    return sample_dates, pool, n


def valid_mask_for_date(obj, date):
    date = pd.Timestamp(date)
    snap = obj.sel(time=date)
    mask = snap.notnull().astype("int8")
    mask = mask.assign_attrs(
        description="1 = valid data, 0 = missing (NaN)",
        source_time=str(np.datetime64(snap["time"].values)),
    )
    return mask


##############################################################################################
# ---------------------- Task Mode Helpers --------------------------------------------
#############################################################################################
def _largest_remainder_counts(total: int, ratios: dict) -> dict:
    """
    Allocate `total` items across modes in proportion to `ratios`.

    Uses the largest-remainder (Hare-Niemeyer) method so the returned counts
    sum to exactly `total`.
    """
    keys = list(ratios)
    raw = {k: total * float(ratios[k]) for k in keys}
    counts = {k: int(np.floor(raw[k])) for k in keys}

    remainder = total - sum(counts.values())
    if remainder > 0:
        # Ties broken by dict order -> deterministic.
        order = sorted(keys, key=lambda k: raw[k] - counts[k], reverse=True)
        for k in order[:remainder]:
            counts[k] += 1

    return counts


def _build_date_slots(n_dates: int, tasks_per_date: float, rng) -> np.ndarray:
    """
    Build an array of date *indices*, length round(n_dates * tasks_per_date).

    Every date appears either floor(tasks_per_date) or ceil(tasks_per_date)
    times, so no date starves and the spread is maximally even. Returns
    indices rather than dates to preserve the caller's element types.
    """
    total = int(round(n_dates * tasks_per_date))
    if total < 1:
        raise ValueError(
            f"_build_date_slots: tasks_per_date={tasks_per_date} with "
            f"{n_dates} dates yields {total} tasks. Increase tasks_per_date."
        )

    full_reps, remainder = divmod(total, n_dates)

    slots = [rng.permutation(n_dates) for _ in range(full_reps)]
    if remainder:
        slots.append(rng.choice(n_dates, size=remainder, replace=False))

    return np.concatenate(slots) if slots else np.empty(0, dtype=int)


def _repair_active_only_duplicates(date_idx, modes, rng, verbose=True) -> list:
    """
    Ensure no date receives more than one 'active_only' task.

    'active_only' is deterministic, so two such tasks on the same date are
    byte-identical duplicates. Offending slots are swapped with a slot of a
    different mode whose date has no 'active_only' yet. Per-mode counts are
    preserved exactly.
    """
    modes = list(modes)
    key = "active_only"

    claimed = set()
    dup_slots = []
    for i, m in enumerate(modes):
        if m != key:
            continue
        d = date_idx[i]
        if d in claimed:
            dup_slots.append(i)
        else:
            claimed.add(d)

    if not dup_slots:
        return modes

    candidates = [j for j in range(len(modes)) if modes[j] != key]
    candidates = [candidates[k] for k in rng.permutation(len(candidates))]

    unrepaired = 0
    cursor = 0
    for i in dup_slots:
        while cursor < len(candidates) and date_idx[candidates[cursor]] in claimed:
            cursor += 1
        if cursor >= len(candidates):
            unrepaired += 1
            continue
        j = candidates[cursor]
        cursor += 1
        modes[i], modes[j] = modes[j], modes[i]
        claimed.add(date_idx[j])

    if unrepaired and verbose:
        print(f"  Warning: {unrepaired} duplicate 'active_only' task(s) could "
              f"not be relocated (too few distinct dates).")

    return modes


def _context_points_for_mode(
    mode, N, active_points, sst_mask, data_processor, rng
) -> np.ndarray:
    """
    Generate context points for a single task under the given mode.

    N is the *total* desired context size; active points count toward it.
    """
    if mode == "fully_random":
        return generate_random_coordinates(
            sst_mask, N=N, data_processor=data_processor, rng=rng
        )

    if mode == "active_only":
        return active_points

    n_active = active_points.shape[1]

    if mode == "active_plus_random":
        chosen = active_points
    elif mode == "subset_active_plus_random":
        k = int(rng.integers(1, n_active + 1))
        sel = np.sort(rng.choice(n_active, size=k, replace=False))
        chosen = active_points[:, sel]
    else:
        raise ValueError(f"_context_points_for_mode: unknown mode '{mode}'")

    n_random = max(N - chosen.shape[1], 0)
    if n_random == 0:
        return chosen

    random_points = generate_random_coordinates(
        sst_mask, N=n_random, data_processor=data_processor, rng=rng
    )
    return np.concatenate([chosen, random_points], axis=1)


def _validate_task_ratios(task_ratios: dict) -> dict:
    """Check keys and non-negativity; return a clean dict in canonical order."""
    missing = set(TASK_MODES) - set(task_ratios)
    extra = set(task_ratios) - set(TASK_MODES)
    if missing or extra:
        raise ValueError(
            f"task_ratios keys must be exactly {TASK_MODES}. "
            f"Missing: {sorted(missing)}. Unexpected: {sorted(extra)}."
        )

    clean = {}
    for k in TASK_MODES:
        v = float(task_ratios[k])
        if v < 0:
            raise ValueError(f"task_ratios['{k}'] must be >= 0, got {v}")
        clean[k] = v

    if sum(clean.values()) <= 0:
        raise ValueError("task_ratios must contain at least one positive value.")

    return clean

