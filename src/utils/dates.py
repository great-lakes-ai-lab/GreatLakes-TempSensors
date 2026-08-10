# src/utils/dates.py
"""Date-range utilities for multi-interval training/validation windows."""

import pandas as pd


def dates_from_intervals(intervals, subsample_factor: int = 1, per_interval_stride: bool = True):
    """
    Build a normalized DatetimeIndex from a list of (start, end) intervals.

    Parameters
    ----------
    intervals : list[tuple[str, str]]
        Canonical list of (start, end) date-string pairs.
    subsample_factor : int
        Stride applied to daily dates.
    per_interval_stride : bool
        If True, stride is applied within each interval (each block represented,
        avoids phase artifacts). If False, intervals are concatenated then strided.

    Returns
    -------
    pd.DatetimeIndex (normalized, sorted, de-duplicated)
    """
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
