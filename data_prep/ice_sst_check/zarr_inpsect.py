"""
inspect_zarr.py — dump everything needed to write the comparison code.
Run and paste the output.
"""
import json
import numpy as np
import xarray as xr
import zarr

# ----------------------------------------------------------------------
# EDIT THESE
SST_PATH = "/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/glsea3_sst.zarr"
ICE_PATH = "/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/ice_concentration.zarr"
# ----------------------------------------------------------------------


def open_any(path):
    """Open zarr with or without consolidated metadata."""
    for kw in ({"consolidated": True}, {"consolidated": False}):
        try:
            return xr.open_zarr(path, mask_and_scale=False, decode_times=True, **kw)
        except Exception as e:
            last = e
    raise last


def describe_coord(ds, name):
    v = ds[name]
    a = v.values
    out = {
        "dtype": str(a.dtype),
        "size": a.size,
        "first": str(a.flat[0]) if a.size else None,
        "last": str(a.flat[-1]) if a.size else None,
        "attrs": dict(v.attrs),
    }
    if a.ndim == 1 and np.issubdtype(a.dtype, np.number) and a.size > 2:
        d = np.diff(a.astype("float64"))
        out["monotonic_increasing"] = bool(np.all(d > 0))
        out["monotonic_decreasing"] = bool(np.all(d < 0))
        out["spacing_min"] = float(d.min())
        out["spacing_max"] = float(d.max())
        out["uniform_spacing"] = bool(np.allclose(d, d[0], rtol=0, atol=1e-9))
    return out


def inspect(path, label):
    print("=" * 78)
    print(f"{label}: {path}")
    print("=" * 78)

    # --- raw zarr metadata (shows true on-disk dtype / fill_value / codecs) ---
    try:
        root = zarr.open(path, mode="r")
        print("\n--- RAW ZARR TREE ---")
        try:
            print(root.tree())
        except Exception:
            print(list(root.array_keys()), list(root.group_keys()))
        print("\n--- RAW ARRAY METADATA ---")
        for k, arr in root.arrays():
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype} chunks={arr.chunks} "
                  f"fill_value={arr.fill_value!r} compressor={getattr(arr, 'compressor', None)}")
            if getattr(arr, "attrs", None):
                print(f"      attrs={json.dumps(dict(arr.attrs), default=str)}")
        print("\n--- RAW ROOT ATTRS ---")
        print(json.dumps(dict(root.attrs), indent=2, default=str)[:4000])
    except Exception as e:
        print(f"  (raw zarr read failed: {e})")

    # --- xarray view (mask_and_scale=False so sentinels stay visible) ---
    ds = open_any(path)
    print("\n--- XARRAY REPR (mask_and_scale=False) ---")
    print(ds)

    print("\n--- DIMS ---")
    print(dict(ds.sizes))

    print("\n--- COORDS ---")
    for c in ds.coords:
        print(f"  {c}: {json.dumps(describe_coord(ds, c), indent=4, default=str)}")

    print("\n--- DATA VARS ---")
    for v in ds.data_vars:
        da = ds[v]
        print(f"  {v}: dims={da.dims} shape={da.shape} dtype={da.dtype} "
              f"chunks={da.chunks}")
        print(f"      attrs    = {json.dumps(dict(da.attrs), default=str)}")
        print(f"      encoding = {json.dumps({k: str(x) for k, x in da.encoding.items()})}")

    print("\n--- GLOBAL ATTRS ---")
    print(json.dumps(dict(ds.attrs), indent=2, default=str)[:4000])

    # --- time axis ---
    tname = next((n for n in ("time", "t", "date", "valid_time") if n in ds.coords), None)
    if tname:
        t = ds[tname].values
        print(f"\n--- TIME ('{tname}') ---")
        print(f"  dtype={t.dtype}  n={t.size}  min={t.min()}  max={t.max()}")
        print(f"  first 3: {t[:3]}")
        print(f"  last 3 : {t[-3:]}")
        if np.issubdtype(t.dtype, np.datetime64):
            ti = xr.CFTimeIndex(t) if not hasattr(t, "astype") else None
            import pandas as pd
            idx = pd.DatetimeIndex(t)
            print(f"  unique gaps (days): {np.unique(np.diff(idx.values).astype('timedelta64[D]'))}")
            print(f"  duplicated timestamps: {int(idx.duplicated().sum())}")
            print(f"  years present: {sorted(set(idx.year))}")
            print(f"  DJF timestep count: {int(idx.month.isin([12, 1, 2]).sum())}")
        print(f"  time encoding: {ds[tname].encoding}")

    ds.close()
    print()


inspect(SST_PATH, "SST")
inspect(ICE_PATH, "ICE")