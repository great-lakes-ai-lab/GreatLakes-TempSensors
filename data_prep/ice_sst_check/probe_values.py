"""
probe_values.py — sample a few timesteps and characterize the value domain.
"""
import numpy as np
import pandas as pd
import xarray as xr

SST_PATH = "/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/glsea3_sst.zarr"
ICE_PATH = "/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/ice_concentration.zarr"

SST_VAR = None   # set if you already know it, else auto-picks first data var
ICE_VAR = None
TIME_DIM = "time"


def open_raw(path):
    try:
        return xr.open_zarr(path, consolidated=True, mask_and_scale=False)
    except Exception:
        return xr.open_zarr(path, consolidated=False, mask_and_scale=False)


def probe(path, var, label, n_steps=3):
    ds = open_raw(path)
    var = var or list(ds.data_vars)[0]
    da = ds[var]
    print("=" * 78)
    print(f"{label}  var='{var}'  dtype={da.dtype}")
    print(f"attrs={dict(da.attrs)}")
    print(f"encoding={ {k: str(v) for k, v in da.encoding.items()} }")

    if TIME_DIM in da.dims:
        idx = pd.DatetimeIndex(ds[TIME_DIM].values)
        # pick a mid-winter step, a mid-summer step, and the middle of record
        picks = []
        djf = np.where(idx.month.isin([1, 2]))[0]
        jja = np.where(idx.month.isin([7, 8]))[0]
        if djf.size:
            picks.append(("winter", int(djf[djf.size // 2])))
        if jja.size:
            picks.append(("summer", int(jja[jja.size // 2])))
        picks.append(("mid-record", da.sizes[TIME_DIM] // 2))
    else:
        picks = [("static", None)]

    for tag, i in picks[:n_steps]:
        sl = da.isel({TIME_DIM: i}) if i is not None else da
        a = np.asarray(sl.values)
        tstamp = str(ds[TIME_DIM].values[i]) if i is not None else "n/a"
        print(f"\n  --- {tag} (index {i}, {tstamp}) shape={a.shape} ---")
        print(f"    NaN count         : {int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 'n/a'}")
        finite = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a.ravel()
        if finite.size:
            print(f"    finite min/max    : {finite.min()} / {finite.max()}")
            qs = np.percentile(finite.astype('float64'),
                               [0, 0.1, 1, 25, 50, 75, 99, 99.9, 100])
            print(f"    percentiles       : {np.round(qs, 4)}")
        # most common values -> exposes sentinels (-99, -999, 255, -1, 0, 0.2 ...)
        vals, counts = np.unique(finite, return_counts=True)
        order = np.argsort(counts)[::-1][:20]
        print(f"    n distinct finite : {vals.size}")
        print("    top-20 values by frequency:")
        for j in order:
            print(f"        {vals[j]!r:>16}  x {int(counts[j]):,}")
        # candidate sentinels
        for s in (-9999, -999, -99, -9, -1, 0, 255, 254, 251, 250):
            c = int(np.sum(finite == s))
            if c:
                print(f"    count(=={s:>6}) : {c:,}")
        # specifically around the suspected SST floor
        if np.issubdtype(a.dtype, np.floating):
            near = finite[(finite > 0.0) & (finite < 0.5)]
            if near.size:
                nv, nc = np.unique(near, return_counts=True)
                print(f"    distinct values in (0.0, 0.5): {list(zip(nv[:25], nc[:25]))}")

    ds.close()
    print()


probe(SST_PATH, SST_VAR, "SST")
probe(ICE_PATH, ICE_VAR, "ICE")
