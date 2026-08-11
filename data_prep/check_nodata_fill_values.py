import xarray as xr
import numpy as np
from pathlib import Path

SOURCES = {
    "glsea":     ("/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/glsea_sst.zarr", "zarr", "sst"),
    "glsea3":     ("/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/glsea3_sst.zarr", "zarr", "sst"),
    "era5":    ("/Users/jagraha/Data/ciglr_globus/greatlakes_era5_1995_2025.zarr", "zarr", None),  # multiple vars
    "ice":     ("/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/ice_concentration.zarr", "zarr", "ice_concentration"),
    "bathy":   ("~/dev/deepsensor_projects/data/bathy/gl_bathy_depth_3arcsec_clean.nc", "netcdf", "z"),
    "lakemask":("~/dev/deepsensor_projects/data/bathy/gl_landmask_3arcsec.nc", "netcdf", "mask"),
    "dist":    ("~/dev/deepsensor_projects/data/bathy/gl_dist_to_land_3arcsec.nc", "netcdf", "dist_to_land"),
    "core":    ("~/dev/deepsensor_projects/data/bathy/gl_core_depth_3arcsec.nc", "netcdf", "core_depth"),
}



def probe(name, path, fmt, var):
    path = str(Path(path).expanduser())
    ds = xr.open_zarr(path) if fmt == "zarr" else xr.open_dataset(path)

    vars_to_check = [var] if var else list(ds.data_vars)

    for v in vars_to_check:
        if v not in ds.data_vars:
            print(f"[{name}:{v}] NOT FOUND (available: {list(ds.data_vars)})")
            continue

        da = ds[v]

        print(f"\n=== {name} : {v} ===")

        # 1. Encoded fill/nodata attributes
        enc = {k: da.encoding.get(k) for k in ("_FillValue", "missing_value") if k in da.encoding}
        att = {k: da.attrs.get(k) for k in ("_FillValue", "missing_value", "nodata", "nodatavals") if k in da.attrs}
        print(f"  encoding fill : {enc}")
        print(f"  attr fill     : {att}")

        # 2. Grab a small slice to compute stats (avoid loading everything)
        if "time" in da.dims:
            sample = da.isel(time=0)
        else:
            sample = da
        sample = sample.compute()
        vals = sample.values

        # 3. Corner probe (top-left cell = typically over land/off-grid = sentinel)
        corner_tl = float(vals.ravel()[0])                # first cell
        corner_tr = float(vals[0, -1]) if vals.ndim >= 2 else np.nan
        corner_bl = float(vals[-1, 0]) if vals.ndim >= 2 else np.nan
        corner_br = float(vals[-1, -1]) if vals.ndim >= 2 else np.nan
        print(f"  corners TL/TR/BL/BR : {corner_tl:.4g} / {corner_tr:.4g} / {corner_bl:.4g} / {corner_br:.4g}")

        # 4. Distribution stats (ignoring existing NaNs)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            print(f"  min/max       : {finite.min():.4g} / {finite.max():.4g}")
            print(f"  has NaN?      : {np.isnan(vals).any()}  (NaN count={int(np.isnan(vals).sum())})")

        # 5. Count suspected sentinels
        for cand in (-1, -99999, -9999, -999, 0):
            n = int((vals == cand).sum())
            if n:
                frac = 100 * n / vals.size
                print(f"  == {cand:>7}  : {n} cells ({frac:.1f}%)")

        # 6. Most common single value (often reveals the sentinel over land)
        u, c = np.unique(vals[np.isfinite(vals)], return_counts=True)
        if u.size:
            top = u[np.argmax(c)]
            print(f"  most common   : {top:.4g} ({100*c.max()/vals.size:.1f}% of grid)")

for name, (path, fmt, var) in SOURCES.items():
    try:
        probe(name, path, fmt, var)
    except Exception as e:
        print(f"[{name}] ERROR: {e}")



################# Exhaustive check of full dataset
SOURCES = {
    "glsea":    ("/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/glsea_sst.zarr", "zarr"),
    "glsea3":   ("/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/glsea3_sst.zarr", "zarr"),
    "era5":     ("/Users/jagraha/Data/ciglr_globus/greatlakes_era5_1995_2025.zarr", "zarr"),
    "ice":      ("/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/ice_concentration.zarr", "zarr"),
    "bathy":    ("~/dev/deepsensor_projects/data/bathy/gl_bathy_depth_3arcsec_clean.nc", "netcdf"),
    "lakemask": ("~/dev/deepsensor_projects/data/bathy/gl_landmask_3arcsec.nc", "netcdf"),
    "dist":     ("~/dev/deepsensor_projects/data/bathy/gl_dist_to_land_3arcsec.nc", "netcdf"),
    "core":     ("~/dev/deepsensor_projects/data/bathy/gl_core_depth_3arcsec.nc", "netcdf"),
}
SENTINELS = [-1, -99999]

def check_source(name, path, fmt):
    path = str(Path(path).expanduser())
    try:
        ds = xr.open_zarr(path) if fmt == "zarr" else xr.open_dataset(path, chunks="auto")
    except Exception as e:
        print(f"[{name}] OPEN ERROR: {e}")
        return

    for v in ds.data_vars:
        da = ds[v]
        line = f"  {name}:{v:<18}"
        for s in SENTINELS:
            # (da == s) is lazy for dask-backed arrays; .any() reduces, .compute() executes
            found = bool((da == s).any().compute())
            n = int((da == s).sum().compute()) if found else 0
            flag = f"⚠️  {n} cells" if found else "clean"
            line += f" | =={s:>7}: {flag}"
        print(line)


print("Scanning full datasets for exact -1 / -99999 sentinels...\n")
for name, (path, fmt) in SOURCES.items():
    check_source(name, path, fmt)

print("\nDone. 'clean' = no exact sentinel values present anywhere in the variable.")