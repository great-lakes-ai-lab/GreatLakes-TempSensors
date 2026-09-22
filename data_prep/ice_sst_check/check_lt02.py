"""
explore_sub_02.py — characterize GLSEA3 SST values at and below 0.2 degC.

Zarr v3, local stores, NaN-only nodata.
"""
import numpy as np
import pandas as pd
import xarray as xr

# ======================================================================
# CONFIG
# ======================================================================
SST_PATH = "/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/glsea3_sst.zarr"
ICE_PATH = "/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/ice_concentration.zarr"
SST_VAR, ICE_VAR = "sst", "ice_concentration"

# Spread across high-ice (2014, 2015), low-ice (2012), and a known-probed date.
DATES = [
    "2012-01-15",   # low-ice winter
    "2014-02-15",   # near-record ice
    "2015-02-20",   # high ice
    "2016-01-31",   # already probed in script 2
    "2019-02-01",
    "2022-02-10",
]

THRESH = np.float32(0.2)      # exact float32 0.2 == 0.20000000298023224
MAX_ROWS = 60                 # cap on printed unique-value rows
INCLUDE_ICE = True            # co-located ice context
LABEL_LAKES = True            # rough bbox lake attribution
OUT_CSV = "/Users/jagraha/dev/deepsensor_projects/tmp/sst_ice_checks/sub02_value_counts.csv"
OUT_CSV_DATE = "/Users/jagraha/dev/deepsensor_projects/tmp/sst_ice_checks/sub02_per_date.csv"


# ======================================================================
# HELPERS
# ======================================================================
def open_store(path):
    """zarr v3 tolerant open; NaN is the only fill so mask_and_scale is moot."""
    for kw in ({"consolidated": True}, {"consolidated": False}):
        try:
            return xr.open_zarr(path, mask_and_scale=False, **kw)
        except Exception as e:
            last = e
    raise last


def date_index(ds):
    """Floor to calendar day: SST is 12:00Z, ice mixes 00:00Z and 12:00Z."""
    return pd.DatetimeIndex(ds["time"].values).floor("D")


def resolve(idx, dates, label):
    """Map date strings -> positional indices, reporting any misses."""
    pos, missing = [], []
    lut = pd.Series(np.arange(len(idx)), index=idx)
    for d in dates:
        key = pd.Timestamp(d)
        if key in lut.index:
            v = lut.loc[key]
            pos.append(int(v if np.ndim(v) == 0 else np.asarray(v).ravel()[0]))
        else:
            missing.append(d)
    if missing:
        print(f"  !! {label}: dates absent from store -> {missing}")
    return pos


LAKE_BOXES = [  # (name, lat_min, lat_max, lon_min, lon_max) -- ROUGH, first match wins
    ("Superior",  46.40, 49.10, -92.20, -84.30),
    ("Michigan",  41.60, 46.20, -88.10, -84.70),
    ("StClair",   42.30, 42.70, -82.95, -82.30),
    ("Erie",      41.30, 43.00, -83.60, -78.80),
    ("Ontario",   43.10, 44.40, -79.90, -76.00),
    ("Huron",     42.90, 46.60, -84.80, -79.60),
]


def label_lake(lat, lon):
    for name, la0, la1, lo0, lo1 in LAKE_BOXES:
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return name
    return "other"


def neighbor_cohesion(mask):
    """Fraction of True cells having >=1 True 4-neighbor. Scattered noise -> ~0."""
    n = int(mask.sum())
    if n == 0:
        return np.nan
    p = np.pad(mask, 1, constant_values=False)
    nb = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
    return float((mask & nb).sum()) / n


def fmt_ice(v):
    if v.size == 0:
        return "n=0"
    return (f"n={v.size:,} min={v.min():.4g} med={np.median(v):.4g} "
            f"max={v.max():.4g} frac>0={float((v > 0).mean()):.3f}")
# ======================================================================
# LOAD
# ======================================================================
sst_ds = open_store(SST_PATH)
sst_idx = date_index(sst_ds)
sst_pos = resolve(sst_idx, DATES, "SST")
# Single fancy-index selection so each 90-deep time chunk is decompressed once.
sst_cube = sst_ds[SST_VAR].isel(time=sst_pos).load()

lat = sst_ds["latitude"].values
lon = sst_ds["longitude"].values

ice_cube = None
if INCLUDE_ICE:
    ice_ds = open_store(ICE_PATH)
    ice_pos = resolve(date_index(ice_ds), DATES, "ICE")
    ice_cube = ice_ds[ICE_VAR].isel(time=ice_pos).load()
    assert np.array_equal(ice_ds["latitude"].values, lat), "lat mismatch"
    assert np.array_equal(ice_ds["longitude"].values, lon), "lon mismatch"

resolved_dates = [str(d.date()) for d in sst_idx[sst_pos]]
print(f"Loaded {len(resolved_dates)} dates: {resolved_dates}\n")




# ======================================================================
# PER-DATE ANALYSIS
# ======================================================================
rows_date, rows_val, sub_masks = [], [], {}

for k, dstr in enumerate(resolved_dates):
    a = np.asarray(sst_cube.isel(time=k).values)
    finite = np.isfinite(a)
    v = a[finite]

    m_le   = finite & (a <= THRESH)
    m_eq   = finite & (a == THRESH)
    m_sub  = finite & (a < THRESH)          # strictly below the spike
    m_zero = finite & (a == np.float32(0.0))
    m_neg  = finite & (a < np.float32(0.0))
    m_band = finite & (a > THRESH) & (a <= np.float32(0.3))   # just above, for contrast

    sub_masks[dstr] = m_sub

    print("=" * 74)
    print(f"{dstr}   valid={int(finite.sum()):,}  NaN={int((~finite).sum()):,}  "
          f"min={v.min():.6g}  max={v.max():.6g}")
    print(f"  <= 0.2        : {int(m_le.sum()):>8,}")
    print(f"     == 0.2     : {int(m_eq.sum()):>8,}   <-- flag spike")
    print(f"     <  0.2     : {int(m_sub.sum()):>8,}   <-- population of interest")
    print(f"        == 0.0  : {int(m_zero.sum()):>8,}")
    print(f"        <  0.0  : {int(m_neg.sum()):>8,}")
    print(f"  (0.2, 0.3]    : {int(m_band.sum()):>8,}   (contrast band)")
    print(f"  cohesion(<0.2): {neighbor_cohesion(m_sub):.3f}  "
          f"cohesion(==0.2): {neighbor_cohesion(m_eq):.3f}")

    # ---- distinct values at or below threshold -----------------------
    vals, counts = np.unique(a[m_le], return_counts=True)
    print(f"\n  distinct values <= 0.2  (n_distinct={vals.size})")
    print(f"    {'value (float32 exact)':<26} {'short':>10} {'count':>10}  note")
    for vv, cc in list(zip(vals, counts))[:MAX_ROWS]:
        note = ""
        if vv == THRESH:
            note = "EXACT 0.2 FLAG"
        elif vv == np.float32(0.0):
            note = "exact zero"
        elif vv < 0:
            note = "NEGATIVE"
        print(f"    {float(vv)!r:<26} {float(vv):>10.4g} {int(cc):>10,}  {note}")
    if vals.size > MAX_ROWS:
        print(f"    ... {vals.size - MAX_ROWS} more rows suppressed")

    # ---- ice context -------------------------------------------------
    ice_sub_gt0 = ice_sub_eq0 = ice_sub_nan = np.nan
    if ice_cube is not None:
        ic = np.asarray(ice_cube.isel(time=k).values)
        i_sub, i_eq = ic[m_sub], ic[m_eq]
        print(f"\n  ice @ (sst <  0.2): {fmt_ice(i_sub[np.isfinite(i_sub)])}")
        print(f"  ice @ (sst == 0.2): {fmt_ice(i_eq[np.isfinite(i_eq)])}")
        fs = np.isfinite(i_sub)
        ice_sub_nan = int((~fs).sum())
        ice_sub_gt0 = int((i_sub[fs] > 0).sum())
        ice_sub_eq0 = int((i_sub[fs] == 0).sum())
        print(f"  sub-0.2 split -> ice>0: {ice_sub_gt0:,}   "
              f"ice==0: {ice_sub_eq0:,}   ice NaN: {ice_sub_nan:,}")

    # ---- locations of strictly-sub-0.2 cells -------------------------
    yy, xx = np.nonzero(m_sub)
    if yy.size:
        print(f"\n  sub-0.2 cell locations (first 25 of {yy.size}):")
        for j in range(min(25, yy.size)):
            la, lo = lat[yy[j]], lon[xx[j]]
            lk = f"  {label_lake(la, lo)}" if LABEL_LAKES else ""
            ice_s = ""
            if ice_cube is not None:
                iv = np.asarray(ice_cube.isel(time=k).values)[yy[j], xx[j]]
                ice_s = f"  ice={iv:8.4f}" if np.isfinite(iv) else "  ice=   NaN"
            print(f"    ({yy[j]:>4},{xx[j]:>4})  lat={la:8.4f} lon={lo:9.4f}"
                  f"  sst={float(a[yy[j], xx[j]])!r:<24}{ice_s}{lk}")
        if LABEL_LAKES:
            lk_counts = pd.Series(
                [label_lake(lat[y], lon[x]) for y, x in zip(yy, xx)]
            ).value_counts()
            print(f"\n  sub-0.2 by (rough) lake:\n{lk_counts.to_string()}")
    print()

    rows_date.append(dict(
        date=dstr, n_valid=int(finite.sum()), n_nan=int((~finite).sum()),
        sst_min=float(v.min()), sst_max=float(v.max()),
        n_le_02=int(m_le.sum()), n_eq_02=int(m_eq.sum()), n_sub_02=int(m_sub.sum()),
        n_eq_0=int(m_zero.sum()), n_neg=int(m_neg.sum()), n_band_02_03=int(m_band.sum()),
        n_distinct_le_02=int(vals.size),
        cohesion_sub=neighbor_cohesion(m_sub), cohesion_eq=neighbor_cohesion(m_eq),
        sub_ice_gt0=ice_sub_gt0, sub_ice_eq0=ice_sub_eq0, sub_ice_nan=ice_sub_nan,
    ))
    for vv, cc in zip(vals, counts):
        rows_val.append(dict(date=dstr, value=float(vv), count=int(cc),
                             is_flag=bool(vv == THRESH)))


# ======================================================================
# CROSS-DATE SYNTHESIS
# ======================================================================
df_date = pd.DataFrame(rows_date)
df_val = pd.DataFrame(rows_val)

print("=" * 74)
print("PER-DATE SUMMARY")
print("=" * 74)
with pd.option_context("display.width", 200, "display.max_columns", 50):
    print(df_date.to_string(index=False))

print("\n" + "=" * 74)
print("UNION OF DISTINCT VALUES <= 0.2 ACROSS ALL DATES")
print("=" * 74)
agg = (df_val.groupby("value")
       .agg(total=("count", "sum"), n_dates=("date", "nunique"))
       .sort_index())
print(agg.to_string())

# Persistent vs transient sub-0.2 cells
if sub_masks:
    stack = np.stack(list(sub_masks.values()))
    freq = stack.sum(axis=0)
    print(f"\nsub-0.2 cell persistence across {stack.shape[0]} dates:")
    for n in range(1, stack.shape[0] + 1):
        c = int((freq == n).sum())
        if c:
            print(f"  cells sub-0.2 on exactly {n} date(s): {c:,}")
    always = np.nonzero(freq == stack.shape[0])
    if always[0].size:
        print(f"  ALWAYS sub-0.2 ({always[0].size} cells) -- suspect bad pixels:")
        for y, x in list(zip(*always))[:20]:
            print(f"    ({y},{x}) lat={lat[y]:.4f} lon={lon[x]:.4f}")

df_date.to_csv(OUT_CSV_DATE, index=False)
agg.to_csv(OUT_CSV)
print(f"\nwrote {OUT_CSV_DATE} and {OUT_CSV}")


# Reverse direction, broken out by snapped ice level.
for k, dstr in enumerate(resolved_dates):
    a = np.asarray(sst_cube.isel(time=k).values)
    ic = np.asarray(ice_cube.isel(time=k).values)
    ok = np.isfinite(a) & np.isfinite(ic)
    lev = np.round(ic[ok] / 5.0) * 5.0          # snap 4.997 -> 5, 99.998 -> 100
    s = a[ok]
    print(f"\n{dstr}")
    print(f"  {'ice_lvl':>8} {'n':>9} {'==0.2':>9} {'frac':>7} {'sst_min':>8} {'sst_max':>8}")
    for L in np.unique(lev):
        m = lev == L
        sl = s[m]
        n_flag = int((sl == np.float32(0.2)).sum())
        print(f"  {L:>8.0f} {m.sum():>9,} {n_flag:>9,} {n_flag/m.sum():>7.3f} "
              f"{sl.min():>8.2f} {sl.max():>8.2f}")