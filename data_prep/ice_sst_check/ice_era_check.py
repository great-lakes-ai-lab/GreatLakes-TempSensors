"""
ice_era_check.py — resolve the ~5% level and the 371 reverse-direction exceptions.
"""
import numpy as np
import pandas as pd
import xarray as xr

SST_PATH = "/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/glsea3_sst.zarr"
ICE_PATH = "/Users/jagraha/dev/deepsensor_projects/data/temporal_inputs/ice_concentration.zarr"
SST_VAR, ICE_VAR = "sst", "ice_concentration"
FLAG = np.float32(0.2)

RUN_A = True   # level set by year, 1995-2025
RUN_B = True   # does GLSEA flag the ~5 level?
RUN_C = True   # explain sst==0.2 & ice==0
RUN_D = True   # locate the 2 ice~30 outliers

DATES_6 = ["2012-01-15", "2014-02-15", "2015-02-20",
           "2016-01-31", "2019-02-01", "2022-02-10"]


def open_store(p):
    for kw in ({"consolidated": True}, {"consolidated": False}):
        try:
            return xr.open_zarr(p, mask_and_scale=False, **kw)
        except Exception as e:
            last = e
    raise last


def didx(ds):
    return pd.DatetimeIndex(ds["time"].values).floor("D")


def pos_of(idx, dates):
    lut = pd.Series(np.arange(len(idx)), index=idx)
    out = {}
    for d in dates:
        k = pd.Timestamp(d)
        if k in lut.index:
            v = lut.loc[k]
            out[d] = int(v if np.ndim(v) == 0 else np.asarray(v).ravel()[0])
    return out


def dilate(mask, n=1):
    m = mask.copy()
    for _ in range(n):
        p = np.pad(m, 1, constant_values=False)
        m = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


ice_ds, sst_ds = open_store(ICE_PATH), open_store(SST_PATH)
ice_idx, sst_idx = didx(ice_ds), didx(sst_ds)
lat, lon = ice_ds["latitude"].values, ice_ds["longitude"].values

# ---------------------------------------------------------------- A
if RUN_A:
    print("=" * 78, "\nA) ICE LEVEL SET BY YEAR (3 dates per winter)\n", "=" * 78)
    targets = [f"{y}-{md}" for y in range(1995, 2026)
               for md in ("01-25", "02-10", "02-25")]
    p = pos_of(ice_idx, targets)
    keys = list(p)
    cube = ice_ds[ICE_VAR].isel(time=[p[k] for k in keys]).load()
    rows = []
    for i, k in enumerate(keys):
        a = np.asarray(cube.isel(time=i).values)
        v = a[np.isfinite(a)]
        lv = np.unique(np.round(v, 2))
        n5 = int(((v > 0) & (v < 12)).sum())
        rows.append(dict(date=k, n_valid=v.size, n_levels=lv.size,
                         has_low=bool(n5 > 0), n_low_0_12=n5,
                         min_nonzero=float(v[v > 0].min()) if (v > 0).any() else np.nan,
                         levels=",".join(f"{x:g}" for x in lv)))
    dfA = pd.DataFrame(rows)
    dfA["year"] = pd.DatetimeIndex(dfA.date).year
    print(dfA.groupby("year").agg(
        dates=("date", "size"), any_low=("has_low", "any"),
        low_cells=("n_low_0_12", "max"),
        min_nonzero=("min_nonzero", "min")).to_string())
    print("\nDistinct level-set signatures:")
    for sig, g in dfA.groupby("levels"):
        yrs = sorted(set(pd.DatetimeIndex(g.date).year))
        print(f"  yrs={yrs[0]}-{yrs[-1]} (n={len(g)}): {sig}")
    dfA.to_csv("/Users/jagraha/dev/deepsensor_projects/tmp/sst_ice_checks/ice_levels_by_year.csv", index=False)

# ---------------------------------------------------------------- B
if RUN_B:
    print("\n" + "=" * 78, "\nB) DOES GLSEA FLAG THE LOW (~5) LEVEL?\n", "=" * 78)
    probe = ["2010-01-26", "2010-01-28", "2010-01-30", "2010-02-01",
             "2008-02-10", "2009-02-10", "2011-02-10"]
    pi, ps = pos_of(ice_idx, probe), pos_of(sst_idx, probe)
    both = [d for d in probe if d in pi and d in ps]
    ic = ice_ds[ICE_VAR].isel(time=[pi[d] for d in both]).load()
    sc = sst_ds[SST_VAR].isel(time=[ps[d] for d in both]).load()
    for i, d in enumerate(both):
        a, b = np.asarray(sc.isel(time=i).values), np.asarray(ic.isel(time=i).values)
        ok = np.isfinite(a) & np.isfinite(b)
        lev, s = np.round(b[ok] / 5.0) * 5.0, a[ok]
        print(f"\n{d}   {'lvl':>6} {'n':>9} {'==0.2':>9} {'frac':>7} "
              f"{'min':>7} {'med':>7} {'max':>7}")
        for L in np.unique(lev):
            m = lev == L
            sl = s[m]
            nf = int((sl == FLAG).sum())
            tag = "  <-- LOW LEVEL" if 0 < L <= 10 else ""
            print(f"{'':11}{L:>6.0f} {m.sum():>9,} {nf:>9,} {nf/m.sum():>7.4f} "
                  f"{sl.min():>7.2f} {np.median(sl):>7.2f} {sl.max():>7.2f}{tag}")

# ---------------------------------------------------------------- C
if RUN_C:
    print("\n" + "=" * 78, "\nC) WHY sst==0.2 WHERE ice==0 ?\n", "=" * 78)
    ps = pos_of(sst_idx, DATES_6)
    rows = []
    for d in DATES_6:
        a = np.asarray(sst_ds[SST_VAR].isel(time=ps[d]).values)
        lag = {}
        for L in (-1, 0, 1):
            dd = str((pd.Timestamp(d) + pd.Timedelta(days=L)).date())
            q = pos_of(ice_idx, [dd])
            if q:
                lag[L] = np.asarray(ice_ds[ICE_VAR].isel(time=q[dd]).values)
        b = lag[0]
        ok = np.isfinite(a) & np.isfinite(b)
        bad = ok & (a == FLAG) & (b == 0)
        n = int(bad.sum())
        r = dict(date=d, n_exceptions=n)
        for k in (1, 2, 3):
            r[f"within_{k}"] = int((bad & dilate(ok & (b > 0), k)).sum())
        for L in (-1, 1):
            if L in lag:
                r[f"lag{L:+d}_ice_gt0"] = int((bad & np.isfinite(lag[L])
                                               & (lag[L] > 0)).sum())
        rows.append(r)
        print(f"{d}: n={n:>4}  adj1={r['within_1']:>4} adj2={r['within_2']:>4} "
              f"adj3={r['within_3']:>4}  "
              f"lag-1={r.get('lag-1_ice_gt0','-')} lag+1={r.get('lag+1_ice_gt0','-')}")
    dfC = pd.DataFrame(rows)
    dfC.to_csv("reverse_exceptions.csv", index=False)
    print("\n", dfC.to_string(index=False))

# ---------------------------------------------------------------- D
if RUN_D:
    print("\n" + "=" * 78, "\nD) ice~30 BUT sst>0.2 ON 2012-01-15\n", "=" * 78)
    d = "2012-01-15"
    a = np.asarray(sst_ds[SST_VAR].isel(time=pos_of(sst_idx, [d])[d]).values)
    b = np.asarray(ice_ds[ICE_VAR].isel(time=pos_of(ice_idx, [d])[d]).values)
    ok = np.isfinite(a) & np.isfinite(b)
    m = ok & (b > 0) & (a != FLAG)
    for y, x in zip(*np.nonzero(m)):
        nb = b[max(0, y-2):y+3, max(0, x-2):x+3]
        print(f"  ({y},{x}) lat={lat[y]:.4f} lon={lon[x]:.4f} "
              f"sst={a[y,x]:.2f} ice={b[y,x]:.3f}  "
              f"nbhd_ice_min={np.nanmin(nb):.1f} max={np.nanmax(nb):.1f}")