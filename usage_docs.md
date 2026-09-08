This is under dev

## Data Sources
- glsea_sst.zarr                  1995-01-01 → 2023-12-31
- glsea3_sst.zarr                 2007-01-01 → 2025-12-31
- ice_concentration.zarr          1995-01-01 → 2025-12-31
- greatlakes_era5_1995_2025.zarr  1995-01-01 → 2024-12-31

Make copies of the yaml templates

## This part for the preprocessing section in hte config yaml
preprocessing:
  fit_range: []  # Date range(s) used to FIT the climatology and DataProcessor scaling.
                 # Leave empty to inherit training.train_range (recommended — prevents
                 # val/test leakage). Supports non-contiguous years, e.g.
                 # [["2016-01-01","2016-12-31"], ["2018-01-01","2018-12-31"]]
  force_reprocess: false

  # ─── Seasonal cycle / anomaly options ────────────────────────────────────
  # Only used when a data source has use_anomalies: true. The cycle is fitted
  # on fit_range only, then subtracted from the FULL record to form anomalies.
  climatology_method: harmonic
    # monthly   — 12 monthly means. Simple and exactly zero-mean over the fit
    #             period, but produces 12 discontinuous steps: anomalies jump
    #             artificially at month boundaries. Fine for a quick baseline.
    # daily_doy — 366 day-of-year means (Andersson et al. style). Highest
    #             fidelity, but needs MANY fit years: each DOY is estimated from
    #             only (n_fit_years) samples. With <10 years, use smoothing or
    #             prefer 'harmonic'.
    # harmonic  — Fourier harmonics least-squares fitted to the daily-DOY
    #             climatology. Smooth, continuous across New Year, and robust
    #             with few fit years. Recommended default.

  n_harmonics: 3
    # Number of annual Fourier harmonics (method: harmonic only).
    # Fits 2*n_harmonics + 1 parameters to 366 day-of-year values.
    #   1  → pure annual sinusoid (7-param equivalent: 3). Symmetric warm/cool
    #        seasons; too smooth for lakes, which warm slower than they cool.
    #   2  → adds semi-annual term. Captures basic spring/fall asymmetry.
    #   3  → RECOMMENDED. Resolves the asymmetric Great Lakes cycle (slow
    #        stratified spring warm-up, rapid autumn overturn) without overfit.
    #   4-6 → sharper shoulder seasons and ice-off transitions. Use if you have
    #        >10 fit years and see systematic residuals in spring.
    #   >6 → starts fitting interannual noise as if it were seasonal signal.
    #        A warning is raised. Not advised with short fit ranges.

  smooth_window: 15
    # Circular rolling-mean width in DAYS (method: daily_doy only). Ignored by
    # the other methods. "Circular" = Dec 31 and Jan 1 are treated as adjacent,
    # so there is no discontinuity at the year boundary.
    #   0 or 1 → no smoothing. Raw DOY means; exactly zero-mean over the fit
    #            period but very noisy unless you have decades of data.
    #   7      → light smoothing. Retains sharp transitions (ice-off, rapid
    #            spring warming) but leaves visible day-to-day wiggle.
    #   15     → RECOMMENDED starting point. Removes most sampling noise while
    #            preserving genuine sub-monthly structure.
    #   31     → heavy smoothing; roughly comparable in smoothness to 'monthly'
    #            but without the step discontinuities.
    #   >45    → over-smoothed. You are approximating a 1-2 harmonic fit at
    #            greater cost; use method: harmonic instead.
    # Rule of thumb: the noise in each DOY mean scales as 1/sqrt(n_fit_years),
    # so fewer fit years → larger window needed.