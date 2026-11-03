# src/pipeline/evaluate.py
"""Generate predictions from a trained model."""

"""Held-out test-set evaluation.

Pure orchestration: loads the best checkpoint, sweeps context-set sizes and
seeds, and delegates all scoring to utils.metrics.compute_weighted_scores so
test numbers are the same metric reported during training.

Produces:
    evaluation/<split>/evaluation_metrics.json
    evaluation/<split>/per_task_scores.csv
    evaluation/<split>/skill_vs_ncontext.png
"""

from pathlib import Path
import json
import calendar

import numpy as np
import pandas as pd

from pipeline.model import load_trained_model
from pipeline.task_builder import gen_tasks
from utils.dates import dates_from_intervals
from utils.metrics import compute_weighted_scores


def _resolve_split_range(config, split: str) -> list:
    ranges = {
        "train": config.training.train_range,
        "val": config.training.val_range,
        "test": config.training.test_range,
    }
    if split not in ranges:
        raise ValueError(f"split must be train/val/test, got '{split}'")
    rng = ranges[split]
    if not rng:
        raise ValueError(
            f"Cannot evaluate on '{split}': training.{split}_range is empty."
        )
    return rng


def run_evaluation(config, bundle, tl_config, split: str = None):
    """
    Evaluate the trained model on a held-out split across context sizes.

    The n_context sweep is the point: a single RMSE is uninterpretable without
    stating how many observations the model was given. This sweep is also the
    random-placement baseline for the later active-learning skill curve.
    """
    ec = config.evaluation
    split = split or ec.split

    split_range = _resolve_split_range(config, split)
    dates = dates_from_intervals(intervals=split_range, subsample_factor=ec.date_subsample_factor, months_to_drop=config.training.test_months_drop)

    if len(dates) == 0:
        raise ValueError(f"No dates resolved from {split}_range={split_range}")

    out_dir = Path(config.paths.run_dir) / "evaluation" / ec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating on split='{split}'")
    print(f"  range:      {split_range}")
    print(f"  dates:      {len(dates)} "
          f"({dates.min().date()} → {dates.max().date()}, "
          f"stride={ec.date_subsample_factor}d)")
    print(f"  n_context:  {ec.n_context_sweep}")
    print(f"  seeds:      {ec.seeds}")

    model = load_trained_model(config, bundle, tl_config.task_loader)

    sweep_rows = []
    all_per_task = []

    # TODO: Should this respect the validation sampling strategy mode?

    for n_context in ec.n_context_sweep:
        for seed in ec.seeds:
            tasks = gen_tasks(
                tl_config,
                dates,
                bundle,
                config,
                n_context=n_context,
                vary_n_context=False,   # fixed size -- the sweep is the variable
                seed=seed,
                progress=False,
            )
            if not tasks:
                print(f"  n_context={n_context} seed={seed}: no tasks, skipping")
                continue

            scores = compute_weighted_scores(
                model, tasks, bundle, tl_config.task_loader,
                return_per_task=True,
            )

            row = {
                "n_context": n_context,
                "seed": seed,
                "n_tasks": scores["n_tasks"],
                "rmse": scores["rmse"],
                "mae": scores["mae"],
                "bias": scores["bias"],
                "nll": scores["nll"],
                "coverage_1sigma": scores["coverage_1sigma"],
                "coverage_2sigma": scores["coverage_2sigma"],
                "mean_std": scores["mean_std"],
            }
            sweep_rows.append(row)

            for r in scores["per_task"]:
                all_per_task.append({**r, "n_context": n_context, "seed": seed})

            print(f"  n_context={n_context:>4} seed={seed:>3} | "
                  f"rmse={row['rmse']:.4f}  nll={row['nll']:.4f}  "
                  f"cov1σ={row['coverage_1sigma']*100:.1f}%  "
                  f"cov2σ={row['coverage_2sigma']*100:.1f}%")

    if not sweep_rows:
        raise RuntimeError("Evaluation produced no results.")

    sweep_df = pd.DataFrame(sweep_rows)
    per_task_df = pd.DataFrame(all_per_task)

    # Aggregate across seeds at each context size
    agg = (
        sweep_df
        .groupby("n_context")
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            nll_mean=("nll", "mean"),
            nll_std=("nll", "std"),
            mae_mean=("mae", "mean"),
            bias_mean=("bias", "mean"),
            cov1_mean=("coverage_1sigma", "mean"),
            cov2_mean=("coverage_2sigma", "mean"),
            mean_std=("mean_std", "mean"),
            n_seeds=("seed", "nunique"),
            n_tasks=("n_tasks", "first"),
        )
        .reset_index()
        .fillna(0.0)
    )

    # Monthly stratification -- seasonal skill varies strongly for lake SST
    # Carries both aggregations: `rmse_pooled` (sqrt of the mean MSE over every task,
    # matcing the headline sweep bumber) and `rmse_mean_over_seeds` (mean of per-seed RMSEs, the center of the +/- 1
    # sigma band in hte figure. They differ slighltly, keeping both lets the plot be reproducible from the JSON alone
    monthly = {}
    if not per_task_df.empty:
        pt = per_task_df.copy()
        pt["month"] = pd.to_datetime(pt["date"]).dt.month
        for n_context, grp in pt.groupby("n_context"):
            m = grp.groupby("month").agg(
                rmse=("mse", lambda s: float(np.sqrt(np.mean(s)))),
                nll=("nll", "mean"),
                n=("mse", "size"),
            )
            monthly[int(n_context)] = {
                int(k): {kk: float(vv) for kk, vv in v.items()}
                for k, v in m.to_dict(orient="index").items()
            }
    monthly_stats = _monthly_stats(per_task_df)
    monthly = {}
    for n_context, grp in monthly_stats.groupby("n_context"):
        monthly[int(n_context)] = {
            int(r["month"]): {
                # back-compatible keys (unchanged meaning)
                "rmse": float(r["rmse_pooled"]),
                "nll": float(r["nll_mean"]),
                "n": int(r["n_tasks"]),
                # new
                "rmse_pooled": float(r["rmse_pooled"]),
                "rmse_mean_over_seeds": float(r["rmse_mean_over_seeds"]),
                "rmse_std_over_seeds": float(r["rmse_std_over_seeds"]),
                "n_seeds": int(r["n_seeds"]),
                "n_tasks": int(r["n_tasks"]),
                "n_tasks_per_seed_mean": float(r["n_tasks_per_seed_mean"]),
                "n_tasks_per_seed_std": float(r["n_tasks_per_seed_std"]),
            }
            for _, r in grp.iterrows()
        }




    # Load training metadata for provenance
    train_meta_path = Path(config.paths.model_dir) / "training_metadata.json"
    train_meta = {}
    if train_meta_path.exists():
        with open(train_meta_path) as f:
            tm = json.load(f)
        train_meta = {
            "best_val_rmse": tm.get("best_val_rmse"),
            "best_epoch": tm.get("best_epoch"),
            "total_epochs": tm.get("total_epochs"),
            "checkpoint_selected_on": tm.get("checkpoint_selected_on", "val"),
        }

    results = {
        "run_name": config.run.name,
        "lake": config.lake,
        "eval_name": ec.name,
        "split": split,
        "split_range": [list(iv) for iv in split_range],
        "date_subsample_factor": ec.date_subsample_factor,
        "n_dates": int(len(dates)),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "n_context_sweep": list(ec.n_context_sweep),
        "seeds": list(ec.seeds),
        "metric": "area-weighted (cos-lat) RMSE, physical units",
        "checkpoint": train_meta,
        "sweep": agg.to_dict(orient="records"),
        "sweep_raw": sweep_df.to_dict(orient="records"),
        "monthly": monthly,
        "monthly_schema": {
            "rmse_pooled": "sqrt(mean(per-task MSE)) over all seeds; matches `sweep` convention",
            "rmse_mean_over_seeds": "mean over seeds of per-seed sqrt(mean(MSE)); plotted centre",
            "rmse_std_over_seeds": "std (ddof=1) across seeds, 0.0 when n_seeds < 2",
            "nll_mean": "mean per-task NLL over all seeds (reported as `nll`)",
            "n": "alias of n_tasks, kept for backward compatibility",
        },
        "evaluated_at": str(pd.Timestamp.now()),
    }

    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    per_task_df.to_csv(out_dir / "per_task_scores.csv", index=False)
    monthly_stats.to_csv(out_dir / "monthly_scores.csv", index=False)

    _plot_skill_curve(config, agg, split, out_dir)
    _plot_monthly_skill(config, per_task_df, split, out_dir, stats=monthly_stats)

    print(f"\n--- Evaluation summary (split={split}) ---")
    for _, r in agg.iterrows():
        print(f"  n_context={int(r['n_context']):>4} | "
              f"rmse={r['rmse_mean']:.4f} ± {r['rmse_std']:.4f} | "
              f"nll={r['nll_mean']:.4f} | "
              f"cov1σ={r['cov1_mean']*100:.1f}% (ideal 68.3%)")
    if train_meta.get("best_val_rmse") is not None:
        print(f"  [reference] best val RMSE during training: "
              f"{train_meta['best_val_rmse']:.4f}")
    print(f"Saved to: {out_dir}")

    return results


def _plot_skill_curve(config, agg, split, out_dir):
    """RMSE / NLL / calibration vs number of context points."""
    import matplotlib.pyplot as plt
    from pipeline.plotting import _finish_plot

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = agg["n_context"].values

    # Panel 1: RMSE
    ax = axes[0]
    ax.plot(x, agg["rmse_mean"], "o-", color="tab:red", label="Area-weighted RMSE")
    if (agg["rmse_std"] > 0).any():
        ax.fill_between(
            x,
            agg["rmse_mean"] - agg["rmse_std"],
            agg["rmse_mean"] + agg["rmse_std"],
            color="tab:red", alpha=0.15,
            label="±1σ across seeds",
        )
    ax.set_xlabel("Number of context points")
    ax.set_ylabel("RMSE (physical units)")
    ax.set_title("Skill vs Context Size\n(random placement baseline)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: NLL
    ax = axes[1]
    ax.plot(x, agg["nll_mean"], "o-", color="tab:blue", label="Gaussian NLL")
    if (agg["nll_std"] > 0).any():
        ax.fill_between(
            x,
            agg["nll_mean"] - agg["nll_std"],
            agg["nll_mean"] + agg["nll_std"],
            color="tab:blue", alpha=0.15, label="±1σ across seeds",
        )
    ax.set_xlabel("Number of context points")
    ax.set_ylabel("NLL (nats)")
    ax.set_title("Predictive Likelihood vs Context Size\n(lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)


    # Panel 3: Calibration
    ax = axes[2]
    ax.plot(x, agg["cov1_mean"] * 100, "o-", color="tab:green", label="Within 1σ")
    ax.plot(x, agg["cov2_mean"] * 100, "s-", color="tab:purple", label="Within 2σ")
    ax.axhline(68.3, color="tab:green", linestyle="--", alpha=0.5,
               label="Ideal 1σ (68.3%)")
    ax.axhline(95.4, color="tab:purple", linestyle="--", alpha=0.5,
               label="Ideal 2σ (95.4%)")
    ax.set_xlabel("Number of context points")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Calibration vs Context Size\n(above ideal = overdispersed)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Held-out Evaluation — {config.lake.upper()} | {config.run.name} | "
        f"split={split}",
        fontsize=12,
    )
    plt.tight_layout()
    _finish_plot(config, out_dir / "skill_vs_ncontext.png")


def _monthly_stats(per_task_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tidy per-(month, n_context) statistics, aggregated over seeds.

    One row per (month, n_context). Returns an empty frame -- never raises --
    if the input is unusable, so callers can guard on `.empty`.

    Two RMSE aggregations are produced deliberately:
      rmse_pooled           sqrt(mean(MSE)) over every task at that cell.
                            Same estimator as the headline sweep RMSE.
      rmse_mean_over_seeds  per-seed RMSE first, then averaged. This is the
                            correct centre for a +/-1 sigma across-seed band;
                            `rmse_pooled` is not (the sqrt is concave, so the
                            pooled value sits slightly below the mean of the
                            per-seed values).
    """
    required = {"date", "n_context", "mse"}
    if per_task_df is None or per_task_df.empty:
        return pd.DataFrame()
    if not required.issubset(per_task_df.columns):
        missing = sorted(required - set(per_task_df.columns))
        print(f"  [monthly] per-task frame missing {missing}; skipping")
        return pd.DataFrame()

    pt = per_task_df.copy()

    # compute_weighted_scores writes str(task.get("time", "unknown"))[:10],
    # so a task without a timestamp lands here as the literal "unknown".
    dt = pd.to_datetime(pt["date"], errors="coerce")
    n_bad = int(dt.isna().sum())
    if n_bad:
        print(f"  [monthly] dropped {n_bad}/{len(pt)} rows with unparseable dates")
    pt = pt.loc[dt.notna()].copy()
    if pt.empty:
        print("  [monthly] no rows with parseable dates; skipping")
        return pd.DataFrame()
    pt["month"] = dt.loc[dt.notna()].dt.month.astype(int)

    if "seed" not in pt.columns:
        pt["seed"] = 0
    if "nll" not in pt.columns:
        pt["nll"] = np.nan

    pt = pt.loc[np.isfinite(pt["mse"].to_numpy(dtype=float))].copy()
    if pt.empty:
        print("  [monthly] no finite MSE values; skipping")
        return pd.DataFrame()

    keys = ["month", "n_context", "seed"]
    per_seed = (
        pt.groupby(keys)
          .agg(
              rmse=("mse", lambda s: float(np.sqrt(np.nanmean(s)))),
              nll=("nll", lambda s: float(np.nanmean(s))),
              n=("mse", "size"),
          )
          .reset_index()
    )

    stats = (
        per_seed.groupby(["month", "n_context"])
        .agg(
            rmse_mean_over_seeds=("rmse", "mean"),
            rmse_std_over_seeds=("rmse", "std"),
            nll_mean=("nll", "mean"),
            nll_std=("nll", "std"),
            n_seeds=("seed", "nunique"),
            n_tasks_per_seed_mean=("n", "mean"),
            n_tasks_per_seed_std=("n", "std"),
            n_tasks=("n", "sum"),
        )
        .reset_index()
    )

    pooled = (
        pt.groupby(["month", "n_context"])["mse"]
          .apply(lambda s: float(np.sqrt(np.nanmean(s))))
          .rename("rmse_pooled")
          .reset_index()
    )
    stats = stats.merge(pooled, on=["month", "n_context"], how="left")

    # std is NaN for a single seed -- that means "no spread measured", not
    # "missing data", so 0.0 is the honest fill here.
    for c in ("rmse_std_over_seeds", "nll_std", "n_tasks_per_seed_std"):
        stats[c] = stats[c].fillna(0.0)

    return stats.sort_values(["n_context", "month"]).reset_index(drop=True)


def _plot_monthly_skill(config, per_task_df, split, out_dir, stats=None):
    """
    Seasonal skill vs context size: 2 rows (RMSE, NLL) x 3 panels
    (lines over months | raw heatmap | change vs smallest context size),
    with a task-count strip above the line column.
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from pipeline.plotting import _finish_plot

    if stats is None:
        stats = _monthly_stats(per_task_df)
    if stats is None or getattr(stats, "empty", True):
        print("  [monthly] nothing to plot; monthly figure skipped")
        return

    months = sorted(int(m) for m in stats["month"].unique())
    ncs = sorted(int(n) for n in stats["n_context"].unique())
    labels = [calendar.month_abbr[m] for m in months]

    def grid(col):
        return (
            stats.pivot_table(index="n_context", columns="month", values=col)
                 .reindex(index=ncs, columns=months)
                 .to_numpy(dtype=float)
        )

    rmse = grid("rmse_mean_over_seeds")
    rmse_sd = grid("rmse_std_over_seeds")
    nll = grid("nll_mean")
    nll_sd = grid("nll_std")
    ntask = grid("n_tasks_per_seed_mean")
    ntask_sd = grid("n_tasks_per_seed_std")

    multi_nc = len(ncs) >= 2
    multi_seed = int(stats["n_seeds"].max()) >= 2
    have_nll = bool(np.isfinite(nll).any())
    annotate = (len(months) * len(ncs)) <= 160  # past this, text is unreadable

    # Dense axis: absent months are closed up, so flag any interior gap rather
    # than letting a compressed axis imply a smoother season than was measured.
    interior = [m for m in range(months[0], months[-1] + 1) if m not in months]

    x = np.arange(len(months))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(ncs)))

    fig = plt.figure(figsize=(19, 11))
    gs = gridspec.GridSpec(
        3, 3, height_ratios=[0.17, 1, 1],
        hspace=0.38, wspace=0.26, figure=fig,
    )

    # --- task-count strip -------------------------------------------------
    ax_bar = fig.add_subplot(gs[0, 0])
    bar_h = np.nanmean(ntask, axis=0)
    # total spread across (n_context, seed): within-cell + between-cell
    bar_e = np.sqrt(np.nanmean(ntask_sd ** 2, axis=0) + np.nanvar(ntask, axis=0))
    ax_bar.bar(x, bar_h, color="0.6", width=0.72)
    if np.any(bar_e > 1e-9):
        # Counts should be identical across context sizes and seeds; visible
        # error bars mean tasks were dropped somewhere (unscorable targets).
        ax_bar.errorbar(x, bar_h, yerr=bar_e, fmt="none",
                        ecolor="tab:red", elinewidth=1.2, capsize=2)
    ax_bar.set_ylabel("tasks\n/ month", fontsize=8)
    ax_bar.tick_params(axis="y", labelsize=7)
    ax_bar.tick_params(labelbottom=False)
    ax_bar.margins(x=0.02)
    ax_bar.grid(True, axis="y", alpha=0.3)

    # --- line panels ------------------------------------------------------
    def lines(ax, M, SD, ylabel, title):
        for i, nc in enumerate(ncs):
            y = np.ma.masked_invalid(M[i])
            ax.plot(x, y, "o-", ms=4, lw=1.6, color=colors[i],
                    label=f"{nc}")
            if multi_seed and np.any(SD[i] > 0):
                lo, hi = M[i] - SD[i], M[i] + SD[i]
                ok = np.isfinite(lo) & np.isfinite(hi)
                ax.fill_between(x, lo, hi, where=ok,
                                color=colors[i], alpha=0.13, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Month (observed only)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if len(ncs) <= 6:
            leg = ax.legend(title="n_context", fontsize=8, title_fontsize=8,
                            ncol=2 if len(ncs) > 3 else 1)
            if not multi_seed:
                leg.set_title("n_context (1 seed:\nno spread shown)")
        else:
            cmap_nc = ListedColormap(colors)
            norm_nc = BoundaryNorm(np.arange(len(ncs) + 1) - 0.5, len(ncs))
            sm = ScalarMappable(cmap=cmap_nc, norm=norm_nc)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, ticks=np.arange(len(ncs)), pad=0.02)
            cb.ax.set_yticklabels([str(n) for n in ncs], fontsize=7)
            cb.set_label("n_context", fontsize=8)

    ax_r_line = fig.add_subplot(gs[1, 0], sharex=ax_bar)
    lines(ax_r_line, rmse, rmse_sd, "RMSE (physical units)",
          "RMSE by month\n(shading = ±1σ across seeds)")

    ax_n_line = fig.add_subplot(gs[2, 0], sharex=ax_bar)
    if have_nll:
        lines(ax_n_line, nll, nll_sd, "NLL (nats)",
              "NLL by month\n(lower is better)")
    else:
        ax_n_line.set_axis_off()
        ax_n_line.text(0.5, 0.5, "NLL unavailable", ha="center", va="center",
                       transform=ax_n_line.transAxes, color="0.4")

    # --- heatmaps ---------------------------------------------------------
    def heat(ax, M, cmap, title, cbar_label, fmt, diverging=False,
             ref_row=False):
        Mm = np.ma.masked_invalid(M)
        if diverging:
            finite = M[np.isfinite(M)]
            lim = float(np.max(np.abs(finite))) if finite.size else 1.0
            lim = lim if lim > 0 else 1.0
            im = ax.imshow(Mm, cmap=cmap, aspect="auto", vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(Mm, cmap=cmap, aspect="auto")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(len(ncs)))
        ax.set_yticklabels(ncs)
        ax.set_xlabel("Month (observed only)")
        ax.set_ylabel("Requested context points")
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(cbar_label, fontsize=9)
        if annotate:
            for i in range(len(ncs)):
                for j in range(len(months)):
                    v = M[i, j]
                    if not np.isfinite(v):
                        continue
                    if ref_row and i == 0:
                        ax.text(j, i, "ref", ha="center", va="center",
                                fontsize=7, color="0.35", style="italic")
                        continue
                    r, g, b, _ = im.cmap(im.norm(v))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=7,
                            color="white" if lum < 0.55 else "black")
        return im

    ax_r_heat = fig.add_subplot(gs[1, 1])
    heat(ax_r_heat, rmse, "magma_r", "RMSE grid\n(dark = worse)",
         "RMSE", "{:.3f}")

    ax_n_heat = fig.add_subplot(gs[2, 1])
    if have_nll:
        heat(ax_n_heat, nll, "viridis_r", "NLL grid\n(dark = worse)",
             "NLL (nats)", "{:.2f}")
    else:
        ax_n_heat.set_axis_off()
        ax_n_heat.text(0.5, 0.5, "NLL unavailable", ha="center", va="center",
                       transform=ax_n_heat.transAxes, color="0.4")

    # --- change vs smallest context size ---------------------------------
    def blank(ax, msg):
        ax.set_axis_off()
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, color="0.4", fontsize=10)

    ax_r_d = fig.add_subplot(gs[1, 2])
    if multi_nc:
        base = rmse[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = 100.0 * (rmse - base) / np.where(base > 0, base, np.nan)
        heat(ax_r_d, pct, "RdBu_r",
             f"RMSE change vs n_context={ncs[0]}\n"
             "(blue = improvement)",
             "Δ RMSE (%)", "{:+.1f}", diverging=True, ref_row=True)
    else:
        blank(ax_r_d, "requires ≥ 2 context sizes")

    ax_n_d = fig.add_subplot(gs[2, 2])
    if multi_nc and have_nll:
        # NLL can be negative, so an absolute difference is the only
        # meaningful "change" here -- percentages would be nonsense.
        dnll = nll - nll[0]
        heat(ax_n_d, dnll, "RdBu_r",
             f"NLL change vs n_context={ncs[0]}\n"
             "(blue = improvement)",
             "Δ NLL (nats)", "{:+.2f}", diverging=True, ref_row=True)
    else:
        blank(ax_n_d, "requires ≥ 2 context sizes"
              if have_nll else "NLL unavailable")

    note = ("centre = mean over seeds of per-seed RMSE "
            f"| {int(stats['n_seeds'].max())} seed(s)")
    if interior:
        note += (" | gap: no data for month(s) "
                 + ", ".join(calendar.month_abbr[m] for m in interior))
    plt.suptitle(
        f"Seasonal Skill vs Context Size — {config.lake.upper()} | "
        f"{config.run.name} | split={split}\n{note}",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    _finish_plot(config, out_dir / "monthly_skill_vs_ncontext.png")