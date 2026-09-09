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
    dates = dates_from_intervals(split_range, ec.date_subsample_factor)

    if len(dates) == 0:
        raise ValueError(f"No dates resolved from {split}_range={split_range}")

    out_dir = Path(config.paths.run_dir) / "evaluation" / split
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
        "evaluated_at": str(pd.Timestamp.now()),
    }

    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    per_task_df.to_csv(out_dir / "per_task_scores.csv", index=False)

    _plot_skill_curve(config, agg, split, out_dir)

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

