# src/pipeline/trainer.py
"""Training loop with validation, checkpointing, and diagnostics."""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from deepsensor.train import Trainer

from pipeline.config import PipelineConfig
from pipeline.model import save_trained_model
from pipeline.task_builder import TaskLoaderConfig
from utils.metrics import point_area_weights, weighted_mse, aggregate_rmse, compute_weighted_rmse

# Canonical metric identifier, written into training_metadata.json
METRIC_NAME = "area_weighted_rmse_per_task_mean"

def train_model(
    model,
    task_loader,
    train_tasks: list,
    val_tasks: list,
    bundle: dict,
    config: PipelineConfig,
    train_task_sampler=None,    # callable(epoch) -> list[Task]
) -> dict:

    if isinstance(task_loader, TaskLoaderConfig):
        task_loader = task_loader.task_loader

    tc = config.training
    trainer = Trainer(model, lr=tc.lr)

    losses = []
    val_rmses = []
    train_rmses = []
    best_val_rmse = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_per_task_details = []

    # How often to compute training RMSE (expensive, so not every epoch)
    train_rmse_interval = max(1, tc.n_epochs // 10)

    start_time = time.time()
    t_sample_total = 0.0
    t_train_total = 0.0
    t_val_total = 0.0

    print(f"Training for {tc.n_epochs} epochs | lr={tc.lr} | "
          f"{len(train_tasks)} train tasks | {len(val_tasks)} val tasks")
    if tc.patience > 0:
        print(f"Early stopping enabled: patience={tc.patience}")
    if train_task_sampler is not None:
        what = ("dates + context points" if tc.train_date_mode == "random" else "context points")
        print(f"Task resampling: ON (new {what} epoch)")
    else:
        print("Task resampling: OFF (static train tasks")

    for epoch in tqdm(range(1, tc.n_epochs + 1), desc="Training"):
        if train_task_sampler is not None:
            _t0 = time.time()
            train_tasks = train_task_sampler(epoch)
            t_train_total += time.time() - _t0
            if epoch == 1:
                n_req = getattr(train_task_sampler, "n_requested", None)
                if n_req:
                    n_skip = n_req - len(train_tasks)
                    pct = 100 * len(train_tasks) / n_req
                    print(f"  Epoch 1: {len(train_tasks)}/{n_req} dates yielded tasks "
                          f"({pct:.1f}%); {n_skip} skipped")
                    if pct < 90:
                        print(f"  WARNING: {n_skip} dates ({100 - pct:.1f}%) produced no task. "
                              f"Re-run with verbose task generation to inspect coverage gaps.")
                    else:
                        print(f"  Epoch 1 sampled {len(train_tasks)} train tasks")
            elif len(train_tasks) == 0:
                raise RuntimeError(f"Epoch {epoch}: task sampler returned 0 tasks.")


        # Train
        _t0 = time.time()
        batch_losses = trainer(train_tasks)
        t_train_total += time.time() - _t0
        epoch_loss = float(np.mean(batch_losses))
        losses.append(epoch_loss)

        # Validate
        _t0 = time.time()
        val_result = compute_weighted_rmse(model, val_tasks, bundle, task_loader, return_per_task=True)
        t_val_total += time.time() - _t0
        val_rmse = val_result["rmse"]
        val_rmses.append(val_rmse)

        # Training RMSE (periodic)
        if epoch % train_rmse_interval == 0 or epoch == 1:
            train_rmse = compute_weighted_rmse(model, train_tasks, bundle, task_loader)["rmse"]
            train_rmses.append({"epoch": epoch, "rmse": train_rmse})

        # Checkpoint best
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            best_per_task_details = val_result["per_task"]
            save_trained_model(model, config)
        else:
            epochs_without_improvement += 1

        # Log periodically
        if epoch % 5 == 0 or epoch == 1:
            train_rmse_str = ""
            if train_rmses:
                train_rmse_str = f", train_rmse={train_rmses[-1]['rmse']:.4f}"
            print(f"  Epoch {epoch:3d}: loss={epoch_loss:.4f}, "
                  f"val_rmse={val_rmse:.4f}{train_rmse_str}, best={best_val_rmse:.4f} "
                  f"(patience: {epochs_without_improvement}/{tc.patience if tc.patience > 0 else '∞'})")

        # Early stopping check
        if tc.patience > 0 and epochs_without_improvement >= tc.patience:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"No improvement for {tc.patience} epochs.")
            break

        # Free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60.0

    print(f"\nTraining complete in {elapsed_min:.1f} minutes. "
          f"Best val RMSE: {best_val_rmse:.4f} (epoch {best_epoch})")
    n_ep = len(losses)
    print(f"  Time breakdown (total / per-epoch):")
    print(f"    task sampling : {t_sample_total:6.1f}s / {t_sample_total / n_ep:5.2f}s "
          f"({100 * t_sample_total / elapsed:4.1f}%)")
    print(f"    training      : {t_train_total:6.1f}s / {t_train_total / n_ep:5.2f}s "
          f"({100 * t_train_total / elapsed:4.1f}%)")
    print(f"    validation    : {t_val_total:6.1f}s / {t_val_total / n_ep:5.2f}s "
          f"({100 * t_val_total / elapsed:4.1f}%)")
    print(f"    other/plots   : {elapsed - t_sample_total - t_train_total - t_val_total:6.1f}s")

    results = {
        "losses": losses,
        "val_rmses": val_rmses,
        "train_rmses": train_rmses,
        "best_val_rmse": float(best_val_rmse),
        "best_epoch": best_epoch,
        "total_epochs": len(losses),
        "early_stopped": tc.patience > 0 and epochs_without_improvement >= tc.patience,
        "training_time_seconds": float(elapsed),
        "training_time_minutes": float(elapsed_min),
        "training_time_minutes": float(elapsed_min),
        "time_sampling_seconds": float(t_sample_total),
        "time_training_seconds": float(t_train_total),
        "time_validation_seconds": float(t_val_total),
        "best_per_task_details": best_per_task_details,
    }

    # Save training metadata and plots
    _save_training_metadata(config, results)
    _save_training_plots(config, results)

    return results


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------
def _select_per_task_examples(per_task, n_best=5, n_worst=5, n_random=10, seed=42):
    """
    Select a readable subset of per-task validation results:
      - n_worst highest-RMSE tasks
      - n_best lowest-RMSE tasks
      - n_random random tasks from the remaining set
    """
    df = pd.DataFrame(per_task).copy()

    if len(df) == 0:
        return df

    df["date_short"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    worst = df.nlargest(min(n_worst, len(df)), "rmse").copy()
    worst["group"] = "Worst"

    best = df.nsmallest(min(n_best, len(df)), "rmse").copy()
    best["group"] = "Best"

    used_idx = set(worst.index).union(set(best.index))
    remaining = df.drop(index=list(used_idx), errors="ignore")

    if len(remaining) > 0 and n_random > 0:
        random = remaining.sample(
            n=min(n_random, len(remaining)),
            random_state=seed,
        ).copy()
        random["group"] = "Random"
    else:
        random = pd.DataFrame(columns=df.columns.tolist() + ["group"])

    selected = pd.concat([worst, random, best], axis=0)

    # Helpful order: worst first, random middle, best last
    group_order = {"Worst": 0, "Random": 1, "Best": 2}
    selected["_group_order"] = selected["group"].map(group_order)
    selected = selected.sort_values(["_group_order", "rmse"], ascending=[True, False])

    return selected


def _save_training_metadata(config: PipelineConfig, results: dict):
    """Save training run metadata to JSON."""
    model_dir = Path(config.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tc = config.training
    metadata = {
        "run_name": config.run.name,
        "run_notes": config.run.notes,
        "lake": config.lake,
        "environment": config.environment,
        "fit_range": list(config.preprocessing.fit_range),
        "train_range": list(tc.train_range),
        "val_range": list(tc.val_range),
        "test_range": list(tc.test_range),
        "train_date_mode": tc.train_date_mode,
        "train_date_stride": tc.train_date_stride,
        "train_date_fraction": tc.train_date_fraction,
        "n_train_dates_per_epoch": tc.n_train_dates_per_epoch,
        "val_date_stride": tc.val_date_stride,
        "n_epochs": tc.n_epochs,
        "lr": tc.lr,
        "patience": tc.patience,
        "total_epochs": results["total_epochs"],
        "early_stopped": results["early_stopped"],
        "internal_density": tc.internal_density,
        "n_context_points": tc.n_context_points,
        "vary_n_context": tc.vary_n_context,
        "resample_tasks_per_epoch": tc.resample_tasks_per_epoch,
        "train_task_seed": tc.train_task_seed,
        "min_n_context": tc.min_n_context,
        "max_n_context": tc.max_n_context,
        "include_bathy_as_context": tc.include_bathy_as_context,
        "bathy_context_sampling": tc.bathy_context_sampling,
        "metric": METRIC_NAME,
        "best_val_rmse": results["best_val_rmse"],
        "best_epoch": results["best_epoch"],
        "final_train_loss": results["losses"][-1],
        "losses": [float(l) for l in results["losses"]],
        "val_rmses": [float(r) for r in results["val_rmses"]],
        "training_time_seconds": results["training_time_seconds"],
        "training_time_minutes": results["training_time_minutes"],
        "time_sampling_seconds": results["time_sampling_seconds"],
        "time_training_seconds": results["time_training_seconds"],
        "time_validation_seconds": results["time_validation_seconds"],
    }

    meta_path = model_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Training metadata saved to: {meta_path}")


def _save_training_plots(config: PipelineConfig, results: dict):
    """Save training curves plots."""
    import matplotlib.pyplot as plt
    from pipeline.plotting import _finish_plot

    model_dir = Path(config.paths.model_dir)
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # epochs = range(1, len(results["losses"]) + 1)
    # losses = results["losses"]
    # val_rmses = results["val_rmses"]

    # ─── Plot 1: Training loss and RMSE summary ───────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    losses = np.array(results["losses"], dtype=float)
    val_rmses = np.array(results["val_rmses"], dtype=float)
    train_rmses = results.get("train_rmses", [])

    epochs = np.arange(1, len(losses) + 1)
    best_epoch = results["best_epoch"]
    best_rmse = results["best_val_rmse"]

    # Panel 1: Training loss
    ax = axes[0]
    ax.plot(
        epochs,
        losses,
        color="tab:blue",
        linewidth=1.3,
        alpha=0.8,
        label="Training loss",
    )

    # Smoothed training loss
    if len(losses) > 10:
        window = max(5, len(losses) // 10)
        smoothed_loss = (
            pd.Series(losses)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .values
        )
        ax.plot(
            epochs,
            smoothed_loss,
            color="tab:blue",
            linewidth=2.4,
            alpha=0.9,
            label=f"Smoothed loss (w={window})",
        )

    ax.axvline(
        x=best_epoch,
        color="green",
        linestyle=":",
        alpha=0.8,
        label=f"Best RMSE epoch ({best_epoch})",
    )

    if results.get("early_stopped", False):
        ax.axvline(
            x=len(losses),
            color="orange",
            linestyle="--",
            alpha=0.8,
            label=f"Early stop ({len(losses)})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: Training RMSE vs validation RMSE
    ax = axes[1]

    ax.plot(
        epochs,
        val_rmses,
        color="tab:red",
        linewidth=1.5,
        label="Validation RMSE",
    )

    if train_rmses:
        train_epochs = np.array([d["epoch"] for d in train_rmses], dtype=int)
        train_rmse_vals = np.array([d["rmse"] for d in train_rmses], dtype=float)

        ax.plot(
            train_epochs,
            train_rmse_vals,
            color="tab:blue",
            marker="o",
            markersize=4,
            linewidth=1.5,
            label="Training RMSE",
        )

        # Generalization gap at epochs where train RMSE was computed
        val_at_train_epochs = np.array([val_rmses[e - 1] for e in train_epochs])
        ax.fill_between(
            train_epochs,
            train_rmse_vals,
            val_at_train_epochs,
            color="red",
            alpha=0.08,
            label="Generalization gap",
        )

        final_gap = val_at_train_epochs[-1] - train_rmse_vals[-1]
        title_suffix = f"\nFinal gap: {final_gap:.4f}"
    else:
        title_suffix = ""

    ax.scatter(
        [best_epoch],
        [best_rmse],
        color="green",
        s=70,
        zorder=5,
        label=f"Best val RMSE: {best_rmse:.4f}",
    )

    ax.axvline(x=best_epoch, color="green", linestyle=":", alpha=0.8)

    if results.get("early_stopped", False):
        ax.axvline(
            x=len(losses),
            color="orange",
            linestyle="--",
            alpha=0.8,
            label=f"Early stop ({len(losses)})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Train vs Validation RMSE{title_suffix}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Training Summary — {config.lake.upper()} | {config.run.name}\n"
        f"lr={config.training.lr}, density={config.training.internal_density}, "
        f"n_context={config.training.n_context_points}",
        fontsize=11,
    )

    plt.tight_layout()
    _finish_plot(config, plots_dir / "training_curves.png")

    # ─── Plot 2: Learning rate / convergence diagnostics ────────────────

    if len(results.get("losses", [])) > 1 and len(results.get("val_rmses", [])) > 1:
        losses = np.array(results["losses"], dtype=float)
        val_rmses = np.array(results["val_rmses"], dtype=float)

        epochs_delta = np.arange(2, len(losses) + 1)

        train_loss_delta = np.diff(losses)
        val_rmse_delta = np.diff(val_rmses)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Panel 1: Training loss delta
        ax = axes[0]

        ax.plot(
            epochs_delta,
            train_loss_delta,
            color="tab:blue",
            linewidth=1.0,
            alpha=0.6,
            label="Δ Training loss",
        )

        if len(train_loss_delta) > 10:
            window = max(5, len(train_loss_delta) // 10)
            train_smooth = (
                pd.Series(train_loss_delta)
                .rolling(window=window, center=True, min_periods=1)
                .mean()
                .values
            )

            ax.plot(
                epochs_delta,
                train_smooth,
                color="tab:blue",
                linewidth=2.4,
                alpha=0.9,
                label=f"Smoothed Δ loss (w={window})",
            )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.8)

        best_epoch = results.get("best_epoch", None)
        if best_epoch is not None and best_epoch > 1:
            ax.axvline(
                x=best_epoch,
                color="green",
                linestyle=":",
                linewidth=1.3,
                alpha=0.8,
                label=f"Best RMSE epoch ({best_epoch})",
            )

        if results.get("early_stopped", False):
            ax.axvline(
                x=len(losses),
                color="orange",
                linestyle="--",
                linewidth=1.3,
                alpha=0.8,
                label=f"Early stop ({len(losses)})",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Δ Training Loss")
        ax.set_title(
            "Training Loss Change per Epoch\n"
            "negative = improving, near 0 = plateau"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2: Validation RMSE delta
        ax = axes[1]

        ax.plot(
            epochs_delta,
            val_rmse_delta,
            color="tab:red",
            linewidth=1.0,
            alpha=0.6,
            label="Δ Validation RMSE",
        )

        if len(val_rmse_delta) > 10:
            window = max(5, len(val_rmse_delta) // 10)
            rmse_smooth = (
                pd.Series(val_rmse_delta)
                .rolling(window=window, center=True, min_periods=1)
                .mean()
                .values
            )

            ax.plot(
                epochs_delta,
                rmse_smooth,
                color="tab:red",
                linewidth=2.4,
                alpha=0.9,
                label=f"Smoothed Δ RMSE (w={window})",
            )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.8)

        if best_epoch is not None and best_epoch > 1:
            ax.axvline(
                x=best_epoch,
                color="green",
                linestyle=":",
                linewidth=1.3,
                alpha=0.8,
                label=f"Best RMSE epoch ({best_epoch})",
            )

        if results.get("early_stopped", False):
            ax.axvline(
                x=len(losses),
                color="orange",
                linestyle="--",
                linewidth=1.3,
                alpha=0.8,
                label=f"Early stop ({len(losses)})",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Δ Validation RMSE")
        ax.set_title(
            "Validation RMSE Change per Epoch\n"
            "negative = improving, near 0 = plateau"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Convergence Diagnostics — {config.lake.upper()} | {config.run.name}",
            fontsize=11,
        )
        plt.tight_layout()

        _finish_plot(config, plots_dir / "convergence_diagnostics.png")
    else:
        print(
            "Skipping convergence diagnostics plot: "
            "requires at least 2 epochs of losses and val_rmses."
        )
    # ─── Plot 3: Per-task validation breakdown (at best epoch) ──────────

    if results.get("best_per_task_details"):
        per_task = results["best_per_task_details"]

        task_rmses = [t["rmse"] for t in per_task]
        task_dates = [t["date"] for t in per_task]
        task_n_context = [t["n_context"] for t in per_task]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Selected per-task RMSE examples
        ax = axes[0, 0]

        selected = _select_per_task_examples(
            per_task,
            n_best=5,
            n_worst=5,
            n_random=10,
            seed=42,
        )

        if len(selected) > 0:
            labels = [
                f"{row['group']}: {row['date_short']} (n={int(row['n_context'])})"
                for _, row in selected.iterrows()
            ]

            color_map = {
                "Worst": "tab:red",
                "Random": "tab:gray",
                "Best": "tab:green",
            }
            colors = [color_map[g] for g in selected["group"]]

            y = np.arange(len(selected))

            ax.barh(y, selected["rmse"], color=colors, alpha=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=7)
            ax.invert_yaxis()

            ax.axvline(
                x=np.mean(task_rmses),
                color="black",
                linestyle="--",
                alpha=0.6,
                label=f"Mean: {np.mean(task_rmses):.4f}",
            )

            ax.set_xlabel("RMSE")
            ax.set_title("Selected Per-Task RMSE\n5 worst, 10 random, 5 best")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3, axis="x")
        else:
            ax.text(0.5, 0.5, "No per-task results", ha="center", va="center")
            ax.set_title("Selected Per-Task RMSE")

        # Panel 2: RMSE histogram
        ax = axes[0, 1]
        ax.hist(task_rmses, bins=min(20, len(task_rmses)), color="steelblue",
                edgecolor="white", alpha=0.8)
        ax.axvline(x=np.mean(task_rmses), color="red", linestyle="--",
                   label=f"Mean: {np.mean(task_rmses):.4f}")
        ax.axvline(x=np.median(task_rmses), color="orange", linestyle="--",
                   label=f"Median: {np.median(task_rmses):.4f}")
        ax.set_xlabel("RMSE")
        ax.set_ylabel("Count")
        ax.set_title("RMSE Distribution Across Tasks")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 3: RMSE vs n_context scatter
        ax = axes[1, 0]
        scatter = ax.scatter(task_n_context, task_rmses, c=task_rmses,
                             cmap="RdYlGn_r", edgecolors="black", linewidths=0.5,
                             s=50, alpha=0.8)
        plt.colorbar(scatter, ax=ax, label="RMSE")

        # Trend line
        if len(set(task_n_context)) > 1:
            z = np.polyfit(task_n_context, task_rmses, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(task_n_context), max(task_n_context), 50)
            ax.plot(x_line, p(x_line), "r--", alpha=0.7,
                    label=f"Trend (slope={z[0]:.4f})")
            ax.legend()

        ax.set_xlabel("N Context Points")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE vs Context Points\n(more context → lower error?)")
        ax.grid(True, alpha=0.3)

        # Panel 4: RMSE by month (seasonal difficulty)
        ax = axes[1, 1]
        try:
            task_months = [pd.Timestamp(d).month for d in task_dates]
            month_data = {}
            for month, rmse in zip(task_months, task_rmses):
                month_data.setdefault(month, []).append(rmse)

            months_sorted = sorted(month_data.keys())
            month_means = [np.mean(month_data[m]) for m in months_sorted]
            month_stds = [np.std(month_data[m]) if len(month_data[m]) > 1 else 0
                          for m in months_sorted]

            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            ax.bar(months_sorted, month_means, yerr=month_stds, capsize=3,
                   color="steelblue", edgecolor="white", alpha=0.8)
            ax.set_xticks(months_sorted)
            ax.set_xticklabels([month_labels[m - 1] for m in months_sorted])
            ax.set_xlabel("Month")
            ax.set_ylabel("Mean RMSE")
            ax.set_title("Seasonal Difficulty\n(which months are hardest?)")
            ax.grid(True, alpha=0.3, axis="y")
        except Exception:
            ax.text(0.5, 0.5, "Could not parse dates\nfor seasonal analysis",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Seasonal Difficulty")

        plt.suptitle(
            f"Per-Task Validation Analysis — {config.lake.upper()} | {config.run.name}\n"
            f"Best epoch {best_epoch} | {len(per_task)} val tasks | "
            f"Mean RMSE: {np.mean(task_rmses):.4f} ± {np.std(task_rmses):.4f}",
            fontsize=11,
        )
        plt.tight_layout()
        _finish_plot(config, plots_dir / "per_task_validation.png")

    print(f"Training plots saved to: {plots_dir}")