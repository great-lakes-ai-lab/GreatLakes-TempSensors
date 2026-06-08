# src/run_greatlakes_deepsensor_pipeline.py
"""CLI entry point for the Great Lakes DeepSensor pipeline."""

import argparse
from pathlib import Path

from pipeline.config import load_config, PipelineConfig, copy_config_to_run_dir
from pipeline.data_loader import load_raw_datasets
from pipeline.preprocessor import load_processed_cache, _cache_exists, preprocess_all
from pipeline.task_builder import build_task_loader, gen_tasks, make_train_val_dates
from pipeline.model import setup_device, build_model, load_trained_model
from pipeline.trainer import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Great Lakes DeepSensor Pipeline"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["preprocess", "train", "diagnostics", "predict", "all"],
        help="Which stage to run (default: all)",
    )

    args = parser.parse_args()

    # 1. Load config
    config = load_config(args.config)

    # Stages that just read existing outputs — no reprocessing, no config copy
    read_only_stages = ["diagnostics", "predict"]
    if args.stage in read_only_stages:
        config.preprocessing.force_reprocess = False
    else:
        copy_config_to_run_dir(config, args.config)

    print(f"Config loaded: lake={config.lake}, env={config.environment}, run={config.run.name}")

    # 2. Setup device
    device = setup_device()

    # 3. Preprocess (loads from cache if available, skips raw load)
    bundle = run_preprocessing(config)
    print("Preprocessing complete.")

    if args.stage == "preprocess":
        print("Done (preprocess only).")
        return

    if args.stage == "diagnostics":
        from pipeline.diagnostics import run_diagnostics
        run_diagnostics(config, bundle)
        return

    # 4. Build TaskLoader
    task_loader = build_task_loader(config, bundle)

    if args.stage == "predict":
        from pipeline.predict import run_predictions
        run_predictions(config, bundle, task_loader)
        return

    # 5. Generate tasks
    train_dates, val_dates = make_train_val_dates(config)
    print(f"Generating tasks: {len(train_dates)} train, {len(val_dates)} val")

    train_tasks = gen_tasks(task_loader, train_dates, bundle, config, seed=42)
    val_tasks = gen_tasks(task_loader, val_dates, bundle, config, seed=123)

    # 6. Build model and train
    model = build_model(config, bundle, task_loader)
    results = train_model(model, task_loader, train_tasks, val_tasks, bundle, config)

    print(f"\nRun '{config.run.name}' complete.")
    print(f"Best RMSE: {results['best_val_rmse']:.4f} at epoch {results['best_epoch']}")


def run_preprocessing(config: PipelineConfig) -> dict:
    """Load from cache if available, otherwise load raw and preprocess."""
    processed_dir = Path(config.paths.processed_dir)
    dp_dir = Path(config.paths.data_processor_dir)

    if not config.preprocessing.force_reprocess and _cache_exists(processed_dir, dp_dir):
        print("Loading from processed cache (skipping raw data load)...")
        return load_processed_cache(config)

    print("Loading raw datasets...")
    raw_datasets = load_raw_datasets(config)
    return preprocess_all(config, raw_datasets)


if __name__ == "__main__":

    # import sys
    #
    # sys.argv = [
    #     "run_greatlakes_deepsensor.py",
    #     "--config", "/Users/jagraha/dev/deepsensor_projects/runs/run05_erie_baseline/config_used.yaml",
    #     "--stage", "predict",
    # ]
    main()