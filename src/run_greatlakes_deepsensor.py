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
        help="Which stages to run. Comma-separated or 'all'. "
             "Options: preprocess, train, predict, diagnostics",
    )

    args = parser.parse_args()

    # 1. Load and validate config
    config = load_config(args.config)
    config.validate()

    # 2. Parse stages
    if args.stage == "all":
        stages = ["preprocess", "train", "predict"]
    else:
        stages = [s.strip() for s in args.stage.split(",")]

    # Only copy config to run dir if we're modifying outputs
    read_only_stages = {"diagnostics", "predict", "active_learning"}
    if all(s in read_only_stages for s in stages):
        config.preprocessing.force_reprocess = False
    else:
        copy_config_to_run_dir(config, args.config)

    print(f"Config loaded: lake={config.lake}, env={config.environment}, run={config.run.name}")
    print(f"Stages to run: {stages}")

    # 3. Setup device
    device = setup_device()

    # 4. Preprocess (loads from cache if available)
    bundle = run_preprocessing(config)
    print("Preprocessing complete.")

    if stages == ["preprocess"]:
        print("Done (preprocess only).")
        return

    # 5. Diagnostics (optional, can run standalone or alongside others)
    if "diagnostics" in stages:
        from pipeline.diagnostics import run_diagnostics
        run_diagnostics(config, bundle)
        if stages == ["diagnostics"]:
            return

    # 6. Build TaskLoader (needed for train and predict)
    print("\n" + "=" * 60)
    print("BUILDING TASKS")
    print("=" * 60)
    tl_config = build_task_loader(config, bundle)

    # 7. Train
    if "train" in stages:
        print("\n" + "=" * 60)
        print("STAGE: TRAINING")
        print("=" * 60)
        train_dates, val_dates = make_train_val_dates(config)
        print(f"Generating tasks: {len(train_dates)} train, {len(val_dates)} val")

        train_tasks = gen_tasks(tl_config, train_dates, bundle, config, seed=42)
        val_tasks = gen_tasks(tl_config, val_dates, bundle, config, seed=123)

        model = build_model(config, bundle, tl_config.task_loader)
        results = train_model(model, tl_config.task_loader, train_tasks, val_tasks, bundle, config)

        print(f"\nBest RMSE: {results['best_val_rmse']:.4f} at epoch {results['best_epoch']}")

    # 8. Predict
    if "predict" in stages:
        print("\n" + "=" * 60)
        print("STAGE: PREDICTION")
        print("=" * 60)
        from pipeline.predict import run_predictions
        run_predictions(config, bundle, tl_config)

    # 9. Active learning
    if "active_learning" in stages:
        print("\n" + "=" * 60)
        print("STAGE: ACTIVE LEARNING")
        print("=" * 60)
        from pipeline.active_learning import run_active_learning
        run_active_learning(config, bundle, tl_config)

    print(f"\nRun '{config.run.name}' complete.")


def run_preprocessing(config: PipelineConfig) -> dict:
    """Load from cache if available, otherwise load raw and preprocess."""
    print("\n" + "=" * 60)
    print("STAGE: PREPROCESSING")
    print("=" * 60)
    processed_dir = Path(config.paths.processed_dir)
    dp_dir = Path(config.paths.data_processor_dir)

    if not config.preprocessing.force_reprocess and _cache_exists(processed_dir, dp_dir):
        print("Loading from processed cache (skipping raw data load)...")
        return load_processed_cache(config)

    print("Loading raw datasets...")
    raw_datasets = load_raw_datasets(config)
    return preprocess_all(config, raw_datasets)


if __name__ == "__main__":

    import sys

    sys.argv = [
        "run_greatlakes_deepsensor.py",
        "--config", "/Users/jagraha/dev/repos/GreatLakes-TempSensors/src/config/config_debug_run_local.yaml",
        "--stage", "all",
    ]
    main()