# src/run_greatlakes_deepsensor_pipeline.py
"""CLI entry point for the Great Lakes DeepSensor pipeline."""

import argparse
from pathlib import Path
import yaml
import warnings
warnings.simplefilter("always", DeprecationWarning)


from pipeline.config import load_config, PipelineConfig, copy_config_to_run_dir, load_al_config
from pipeline.data_loader import load_raw_datasets
from pipeline.preprocessor import load_processed_cache, _cache_exists, preprocess_all
from pipeline.task_builder import build_task_loader, gen_tasks, make_train_val_dates, make_train_date_sampler
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

    # 1. Load config (auto-detect AL overlay vs. full pipeline config)
    with open(args.config) as f:
        raw_peek = yaml.safe_load(f)

    is_al_config = "run_ref" in raw_peek

    if is_al_config:
        config = load_al_config(args.config)  # already validated internally
    else:
        config = load_config(args.config)
        config.validate()

    # 2. Parse stages
    if args.stage == "all":
        if is_al_config:
            # An AL overlay config only makes sense for AL-related stages
            stages = ["active_learning"]  # (+ "skill_curve" once added)
        else:
            stages = ["preprocess", "train", "predict"]
    else:
        stages = [s.strip() for s in args.stage.split(",")]

    # ── Active learning stage with train/predict config ──
    al_only_stages = {"active_learning", "skill_curve"}
    if not is_al_config and any(s in al_only_stages for s in stages):
        raise ValueError(
            f"Stage(s) {sorted(set(stages) & al_only_stages)} require an active-learning "
            f" config (one with a top-level 'run_ref'). "
            f"You passed a full-pipeline config. See src/config/al_config_template.yaml."
            f"To run active learing you must first have a trained model and use the AL template to point at the run dir"
        )

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
        train_dates, val_dates = make_train_val_dates(config)   # with the revamp of the date selection this really just returns the val dates. train dates gets derived
        # print(f"Generating tasks: {len(train_dates)} train, {len(val_dates)} val")

        print(f"Generating {len(val_dates)} validation tasks")
        val_tasks = gen_tasks(tl_config, val_dates, bundle, config, seed=123)

        tc = config.training

        if tc.train_date_mode == "random":
            date_sampler, _, _ = make_train_date_sampler(config)
        else:
            date_sampler = None

        if tc.resample_tasks_per_epoch:
            def sample_train_tasks(epoch):
                dates_ep = date_sampler(epoch) if date_sampler else train_dates
                sample_train_tasks.n_requested = len(dates_ep)
                return gen_tasks(tl_config, dates_ep, bundle, config, seed=tc.train_task_seed + epoch, progress=False, verbose=False)
            sample_train_tasks.n_requested = None
            train_tasks = sample_train_tasks(0)
            train_task_sampler = sample_train_tasks
        else:
            train_tasks = gen_tasks(tl_config, train_dates, bundle, config, seed=42)
            train_task_sampler = None

        model = build_model(config, bundle, tl_config.task_loader)
        results = train_model(model, tl_config.task_loader, train_tasks, val_tasks, bundle, config, train_task_sampler=train_task_sampler)


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
        "--config", "/Users/jagraha/dev/repos/GreatLakes-TempSensors/src/config/config_debug_local.yaml",
        # "--config", "/Users/jagraha/dev/deepsensor_projects/runs/run00_resume_dev_usable_model/al_config.yaml",
        "--stage", "train",
    ]
    main()