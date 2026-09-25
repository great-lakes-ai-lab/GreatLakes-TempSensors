# src/run_greatlakes_deepsensor.py
"""CLI entry point for the Great Lakes DeepSensor pipeline."""

import argparse
from pathlib import Path
import yaml
import warnings



from pipeline.config import load_config, PipelineConfig, copy_config_to_run_dir, load_al_config
from pipeline.data_loader import load_raw_datasets
from pipeline.preprocessor import load_processed_cache, _cache_exists, preprocess_all
from pipeline.task_builder import build_task_loader, gen_tasks, make_train_val_dates, make_train_date_sampler
from pipeline.model import setup_device, build_model, load_trained_model
from pipeline.trainer import train_model
from utils.seeds import derive_rng


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
             "Options: preprocess, train, predict, evaluate, diagnostics, active_learning",
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
            stages = ["active_learning", "skill_curve"]
        else:
            stages = ["preprocess", "train", "evaluate"]
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
    read_only_stages = {"diagnostics", "predict", "evaluate", "active_learning", "skill_curve"}
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
        raise ValueError("Diagnostics currently deprecated. Need to adjust for post processed coord names")
        from pipeline.diagnostics import run_diagnostics
        run_diagnostics(config, bundle)
        if stages == ["diagnostics"]:
            return



    # 6. Build TaskLoader (needed for train and predict)
    print("\n" + "=" * 60)
    print("BUILDING TASKS")
    print("=" * 60)
    tl_config = build_task_loader(config, bundle)

    if 'model_preview' in stages:
        from pipeline.preview import run_preview
        run_preview(config, bundle, tl_config)

        from utils.dates import dates_from_intervals
        # Get a list of the # of train, val, and eval tasks

        print("\n" + "=" * 60)
        print("STAGE: PREVIEW")
        print("=" * 60)
        train_dates, val_dates = make_train_val_dates(config)
        tc = config.training

        print("Validation Task Sampling:")
        print("With the current configuration there will be:")
        print(f"              {len(val_dates)} validation tasks. Striding {tc.val_date_stride} days")
        print(f"              first date: {val_dates[0].date()}, last date: {val_dates[-1].date()}")

        print("\n" + "-" * 60)
        print("Training Task Sampling:")
        if tc.train_date_mode == "random":
            date_sampler, _, _ = make_train_date_sampler(config)
        else:
            print(f"              {len(train_dates)} train tasks. Striding {tc.train_date_stride} days")

        print("\n" + "-" * 60)
        print("Evaluation Tasks")
        ec = config.evaluation
        eval_dates = dates_from_intervals(tc.test_range, ec.date_subsample_factor)
        print(f"                 Evaluation will use {len(eval_dates)} dates. Striding {ec.date_subsample_factor} days ")
        print(f"                  first date: {eval_dates[0].date()}, last date: {eval_dates[-1].date()}")
        # Plot out what the receptive field looks like
        model = build_model(config, bundle, tl_config.task_loader)

    # 7. Train
    if "train" in stages:
        print("\n" + "=" * 60)
        print("STAGE: TRAINING")
        print("=" * 60)
        tc = config.training
        train_dates, val_dates = make_train_val_dates(config)   # with the revamp of the date selection this really just returns the val dates. train dates gets derived
        # print(f"Generating tasks: {len(train_dates)} train, {len(val_dates)} val")

        print(f"Generating {len(val_dates)} validation tasks")
        # Should the N context be the same for all validation tasks?
        # TODO Consider making flavors of context points (fully random, the active plus random, subset active + random, active only)
        #   perhaps a percentage of tasks and flavor ie {random: 40, active + random: 30, active subset + random: 20, active only: 10}
        val_tasks = gen_tasks(tl_config, val_dates, bundle, config, seed=derive_rng(tc.train_task_seed, "val"), vary_n_context=False, n_context=tc.n_context_points)

        if tc.train_date_mode == "random":
            date_sampler, _, _ = make_train_date_sampler(config)
        else:
            date_sampler = None

        if tc.resample_tasks_per_epoch:
            _cache = {}
            def sample_train_tasks(epoch):
                if epoch in _cache:
                    return _cache.pop(epoch)
                dates_ep = date_sampler(epoch) if date_sampler else train_dates
                tasks = gen_tasks(
                    tl_config, dates_ep, bundle, config,
                    seed=derive_rng(tc.train_task_seed, epoch, "context"),
                    progress=False, verbose=False,
                )
                if not tasks:
                    raise RuntimeError(
                        f"Epoch {epoch}: 0 tasks from {len(dates_ep)} dates."
                    )
                if epoch == 1:
                    pct = 100 * len(tasks) / len(dates_ep)
                    print(f"  Task coverage: {len(tasks)}/{len(dates_ep)} dates "
                          f"yielded tasks ({pct:.1f}%)")
                    if pct < 90:
                        print(f"  WARNING: {len(dates_ep) - len(tasks)} dates "
                              f"produced no task. Re-run gen_tasks with "
                              f"verbose=True to inspect.")
                return tasks
            train_tasks = sample_train_tasks(1)
            _cache[1] = train_tasks
            train_task_sampler = sample_train_tasks
        else:
            train_tasks = gen_tasks(tl_config, train_dates, bundle, config, seed=derive_rng(tc.train_task_seed, "fixed_train"))
            train_task_sampler = None

        model = build_model(config, bundle, tl_config.task_loader)
        # TODO plot export of the model receptive field
        results = train_model(model, tl_config.task_loader, train_tasks, val_tasks, bundle, config, train_task_sampler=train_task_sampler)


        print(f"\nBest RMSE: {results['best_val_rmse']:.4f} at epoch {results['best_epoch']}")

    # 8. Predict
    if "predict" in stages:
        print("\n" + "=" * 60)
        print("STAGE: PREDICTION")
        print("=" * 60)
        from pipeline.predict import run_predictions
        run_predictions(config, bundle, tl_config)

    # 8.5 Evaluate on held-out split
    if "evaluate" in stages:
        print("\n" + "=" * 60)
        print("STAGE: EVALUATION")
        print("=" * 60)
        from pipeline.evaluate import run_evaluation
        run_evaluation(config, bundle, tl_config)

    # 9. Active learning
    if "active_learning" in stages:
        print("\n" + "=" * 60)
        print("STAGE: ACTIVE LEARNING")
        print("=" * 60)
        from pipeline.active_learning import run_active_learning
        run_active_learning(config, bundle, tl_config)

    # 10. Skill curve
    if "skill_curve" in stages:
        from pipeline.skill_curve import run_skill_curve
        run_skill_curve(config, bundle, tl_config)

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
        # "--config", "/Users/jagraha/dev/deepsensor_projects/runs/Erie_Eval_Pipeline_Modest/config_used.yaml",
        # "--config", "/Users/jagraha/dev/deepsensor_projects/runs/sep16_yml_test/al_config_yml_test.yaml",
        # "--stage", "skill_curve",
        "--stage", "model_preview"
    ]
    main()