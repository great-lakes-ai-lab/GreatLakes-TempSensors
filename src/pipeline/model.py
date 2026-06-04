# src/pipeline/model.py
"""ConvNP model initialization, saving, and loading."""
from pathlib import Path
import shutil

import torch
import deepsensor.torch
from deepsensor.model import ConvNP
from deepsensor.train import set_gpu_default_device
from deepsensor_greatlakes.model import save_model, load_convnp_model

from pipeline.config import PipelineConfig


def detect_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_device() -> str:
    """Configure the best available device and set DeepSensor default."""
    device = detect_device()

    if device in ("cuda", "mps"):
        set_gpu_default_device()

    if device == "cuda":
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print("Using MPS (Apple Silicon)")
    else:
        print("Using CPU")

    return device


def build_model(config: PipelineConfig, bundle: dict, task_loader) -> ConvNP:
    """
    Instantiate a fresh ConvNP model.

    Parameters
    ----------
    config : PipelineConfig
    bundle : dict
        Must contain 'data_processor'
    task_loader : TaskLoader

    Returns
    -------
    ConvNP model instance
    """
    model = ConvNP(
        bundle["data_processor"],
        task_loader,
        internal_density=config.training.internal_density,
    )

    print(f"ConvNP model built (internal_density={config.training.internal_density})")
    return model


def save_trained_model(model: ConvNP, config: PipelineConfig):
    """Save model weights and config to the run's model directory."""
    model_dir = Path(config.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    save_model(model, str(model_dir))

    # Copy DataProcessor config alongside model for reproducibility
    dp_source = Path(config.paths.data_processor_dir)
    dp_dest = model_dir / "data_processor"

    if dp_source.exists():
        if dp_dest.exists():
            shutil.rmtree(dp_dest)
        shutil.copytree(str(dp_source), str(dp_dest))

    print(f"Model saved to: {model_dir}")


def load_trained_model(config: PipelineConfig, bundle: dict, task_loader) -> ConvNP:
    model_dir = Path(config.paths.model_dir)

    model = load_convnp_model(
        str(model_dir),
        bundle["data_processor"],
        task_loader,
    )

    print(f"Model loaded from: {model_dir}")
    return model