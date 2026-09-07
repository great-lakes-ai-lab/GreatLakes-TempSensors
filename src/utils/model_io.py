# src/utils/model_io.py
"""ConvNP model save/load utilities."""

import os
import json
import re
import pprint

import torch
import torch.nn as nn

from deepsensor.model import ConvNP
from deepsensor.data import DataProcessor, TaskLoader


def save_model(model, model_ID: str):
    """
    Save ConvNP model weights and config to disk.

    Parameters
    ----------
    model : ConvNP
        Trained model instance.
    model_ID : str
        Directory path to save model files.
    """
    os.makedirs(model_ID, exist_ok=True)
    torch.save(model.model.state_dict(), os.path.join(model_ID, "model.pt"))
    config_fpath = os.path.join(model_ID, "model_config.json")
    with open(config_fpath, "w") as f:
        json.dump(model.config, f, indent=4, sort_keys=False, default=str)


def load_convnp_model(model_ID: str, data_processor: DataProcessor, task_loader: TaskLoader):
    """
    Load a saved ConvNP model from disk.

    Parameters
    ----------
    model_ID : str
        Directory path containing model.pt and model_config.json.
    data_processor : DataProcessor
        Fitted DataProcessor instance.
    task_loader : TaskLoader
        Configured TaskLoader instance.

    Returns
    -------
    ConvNP
        Model with loaded weights.
    """
    config_fpath = os.path.join(model_ID, "model_config.json")
    with open(config_fpath, "r") as f:
        config_raw = json.load(f)

    deserialized_config = _deserialize_config(config_raw)

    # Prepare config for constructing the underlying neural process
    config_for_nps_constructor = deserialized_config.copy()

    for key in ('family', 'neural_process_type', 'data_processor', 'task_loader'):
        config_for_nps_constructor.pop(key, None)

    print("Attempting to instantiate ConvNP model (randomly initialized initially):")
    print("Architectural config:", config_for_nps_constructor)

    try:
        loaded_convnp_model = ConvNP(
            data_processor,
            task_loader,
            **config_for_nps_constructor,
        )
    except Exception as e:
        print(f"Error when instantiating ConvNP: {e}")
        debug_info = {
            'data_processor_arg': data_processor,
            'task_loader_arg': task_loader,
            'architectural_kwargs': config_for_nps_constructor,
        }
        pprint.pprint(debug_info)
        raise

    model_weights_fpath = os.path.join(model_ID, "model.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loaded_convnp_model.model.load_state_dict(
        torch.load(model_weights_fpath, map_location=device, weights_only=True)
    )
    loaded_convnp_model.model.to(device)
    loaded_convnp_model.config = deserialized_config

    return loaded_convnp_model


# --- Private helpers ---

def _convert_string_to_numeric_if_possible(value):
    """Attempt to convert string representations back to numeric types."""
    if isinstance(value, str):
        if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value):
            try:
                return float(value)
            except ValueError:
                pass
    return value


def _deserialize_config(config_data):
    """Recursively deserialize a config dict from JSON, restoring types."""
    if isinstance(config_data, dict):
        return {key: _deserialize_config(value) for key, value in config_data.items()}
    elif isinstance(config_data, list):
        return [_deserialize_config(item) for item in config_data]
    else:
        converted_val = _convert_string_to_numeric_if_possible(config_data)
        if isinstance(converted_val, str):
            if converted_val == "<class 'torch.nn.modules.activation.ReLU'>":
                return nn.ReLU
            elif converted_val == "<class 'torch.nn.modules.activation.LeakyReLU'>":
                return nn.LeakyReLU
            return converted_val
        return converted_val