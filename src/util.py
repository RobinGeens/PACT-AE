import os
from enum import StrEnum
from typing import Any, TypeAlias

import numpy as np
from zigzag.cost_model.cost_model import CostModelEvaluation

from src.config import ModelConfig, QuantConfig

CME_T: TypeAlias = CostModelEvaluation
ARRAY_T: TypeAlias = np.ndarray[Any, Any]


class Stage(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"


def get_experiment_id(model: ModelConfig, stage: Stage, quant: QuantConfig, accelerator_name: str):
    """Generate the name of the experiment"""
    assert "yaml" not in accelerator_name and "/" not in accelerator_name
    seq_len = model.prefill_size if stage == Stage.PREFILL else model.decode_size
    return f"{model.parameterized_name}_L={seq_len}_{quant.name}_{stage}_{accelerator_name}"


def get_onnx_path(model: ModelConfig, stage: Stage, quant: QuantConfig):
    ONNX_DIR = "outputs/onnx"
    name = f"{model.parameterized_name}_PREFILL_SIZE={model.prefill_size}_DECODE_SIZE={model.decode_size}_{quant.name}_{stage}.onnx"
    return f"{ONNX_DIR}/{name}"


def get_accelerator_path(accelerator_name: str):
    DEFAULT_DIR = "inputs/single_core_system"
    assert not os.path.splitext(accelerator_name)[1]  # Gives the file extension or False if no extension
    assert not os.path.dirname(os.path.normpath(accelerator_name))
    return f"{DEFAULT_DIR}/{accelerator_name}.yaml"


def get_accelerator_name_and_path(accelerator_name_or_path: str):
    """Given either the full path of an accelerator yaml file, or just the name of an accelerator saved in the default
    path, return both the name (without path or extension) and the full path."""
    path_and_name, extension = os.path.splitext(accelerator_name_or_path)
    # This is a path
    if extension == ".yaml":
        name = os.path.basename(path_and_name)
        return name, accelerator_name_or_path
    # This is just a name
    elif not extension:
        full_path = get_accelerator_path(accelerator_name_or_path)
        return accelerator_name_or_path, full_path
    else:
        raise ValueError("Argument is not a name or a yaml file")


def accelerator_to_pretty_name(accelerator: str):
    name, _ = get_accelerator_name_and_path(accelerator)
    if name.startswith("generic_array"):
        return "Cloud Architecture"
    if name.startswith("generic_array_edge"):
        return "Edge Architecture"
    # Default
    return name
