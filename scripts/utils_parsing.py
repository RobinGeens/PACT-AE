import os
from collections import defaultdict
from math import ceil

import numpy as np
import pandas as pd
import seaborn
from zigzag.datatypes import LayerOperand
from zigzag.utils import open_yaml

from scripts.utils_roofline import get_roofline_latency
from src.config import MambaConfig, ModelConfig, QuantConfig
from src.config_library import MAMBA1_2_8B, OPT_2_7B
from src.export_onnx import export_model_to_onnx
from src.util import Stage, get_onnx_path
from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.mapping_parser import MappingParser
from stream.parser.onnx.model import ONNXModelParser
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ONNXWorkload

LAYER_TO_TYPE = {
    # Transformer exclusive
    "norm": "Normalization",
    "Relu": "Activation",
    "Div": "Elementwise",
    "sa/key_proj": "Projection",
    "sa/value_proj": "Projection",
    "sa/query_proj": "Projection",
    "sa/out_proj": "Projection",
    "net/up_proj": "Projection",
    "net/down_proj": "Projection",
    "sa/mul_qk_t": "Attention",
    "sa/mul_logits_v": "Attention",
    "sa": "Attention",
    # Mamba (and some also in Transformer)
    "dt_proj": "Projection",
    "mul_dBx": "SSM",
    "mul_delta_B": "SSM",
    "mul_h_C": "SSM",
    "mul_D_x": "SSM",
    "add_y_Dx": "SSM",
    "mul_delta_A": "SSM",
    "mul_h_dA": "SSM",
    "add_dBx": "SSM",
    "Conv": "Projection",
    "in_proj": "Projection",
    "x_proj": "Projection",
    "out_proj": "Projection",
    "Add": "Elementwise",
    "Exp": "Elementwise",
    "act": "Activation",
    "mul_y_z": "Elementwise",
}

TYPE_ORDER = [
    "Normalization",
    "Activation",
    "Elementwise",
    "Projection",
    "Attention",
    "SSM",
]

COLOR_PALETTE = seaborn.color_palette("pastel")

OP_TYPE_COLORS = {
    "Normalization": COLOR_PALETTE[2],
    "Activation": COLOR_PALETTE[6],
    "Elementwise": COLOR_PALETTE[4],
    "Projection": COLOR_PALETTE[3],
    "Attention": COLOR_PALETTE[0],
    "SSM": COLOR_PALETTE[1],
}

# OPERATOR TYPES TO IGNORE
IGNORE_OP_TYPES = []  # ['Normalization', 'Activation']

ACCELERATOR_PATH = "inputs/multicore_system/single_core_vec_array.yaml"
MAPPING_PATH = "inputs/mapping_multicore/single_core_vec_array_mamba.yaml"


def layer_to_type(layer_name: str):
    types = []
    for name, type in LAYER_TO_TYPE.items():
        if name in layer_name:
            types.append(type)
    if len(types) > 1:
        # print(f"WARNING: Multiple types found for layer {layer_name}: {types}. Returning the first one.")
        return types[0]
    if len(types) == 0:
        raise ValueError(f"No type found for layer {layer_name}")
    return types[0]


def parse_accelerator_from_yaml(yaml_path: str) -> Accelerator:
    accelerator_data = open_yaml(yaml_path)

    validator = AcceleratorValidator(accelerator_data, yaml_path)
    accelerator_data = validator.normalized_data
    validate_success = validator.validate()
    if not validate_success:
        raise ValueError("Failed to validate user provided accelerator.")

    factory = AcceleratorFactory(accelerator_data)
    return factory.create()


def parse_workload(onnx_path: str):
    accelerator = parse_accelerator_from_yaml(ACCELERATOR_PATH)
    mapping_parser = MappingParser(MAPPING_PATH)
    all_mappings = mapping_parser.run()
    onnx_model_parser = ONNXModelParser(onnx_path, all_mappings, accelerator)
    onnx_model_parser.run()
    workload = onnx_model_parser.workload
    return workload


def generate_workload(model: ModelConfig, quant: QuantConfig, prefill_size: int, decode_size: int, stage: Stage):
    model.prefill_size = prefill_size
    model.decode_size = decode_size
    model.batch_size = 1
    model = model.to_single_layer_config()
    onnx_path = get_onnx_path(model, stage, quant)
    if not os.path.exists(onnx_path):
        export_model_to_onnx(model, quant, path=onnx_path, stage=stage)
    return parse_workload(onnx_path)


def get_ops_per_op_type(workload: ONNXWorkload):
    nodes = workload.node_list
    nodes = [n for n in nodes if isinstance(n, ComputationNode)]
    result = defaultdict(lambda: 0)
    for n in nodes:
        result[layer_to_type(n.name)] += n.total_mac_count
    return result


def get_ops_per_layer(workload: ONNXWorkload, model: ModelConfig, decode_size: int, stage: Stage):
    """Get the number of operations per layer"""
    nodes = workload.node_list
    nodes = [n for n in nodes if isinstance(n, ComputationNode)]
    result = {}
    scaling_factor = decode_size if (stage == Stage.DECODE and isinstance(model, MambaConfig)) else 1
    for n in nodes:
        result[n.name] = n.total_mac_count * scaling_factor
    return result


def get_ai_per_layer(workload: ONNXWorkload):
    """Get the arithmetic intensity per layer"""
    # nodes_involving_state = ["mul_h_dA", "add_dBx", "mul_h_C"]
    nodes_involving_state = []
    state_operands = [LayerOperand("I"), LayerOperand("O")]
    nodes = workload.node_list
    nodes = [n for n in nodes if isinstance(n, ComputationNode)]
    result = {}
    for n in nodes:
        total_mac_count = n.total_mac_count
        if any((name in n.name) for name in nodes_involving_state):
            total_mem_access = 0
            for op, size_bit in n.operand_size_bit.items():
                if op not in state_operands:
                    total_mem_access += size_bit
        else:
            total_mem_access = sum(n.operand_size_bit.values())
        ai = total_mac_count / total_mem_access
        result[n.name] = ai
    return result


def get_average_ai_per_op_type(workload: ONNXWorkload, model: ModelConfig, decode_size: int, stage: Stage):
    nodes = workload.node_list
    nodes = [n for n in nodes if isinstance(n, ComputationNode)]
    ai_per_layer = get_ai_per_layer(workload)
    ops_per_layer = get_ops_per_layer(workload, model, decode_size, stage)
    op_types = set([layer_to_type(n.name) for n in nodes])
    result = defaultdict(lambda: 0)
    for op_type in op_types:
        ai_values = [v for k, v in ai_per_layer.items() if op_type == layer_to_type(k)]
        ops_values = [v for k, v in ops_per_layer.items() if op_type == layer_to_type(k)]
        weighted_ai = np.average(ai_values, weights=ops_values)
        result[op_type] = weighted_ai
    return result


def get_average_ai_per_model(workload: ONNXWorkload, model: ModelConfig, decode_size: int, stage: Stage):
    ops_per_op_type = get_ops_per_op_type(workload)
    avg_ai_per_op_type = get_average_ai_per_op_type(workload, model, decode_size, stage)
    avg_ai = np.average(
        [avg_ai_per_op_type[op] for op in ops_per_op_type], weights=[ops_per_op_type[op] for op in ops_per_op_type]
    )
    return avg_ai


def get_mem_access_per_op(workload: ONNXWorkload, model: ModelConfig, decode_size: int, stage: Stage):
    """Get the number of memory accesses per op type in bits"""
    nodes = workload.node_list
    nodes = [n for n in nodes if isinstance(n, ComputationNode)]
    result = defaultdict(lambda: 0)
    scaling_factor = decode_size if (stage == Stage.DECODE and isinstance(model, MambaConfig)) else 1
    for n in nodes:
        total_mem_access = sum(n.operand_size_bit.values())
        # n.operand_size_bit, {k: v/prefill_size for k, v in n.operand_size_bit.items()},
        result[layer_to_type(n.name)] += total_mem_access * scaling_factor
    return result


def generate_operation_dataframe(
    all_op_types: list[str], models: list[ModelConfig], seq_lens: list[int], stages: list[Stage], quant: QuantConfig
):
    data = []
    for model in models:
        for seq_len in seq_lens:
            for stage in stages:
                prefill_size, decode_size = get_prefill_and_decode_size(model, seq_len, stage)
                workload = generate_workload(model, quant, prefill_size, decode_size, stage)
                nb_ops = get_ops_per_op_type(workload)
                row = {"model": str(model), "seq_len": seq_len, "stage": stage}
                row.update({op_type: nb_ops.get(op_type, 0) for op_type in all_op_types})
                data.append(row)
    df = pd.DataFrame(data)
    df = add_ops_per_token(df, all_op_types, models, seq_lens)
    return df


def add_ops_per_token(df, all_op_types: list[str], models, seq_lens):
    for model in models:
        for seq_len in seq_lens:
            df_model_seq_len = df[(df["model"] == str(model)) & (df["seq_len"] == seq_len)]
            # Sum all columns of all_op_types
            total_ops = df_model_seq_len[all_op_types].sum(axis=1)
            # Divide by the number of tokens
            ops_per_token = total_ops / seq_len
            # Add total_ops column to dataframe if it doesn't exist and set it
            if "total_ops" not in df.columns:
                df["total_ops"] = 0
            df.loc[(df["model"] == str(model)) & (df["seq_len"] == seq_len), "total_ops"] = total_ops.iloc[0]
            # Add ops_per_token column to dataframe if it doesn't exist and set it
            if "ops_per_token" not in df.columns:
                df["ops_per_token"] = 0
            df.loc[(df["model"] == str(model)) & (df["seq_len"] == seq_len), "ops_per_token"] = int(
                ops_per_token.iloc[0]
            )
    return df


def scale_ops_with_num_layers(df_op):
    exclude_columns = ["model", "seq_len", "stage"]
    # Copy the df
    df = df_op.copy()
    df.loc[df["model"] == "OPT-2.7B", ~df.columns.isin(exclude_columns)] *= OPT_2_7B.num_layer
    df.loc[df["model"] == "Mamba1-2.7B", ~df.columns.isin(exclude_columns)] *= MAMBA1_2_8B.num_layer
    return df


def generate_total_ops_and_mem_accesses_dataframe(df_op, df_ai):
    data = []
    for i in range(len(df_op)):
        op_row = df_op.iloc[i]
        ai_row = df_ai.iloc[i]
        assert all(op_row[["model", "seq_len", "stage"]] == ai_row[["model", "seq_len", "stage"]]), "Rows do not match"
        model = op_row["model"]
        stage = op_row["stage"]
        seq_len = op_row["seq_len"]

        ops = get_non_nan_series(op_row).to_dict()
        total_ops = sum(ops.values())
        ais = get_non_nan_series(ai_row).to_dict()
        mem_accesses = {k: ops[k] / ais[k] for k in ops.keys()}
        total_mem_accesses = sum(mem_accesses.values())
        row = {"model": model, "seq_len": seq_len, "stage": stage}
        row.update({"total_ops": total_ops, "total_mem_access": total_mem_accesses})
        row.update(
            {"total_ops_per_token": total_ops / seq_len, "total_mem_access_per_token": total_mem_accesses / seq_len}
        )
        data.append(row)
    df = pd.DataFrame(data)
    return df


def generate_arithmetic_intensity_dataframe(df_ops, df_mem):
    df = df_ops.copy()
    df["mem_access_per_token"] = df_mem["mem_access_per_token"]
    df["arithmetic_intensity"] = df["ops_per_token"] / df_mem["mem_access_per_token"]
    return df


def get_prefill_and_decode_size(model: ModelConfig, seq_len: int, stage: Stage):
    if stage == Stage.PREFILL:
        prefill_size = seq_len
        decode_size = 1
    else:
        if isinstance(model, MambaConfig):
            prefill_size = seq_len
            decode_size = 1
        else:
            prefill_size = seq_len
            decode_size = 1
    return prefill_size, decode_size


def get_workloads(models, seq_lens, stages, quant):
    workloads = []
    for model in models:
        for seq_len in seq_lens:
            for stage in stages:
                prefill_size, decode_size = get_prefill_and_decode_size(model, seq_len, stage)
                workload = generate_workload(model, quant, prefill_size, decode_size, stage)
                workloads.append(workload)
    return workloads


def get_unique_operation_types(workloads):
    all_op_types = set()
    for workload in workloads:
        nb_ops = get_ops_per_op_type(workload)
        all_op_types |= set(nb_ops.keys())
    all_op_types = sorted(all_op_types, key=lambda x: TYPE_ORDER.index(x) if x in TYPE_ORDER else len(TYPE_ORDER))
    return all_op_types


def load_dataframes(op_csv_path, ai_csv_path, latency_csv_path):
    print(f"Loading dataframes from {os.path.dirname(op_csv_path)}")
    df_op = pd.read_csv(op_csv_path)
    df_ai = pd.read_csv(ai_csv_path)
    df_latency = pd.read_csv(latency_csv_path)
    return df_op, df_ai, df_latency


def get_row_layer(model, seq_len, stage, x: dict[str, float]) -> dict[str, float]:
    row = {"model": str(model), "seq_len": seq_len, "stage": str(stage)}
    for layer_name, value in x.items():
        row[layer_name] = value
    return row


def get_row_op(model, seq_len, stage, all_op_types, x: dict[str, float]) -> dict[str, float]:
    row = {"model": str(model), "seq_len": seq_len, "stage": str(stage)}
    row.update({op_type: x.get(op_type, 0) for op_type in all_op_types})
    return row


def get_non_nan_series(df: pd.Series) -> pd.Series:
    # Remove 'model','seq_len' and 'stage' from series
    return df.iloc[3:].dropna()


def get_latency_per_layer(
    ops_per_layer: dict[str, float],
    ai_per_layer: dict[str, float],
    scaling_factor: int = 1,
    peak_ops_per_cycle: int = 8192,
    peak_memory_bandwidth: int = 2048,
) -> dict[str, float]:
    """Simple latency analysis based on the roofline model"""
    latencies = {}
    for layer in ai_per_layer:
        ai = ai_per_layer[layer]
        ops = ops_per_layer[layer]
        latency = get_roofline_latency(ops, ai, peak_ops_per_cycle, peak_memory_bandwidth)
        latencies[layer] = ceil(latency * scaling_factor)
    return latencies


def separate_ops_per_op(ops_per_layer: dict[str, float]) -> dict[str, float]:
    """Seperate ops per operator type"""
    result = defaultdict(lambda: 0.0)
    for layer, ops in ops_per_layer.items():
        op_type = layer_to_type(layer)
        if op_type in IGNORE_OP_TYPES:
            continue
        result[op_type] += ops
    return dict(result)


def separate_latency_per_op(latency_per_layer: dict[str, float]) -> dict[str, float]:
    """Seperate latency per operator type as defined in op_distribution.py"""
    result = defaultdict(lambda: 0.0)
    for layer, latency in latency_per_layer.items():
        op_type = layer_to_type(layer)
        if op_type in IGNORE_OP_TYPES:
            continue
        result[op_type] += latency
    return dict(result)
