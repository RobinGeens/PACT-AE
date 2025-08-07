import itertools
import os
import sys

import pandas as pd

sys.path.append(os.getcwd())
from scripts.utils_parsing import (
    OP_TYPE_COLORS,
    TYPE_ORDER,
    get_ai_per_layer,
    get_latency_per_layer,
    get_non_nan_series,
    get_ops_per_layer,
    get_row_layer,
    get_row_op,
    get_unique_operation_types,
    get_workloads,
    load_dataframes,
    separate_latency_per_op,
)
from scripts.utils_roofline import convert_df_to_ms, convert_to_ms
from src.config import ModelConfig
from src.config_library import MAMBA1_2_8B, OPT_2_7B, W32A32
from src.plot_util import BarPlotter, set_custom_mpl_style
from src.util import Stage

# MODEL CONFIGURATIONS
MODELS: list[ModelConfig] = [OPT_2_7B, MAMBA1_2_8B]
QUANT = W32A32
SEQ_LENS = [
    64,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
]
STAGES = [Stage.PREFILL, Stage.DECODE]


# HARDWARE CONSTANTS
PEAK_OPS_CYCLE = 32 * 16 * 16  # ops per cycle
PEAK_BANDWIDTH = 2048  # bits per cycle
FREQUENCY = int(1e9)  # 1 GHz

# PLOT SETTINGS
set_custom_mpl_style()

# CSV SAVE LOCATIONS
OP_CSV_PATH = "outputs/csv/op_distribution.csv"
AI_CSV_PATH = "outputs/csv/ai_distribution.csv"
LATENCY_CSV_PATH = "outputs/csv/latency_distribution.csv"


def plot_stacked_bars_by_group(df: pd.DataFrame, all_op_types: list, fig_path: str):
    """Plot the stacked bar chart for latency by operator type"""
    # Plot
    colors = [OP_TYPE_COLORS[op_type] for op_type in all_op_types]
    plotter = BarPlotter(
        groups=[f"L={seq_len}" for seq_len in SEQ_LENS],
        bars=[model.name for model in MODELS],
        sections=all_op_types,
        ylabel="Latency (ms)",
        xtick_rotation=0,
        xtick_ha="center",
        colors=colors,
        figsize=(8, 8),
    )
    plotter.plot_two_subplots(
        df,
        filename=fig_path,
        y_labels=["Time to first token (s)", "Time between tokens (ms)"],
        scaling_factors=[1e3, 1],
        add_legends=[False, False],
    )


def generate_dataframes(workloads, all_op_types):
    data_op = []
    data_ai = []
    data_latency = []
    for workload, (model, seq_len, stage) in zip(workloads, list(itertools.product(MODELS, SEQ_LENS, STAGES))):
        num_layers = model.num_layer
        ai_per_layer = get_ai_per_layer(workload)
        ops_per_layer = get_ops_per_layer(workload, model=model, decode_size=1, stage=stage)
        latency_per_layer = get_latency_per_layer(
            ai_per_layer,
            ops_per_layer,
            scaling_factor=num_layers,
            peak_ops_per_cycle=PEAK_OPS_CYCLE,
            peak_memory_bandwidth=PEAK_BANDWIDTH,
        )
        latency_per_group = separate_latency_per_op(latency_per_layer)
        data_op.append(get_row_layer(model, seq_len, stage, ops_per_layer))
        data_ai.append(get_row_layer(model, seq_len, stage, ai_per_layer))
        data_latency.append(get_row_op(model, seq_len, stage, all_op_types, latency_per_group))
    df_op = pd.DataFrame(data_op)
    df_ai = pd.DataFrame(data_ai)
    df_cycles = pd.DataFrame(data_latency)
    df_ms = convert_df_to_ms(df_cycles, all_op_types, FREQUENCY)
    return df_op, df_ai, df_ms


def get_unique_op_types_from_df(df: pd.DataFrame) -> list:
    op_types = list(df.columns[3:])
    sorted_op_types = sorted(op_types, key=lambda x: TYPE_ORDER.index(x) if x in TYPE_ORDER else len(TYPE_ORDER))
    return sorted_op_types


def save_dataframes(df_op: pd.DataFrame, df_ai: pd.DataFrame, df_latency: pd.DataFrame):
    print(f"Saving dataframes to {os.path.dirname(OP_CSV_PATH)}")
    os.makedirs(os.path.dirname(OP_CSV_PATH), exist_ok=True)
    df_op.to_csv(OP_CSV_PATH, index=False)
    df_ai.to_csv(AI_CSV_PATH, index=False)
    df_latency.to_csv(LATENCY_CSV_PATH, index=False)


def main():
    if os.path.exists(OP_CSV_PATH):
        df_op, _, df_latency = load_dataframes(OP_CSV_PATH, AI_CSV_PATH, LATENCY_CSV_PATH)
        all_op_types = get_unique_op_types_from_df(df_latency)
    else:
        print("Generating dataframes for operation counts...")
        workloads = get_workloads(MODELS, SEQ_LENS, stages=STAGES, quant=QUANT)
        all_op_types = get_unique_operation_types(workloads)
        df_op, df_ai, df_latency = generate_dataframes(workloads, all_op_types)
        save_dataframes(df_op, df_ai, df_latency)


def plot_latency_bars():
    DECODE_LAYERS_OPT = 32  # 32
    DECODE_LAYERS_MAMBA = 64  # 64
    latency_data = []
    df_op, df_ai, df_latency = load_dataframes(OP_CSV_PATH, AI_CSV_PATH, LATENCY_CSV_PATH)
    for i in range(len(df_op)):
        op_row = df_op.iloc[i]
        ai_row = df_ai.iloc[i]
        assert all(op_row[["model", "seq_len", "stage"]] == ai_row[["model", "seq_len", "stage"]]), "Rows do not match"
        model = op_row["model"]
        stage = op_row["stage"]
        seq_len = op_row["seq_len"]
        if "mamba" in model.lower():
            if stage == "prefill":
                num_layer = MAMBA1_2_8B.num_layer
            else:
                num_layer = DECODE_LAYERS_MAMBA
        else:
            if stage == "prefill":
                num_layer = OPT_2_7B.num_layer
            else:
                num_layer = DECODE_LAYERS_OPT
        ops = get_non_nan_series(op_row).to_dict()
        ais = get_non_nan_series(ai_row).to_dict()
        latency_per_layer = get_latency_per_layer(ops, ais, scaling_factor=num_layer)
        latency_per_op = separate_latency_per_op(latency_per_layer)
        latency_per_op_ms = {k: convert_to_ms(v, FREQUENCY) for k, v in latency_per_op.items()}
        row = {"model": model, "seq_len": seq_len, "stage": stage}
        row.update(latency_per_op_ms)
        latency_data.append(row)
    df_latency = pd.DataFrame(latency_data)
    # Replace all NaN values with 0
    df_latency.fillna(0, inplace=True)
    all_op_types = get_unique_op_types_from_df(df_latency)
    plot_stacked_bars_by_group(df_latency, all_op_types, "outputs/figures/latency.png")


if __name__ == "__main__":
    main()
    plot_latency_bars()
