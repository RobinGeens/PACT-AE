import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from scripts.simple_latency_analysis import AI_CSV_PATH, LATENCY_CSV_PATH, MODELS, OP_CSV_PATH, SEQ_LENS, STAGES
from scripts.utils_parsing import OP_TYPE_COLORS, TYPE_ORDER, get_non_nan_series, layer_to_type, load_dataframes
from scripts.utils_roofline import get_roofline_performance
from src.plot_util import BarPlotter, set_custom_mpl_style

# Constants
SEQ_LEN_TO_SHOW = 2048
PEAK_OPS_CYCLE = 32 * 16 * 16
MEM_SCALING = 8  # 1 for ops/bit, 8 for ops/byte
PEAK_BANDWIDTH = 2048 / MEM_SCALING
OI_UNIT = "ops/bit" if MEM_SCALING == 1 else "ops/byte"

# PLOT SETTINGS
set_custom_mpl_style()
MARKER_SHAPES = ["s", "d"]  # corresponds to models
LINE_STYLES = ["dotted", "-"]  # corresponds to models
AI_RANGE = np.logspace(np.log10(0.08), np.log10(300), 1000)
XTICK_FONTSIZE = 12
YTICK_FONTSIZE = 12
XLABEL_FONTSIZE = 16
YLABEL_FONTSIZE = 16
AXIS_TITLE_FONTSIZE = 16
MARKER_SIZE = 15
LINE_WIDTH = 3


def load_dataframe(file_path):
    """Load the dataframe from a CSV file."""
    return pd.read_csv(file_path)


def get_performance_dataframe(df):
    """Process the dataframe to calculate the roofline points."""
    data = []
    for i, (model, seq_len, stage) in enumerate(df[["model", "seq_len", "stage"]].values):
        ai_per_op_type = get_non_nan_series(df.iloc[i]).to_dict()
        performance_per_op_type = {
            op_type: get_roofline_performance(ai, PEAK_OPS_CYCLE, PEAK_BANDWIDTH)
            for op_type, ai in ai_per_op_type.items()
        }
        row = {"model": model, "seq_len": seq_len, "stage": stage}
        row.update(performance_per_op_type)
        data.append(row)
    df = pd.DataFrame(data)
    return df


def get_ops_and_average_ai_per_op_type(ops_dict: dict[str, int], ai_dict: dict[str, float]):

    scaled_ais_dict = {k: v * ops_dict[k] for k, v in ai_dict.items()}
    op_type_to_layer = {}
    for layer in ops_dict:
        op_type = layer_to_type(layer)
        if op_type not in op_type_to_layer:
            op_type_to_layer[op_type] = []
        op_type_to_layer[op_type].append(layer)
    avg_ai_per_op_type = {}
    total_ops_per_op_type = {}
    for op_type in op_type_to_layer:
        total_ai = 0
        total_ops = 0
        for layer in op_type_to_layer[op_type]:
            total_ai += scaled_ais_dict[layer]
            total_ops += ops_dict[layer]
        avg_ai_per_op_type[op_type] = total_ai / total_ops
        total_ops_per_op_type[op_type] = total_ops
    total_ops = sum(total_ops_per_op_type.values())
    return total_ops_per_op_type, avg_ai_per_op_type


def generate_df_total_ops_and_avg_ai_per_op_type(df_op, df_ai):
    data_ops = []
    data_ai = []
    for model in MODELS:
        for seq_len in SEQ_LENS:
            for stage in STAGES:
                row_op = df_op[
                    (df_op["model"] == model.name) & (df_op["seq_len"] == seq_len) & (df_op["stage"] == stage)
                ].iloc[0]
                row_ai = df_ai[
                    (df_ai["model"] == model.name) & (df_ai["seq_len"] == seq_len) & (df_ai["stage"] == stage)
                ].iloc[0]
                op_dict = get_non_nan_series(row_op).to_dict()
                ai_dict = get_non_nan_series(row_ai).to_dict()
                ai_dict = {k: v * MEM_SCALING for k, v in ai_dict.items()}
                ops_per_op_type, ai_per_op = get_ops_and_average_ai_per_op_type(op_dict, ai_dict)
                row_total_ops = {"model": model.name, "seq_len": seq_len, "stage": stage}
                row_total_ops.update(ops_per_op_type)
                data_ops.append(row_total_ops)
                row_avg_ai = {"model": model.name, "seq_len": seq_len, "stage": stage}
                row_avg_ai.update(ai_per_op)
                data_ai.append(row_avg_ai)
    return pd.DataFrame(data_ops), pd.DataFrame(data_ai)


def plot_op_distribution(df):
    """Plot the distribution of operations per operator type.
    Different df rows are different model, seq_len and stage configurations.
    For now, we only plot the prefill stage.
    The stacked bars are normalized to 100% for each combination.
    The bars are grouped based on the sequence length and two bars are shown per group for the two models.
    """
    # Replace all NaN with 0
    op_types = TYPE_ORDER
    df = df.fillna(0)
    df_normalized = df.copy()
    # Normalize the data to 100%
    df_normalized[op_types] = df[op_types].div(df[op_types].sum(axis=1), axis=0) * 100
    # Plot
    colors = [OP_TYPE_COLORS[op_type] for op_type in op_types]
    plotter = BarPlotter(
        groups=[f"L={seq_len}" for seq_len in SEQ_LENS],
        bars=[model.name for model in MODELS],
        sections=op_types,
        ylabel="Operation Distribution (%)",
        xtick_rotation=0,
        xtick_ha="center",
        legend_cols=len(op_types),
        legend_fontsize=13,
        figsize=(8, 8),
        title="",
        legend_loc="lower center",
        bbox_to_anchor=(0.486, 1.075),
        colors=colors,
    )
    plotter.plot_two_subplots(
        df_normalized,
        filename="outputs/figures/op_distribution.png",
        y_labels=["Operation Distribution (%)", "Operation Distribution (%)"],
        scaling_factors=[1, 1],
        add_legends=[False, False],
    )


def plot_roofline_boundaries(ax):
    """Plot the roofline boundaries for memory and compute bound regions."""

    # Calculate the performance for the roofline boundaries
    compute_bound = np.full_like(AI_RANGE, PEAK_OPS_CYCLE)
    memory_bound = AI_RANGE * PEAK_BANDWIDTH

    # Find the crossover point
    crossover_index = np.where(memory_bound >= PEAK_OPS_CYCLE)[0][0]

    # Adjust memory bound to only show until the crossover point
    memory_bound[crossover_index:] = np.nan
    compute_bound[:crossover_index] = np.nan

    # Plot the roofline boundaries
    ax.plot(AI_RANGE, compute_bound, "k--")
    ax.plot(AI_RANGE, memory_bound, "k--")

    # Fill the regions
    ax.fill_between(AI_RANGE[:crossover_index], 0, memory_bound[:crossover_index], color="gray", alpha=0.3)
    ax.fill_between(AI_RANGE[crossover_index:], 0, compute_bound[crossover_index:], color="gray", alpha=0.3)

    # Set log scale for better visualization
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(AI_RANGE[0], AI_RANGE[-1])
    ax.set_ylim(1, max(PEAK_OPS_CYCLE, PEAK_BANDWIDTH * AI_RANGE[-1]))


# Add the function call in the plot_roofline function
def plot_roofline(df_ai, df_performance):
    """Plot the roofline graph with vertical lines for each arithmetic intensity."""
    fig, axs = plt.subplots(nrows=len(STAGES), ncols=1, figsize=(8, 8), sharex=True)

    patches = []
    labels = []
    handles = []
    for i, ax in enumerate(axs):
        # Plot the roofline boundaries
        plot_roofline_boundaries(ax)
        stage = STAGES[i]
        # Get the dataframes for this stage
        df_ai_stage = df_ai[df_ai["stage"] == stage]
        df_performance_stage = df_performance[df_performance["stage"] == stage]
        for j, model in enumerate(MODELS):
            # Get the dataframes for this model
            df_ai_model = df_ai_stage[df_ai_stage["model"] == model.name]
            df_performance_model = df_performance_stage[df_performance_stage["model"] == model.name]
            assert len(df_ai_model) == len(df_performance_model) == 1
            seq_len = df_ai_model["seq_len"].iloc[0]

            # Iterate over the different operator types for this model
            ais = get_non_nan_series(df_ai_model.iloc[0]).to_dict()
            performances = get_non_nan_series(df_performance_model.iloc[0]).to_dict()
            # Plot the data points with matching colors and correct marker shape
            marker_type = MARKER_SHAPES[j]
            line_style = LINE_STYLES[j]
            for k, op_type in enumerate(TYPE_ORDER):
                if op_type not in ais:
                    continue
                ai = ais[op_type]
                perf = performances[op_type]
                color = OP_TYPE_COLORS[op_type]
                label = op_type if op_type not in labels else ""
                h = ax.plot(ai, perf, marker_type, color=color, ms=MARKER_SIZE)
                ax.axvline(x=ai, color=color, linestyle=line_style, lw=LINE_WIDTH)
                if label:
                    patches.append(mpatches.Patch(color=color, label=label))
                    labels.append(label)
                if k == 0:
                    handles.append(h)
        ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)
        ax.set_ylabel("Performance (GOPS)", fontsize=YLABEL_FONTSIZE)  # same as ops/cycle at 1 GHz
        ax.set_title(f"{stage.capitalize()}, Sequence length = {seq_len}", fontsize=AXIS_TITLE_FONTSIZE)
    axs[-1].set_xlabel(f"Operational Intensity ({OI_UNIT})", fontsize=XLABEL_FONTSIZE)

    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig("outputs/figures/roofline.png", bbox_inches="tight")


def main():
    df_op, df_ai, _ = load_dataframes(OP_CSV_PATH, AI_CSV_PATH, LATENCY_CSV_PATH)
    df_total_ops_per_op_type, df_avg_ai_per_op_type = generate_df_total_ops_and_avg_ai_per_op_type(df_op, df_ai)
    # print("Total ops per op type")
    # print(df_total_ops_per_op_type)
    # print("Average AI per op type")
    # print(df_avg_ai_per_op_type)
    plot_op_distribution(df_total_ops_per_op_type)
    df_avg_ai_per_op_type = df_avg_ai_per_op_type[(df_avg_ai_per_op_type["seq_len"] == SEQ_LEN_TO_SHOW)]
    df_performance_per_op_type = get_performance_dataframe(df_avg_ai_per_op_type)
    plot_roofline(df_avg_ai_per_op_type, df_performance_per_op_type)


if __name__ == "__main__":
    main()
