import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec

sys.path.append(os.getcwd())

from scripts.simple_latency_analysis import OP_CSV_PATH
from scripts.sweep_layer_stacks import get_configurations, get_dump_path, layer_stack_sweep
from scripts.util import convert_latency_result, get_layer_stack_SSM, stack_to_short_str
from src.config import Mamba1Config
from src.plot_util import set_custom_mpl_style


def print_results(results: dict[tuple[int, int, str], float]):
    all_D_L = sorted(list({(key[0], key[1]) for key in results}))
    all_stack = layer_stack_sweep

    all_D_L_str = [f"(D={curr_D}, L={curr_L})" for curr_D, curr_L in all_D_L]
    print("\t".join(all_D_L_str))

    for stack in all_stack:
        results_row = [results.get((curr_D, curr_L, str(stack)), 0) for curr_D, curr_L in all_D_L]
        results_row_format = [f"{res:.3f}".replace(".", ",") for res in results_row]
        print("\t\t".join(results_row_format), f"\tstack={stack_to_short_str(stack)}\t")


def create_plot(
    latency_results: dict[tuple[int, int, str], float],
    utilization_results: dict[tuple[int, int, str], float],
):
    # Input data
    all_D_L = sorted(list({(key[0], key[1]) for key in results}))
    all_stack = [str(s) for s in layer_stack_sweep]
    all_stack_print = [stack_to_short_str(s) for s in layer_stack_sweep]
    x = np.arange(len(all_stack))

    # Set up fig
    set_custom_mpl_style()
    plt.figure(figsize=(8, 6))  # type: ignore
    sps1, sps2 = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.1)
    ax1 = brokenaxes(ylims=((0, 5), (41.5, 46)), hspace=0.1, diag_color="grey", d=0.01, subplot_spec=sps1)
    ax2 = plt.subplot(sps2)  # type: ignore

    # Constants
    xlabel_with_star = ["UF", "A", "BS", "All"]
    xlabels = [f"{x}*" if x in xlabel_with_star else x for x in all_stack_print]
    nb_categories = len(all_D_L)
    bar_width = 1 / (nb_categories + 2)
    colors = seaborn.color_palette("pastel", nb_categories)
    xlims = (-0.5, len(all_stack) - 0.5)

    # Top plot
    for i, (curr_D, curr_L) in enumerate(all_D_L):
        latencies = np.array([latency_results.get((curr_D, curr_L, stack), 1) for stack in all_stack])
        latencies = latencies / curr_L
        ax1.plot(  # type: ignore
            x,
            latencies,
            label=f"{curr_L}{' (decode)' if curr_L == 1 else ''}",
            color=colors[i],
            marker=".",
            linewidth=3,
        )

    ax1.tick_params(labelbottom=False)  # type: ignore
    ax1.set_xticks(ticks=x)  # type: ignore
    ax1.set_xlim(xlims)  # type: ignore
    ax1.set_ylabel("Average latency [ms/token]", fontsize=14)  # type: ignore
    ax1.tick_params(axis="y", labelsize=12)  # type: ignore
    ax1.legend(title="Seq. length (L)", title_fontsize=12, fontsize=12)  # type: ignore

    # Bottom plot
    for i, (curr_D, curr_L) in enumerate(all_D_L):
        utilizations = np.array([utilization_results.get((curr_D, curr_L, stack), 0) for stack in all_stack])
        ax2.bar(  # type: ignore
            -0.5 + x + (i + 1) * bar_width,
            utilizations,
            width=bar_width,
            bottom=0,
            label=f"{curr_L}{' (decode)' if curr_L == 1 else ''}",
            color=colors[i],
            edgecolor="black",
        )

    ax2.set_xlabel("Fusion scheme", fontsize=14)  # type: ignore
    ax2.set_xticks(ticks=x, labels=xlabels, rotation=0, ha="center", fontsize=14)  # type: ignore
    ax2.tick_params(axis="both", which="both", labelsize=12, width=2)  # Make axis lines thicker
    ax2.set_xlim(xlims)
    ax2.set_ylabel("Hardware\nutilization", fontsize=14)  # type: ignore
    ax2.set_ylim(0, 1)

    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/layer_stack_sweep.png", bbox_inches="tight", pad_inches=0)  # type: ignore
    # plt.savefig("outputs/figures/layer_stack_sweep.pdf", bbox_inches="tight", pad_inches=0)  # type: ignore


def get_total_ops(ops_df: pd.DataFrame, model: Mamba1Config) -> int:
    """Fetch the total number of operations from the CSV file, in order to compute hardware utilization."""
    if model.prefill_size == 1:
        stage = "decode"
        seq_len = 512
    else:
        stage = "prefill"
        seq_len = model.prefill_size

    ops_df = ops_df[ops_df["model"] == model.name]
    ops_df = ops_df[ops_df["seq_len"] == seq_len]
    ops_df = ops_df[ops_df["stage"] == stage]
    ops_columns = [col for col in ops_df.columns if col.startswith("/")]
    total_ops = ops_df[ops_columns].sum(axis=1)  # type: ignore

    try:
        return total_ops.iloc[0]  # type: ignore
    except Exception:
        return 0


def get_ssm_utilization(model: Mamba1Config, dump_path: str):
    LAYER_IDS = tuple(range(18, 22)) + get_layer_stack_SSM(model)
    files_in_dump_file = os.listdir(dump_path)
    json_file = next(f for f in files_in_dump_file if "schedule" in f)

    with open(f"{dump_path}/{json_file}", "r") as f:
        data = json.load(f)

    total_cycle = 0
    ideal_cycle = 0

    for layer in data:
        if "cat" not in layer or layer["cat"] != "compute":
            continue

        try:
            layer_id = int(layer["name"].split("Id: ")[1].split(",")[0])
            if layer_id in LAYER_IDS:
                # Process the layer if its ID is in LAYER_IDS
                total_cycle += float(layer["args"]["Runtime"])
                ideal_cycle += float(layer["args"]["Runtime"]) * float(layer["args"]["SpatialUtilizationWithTemporal"])
        except IndexError:
            print(f"ID not found in {layer['name']}")

    # print(f"Utilization for {dump_path}: {ideal_cycle} / {total_cycle}")

    return ideal_cycle / total_cycle


if __name__ == "__main__":

    results: dict[tuple[int, int, str], float] = {}
    utilizations: dict[tuple[int, int, str], float] = {}
    ssm_utilizations: dict[tuple[int, int, str], float] = {}

    configs = list(get_configurations())
    try:
        ops_df = pd.read_csv(OP_CSV_PATH)  # type: ignore
    except FileNotFoundError:
        print(f"File {OP_CSV_PATH} not found. Please run simple_latency_analysis.py first.")
        sys.exit(1)

    for args in configs:
        model, layer_stacks = args
        dump_path = get_dump_path(model, layer_stacks)
        results_path = f"{dump_path}/results.json"

        try:
            with open(results_path, "r") as f:
                latency_cc = json.load(f)["latency"]

            full_latency_ms = convert_latency_result(model, latency_cc)
            total_ops = get_total_ops(ops_df, model)
            total_utilization = (total_ops / 8192) / latency_cc
            ssm_utilization = get_ssm_utilization(model, dump_path)
            results[(model.d_model, model.prefill_size, str(layer_stacks))] = full_latency_ms
            utilizations[(model.d_model, model.prefill_size, str(layer_stacks))] = total_utilization
            ssm_utilizations[(model.d_model, model.prefill_size, str(layer_stacks))] = ssm_utilization
        except FileNotFoundError:
            print(f"{results_path} not found")

    create_plot(results, utilizations)
    # print_results(results)
    # print("Hardware utilizations in SSM:")
    # print_results(ssm_utilizations)
    # print("Hardware utilizations overall:")
    # print_results(utilizations)

    print(f">>>{len(results)} runs out of {len(configs)} completed<<<")
