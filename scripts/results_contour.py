import json
import os
import sys
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn

sys.path.append(os.getcwd())

from scripts.sweep_contour import get_configurations, get_dump_path
from scripts.util import (
    compute_mem_size_MB,
    convert_latency_result,
    get_intra_core_split_D,
)
from src.plot_util import set_custom_mpl_style


def create_plots(results: dict[tuple[int, int, float, float, Literal[1, None]], float]):
    """keys = D, L,  mem_area_pct, area_reduction, D_split"""

    all_L = sorted(list({key[1] for key in results}))
    d_model = 2560

    for prefill_size in all_L:
        figure_path = f"outputs/figures/contour_plot_L={prefill_size}.png"
        results_this_plot: dict[tuple[float, float, Literal[1, None]], float] = {
            (mem_area_pct, area_reduction, D_split): v
            for (D, L, mem_area_pct, area_reduction, D_split), v in results.items()
            if L == prefill_size and D == d_model
        }
        create_plot(results_this_plot, prefill_size, figure_path)


def create_plot(results: dict[tuple[float, float, Literal[1, None]], float], prefill_size: int, figure_file: str):
    """keys = mem_area_pct, area_reduction, D_split"""
    assert figure_file.endswith(".png")

    # Inputs
    all_D_split = (1, None)
    all_mem_pct = sorted(list({key[0] for key in results}))  # x-axis
    all_area_pct = sorted(list({key[1] for key in results}), reverse=False)  # legend

    # Figure setup
    set_custom_mpl_style()
    fig, ax = plt.subplots(figsize=(8, 6))  # type: ignore

    # Constants
    colors = seaborn.color_palette("pastel", len(all_area_pct))
    max_value = max(results.values()) / prefill_size
    min_value = min(results.values()) / prefill_size
    xticks = np.arange(11) / 10
    linestile = {1: "-", None: "--"}

    # Plot
    for D_split in all_D_split:
        for i, area_pct in enumerate(all_area_pct):
            latencies = np.array([results.get((mem_pct, area_pct, D_split), 0) for mem_pct in all_mem_pct])
            latencies = latencies / prefill_size

            ax.plot(
                all_mem_pct,
                latencies,
                label=f"{int(area_pct * 222)} ({100*area_pct:.0f}%)" if D_split == 1 else "",
                marker="",
                color=colors[i],
                linewidth=3,
                linestyle=linestile[D_split],
            )

            # Plot the tiling factor for the split and print extra info
            seen_factors: set[int] = set()
            if D_split is None:
                for mem_pct in sorted(all_mem_pct, reverse=True):
                    mem_size_MB = compute_mem_size_MB(mem_pct, area_pct)
                    model.prefill_size = prefill_size
                    model.d_model = 2560
                    # pe_size = _find_closest_factors(compute_total_PE(mem_pct, area_pct))
                    # dram_bw = compute_dram_bandwidth(area_pct)
                    # print(
                    #     f"{(100*area_pct):.1f}% area {(100*mem_pct):.1f}% mem =>\t{pe_size} PE,\t{mem_size_MB:.1f} MB\t {dram_bw} bit/cc"
                    # )
                    split_factor = get_intra_core_split_D(model, mem_size_MB)
                    if split_factor in seen_factors:
                        continue
                    seen_factors.add(split_factor)
                    ax.text(
                        mem_pct,
                        latencies[list(all_mem_pct).index(mem_pct)] * 1.02,
                        f"{split_factor}L",
                        fontsize=10,
                        ha="right",
                        color="grey",
                    )

    # Add full/dashed line
    ax.plot([], [], "k-", label="Fuse-All", linewidth=2)
    ax.plot([], [], "k--", label="Mem-Aware", linewidth=2)
    # Show MARCA point
    try:
        ax.plot(0.8, results[(0.8, 1, None)] / prefill_size, "ro", label="MARCA", markersize=9)
    except KeyError:
        print("MARCA point not yet found in results")

    ax.yaxis.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax.yaxis.grid(which="major", linestyle="-", linewidth="1", color="gray")

    # ax.minorticks_on()
    ax.set_yscale("log")
    ax.set_xlabel("Memory percentage", fontsize=14)
    ax.set_xticks(xticks, labels=[f"{(100*x):.0f}%" for x in xticks])
    ax.tick_params(axis="both", which="both", labelsize=12, width=2)
    ax.set_ylabel("Average latency [ms/token]", fontsize=14)
    ax.set_ylim(min_value * 0.95, max_value * 1.05)
    if prefill_size == 1 and False:
        ax.legend(
            title="Total area [mm²]   Fusion scheme",
            title_fontsize=12,
            fontsize=12,
            loc="upper left",
            ncol=4,
            alignment="left",
        )

    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig(figure_file, bbox_inches="tight", pad_inches=0)  # type: ignore
    # plt.savefig(figure_file.replace("png", "pdf"), bbox_inches="tight", pad_inches=0)  # type: ignore
    # plt.savefig(figure_file.replace("png", "svg"), bbox_inches="tight", pad_inches=0)  # type: ignore


if __name__ == "__main__":

    results: dict[tuple[int, int, float, float, Literal[1, None]], float] = {}
    configs = list(get_configurations())

    for args in configs:
        model, mem_area_percentage, area_reduction, D_split = args
        dump_path = get_dump_path(args)

        if D_split is None:
            mem_size_MB = compute_mem_size_MB(mem_area_percentage, area_reduction)
            optimal_D_split = get_intra_core_split_D(model, mem_size_MB)
            if optimal_D_split == 1:
                # In this case, the run has returned `DontRunException` ->  take results from D_split=1
                dump_path = get_dump_path((model, mem_area_percentage, area_reduction, 1))

        results_path = f"{dump_path}/results.json"

        try:
            with open(results_path, "r") as f:
                latency = json.load(f)["latency"]

            full_latency_ms = convert_latency_result(model, latency)
            results[(model.d_model, model.prefill_size, mem_area_percentage, area_reduction, D_split)] = full_latency_ms
        except FileNotFoundError:
            print(f"{results_path} not found")
        except json.JSONDecodeError:
            print(f"{results_path} is invalid")

    create_plots(results)

    print(f">>>{len(results)} runs out of {len(list(configs))} completed<<<")
