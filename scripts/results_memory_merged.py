import json
import os
import sys
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn

sys.path.append(os.getcwd())
from brokenaxes import brokenaxes  # type: ignore

from scripts.sweep_mem_no_split import get_configurations as get_configurations_F
from scripts.sweep_mem_no_split import get_dump_path as get_dump_path_F
from scripts.sweep_mem_split_D import (
    get_configurations as get_configurations_AF,  # F = Full fusion, AF = Adaptive-Full
)
from scripts.sweep_mem_split_D import (
    get_dump_path as get_dump_path_AF,
)
from scripts.util import convert_latency_result, get_intra_core_split_D
from src.config_library import MAMBA1_2_8B
from src.plot_util import set_custom_mpl_style


def create_plot(results_F: dict[tuple[int, int, int, int], float], results_AF: dict[tuple[int, int, int, int], float]):
    # Input data
    model = MAMBA1_2_8B
    nb_cores = 1
    all_D_L = sorted(list({(key[0], key[1]) for key in results_F}))
    all_mem_size = sorted(list({key[3] for key in results_F}))

    # Figure setup
    set_custom_mpl_style()
    plt.figure(figsize=(8, 6))  # type: ignore
    bax = brokenaxes(
        xlims=((0, 8), (23, 25)),
        ylims=((0, 6), (42, 47)),
        hspace=0.08,
        wspace=0.08,
        d=0.01,
        diag_color="k",
        xscale="linear",
        yscale="linear",
    )
    # Constants
    colors = seaborn.color_palette("pastel", len(all_D_L))
    # xlim = (0, all_mem_size[-1] + 1)

    # Plot F
    for i, (curr_D, curr_L) in enumerate(all_D_L):
        latencies = np.array(
            [results_F.get((curr_D, curr_L, nb_cores, mem_size_MB), 0) for mem_size_MB in all_mem_size]
        )
        latencies = latencies / curr_L

        bax.step(  # type: ignore
            all_mem_size,
            latencies,
            color=colors[i],
            label=f"{curr_L}{' (decode)' if curr_L == 1 else ''}",
            where="pre",
            marker="",
            linestyle="-",
            linewidth=3,
        )

    # Plot AF
    for i, (curr_D, curr_L) in enumerate(all_D_L):
        latencies = np.array(
            [results_AF.get((curr_D, curr_L, nb_cores, mem_size_MB), 0) for mem_size_MB in all_mem_size]
        )
        latencies = latencies / curr_L

        bax.step(  # type: ignore
            all_mem_size,
            latencies,
            color=colors[i],
            label="",  # f"{curr_L}{' (decode)' if curr_L == 1 else ''}",
            where="pre",
            marker="",
            linestyle="--",
            linewidth=3,
        )

    # Vertical line for equation point
    x = (5 * model.d_inner * model.d_state + model.d_inner + model.d_state) * 32 / (8 * 1024**2)
    bax.axvline(
        x=x,
        color="red",
        linestyle="--",
        linewidth=1,
    )
    bax.text(
        x * 1.02,
        4.5,
        "Required mem. \nsize (Eq. 2)",
        color="red",
        fontsize=12,
    )

    # Vertical lines for number of tiles
    seen_factors: set[int] = set()
    for mem_size in sorted(all_mem_size, reverse=True):
        try:
            x = mem_size
            nb_split = get_intra_core_split_D(model, mem_size)
            if nb_split in seen_factors or (nb_split & (nb_split - 1) != 0) or nb_split == 32:
                continue
            seen_factors.add(nb_split)
            bax.axvline(x=x, color="grey", linestyle="--", linewidth=1)  # type: ignore
            bax.text(  # type: ignore
                x,
                1.3,  # 0.05,
                f"{nb_split}L",
                fontsize=9,
                ha="right",
                color="grey",
                rotation="vertical",
                va="bottom",
            )
        except ValueError:
            print(f"x={mem_size} not in plot")

    # Full/dashed line in legend
    bax.plot([], [], "k-", label="Fuse-All", linewidth=2)
    bax.plot([], [], "k--", label="Mem-Aware", linewidth=2)

    # Plot details
    bax.set_xlabel("On-chip memory size [MB]", fontsize=14)  # type: ignore
    bax.minorticks_off()
    bax.tick_params(axis="both", which="both", labelsize=11, width=2)  # Make axis lines thicker
    bax.set_ylabel("Average latency [ms/token]", fontsize=14)  # type: ignore
    bax.legend(title="Seq. length (L)", title_fontsize=12, fontsize=12)  # type: ignore

    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/memory_sweep_merged.png", bbox_inches="tight", pad_inches=0)  # type: ignore
    # plt.savefig("outputs/figures/memory_sweep_merged.pdf", bbox_inches="tight", pad_inches=0)  # type: ignore


def read_results_from_json(fusion_type: Literal["F", "AF"]):
    configs = list(get_configurations_F()) if fusion_type == "F" else list(get_configurations_AF())
    dump_path_getter = get_dump_path_F if fusion_type == "F" else get_dump_path_AF

    results: dict[tuple[int, int, int, int], float] = {}

    for model, nb_cores, mem_size_MB in configs:
        dump_path = dump_path_getter(model, nb_cores, mem_size_MB)
        results_path = f"{dump_path}/results.json"

        try:
            with open(results_path, "r") as f:
                latency = json.load(f)["latency"]

            full_latency_ms = convert_latency_result(model, latency)
            results[(model.d_model, model.prefill_size, nb_cores, mem_size_MB)] = full_latency_ms
        except FileNotFoundError:
            print(f"{results_path} not found")
        except json.JSONDecodeError:
            print(f"{results_path} is invalid")

    print(f">>>{len(results)} runs out of {len(configs)} completed<<<")

    return results


if __name__ == "__main__":

    results_F = read_results_from_json("F")
    results_AF = read_results_from_json("AF")

    create_plot(results_F, results_AF)
