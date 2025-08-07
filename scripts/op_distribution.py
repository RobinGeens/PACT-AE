import os
import sys

import pandas as pd

sys.path.append(os.getcwd())
from scripts.simple_latency_analysis import (
    AI_CSV_PATH,
    LATENCY_CSV_PATH,
    MODELS,
    OP_CSV_PATH,
    SEQ_LENS,
    load_dataframes,
)
from scripts.utils_parsing import generate_total_ops_and_mem_accesses_dataframe, scale_ops_with_num_layers
from src.plot_util import BarPlotter

# CSV SAVE LOCATIONS
TOTAL_CSV_PATH = "outputs/csv/total.csv"


def plot_ops_and_mem_accesses(df_total: pd.DataFrame):
    # Plot
    plotter = BarPlotter(
        groups=[f"L={seq_len}" for seq_len in SEQ_LENS],
        bars=[model.name for model in MODELS],
        sections=["ops", "mem_accesses"],
        ylabel="Number of operations and memory accesses",
        xtick_rotation=0,
        xtick_ha="center",
        xtick_fontsize=10,
        ytick_fontsize=10,
    )
    os.makedirs("outputs/figures", exist_ok=True)
    # plotter.plot_line_four_subplots(df_total, filename="outputs/figures/ops_and_mem_accesses.pdf")
    plotter.plot_line_four_subplots(df_total, filename="outputs/figures/ops_and_mem_accesses.png")


def main():
    if not os.path.exists(TOTAL_CSV_PATH):
        # Assume the csv exit (can be generated using simple_latency_analysis.py)
        df_op, df_ai, _ = load_dataframes(OP_CSV_PATH, AI_CSV_PATH, LATENCY_CSV_PATH)
        # Scale ops with num layers
        df_op = scale_ops_with_num_layers(df_op)
        # Get memory accesses dataframe from ops and ai
        df_total = generate_total_ops_and_mem_accesses_dataframe(df_op, df_ai)
        # Save to csv
        os.makedirs(os.path.dirname(TOTAL_CSV_PATH), exist_ok=True)
        df_total.to_csv(TOTAL_CSV_PATH, index=False)
    else:
        df_total = pd.read_csv(TOTAL_CSV_PATH)
    # print(df_total)
    plot_ops_and_mem_accesses(df_total)


if __name__ == "__main__":
    main()
