import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

from src.util import ARRAY_T

PLOT_STYLE = "seaborn-v0_8-whitegrid"


def set_custom_mpl_style():
    plt.style.use(PLOT_STYLE)
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.facecolor"] = "#ebe2e2"  # gray
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["xtick.major.size"] = 1
    plt.rcParams["ytick.major.size"] = 1


class BarPlotter:
    def __init__(
        self,
        groups: list[str],
        bars: list[str],
        sections: list[str],
        *,
        supergroups: list[str] | None = None,
        # Layout
        bar_width: float = 0.6,
        bar_spacing: float = 0.1,
        group_spacing: float = 1,
        group_name_dy: float = -4,
        group_name_offset: float | None = None,
        scale: str = "linear",
        # Labels
        group_name_fontsize: int = 14,
        group_name_color: str = "black",
        xtick_labels: list[str] | None = None,
        xtick_rotation: int = 45,
        xtick_fontsize: int = 14,
        ytick_fontsize: int = 12,
        xtick_ha: str = "right",
        bar_text: list[str] | None = None,
        ylabel: str = "",
        title: str = "",
        legend_cols: int = 1,
        legend_loc: str = "lower center",
        legend_fontsize: int = 12,
        # Other
        colors: list[str] | list[tuple[float, float, float]] | None = None,
        # Custom additions
        horizontal_line: float | None = None,
        figsize: tuple[int, int] = (12, 6),
        bbox_to_anchor: tuple[float, float] = (0.5, 1.03),
    ):
        assert xtick_labels is None or len(xtick_labels) == len(groups) * len(bars)
        assert bar_text is None or len(bar_text) == len(groups) * len(bars)
        self.groups = groups
        self.models = bars
        self.sections = sections
        self.supergroups = supergroups
        # Layout
        self.bar_width = bar_width
        self.bar_spacing = bar_spacing
        self.group_spacing = group_spacing
        self.group_name_dy = group_name_dy
        self.group_name_offset = (
            (len(self.models) * (self.bar_width + self.bar_spacing)) * 0.4
            if group_name_offset is None
            else group_name_offset
        )
        self.scale = scale

        # Labels
        self.group_name_fontsize = group_name_fontsize
        self.group_name_color = group_name_color
        self.xtick_labels = xtick_labels if xtick_labels is not None else len(groups) * bars
        self.xtick_rotation = xtick_rotation
        self.xtick_fontsize = xtick_fontsize
        self.ytick_fontsize = ytick_fontsize
        self.xtick_ha = xtick_ha
        self.bar_text = bar_text
        # Offset from bar center
        self.xtick_offset = self.bar_width / 2 if xtick_ha == "right" else 0
        self.ylabel = ylabel
        self.title = title
        self.legend_cols = legend_cols
        self.legend_loc = legend_loc
        self.legend_fontsize = legend_fontsize
        self.bbox_to_anchor = bbox_to_anchor

        # Other
        colors_default = seaborn.color_palette("pastel", len(self.sections))
        colors_default = colors_default[2:] + colors_default[:2]  # Because green is at idx 2
        self.colors = colors_default if colors is None else colors
        # Use this setting to saturate the bars and print their actual value on top
        self.nb_saturated_bars = 0
        self.horizontal_line = horizontal_line
        self.figsize = figsize

    def construct_subplot(self, ax: Any, df: pd.DataFrame, add_xticks_and_label: bool = True, add_legend: bool = True):
        # Number of groups is number of different seq_len fields in df
        assert len(df) == len(self.groups) * len(self.models)

        indices = np.arange(len(self.groups)) * (
            len(self.models) * (self.bar_width + self.bar_spacing) + self.group_spacing
        )
        group_name_positions = indices + self.group_name_offset

        # Make bars
        max_height = 0
        for i, model in enumerate(self.models):
            bottom = np.zeros(len(self.groups))
            positions = indices + i * (self.bar_width + self.bar_spacing)
            df_model = df[df["model"] == model]
            for j, section in enumerate(self.sections):
                heights = df_model[section].values
                ax.bar(
                    positions,
                    heights,
                    self.bar_width,
                    bottom=bottom,
                    label=f"{section}" if i == 0 else "",
                    color=self.colors[j],
                    edgecolor="black",
                )
                bottom += heights
            max_height = max(max_height, np.max(bottom))

            # Add text above bar
            if self.bar_text:
                bar_heights = bottom
                labels_this_bar = (self.bar_text[i + len(self.models) * n_group] for n_group in range(len(self.groups)))

                for x, y in zip(positions, bar_heights):
                    ax.text(x=x, y=y, va="bottom", ha="center", s=next(labels_this_bar))

        # Custom xticks and labels
        xtick_positions = [i + self.bar_width / 2 for i in indices]
        if add_xticks_and_label:
            xtick_labels = [s.lstrip("L=") for s in self.groups]
            x_label = "Sequence length (tokens)"
        else:
            xtick_labels = ["" for i in indices]
            x_label = ""
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=self.xtick_fontsize, ha=self.xtick_ha, rotation=self.xtick_rotation)
        ax.set_xlabel(x_label, fontsize=16)

        ax.tick_params(axis="y", labelsize=self.ytick_fontsize)

        if self.supergroups is not None:
            groups_per_supergroup = len(self.groups) // len(self.supergroups)
            # e.g. 3 groups in each supergroup -> put supergroup name under group name at idx=1 within the supergroup
            supergroup_idx_within_group = groups_per_supergroup // len(self.supergroups)
            # Supergroup names
            for i, supergroup in enumerate(self.supergroups):
                x_coordinate = group_name_positions[i * groups_per_supergroup + supergroup_idx_within_group]
                ax.annotate(
                    supergroup,
                    xy=(x_coordinate, 0),  # Reference in coordinate system
                    xycoords="data",  # Use coordinate system of data points
                    xytext=(0, self.group_name_dy - 0.9),  # Offset from reference
                    textcoords="offset fontsize",  # Offset value is relative to fontsize
                    ha="center",
                    va="top",
                    weight="normal",
                    fontsize=14,
                    rotation=0,
                )

        # Add horizontal line
        if self.horizontal_line:
            ax.axhline(y=self.horizontal_line, color="r")

        # Set y limit based on the maximum total latency value
        ax.set_ylim([0, max_height * 1.1])

        # Add labels and title
        ax.set_ylabel(self.ylabel, fontsize=16)
        ax.set_title(self.title, fontsize=18)
        if add_legend:
            ax.legend(
                ncol=self.legend_cols,
                fontsize=self.legend_fontsize,
                loc=self.legend_loc,
                bbox_to_anchor=self.bbox_to_anchor,
            )

    def construct_subplot_broken_axis(self, ax: Any, data: ARRAY_T):
        assert data.shape == (len(self.groups), len(self.models), len(self.sections))

        indices = np.arange(len(self.groups)) * (
            len(self.models) * (self.bar_width + self.bar_spacing) + self.group_spacing
        )

        # Make bars
        for i, _ in enumerate(self.models):
            bottom = np.zeros(len(self.groups))
            for j, _ in enumerate(self.sections):
                positions = indices + i * (self.bar_width + self.bar_spacing)
                heights = data[:, i, j]
                ax.bar(
                    positions,
                    heights,
                    self.bar_width,
                    bottom=bottom,
                    color=self.colors[j],
                    edgecolor="black",
                )
                bottom += heights

    def construct_subplot_line(self, ax: Any, df: pd.DataFrame, x_axis: str, y_axis: str, log_x=False, log_y=False):
        """
        Draw a line for each model with x axis and y axis attributes as given.
        """
        assert len(df) == len(self.models) * len(self.groups)
        for i, model in enumerate(self.models):
            df_model = df[df["model"] == model]
            x = np.array([int(i) for i in df_model[x_axis]])
            y = np.array(df_model[y_axis])
            ax.plot(x, y, label=model, marker="o")
        if log_x:
            ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(x, fontsize=self.xtick_fontsize, ha=self.xtick_ha, rotation=self.xtick_rotation)
        ax.set_xlabel("Sequence length", fontsize=16)
        y_label = y_axis
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_title(self.title, fontsize=16)
        ax.legend(ncol=self.legend_cols, fontsize=14, loc=self.legend_loc)

    def construct_subplot_line_two_y_axes(
        self,
        ax: Any,
        df: pd.DataFrame,
        x_axis: str,
        y_axes: list[str],
        y_axes_labels: list[str],
        marker_shapes: list[str],
        log_x=False,
        log_y=False,
    ):
        """
        Draw a line for each model with x axis and y axis attributes as given.
        """
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(y_axes)]
        assert len(df) == len(self.models) * len(self.groups)
        assert len(self.models) == len(marker_shapes)
        assert len(y_axes) == 2
        for j, (y_axis, y_axis_label) in enumerate(zip(y_axes, y_axes_labels)):
            if j == 1:
                ax = ax.twinx()
            color = colors[j]
            for i, (model, marker_shape) in enumerate(zip(self.models, marker_shapes)):
                df_model = df[df["model"] == model]
                x = np.array([int(i) for i in df_model[x_axis]])
                y = np.array(df_model[y_axis])
                label = model
                ax.plot(x, y, label=label, marker=marker_shape, color=color)
            if log_y:
                ax.set_yscale("log")
            ax.set_ylabel(y_axis_label, fontsize=16, color=color)
            ax.tick_params(axis="y", labelcolor=color)
        if log_x:
            ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels(x, fontsize=self.xtick_fontsize, ha=self.xtick_ha, rotation=self.xtick_rotation)
        ax.set_xlabel("Sequence length", fontsize=16)
        ax.set_title(self.title, fontsize=16)
        ax.legend(ncol=self.legend_cols, fontsize=14, loc=self.legend_loc)

    def construct_subplot_line_for_four_subplots(
        self, ax: Any, df: pd.DataFrame, x_axis: str, y_axis: str, y_axis_label: str, log_x=False, log_y=False
    ):
        """
        Draw a line for each model with x axis and y axis attributes as given.
        """
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(self.models)]
        assert len(df) == len(self.models) * len(self.groups)
        for i, model in enumerate(self.models):
            df_model = df[df["model"] == model]
            x = np.array(range(len(df_model[x_axis])))  # To get equally spaced x-ticks
            x_labels = np.array([int(i) for i in df_model[x_axis]])
            y = np.array(df_model[y_axis])
            label = "Transformer" if "OPT" in model else "SSM"
            ax.plot(x, y, label=label, marker="o", color=colors[i])
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(y_axis_label, fontsize=16)
        if log_x:
            ax.set_xscale("log", base=2)
        # Determine the power for scientific notation
        y_min, y_max = ax.get_ylim()
        y_min = max(y_min, 0)

        ax.set_ylabel(y_axis_label, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=self.xtick_fontsize, ha=self.xtick_ha, rotation=self.xtick_rotation)
        ax.tick_params(axis="y", labelsize=self.ytick_fontsize)
        ax.set_title(self.title, fontsize=16)
        ax.legend(ncol=self.legend_cols, fontsize=14, loc=self.legend_loc)
        # ax.set_ylim(bottom=0)

    def plot(self, df: pd.DataFrame, filename: str) -> None:
        plt.rc("font", family="DejaVu Serif")  # type: ignore
        _, ax = plt.subplots(figsize=self.figsize)  # type: ignore
        self.construct_subplot(ax, df)
        plt.yscale(self.scale)  # type: ignore
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight", transparent=False)  # type: ignore

    def plot_two_subplots(
        self,
        df: pd.DataFrame,
        filename: str,
        y_labels: list[str] = [],
        scaling_factors: list[float] = [1e3, 1],
        add_legends: list[bool] = [True, True],
    ) -> None:
        plt.rc("font", family="DejaVu Serif")  # type: ignore
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=self.figsize)  # type: ignore
        # Plot prefill subplot
        df_prefill = df[df["stage"] == "prefill"]
        df_prefill.iloc[:, 3:] /= scaling_factors[0]
        self.ylabel = y_labels[0] if y_labels else self.ylabel
        self.title = "Prefill"
        self.construct_subplot(ax0, df_prefill, add_xticks_and_label=False, add_legend=add_legends[0])
        # ax0.set_title('Prefill', fontsize=18)
        # Plot decode subplot
        df_decode = df[df["stage"] == "decode"]
        df_decode.iloc[:, 3:] /= scaling_factors[1]
        self.ylabel = y_labels[1] if y_labels else self.ylabel
        self.title = "Decode"
        self.construct_subplot(ax1, df_decode, add_legend=add_legends[1])
        # ax1.set_title('Decode', fontsize=18)
        plt.yscale(self.scale)  # type: ignore
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches="tight", transparent=False)  # type: ignore

    def plot_line(self, df: pd.DataFrame, x_axis: str, y_axis: str, filename: str) -> None:
        plt.rc("font", family="DejaVu Serif")  # type: ignore
        _, ax = plt.subplots(figsize=(12, 6))  # type: ignore
        self.construct_subplot_line(ax, df, x_axis, y_axis, log_x=True)
        plt.yscale(self.scale)  # type: ignore
        plt.tight_layout()
        plt.savefig(filename, transparent=False)  # type: ignore

    def plot_line_two_subplots(self, df: pd.DataFrame, filename: str) -> None:
        plt.rc("font", family="DejaVu Serif")  # type: ignore
        _, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))  # type: ignore
        # Plot prefill subplot
        df_prefill = df[df["stage"] == "prefill"]
        # Convert all columns from 3rd onwards to seconds (divide by 1e3)
        df_prefill.iloc[:, 3:] /= 1e3
        self.ylabel = "Time to first token (s)"
        marker_shapes = ["o", "s"]
        y_axes = ["total_ops", "total_mem_access"]
        y_axes_labels = ["Total Operations (ops)", "Total Mem Accesses (bits)"]
        self.construct_subplot_line_two_y_axes(
            ax0, df_prefill, "seq_len", y_axes, y_axes_labels, marker_shapes, log_x=True
        )
        ax0.set_title("Prefill Stage", fontsize=18)
        # Plot decode subplot
        df_decode = df[df["stage"] == "decode"]
        self.ylabel = "Time between tokens (ms)"
        self.construct_subplot_line_two_y_axes(
            ax1, df_decode, "seq_len", y_axes, y_axes_labels, marker_shapes, log_x=True
        )
        ax1.set_title("Decode Stage", fontsize=18)
        plt.yscale(self.scale)  # type: ignore
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight", transparent=False)  # type: ignore

    def plot_line_four_subplots(self, df: pd.DataFrame, filename: str) -> None:
        plt.rc("font", family="DejaVu Serif")  # type: ignore
        _, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 5), sharex=True)  # type: ignore
        # Plot prefill ops subplot
        df_prefill = df[df["stage"] == "prefill"]
        # Convert all columns from 3rd onwards to seconds (divide by 1e3)
        df_prefill.iloc[:, 3:] /= 1e3
        df_decode = df[df["stage"] == "decode"]
        dfs = [df_prefill, df_decode, df_prefill, df_decode]
        y_axes = ["total_ops", "total_ops", "total_mem_access", "total_mem_access"]
        y_axes_labels = ["Operations", "", "Mem Accesses (bits)", ""]
        for i, ax in enumerate(axs.flat):
            self.construct_subplot_line_for_four_subplots(
                ax, dfs[i], "seq_len", y_axes[i], y_axes_labels[i], log_x=False
            )
            if i == 0:
                ax.set_title("Prefill Stage", fontsize=16, pad=10)
            elif i == 1:
                ax.set_title("Decode Stage", fontsize=16, pad=10)
            if i in [2, 3]:
                ax.set_xlabel("Sequence length (tokens)", fontsize=14)

        plt.yscale(self.scale)  # type: ignore
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight", transparent=False)  # type: ignore


class BarPlotterSubfigures:
    def __init__(
        self,
        bar_plotters: list[BarPlotter],
        *,
        subplot_rows: int = 1,
        subplot_cols: int = 1,
        width_ratios: list[int | float] | None = None,
        title: str = "",
    ):
        self.nb_plots = subplot_rows * subplot_cols
        assert width_ratios is None or len(width_ratios) == subplot_cols
        assert len(bar_plotters) == self.nb_plots
        self.bar_plotters = bar_plotters
        self.subplot_rows = subplot_rows
        self.subplot_cols = subplot_cols
        self.width_ratios = width_ratios if width_ratios is not None else subplot_cols * [1]
        self.title = title

    def plot(self, data: list[ARRAY_T], filename: str) -> None:
        assert len(data) == self.nb_plots

        fig, axises = plt.subplots(  # type: ignore
            nrows=self.subplot_rows, ncols=self.subplot_cols, width_ratios=self.width_ratios, figsize=(12, 6)
        )

        for ax, data_subplot, plotter in zip(axises, data, self.bar_plotters):  # type: ignore
            plotter.construct_subplot(ax, data_subplot)  # type: ignore

        fig.suptitle(self.title)  # type: ignore
        # plt.rc("font", family="DejaVu Serif")
        plt.tight_layout(pad=0.5, w_pad=0)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, transparent=False)  # type: ignore
