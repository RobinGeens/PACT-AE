import os
import sys
from dataclasses import dataclass

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn

sys.path.append(os.getcwd())
from src.config_library import MAMBA1_2_8B

L = 1
D = MAMBA1_2_8B.d_inner
N = MAMBA1_2_8B.d_state
PRECISION = 32  # bits

# order of NODES is used for schedule
NODES = [
    "A",
    "Δ",
    "h_t",
    "ΔA",
    "Exp(ΔA)",
    "B",
    "ΔB",
    "x",
    "ΔB*x",
    "Exp(ΔA)*h_t",
    "h_t+1",
    "C",
    "C*h_t+1",
    # 'D',
    # 'D*x',
    # 'y_t+1',
    "end",
]

COLOR_PALETTE = seaborn.color_palette("pastel")

CONSTANT_COLOR = COLOR_PALETTE[7]
TENSOR_COLORS = {
    "A": CONSTANT_COLOR,  # COLOR_PALETTE[1],
    "Δ": CONSTANT_COLOR,
    "h_t": COLOR_PALETTE[0],
    "ΔA": COLOR_PALETTE[1],
    "Exp(ΔA)": COLOR_PALETTE[4],
    "B": CONSTANT_COLOR,
    "ΔB": COLOR_PALETTE[6],
    "x": CONSTANT_COLOR,
    "ΔB*x": COLOR_PALETTE[8],
    "Exp(ΔA)*h_t": COLOR_PALETTE[9],
    "h_t+1": COLOR_PALETTE[0],
    "C": CONSTANT_COLOR,
    "C*h_t+1": COLOR_PALETTE[5],
    # 'D': 'k',
    # 'D*x': 'w',
    # 'y_t+1': 'w',
    "end": CONSTANT_COLOR,
}

NODE_SIZES = {
    "Δ": L * D,
    "A": D * N,
    "B": L * N,
    "x": L * D,
    "h_t": L * D * N,
    "C": L * N,
    "D": D,
    "ΔA": L * D * N,
    "Exp(ΔA)": L * D * N,
    "ΔB": L * D * N,
    "ΔB*x": L * D * N,
    "Exp(ΔA)*h_t": L * D * N,
    "h_t+1": L * D * N,
    "C*h_t+1": L * D,
    "D*x": L * D,
    "y_t+1": L * D,
    "end": 0,
}  # in elements


TENSOR_HEIGHT_IN_FIGURE = {
    L * D * N: L * D * N,
    D * N: D * N,
    L * D: L * D * N / 7,
    D: L * D * N / 7,
    L * N: L * D * N / 7,
}

CONSTANT_NODES = [
    "Δ",
    "A",
    "B",
    "x",
    "C",
    # 'D',
    "h_t",
    "end",
]

EDGES = [
    ("Δ", "ΔA"),
    ("A", "ΔA"),
    ("ΔA", "Exp(ΔA)"),
    ("Δ", "ΔB"),
    ("B", "ΔB"),
    ("ΔB", "ΔB*x"),
    ("x", "ΔB*x"),
    ("Exp(ΔA)", "Exp(ΔA)*h_t"),
    ("h_t", "Exp(ΔA)*h_t"),
    ("Exp(ΔA)*h_t", "h_t+1"),
    ("ΔB*x", "h_t+1"),
    ("h_t+1", "C*h_t+1"),
    ("C", "C*h_t+1"),
    ("C*h_t+1", "end"),
    # ('D', 'D*x'),
    # ('x', 'D*x'),
    # ('C*h_t+1', 'D*x'),
    # ('D*x', 'y_t+1'),
    ("h_t+1", "end"),
    # ('y_t+1', 'end'),
    ("A", "end"),  # to keep A in sram always
]

LATENCIES = {
    "Δ": 0,
    "A": 0,
    "B": 0,
    "x": 0,
    "h_t": 0,
    "C": 0,
    "D": 0,
    "ΔA": 10,
    "Exp(ΔA)": 10,
    "ΔB": 10,
    "ΔB*x": 10,
    "Exp(ΔA)*h_t": 10,
    "h_t+1": 10,
    "C*h_t+1": 10,
    # 'D*x': 10,
    # 'y_t+1': 10,
    "end": 0,
}


def generate_simple_workload_graph():
    # Check that each entry in EDGES is a node in NODES
    for edge in EDGES:
        assert edge[0] in NODES, f"Node {edge[0]} not in NODES"
        assert edge[1] in NODES, f"Node {edge[1]} not in NODES"
    # Generate a simple networkx DiGraph with NODES and EDGES
    G = nx.DiGraph()
    G.add_nodes_from(NODES)
    G.add_edges_from(EDGES)
    return G


@dataclass
class Lifetime:
    tensor: str
    start: int
    end: int


def get_start(node):
    return sum(LATENCIES[n] for n in NODES[: NODES.index(node)])


def get_end(node):
    return get_start(node) + LATENCIES[node]


def get_tensor_lifetimes(workload):
    """Get tensor lifetimes based on the workload with schedule order as defined in NODES.
    A tensor is alive as soon as it is produced and is dead after its last consumption.
    The producer is always the node name (as we name the output tensor identical to the node name)
    The last consumer is the last node in NODES that has the tensor as input
    Returns a list of Lifetime objects"""
    lifetimes = []
    tensor_to_first_producer = {}
    tensor_to_last_consumer = {}
    assert NODES[-1] == "end", "Last node in NODES must be 'end'"
    for tensor in NODES[:-1]:
        # PRODUCER
        first_producer = tensor
        tensor_to_first_producer[tensor] = first_producer
        # LAST CONSUMER
        succ_idxs = list(NODES.index(succ) for succ in workload.successors(tensor))
        max_succ_idx = max(succ_idxs)
        last_consumer = NODES[max_succ_idx]
        tensor_to_last_consumer[tensor] = last_consumer
    for tensor in NODES[:-1]:
        tensor = tensor
        start = get_start(tensor_to_first_producer[tensor])
        end = get_end(tensor_to_last_consumer[tensor])
        lifetimes.append(Lifetime(tensor, start, end))
    return lifetimes


def convert_bits_to_MiB(bits):
    return bits / 8 / 1024 / 1024


def get_cumulative_memory_usage_MiB(lifetimes):
    # Create a list of events (start and end of lifetimes)
    events = []
    for lifetime in lifetimes:
        events.append((lifetime.start, NODE_SIZES[lifetime.tensor] * PRECISION))
        events.append((lifetime.end, -NODE_SIZES[lifetime.tensor] * PRECISION))

    # Simplify the events if two are at the same time
    simplified_events = []
    unique_timesteps = set(event[0] for event in events)
    for timestep in unique_timesteps:
        memory_change = sum(event[1] for event in events if event[0] == timestep)
        simplified_events.append((timestep, memory_change))
    events = simplified_events

    # Sort events by time
    events.sort()

    # Calculate cumulative memory usage
    times = [event[0] for event in events]
    memory_changes = [event[1] for event in events]
    cumulative_memory = np.cumsum(memory_changes)

    # Convert to MiB using np apply
    cumulative_memory = np.apply_along_axis(convert_bits_to_MiB, 0, cumulative_memory)

    return times, cumulative_memory


def get_max_timestep_and_memory_usage_MiB(times, cumulative_memory):
    max_memory = max(cumulative_memory)
    max_timestep = times[np.argmax(cumulative_memory)]
    return max_timestep, max_memory


def plot_cumulative_memory_usage(times, cumulative_memory):
    # Print max memory usage
    max_timestep, max_memory = get_max_timestep_and_memory_usage_MiB(times, cumulative_memory)
    # print(f"Max memory usage: {max_memory} MiB at timestep {max_timestep}")
    # Plot the cumulative memory usage
    fig, ax = plt.subplots()
    ax.step(times, cumulative_memory, where="pre")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Memory Usage (MiB)")
    ax.set_title("Cumulative Memory Usage Over Time")
    ax.grid(True)
    fig.savefig("outputs/figures/cumulative_memory_usage.png", bbox_inches="tight")
    return max_timestep


def add_schedule_rectangles(ax, relevant_nodes):
    # Add a rectangle patch outside of the ax bounds with correct color for each node in relevant_nodes
    nb_nodes = len(relevant_nodes)
    x_fig = 0.0
    x_shift_per_node = 1 / nb_nodes
    for i, node in enumerate(relevant_nodes):
        color = TENSOR_COLORS[node]
        rect = patches.Rectangle(
            (x_fig, 1.02),
            x_shift_per_node,
            0.15,
            facecolor=color,
            edgecolor="black",
            transform=ax.transAxes,
            clip_on=False,
        )
        if node == "Exp(ΔA)*h_t":
            node = "Exp(ΔA)\n* h_t"
        ax.text(
            x_fig + x_shift_per_node / 2,
            1.095,
            node,
            ha="center",
            va="center",
            rotation=0,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        x_fig += x_shift_per_node
    # Add label to the schedule all the way on the left rotated 90 degrees
    ax.text(-0.043, 1.095, "Schedule", ha="center", va="center", rotation=90, transform=ax.transAxes)


def plot_tensor_lifetime_with_schedule(lifetimes: list[Lifetime], max_timestep: int, tensors_at_max: list[str]):
    # Plot the tensor lifetimes as a Gantt chart with the schedule on top
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.grid(True, axis="x")
    total_size = 0
    mid_ys = {}
    for i, lifetime in enumerate(lifetimes):
        tensor = lifetime.tensor
        start = lifetime.start
        end = lifetime.end
        mid_x = (start + end) / 2
        size = TENSOR_HEIGHT_IN_FIGURE[NODE_SIZES[lifetime.tensor]]
        mid_y = total_size + size / 2
        color = TENSOR_COLORS[tensor]
        ax.add_patch(
            plt.Rectangle((start, total_size), end - start, size, facecolor=color, edgecolor="black", zorder=1)
        )
        ax.text(mid_x, mid_y, lifetime.tensor, ha="center", va="center")
        total_size += size
        mid_ys[lifetime.tensor] = mid_y
    ax.set_xlim(0, max(lifetime.end for lifetime in lifetimes))
    ax.set_ylim(0, total_size)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel("Tensor Lifetimes")
    ax.set_ylabel("Tensor Size", labelpad=10)
    # Add the schedule on top
    relevant_nodes = [n for n in NODES if n not in CONSTANT_NODES]  # Skip the constant nodes
    ax.set_xticks([get_start(node) for node in relevant_nodes])
    ax.set_xticklabels([" " for node in relevant_nodes])
    add_schedule_rectangles(ax, relevant_nodes)
    # Add vertical red line at max_timestep
    max_timestep += 3  # manually place it a bit later
    ax.axvline(x=max_timestep, color="red", linestyle="--", zorder=5)
    # Add double-ended arrow line vertically at x = 70 going from 0 to L*D*N and annotate it with "D*N" vertically
    ax.annotate("", xy=(55, 0), xytext=(55, L * D * N), arrowprops=dict(arrowstyle="<->", color="red"))
    ax.text(56.2, L * D * N / 2, "D*N", color="red", ha="left", va="center")
    # Add 'Max memory usage' annotation alongside the red line vertically at height of 6.5*L*D*N
    ax.text(max_timestep - 0.2, 7 * L * D * N, "Max mem usage", color="red", ha="right", va="center", rotation=90)

    os.makedirs("outputs/figures", exist_ok=True)
    # fig.savefig("outputs/figures/tensor_lifetimes.pdf", bbox_inches="tight")
    fig.savefig("outputs/figures/tensor_lifetimes.png", bbox_inches="tight")


def get_alive_tensors(lifetimes, max_timestep):
    # Get the tensors alive at max_timestep
    alive_tensors = []
    for lifetime in lifetimes:
        if lifetime.start <= max_timestep and lifetime.end >= max_timestep:
            alive_tensors.append(lifetime.tensor)
    return alive_tensors


def main():
    workload = generate_simple_workload_graph()
    # Tensor lifetimes
    lifetimes: list[Lifetime] = get_tensor_lifetimes(workload)
    # Cumulative memory usage
    times, cumulative_memory = get_cumulative_memory_usage_MiB(lifetimes)
    plot_cumulative_memory_usage(times, cumulative_memory)
    max_timestep, _ = get_max_timestep_and_memory_usage_MiB(times, cumulative_memory)
    tensors_at_max: list[str] = get_alive_tensors(lifetimes, max_timestep)
    # print(f"Alive tensors at max timestep ({max_timestep})")
    # for tensor in tensors_at_max:
    #     tensor_size_MiB = convert_bits_to_MiB(NODE_SIZES[tensor] * PRECISION)
    #     print(f"{tensor} ({tensor_size_MiB} MiB)")
    plot_tensor_lifetime_with_schedule(lifetimes, max_timestep, tensors_at_max)


if __name__ == "__main__":
    main()
