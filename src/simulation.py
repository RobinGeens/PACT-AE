import os
import shutil
from datetime import datetime
from typing import Literal

import numpy as np
import yaml
from zigzag.datatypes import Constants

from src.config import ModelConfig, QuantConfig
from src.export_onnx import Stage, export_model_to_onnx
from src.plot_util import BarPlotter
from src.stream_wrapper import stream_wrapper_co
from src.util import (
    get_accelerator_name_and_path,
    get_experiment_id,
    get_onnx_path,
)
from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.workload.computation.computation_node import ComputationNode


def run_simulation_multicore(
    model: ModelConfig,
    stage: Stage,
    quant: QuantConfig,
    accelerator_name_or_path: str,
    mapping_path: str,
    output_dir: str,
    *,
    experiment_id: str | None = None,
    onnx_path: str | None = None,
    dump_path: str | None = None,
    skip_if_dump_exists: bool = True,
    skip_if_result_exists: bool = True,
    # Stream specific
    mode: Literal["fused"] | Literal["lbl"] = "lbl",
    layer_stacks: list[tuple[int, ...]] = [],
    lpf_limit: int = 6,
    latency_attr: Literal["ideal_temporal_cycle", "latency_total1", "latency_total2"] = "latency_total2",
):

    assert model.num_layer >= 1, "This should be a full configuration, not a `simulatable` one"
    model_for_simulation = model.to_single_layer_config()
    accelerator_name, accelerator_path = get_accelerator_name_and_path(accelerator_name_or_path)
    experiment_config_id = get_experiment_id(model_for_simulation, stage, quant, accelerator_name)

    # Replace default arguments
    if experiment_id is None:
        experiment_id = experiment_config_id
    if onnx_path is None:
        onnx_path = get_onnx_path(model_for_simulation, stage, quant)
    if dump_path is None:
        dump_path = os.path.join(output_dir, experiment_id)

    # Skip simulation if the parameters have been simulated before (successfully or failed)
    if skip_if_dump_exists and os.path.isdir(dump_path):
        return
    # Skip simulation if the parameters have been simulated before (successfully)
    scme_path = os.path.join(dump_path, "scme.pickle")
    results_path = os.path.join(dump_path, "results.json")
    if skip_if_result_exists and (os.path.isfile(scme_path) or os.path.isfile(results_path)):
        return

    # Save input configuration
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    copy_all_hardware_files(accelerator_path, output_dir)
    copy_mapping(mapping_path, output_dir)
    with open(f"{output_dir}/experiment_info.txt", "w") as f:
        f.write(
            f"""{40*'='}
        time={datetime.now()}
        model={model_for_simulation.parameterized_name}
        L={model_for_simulation.prefill_size}
        {quant=}
        stage={str(stage)}
        {onnx_path=}
        {mode=}
        {layer_stacks=}
        {lpf_limit=}
        {latency_attr=}
        """
        )

    # Generate ONNX workload
    if not os.path.exists(onnx_path):
        export_model_to_onnx(model_for_simulation, quant, path=onnx_path, stage=stage)

    print(f"Launching Stream: {experiment_id}")
    time_start = datetime.now()
    scme = stream_wrapper_co(
        hardware=accelerator_path,
        workload=onnx_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        cache_path=output_dir,
        dump_path=output_dir,
        experiment_config_id=experiment_config_id,
        lpf_limit=lpf_limit,
        latency_attr=latency_attr,
        skip_CO=True,
    )
    time_end = datetime.now()
    print(f"Execution time for run {experiment_id}: {time_end - time_start}")

    # Save layer names
    with open(f"{output_dir}/layer_info.txt", "w") as f:
        id_to_name = {node.id: node.name for node in scme.workload.nodes()}  # type: ignore
        for id, name in id_to_name.items():  # type: ignore
            f.write(f"Layer {id}:\t{name}\n")


def create_memory_report(scme: StreamCostModelEvaluation, fig_path: str):
    nodes: list[ComputationNode] = []
    nb_tiles: dict[str, int] = {}

    # Get all identical tiles
    for node in scme.workload.node_list:
        if isinstance(node, ComputationNode):
            if not any(node.has_same_performance(other) for other in nodes):
                nodes.append(node)
                nb_tiles[node.short_name] = 1
            else:
                nb_tiles[node.short_name] += 1

    # bars = [f"{nb_tiles[n.short_name]}x  {n.short_name}" for n in nodes]
    bars = [n.short_name for n in nodes]
    bar_text = [f"{nb_tiles[n.short_name]}x" for n in nodes]
    operands = [Constants.LAYER_OP_I, Constants.LAYER_OP_W, Constants.OUTPUT_LAYER_OP]
    sections = ["I", "W", "O"]

    data_to_plot = np.array([[[n.operand_size_bit[op] for op in operands] for n in nodes]])

    mem_capacity = scme.accelerator.get_core(0).memory_hierarchy.mem_level_list[-1].memory_instance.size

    plotter = BarPlotter(
        groups=[""],
        bars=bars,
        sections=sections,
        bar_text=bar_text,
        title="Memory usage per tile",
        horizontal_line=mem_capacity,
    )
    plotter.plot(data_to_plot, fig_path)


def copy_all_hardware_files(accelerator_path: str, dump_folder: str):
    if dump_folder not in accelerator_path:
        shutil.copyfile(accelerator_path, f"{dump_folder}/accelerator.yaml")

    with open(accelerator_path, "r") as f:
        core_data = yaml.safe_load(f)

    if "cores" in core_data:
        for core_path in set(core_data["cores"].values()):
            if dump_folder not in core_path:
                shutil.copyfile(core_path, f"{dump_folder}/{core_path.split('/')[-1]}")


def copy_mapping(mapping_path: str, dump_folder: str):
    shutil.copyfile(mapping_path, f"{dump_folder}/mapping.yaml")
