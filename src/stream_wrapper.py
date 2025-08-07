import json
import os
from typing import Literal

from zigzag.stages.main import MainStage

from stream.api import _sanity_check_inputs
from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.stages.allocation.constraint_optimization_allocation import ConstraintOptimizationAllocationStage
from stream.stages.estimation.stream_cost_model_evaluation import StreamCostModelEvaluationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.scheduling_order_generation import SchedulingOrderGenerationStage
from stream.stages.generation.tiled_workload_generation import (
    TiledWorkloadGenerationStage,
)
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.set_fixed_allocation_performance import SetFixedAllocationPerformanceStage
from stream.stages.stage import StageCallable
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage  # type: ignore
from stream.visualization.perfetto import convert_scme_to_perfetto_json

STAGES_SKIP_CO = (
    AcceleratorParserStage,
    StreamONNXModelParserStage,
    LayerStacksGenerationStage,
    TilingGenerationStage,
    TiledWorkloadGenerationStage,
    ZigZagCoreMappingEstimationStage,
    SchedulingOrderGenerationStage,  # ! Also sets core allocations needed for SetFixedAllocationPerformanceStage
    SetFixedAllocationPerformanceStage,
    StreamCostModelEvaluationStage,
)

STAGES_CO = (
    AcceleratorParserStage,
    StreamONNXModelParserStage,
    LayerStacksGenerationStage,
    TilingGenerationStage,
    TiledWorkloadGenerationStage,
    ZigZagCoreMappingEstimationStage,
    SetFixedAllocationPerformanceStage,
    ConstraintOptimizationAllocationStage,
)


def stream_wrapper_co(
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    cache_path: str,  # For all experiments
    dump_path: str,  # For this experiments
    experiment_config_id: str,
    lpf_limit: int,
    latency_attr: str,
    skip_CO: bool = True,
):

    _sanity_check_inputs(hardware, workload, mapping, mode, cache_path)
    # _sanity_check_gurobi_license()

    # Output paths
    tiled_workload_path = os.path.join(cache_path, f"{experiment_config_id}-tiled_workload.pickle")
    cost_lut_path = os.path.join(cache_path, f"{experiment_config_id}-cost_lut.pickle")
    if not skip_CO:
        tiled_workload_post_co_path = os.path.join(cache_path, f"{experiment_config_id}-tiled_workload_post_co.pickle")
        cost_lut_post_co_path = os.path.join(cache_path, f"{experiment_config_id}-cost_lut_post_co.pickle")
    else:
        tiled_workload_post_co_path = tiled_workload_path
        cost_lut_post_co_path = cost_lut_path

    results_path = os.path.join(dump_path, "results.json")
    # scme_path = os.path.join(dump_path, "scme.pickle")
    steady_state_visualization_path = dump_path

    stages: tuple[StageCallable] = STAGES_SKIP_CO if skip_CO else STAGES_CO

    mainstage = MainStage(
        list(stages),
        accelerator=hardware,  # required by AcceleratorParserStage
        workload_path=workload,  # required by ModelParserStage
        mapping_path=mapping,  # required by ModelParserStage
        loma_lpf_limit=lpf_limit,  # required by LomaEngine
        mode=mode,
        layer_stacks=layer_stacks,
        tiled_workload_path=tiled_workload_path,
        allocations_path=dump_path,
        cost_lut_path=cost_lut_path,
        steady_state_visualization_path=steady_state_visualization_path,
        tiled_workload_post_co_path=tiled_workload_post_co_path,
        cost_lut_post_co_path=cost_lut_post_co_path,
        operands_to_prefetch=[],
        latency_attr=latency_attr,
    )
    # Launch the MainStage
    answers = mainstage.run()
    scme = answers[0][0]
    save_results_concise(scme, results_path)

    # Plotting schedule timeline of best SCME
    cost_lut = CostModelEvaluationLUT(cost_lut_post_co_path)

    convert_scme_to_perfetto_json(scme, cost_lut, json_path=f"{dump_path}/{experiment_config_id}-schedule.json")

    # Plotting memory usage of best SCME
    plot_memory_usage(scme, (0,), (100,), fig_path=f"{dump_path}/{experiment_config_id}-schedule.png")

    return scme


def save_results_concise(scme: StreamCostModelEvaluation, results_path: str):
    results = {"latency": int(scme.latency)}
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
