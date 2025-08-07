import datetime
import logging
import os
import sys
from copy import deepcopy
from itertools import product
from typing import Generator, TypeAlias

import numpy as np

sys.path.append(os.getcwd())

from run_settings import CLEAR_INCOMPLETE_RUNS, N_JOBS, RUN_PARALLEL, SKIP_IF_RESULT_EXISTS
from scripts.util import (
    LOG_FILE,
    MODEL,
    QUANT,
    clear_incomplete_runs,
    compute_dram_bandwidth,
    compute_mem_size_MB,
    generate_accelerator,
    generate_all_onnx_models,
    generate_mapping,
    get_intra_core_split_D,
    get_layer_stacks,
    run_experiments,
    run_experiments_parallel,
)
from src.config import Mamba1Config
from src.simulation import run_simulation_multicore
from src.util import Stage

output_dir = "outputs/sweep_contour"
template_dump_dir = os.path.join(output_dir, "templates")
experiment_name = "CONTOUR SWEEP"
D_split_str = {1: "no_D_split", None: "optimal_D_split"}

# ---------- Configuration dashboard ----------

nb_x_values = 20
seq_lengths = [1, 64, 1024]
mem_area_sweep = np.array(range(1, nb_x_values)) / nb_x_values
D_split_sweep = (1, None)
area_reduction_sweep = [0.125, 0.5, 1.0, 1.25]

ARGS_T: TypeAlias = tuple[Mamba1Config, float, float, int | None]

# ---------- Configuration dashboard end ----------


class DontRunException(Exception):
    """Implement early stopping: if the optimal split is 1, the result is the same as the no-split case"""


def get_experiment_id(args: ARGS_T):
    model, mem_area_percentage, area_reduction, D_split = args
    return f"D={model.d_model}_L={model.prefill_size}_{int(mem_area_percentage*100):.1f}pctMem_{(area_reduction*100):.1f}pctArea_{D_split_str[D_split]}"


def get_dump_path(args: ARGS_T):
    return os.path.join(output_dir, get_experiment_id(args))


def get_model_combinations() -> Generator[Mamba1Config, None, None]:
    """Make a copy of the default model and change the parameters to sweep"""
    for seq_len in seq_lengths:
        curr_model = deepcopy(MODEL)
        curr_model.prefill_size = seq_len
        yield curr_model


def get_configurations():
    return product(get_model_combinations(), mem_area_sweep, area_reduction_sweep, D_split_sweep)


def get_mapping(args: ARGS_T):
    model, mem_area_percentage, area_reduction, D_split = args

    gen_mapping_file = os.path.join(template_dump_dir, f"{get_experiment_id(args)}.yaml")
    mem_size_MB = compute_mem_size_MB(mem_area_percentage, area_reduction)

    if D_split is None:
        intra_core_splits_D = get_intra_core_split_D(model, mem_size_MB=mem_size_MB)
        if intra_core_splits_D == 1:
            raise DontRunException

    generate_mapping(
        model,
        output_mapping_file=gen_mapping_file,
        intra_core_splits_L=model.prefill_size,  # Always take size-1 splits
        intra_core_splits_N=1,
        intra_core_splits_D=D_split,
        inter_core_dim="N",
        nb_cores=1,
        mem_size_MB=mem_size_MB,
    )
    return gen_mapping_file


def get_accelerator(args: ARGS_T):
    model, mem_area_percentage, area_reduction, D_split = args

    dram_bw = compute_dram_bandwidth(area_reduction)
    gen_core_file = os.path.join(
        template_dump_dir,
        f"core_gen_{int(mem_area_percentage*100)}pctMem_{int(area_reduction*100)}pctArea.yaml",
    )
    gen_offchip_core_file = os.path.join(
        template_dump_dir,
        f"{int(area_reduction*100)}pctArea.yaml",
    )
    gen_accelerator_file = os.path.join(
        template_dump_dir,
        f"accelerator_gen_{int(mem_area_percentage*100)}pctMem_{int(area_reduction*100)}pctArea.yaml",
    )
    generate_accelerator(
        gen_core_file=gen_core_file,
        gen_accelerator_file=gen_accelerator_file,
        nb_cores=1,
        mem_area_percentage=mem_area_percentage,
        area_fraction=area_reduction,
        gen_offchip_core_file=gen_offchip_core_file,
        dram_bits_per_cc=dram_bw,
    )
    return gen_accelerator_file


def run_single_experiment(args: ARGS_T):
    """Generate results for all accelerator variations"""
    model = args[0]
    layer_stacks = get_layer_stacks(model)
    accelerator_path = get_accelerator(args)
    dump_path = get_dump_path(args)
    try:
        mapping_path = get_mapping(args)
    except DontRunException:
        print(f"Skipping {dump_path} because optimal split is 1")
        return

    try:
        run_simulation_multicore(
            model=model,
            stage=Stage.PREFILL,
            quant=QUANT,
            accelerator_name_or_path=accelerator_path,
            mapping_path=mapping_path,
            output_dir=dump_path,
            mode="fused",
            layer_stacks=layer_stacks,
            lpf_limit=6,
            dump_path=dump_path,
            skip_if_dump_exists=False,
            skip_if_result_exists=SKIP_IF_RESULT_EXISTS,
        )
    except AttributeError as e:
        print(
            f"EXCEPTION FOR RUN {dump_path}: AttributeError. Cause might be outdated cache. Try removing  the dump path. (exception : {e})"
        )
    except Exception as e:
        print(f"EXCEPTION FOR RUN {dump_path}: {e}")


def run_and_capture_output(args: ARGS_T):
    dump_path = get_dump_path(args)
    output_file = os.path.join(dump_path, "out.log")
    os.makedirs(dump_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} - Starting run {dump_path}\n")

    with open(output_file, "w") as f:
        sys.stdout = f
        sys.stderr = f
        handler = logging.StreamHandler(f)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.handlers = [handler]
        run_single_experiment(args)

    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} - Finished run {dump_path}\n")


if __name__ == "__main__":
    """Boilerplate code"""
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} - Starting {experiment_name}\n")

    generate_all_onnx_models(get_model_combinations())

    if CLEAR_INCOMPLETE_RUNS:
        clear_incomplete_runs(output_dir, exception_dirs=[template_dump_dir])  # -> will clear cached workload
    print(f"Running {experiment_name} with {len(list(get_configurations()))} configurations")
    if RUN_PARALLEL:
        run_experiments_parallel(
            fn=run_and_capture_output, configs=get_configurations(), n_jobs=N_JOBS, do_shuffle=False
        )
    else:
        run_experiments(fn=run_single_experiment, configs=get_configurations())

    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} - Finished {experiment_name}\n")
