import datetime
import logging
import os
import sys
from copy import deepcopy
from itertools import product
from typing import Generator

sys.path.append(os.getcwd())

from run_settings import CLEAR_INCOMPLETE_RUNS, N_JOBS, RUN_PARALLEL, SKIP_IF_RESULT_EXISTS
from scripts.util import (
    LOG_FILE,
    MODEL,
    QUANT,
    clear_incomplete_runs,
    generate_accelerator,
    generate_all_onnx_models,
    generate_mapping,
    get_layer_stacks,
    run_experiments,
    run_experiments_parallel,
)
from src.config import Mamba1Config
from src.simulation import run_simulation_multicore
from src.util import Stage

output_dir = "outputs/sweep_mem_no_split"
template_dump_dir = os.path.join(output_dir, "templates")
experiment_name = "MEMORY SWEEP WITHOUT N SPLIT"

# ---------- Configuration dashboard ----------

seq_lengths = [1, 64, 512, 1024]
nb_core_sweep = [1]
mem_size_sweep: list[float] = [0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 6, 8, 10, 12, 18, 24, 30]

# ---------- Configuration dashboard end ----------


def get_experiment_id(model: Mamba1Config, nb_cores: int, mem_size_MB: int):
    return f"D={model.d_model}_L={model.prefill_size}_{nb_cores}core_{mem_size_MB}MB"


def get_dump_path(model: Mamba1Config, nb_cores: int, mem_size_MB: int):
    return os.path.join(output_dir, get_experiment_id(model, nb_cores, mem_size_MB))


def get_model_combinations() -> Generator[Mamba1Config, None, None]:
    """Make a copy of the default model and change the parameters to sweep"""
    for seq_len in seq_lengths:
        curr_model = deepcopy(MODEL)
        curr_model.prefill_size = seq_len
        yield curr_model


def get_configurations():
    return product(get_model_combinations(), nb_core_sweep, mem_size_sweep)


def get_mapping(model: Mamba1Config, nb_cores: int, mem_size_MB: int):
    gen_mapping_file = os.path.join(template_dump_dir, f"{get_experiment_id(model, nb_cores, mem_size_MB)}.yaml")
    generate_mapping(
        model,
        output_mapping_file=gen_mapping_file,
        intra_core_splits_L=model.prefill_size,
        intra_core_splits_N=1,
        intra_core_splits_D=1,
        inter_core_dim="D",
        nb_cores=nb_cores,
        mem_size_MB=mem_size_MB,
    )
    return gen_mapping_file


def get_accelerator(nb_cores: int, mem_size_MB: int):
    size_str = f"{int(mem_size_MB)}MB" if mem_size_MB % 1 == 0 else f"{int(mem_size_MB / 1024)}kB"
    gen_core_file = os.path.join(template_dump_dir, f"core_gen_{nb_cores}core_{size_str}.yaml")
    gen_accelerator_file = os.path.join(template_dump_dir, f"accelerator_gen_{nb_cores}core_{size_str}.yaml")
    generate_accelerator(
        gen_core_file=gen_core_file,
        gen_accelerator_file=gen_accelerator_file,
        nb_cores=nb_cores,
        mem_area_percentage=None,
        mem_size_MB=mem_size_MB,
    )
    return gen_accelerator_file


def run_single_experiment(args: tuple[Mamba1Config, int, int]):
    """Generate results for all accelerator variations"""
    model, nb_cores, mem_area_percentage = args
    layer_stacks = get_layer_stacks(model)
    accelerator_path = get_accelerator(nb_cores, mem_area_percentage)
    mapping_path = get_mapping(model, nb_cores, mem_area_percentage)
    dump_path = get_dump_path(model, nb_cores, mem_area_percentage)

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


def run_and_capture_output(args: tuple[Mamba1Config, int, int]):
    model, nb_cores, mem_area_percentage = args
    dump_path = get_dump_path(model, nb_cores, mem_area_percentage)
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
    """Boilerplate code :)"""
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} - Starting {experiment_name}\n")

    generate_all_onnx_models(get_model_combinations())

    if CLEAR_INCOMPLETE_RUNS:
        clear_incomplete_runs(output_dir, exception_dirs=[template_dump_dir])  # -> will clear cached workload
    print(f"Running {experiment_name} with {len(list(get_configurations()))} configurations")
    if RUN_PARALLEL:
        run_experiments_parallel(fn=run_and_capture_output, configs=get_configurations(), n_jobs=N_JOBS)
    else:
        run_experiments(fn=run_single_experiment, configs=get_configurations())

    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} - Finished {experiment_name}\n")
