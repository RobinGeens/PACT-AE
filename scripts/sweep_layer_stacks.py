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
    LAYER_STACKS_SHORT_T,
    LAYER_STACKS_T,
    LOG_FILE,
    MODEL,
    QUANT,
    clear_incomplete_runs,
    generate_accelerator,
    generate_all_onnx_models,
    generate_mapping,
    get_layer_stack_SSM,
    run_experiments,
    run_experiments_parallel,
    stack_to_full_str,  # type: ignore
)
from src.config import Mamba1Config
from src.simulation import run_simulation_multicore
from src.util import Stage

output_dir = "outputs/sweep_layer_stacks"
template_dump_dir = os.path.join(output_dir, "templates")
experiment_name = "LAYER STACK SWEEP"

# ---------- Configuration dashboard ----------

seq_lengths = [1, 64, 512, 1024, 2048]
hidden_dimensions = [2560]
layer_stack_sweep: list[LAYER_STACKS_SHORT_T] = [
    [],  # UF
    # [(18,), (19,), (20,), (21,), ("SSM",)],  # LBL but keep state local
    [(18, 19)],  # A
    [(18,), (19,), (20, 21)],  # B
    # [(18, 19), (20,), (21,), ("SSM",)],  # Also keep state local
    [(18, 19), (20, 21)],  # A-B
    [(20,), (21,), (18, 19, "SSM")],  # AS
    # [(18,), (20,), (19, 21, "SSM")],  # ABS
    [(18,), (19,), (20, 21, "SSM")],  # BS
    [(18, 19, "SSM"), (20, 21)],  # AS-B
    [(18, 19), (20, 21, "SSM")],  # A-BS
    [(18, 19, 20, 21, "SSM")],  # FA
]

# ---------- Configuration dashboard end ----------


def get_experiment_id(model: Mamba1Config, layer_stacks: LAYER_STACKS_SHORT_T):
    stack_str = stack_to_full_str(layer_stacks).replace("\n", "").replace("+", "_")
    return f"D={model.d_model}_L={model.prefill_size}_stacks={stack_str}"


def get_dump_path(model: Mamba1Config, layer_stacks: LAYER_STACKS_SHORT_T):
    return os.path.join(output_dir, get_experiment_id(model, layer_stacks))


def get_model_combinations() -> Generator[Mamba1Config, None, None]:
    """Make a copy of the default model and change the parameters to sweep"""
    for hidden_d, seq_len in product(hidden_dimensions, seq_lengths):
        curr_model = deepcopy(MODEL)
        curr_model.prefill_size = seq_len
        curr_model.d_model = hidden_d
        yield curr_model


def get_configurations():
    return product(get_model_combinations(), layer_stack_sweep)


def get_mapping(model: Mamba1Config, layer_stacks: LAYER_STACKS_SHORT_T):
    gen_mapping_file = os.path.join(template_dump_dir, f"{get_experiment_id(model, layer_stacks)}.yaml")
    generate_mapping(
        model,
        nb_cores=1,
        output_mapping_file=gen_mapping_file,
        intra_core_splits_L=model.prefill_size,
        intra_core_splits_N=1,
        intra_core_splits_D=1,
        inter_core_dim="D",
    )
    return gen_mapping_file


def fill_SSM_stack(model: Mamba1Config, layer_stacks: LAYER_STACKS_SHORT_T) -> LAYER_STACKS_T:
    filled_stacks: LAYER_STACKS_T = [(i,) for i in range(18)]
    for stack in layer_stacks:
        new_stack = ()
        for s in stack:
            if s == "SSM":
                new_stack += get_layer_stack_SSM(model)
            else:
                new_stack += (s,)
        filled_stacks.append(new_stack)
    return filled_stacks


def get_accelerator(nb_cores: int):
    mem_size_MB = 24
    gen_core_file = os.path.join(template_dump_dir, f"core_gen_{nb_cores}core_{mem_size_MB}MB.yaml")
    gen_accelerator_file = os.path.join(template_dump_dir, f"accelerator_gen_{nb_cores}core_{mem_size_MB}MB.yaml")
    generate_accelerator(
        gen_core_file=gen_core_file,
        gen_accelerator_file=gen_accelerator_file,
        nb_cores=nb_cores,
        mem_area_percentage=None,
        mem_size_MB=mem_size_MB,
    )
    return gen_accelerator_file


def run_single_experiment(args: tuple[Mamba1Config, LAYER_STACKS_SHORT_T]):
    model, layer_stacks = args
    dump_path = get_dump_path(model, layer_stacks)
    mapping_path = get_mapping(model, layer_stacks)
    accelerator_path = get_accelerator(1)
    layer_stacks = fill_SSM_stack(model, layer_stacks)

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


def run_and_capture_output(args: tuple[Mamba1Config, LAYER_STACKS_SHORT_T]):
    model, layer_stacks = args
    dump_path = get_dump_path(model, layer_stacks)
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
