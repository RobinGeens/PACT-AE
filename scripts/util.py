import concurrent
import concurrent.futures
import os
import shutil
from math import ceil, sqrt
from random import shuffle
from typing import Any, Callable, Iterator, Literal, TypeAlias

from src.config import Mamba1Config
from src.config_library import MAMBA1_2_8B, W32A32
from src.export_onnx import export_model_to_onnx
from src.modify_architecture import AcceleratorModifier, CoreModifier, MappingModifier
from src.util import Stage, get_onnx_path

# Base model info
MODEL = MAMBA1_2_8B
MODEL.batch_size = 1
QUANT = W32A32

# Base HW info
SRAM_SIZE_MB = 24
NB_PE = 8192
UNIT_SRAM_BW = 9999999999  # 1 core
UNIT_DRAM_BW = 2048
SRAM_SIZE_FULL_AREA = SRAM_SIZE_MB * 8 * 1024**2 / 0.8  # 80% of the area is SRAM
NB_PE_FULL_AREA = NB_PE / 0.2  # 20% of the area is PEs


BASE_MAPPING = "inputs/mapping_multicore/mamba1.yaml"
MARCA_ACCELERATOR_PATH = "inputs/multicore_system/marca.yaml"
MARCA_OFFCHIP_CORE_PATH = "inputs/cores/dram_marca.yaml"
LOG_FILE = "experiment_log.log"
base_core = "inputs/cores/marca.yaml"


LAYER_STACKS_T: TypeAlias = list[tuple[int, ...]]
LAYER_STACKS_SHORT_T: TypeAlias = list[tuple[int | Literal["SSM"], ...]]


def convert_latency_result(model: Mamba1Config, latency: int):
    """Convert the stream result [cycles] (for a single layer) to the full inference latency [ms].
    Assumes f_clk = 1GHz"""
    assert model.num_layer > 1, "Wrong model is used"
    return latency * 1e-6 * model.num_layer


def get_layer_stacks(model: Mamba1Config):
    """Fuses dA, exp, dB, dBx and SSM"""
    base_stacks = [(i,) for i in range(18)]
    ssm_stack = (18, 19, 20, 21) + get_layer_stack_SSM(model)
    return base_stacks + [ssm_stack]


def get_layer_stack_SSM(model: Mamba1Config):
    """SSM super-node has 3 splits, 1 concat, 3 compute nodes per L (A*h, h+dBx and h*C)"""
    nb_ssm_nodes = 4 + 3 * model.prefill_size
    return tuple(range(22, 22 + nb_ssm_nodes))


def generate_all_onnx_models(models: Iterator[Mamba1Config]):
    for model in models:
        model_for_simulation = model.to_single_layer_config()
        onnx_path = get_onnx_path(model_for_simulation, Stage.PREFILL, QUANT)
        if not os.path.exists(onnx_path):
            export_model_to_onnx(model_for_simulation, QUANT, path=onnx_path, stage=Stage.PREFILL)
    print("All models exported to ONNX")


def generate_mapping(
    model: Mamba1Config,
    output_mapping_file: str,
    nb_cores: int,
    intra_core_splits_L: int | None,
    intra_core_splits_N: int | None,
    intra_core_splits_D: int | None,
    base_mapping: str = BASE_MAPPING,
    inter_core_dim: str | None = None,
    mem_size_MB: float = SRAM_SIZE_MB,
):
    """Create a new mapping file that is a copy of `base mapping` with the following modifications:
    - Set core allocation to be valid with nb of cores
    - Set inter-core dimension to `(inter_core_dim, nb_cores)`
    - Set intra-core tiling for `L` to `intra_core_splits_L` for all layers.
        If not provided, it is computed, it is computed automatically, given that `intra_core_splits_N` == 1.
    - Set intra-core tiling for `N` to `intra_core_splits_N` for all layers.
        If not provided, it is computed, it is computed automatically, given that `intra_core_splits_L` == L.
    """
    assert nb_cores > 0
    os.makedirs(os.path.dirname(output_mapping_file), exist_ok=True)
    modifier = MappingModifier(base_mapping)

    # Set intra core tiling
    if intra_core_splits_L is None:
        assert (
            intra_core_splits_N == 1 and intra_core_splits_D == 1
        ), "Cannot generate tiling for both N, D and L simultaneously"
        intra_core_splits_L = get_intra_core_split_L(model, mem_size_MB=mem_size_MB)
    if intra_core_splits_N is None:
        assert (
            intra_core_splits_L == model.prefill_size and intra_core_splits_D == 1
        ), "Cannot generate tiling for both N, D and L simultaneously"
        intra_core_splits_N = get_intra_core_split_N(model, mem_size_MB=mem_size_MB)
    if intra_core_splits_D is None:
        assert (
            intra_core_splits_L == model.prefill_size and intra_core_splits_N == 1
        ), "Cannot generate tiling for both N, D and L simultaneously"
        intra_core_splits_D = get_intra_core_split_D(model, mem_size_MB=mem_size_MB)

    if model.prefill_size % (intra_core_splits_L) != 0:
        raise ValueError(f"L={model.prefill_size} not divisible by splits {intra_core_splits_L}")
    if model.d_state % (intra_core_splits_N) != 0:
        raise ValueError(f"N={model.d_state} not divisible by splits {intra_core_splits_N}")

    if intra_core_splits_N > 1:
        modifier.modify_intra_core_tiling("N", intra_core_splits_N)  # The order of the two calls matters!
    if intra_core_splits_D > 1:
        modifier.modify_intra_core_tiling("D", intra_core_splits_D)  # The order of the two calls matters!
    modifier.modify_intra_core_tiling("L", intra_core_splits_L)  # Tilings of size 1 are ignored

    # Set inter core tiling
    if inter_core_dim is not None:
        modifier.modify_inter_core_tiling(inter_core_dim, nb_cores)

    # Set proper core allocation
    core_allocation = list(range(nb_cores))
    modifier.modify_core_allocation(core_allocation)
    modifier.save(output_mapping_file)


def get_intra_core_split_L(model: Mamba1Config, mem_size_MB: float):
    """Assumes intra-core tiling (N, 1)"""
    return get_intra_core_split(
        model,
        total_size_of_split_dim=model.prefill_size,
        mem_size_MB=mem_size_MB,
        other_assumed_splits=1,
    )


def get_intra_core_split_N(model: Mamba1Config, mem_size_MB: float):
    """Assumes intra-core tiling (L, model.prefill_size)"""
    return get_intra_core_split(
        model,
        total_size_of_split_dim=model.d_state,
        mem_size_MB=mem_size_MB,
        other_assumed_splits=model.prefill_size,
    )


def get_intra_core_split_D(model: Mamba1Config, mem_size_MB: float):
    """Assumes intra-core tiling (L, model.prefill_size)"""
    return get_intra_core_split(
        model,
        total_size_of_split_dim=model.d_inner,
        mem_size_MB=mem_size_MB,
        other_assumed_splits=model.prefill_size,
    )


def get_intra_core_split(
    model: Mamba1Config, total_size_of_split_dim: int, mem_size_MB: float, other_assumed_splits: int
):
    """Compute nb_intra_core_splits such that the tiles would fit in the SRAM. Tensor size: (B, L, D, N)
    Assumes that the tensors are already split up in `other_assumed_splits`.
    e.g. when computing N split, assume an L split of prefill_size
    """
    assert other_assumed_splits > 0
    assert mem_size_MB < 1024, "Memory size should be in MB"

    def make_divisor_of(n: int, d: int):
        """Return `x >= n` such that `d % n == 0`"""
        # number of splits exceeds dimension -> return max allowed
        if n > d:
            return d
        while d % n != 0:
            n += 1
        return n

    NB_TENSORS_TO_FIT = 5 + (2 / 64)
    MARGIN = 0.01
    mem_size_bit = mem_size_MB * 8 * 1024**2

    # 2x because input and output tile should both fit
    total_data_bits = (
        NB_TENSORS_TO_FIT * model.batch_size * model.prefill_size * model.d_inner * model.d_state * QUANT.act_bits
    )
    total_data_bits = total_data_bits * (1 + MARGIN) / other_assumed_splits

    nb_intra_core_splits = ceil(total_data_bits / mem_size_bit)
    nb_intra_core_splits = make_divisor_of(nb_intra_core_splits, total_size_of_split_dim)
    # print(f"Generated intra-core tiling factor: {nb_intra_core_splits}")
    return nb_intra_core_splits


def clear_incomplete_runs(base_folder: str, exception_dirs: list[str] = []):
    if not os.path.isdir(base_folder):
        return

    subfolders = [
        os.path.join(base_folder, f) for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))
    ]
    for subfolder in subfolders:
        # Run is successful if it generated an scme
        if subfolder not in exception_dirs and not os.path.isfile(os.path.join(subfolder, "scme.pickle")):
            # Sanity check: the dump folders of incomplete runs should have this file
            assert os.path.isfile(os.path.join(subfolder, "experiment_info.txt")) or os.path.isfile(
                os.path.join(subfolder, "out.log")
            )
            print(f"Removing {subfolder}")
            shutil.rmtree(subfolder)


def generate_accelerator(
    gen_core_file: str,
    gen_accelerator_file: str,
    nb_cores: int,
    area_fraction: float = 1,
    # Option 1: modify % of area for memory and PE
    mem_area_percentage: float | None = None,
    # Option 2: modify total memory size and PE count
    mem_size_MB: float | None = None,
    # Optional: modify DRAM
    dram_bits_per_cc: int = 2048,
    gen_offchip_core_file: str | None = None,
):
    """Generate an accelerator as an adaptation of MARCA
    - the total area will be MARCA's area times `area_factor` (222mm^2)
    - the total number of PEs will be the number of PEs that would fit on the total area, times `1 - mem_area_percentage`
    - the total number of cores will be `nb_cores`, where each core has `1/nb_cores * total_pe` PEs
    - the total SRAM size will be the given size in MB (if given) OR the SRAM size that would fit on the total area,
        times `mem_area_percentage`
    """
    assert nb_cores > 0
    os.makedirs(os.path.dirname(gen_core_file), exist_ok=True)
    if mem_area_percentage and not mem_size_MB:
        total_sram_size = int(area_fraction * SRAM_SIZE_FULL_AREA * mem_area_percentage)
        total_pe = ceil(area_fraction * NB_PE_FULL_AREA * (1 - mem_area_percentage))
        memory_log_str = f"{mem_area_percentage * 100}%"
    elif mem_size_MB and not mem_area_percentage:
        total_sram_size = int(mem_size_MB * 8 * 1024**2)
        total_pe = 8192
        memory_log_str = f"{mem_size_MB} MB"
    else:
        raise ValueError("Either mem_size_MB or mem_area_percentage should be provided, but not both")

    # Synthesize one core
    core_mod = CoreModifier(base_core)
    array_sizes = _find_closest_factors(total_pe // nb_cores)
    sram_size = total_sram_size // nb_cores
    sram_bw = (32 // nb_cores) * UNIT_SRAM_BW
    print(
        f"GENERATING ACCELERATOR: {nb_cores} cores and {memory_log_str} memory => {array_sizes} PEs; {sram_size / 8 / 1024} kB SRAM"
    )
    core_mod.modify_operational_array_size(array_sizes)
    core_mod.modify_sram_size(sram_size)
    core_mod.modify_sram_bandwidth(sram_bw)
    core_mod.save_modified(gen_core_file)

    # Synthesize DRAM core
    if gen_offchip_core_file:
        core_mod_offchip = CoreModifier(MARCA_OFFCHIP_CORE_PATH)
        core_mod_offchip.modify_dram_bandwidth(dram_bits_per_cc)
        core_mod_offchip.save_modified(gen_offchip_core_file)
        offchip_core = gen_offchip_core_file
    else:
        assert dram_bits_per_cc == UNIT_DRAM_BW, "Provide offchip core path if DRAM bandwidth is modified"
        offchip_core = MARCA_OFFCHIP_CORE_PATH

    # Construct accelerator system
    accel_mod = AcceleratorModifier()
    accel_mod.construct(
        core_path=gen_core_file,
        nb_cores=nb_cores,
        offchip_path=offchip_core,
    )
    accel_mod.save(gen_accelerator_file)
    return gen_accelerator_file


def compute_mem_size_MB(mem_area_percentage: float, area_reduction: float):
    return area_reduction * SRAM_SIZE_FULL_AREA * mem_area_percentage / 8 / 1024**2


def compute_total_PE(mem_area_percentage: float, area_reduction: float):
    return int(area_reduction * NB_PE_FULL_AREA * (1 - mem_area_percentage))


def compute_dram_bandwidth(area_reduction: float):
    """DRAM bandwidth (in bit/cc) scales with the chip's circumference, i.e. the square root of the area reduction"""
    return int(UNIT_DRAM_BW * sqrt(area_reduction))


def _find_closest_factors(x: int):
    """Finds two factors a and b of x such that a * b = x and |a - b| is minimized."""
    PERCENTAGE_OFF = 0.05

    def _is_power_of_two(n: int) -> bool:
        return (n & (n - 1)) == 0

    def _get_nice_number(n: int):
        for offset in range(1, ceil(PERCENTAGE_OFF * x)):
            for sign in [-1, 1]:
                if _is_power_of_two(n + sign * offset):
                    return n + sign * offset
        return n

    # If a nearby number if a power of 2, take that one
    x = _get_nice_number(x)

    # Start from the square root of x and search downwards
    sqrt_x = int(sqrt(x))
    for a in range(sqrt_x, 0, -1):
        if x % a == 0:
            b = x // a
            # a will always be smaller than b
            assert a <= b, f"Problem with finding factors for array sizes: {a} <= {b}"
            if b / a < 3:
                return [a, b]

    return [sqrt_x, sqrt_x]


def run_experiments(fn: Callable[..., None], configs: Iterator[Any], do_shuffle: bool = True):
    configurations = list(configs)
    if do_shuffle:
        shuffle(configurations)

    for config in configurations:
        fn(config)


def run_experiments_parallel(
    fn: Callable[..., None], configs: Iterator[Any], n_jobs: int, do_shuffle: bool = True
) -> None:
    configurations = list(configs)
    if do_shuffle:
        shuffle(configurations)

    # Parallel(n_jobs=N_JOBS, backend="loky")(delayed(run_and_capture_output)(config) for config in configurations)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(fn, config) for config in configurations]
        for future, config in zip(concurrent.futures.as_completed(futures), configurations):
            try:
                future.result()
            except Exception as e:
                print(f"Exception for {config}: {e}")


def stack_to_full_str(stack: LAYER_STACKS_SHORT_T) -> str:
    match stack:
        case []:
            return "Unfused"
        case [(18,), (19,), (20,), (21,), ("SSM",)]:
            return "Local state"
        case [(18, 19)]:
            return "dA-Exp"
        case [(18, 19), (20,), (21,), ("SSM",)]:
            return "dA-Exp\n+ Local state"
        case [(18,), (19,), (20, 21)]:
            return "dB-dBx"
        case [(18, 19), (20, 21)]:
            return "dA-Exp\n+ dB-dBx"
        case [(20,), (21,), (18, 19, "SSM")]:
            return "dA-Exp-State"
        case [(18,), (19,), (20, 21, "SSM")]:
            return "dB-dBx-State"
        case [(18,), (20,), (19, 21, "SSM")]:
            return "Exp-dBx-State"
        case [(18, 19, "SSM"), (20, 21)]:
            return "dA-Exp-State\n+ dB-dBx"
        case [(18, 19), (20, 21, "SSM")]:
            return "dA-Exp +\ndB-dBx-State"
        case [(18, 19, 20, 21, "SSM")]:
            return "Full SSM"
        case _:
            return str(stack)


def stack_to_short_str(stack: LAYER_STACKS_SHORT_T) -> str:
    match stack:
        case []:
            return "UF"
        case [(18,), (19,), (20,), (21,), ("SSM",)]:
            return "Local state"
        case [(18, 19)]:
            return "A"
        case [(18, 19), (20,), (21,), ("SSM",)]:
            return "A + Local state"
        case [(18,), (19,), (20, 21)]:
            return "B"
        case [(18, 19), (20, 21)]:
            return "A-B"
        case [(20,), (21,), (18, 19, "SSM")]:
            return "AS"
        case [(18,), (19,), (20, 21, "SSM")]:
            return "BS"
        case [(18,), (20,), (19, 21, "SSM")]:
            return "ABS"  # TODO find better name, this is only half of A and half of B
        case [(18, 19, "SSM"), (20, 21)]:
            return "AS-B"
        case [(18, 19), (20, 21, "SSM")]:
            return "A-BS"
        case [(18, 19, 20, 21, "SSM")]:
            return "All"
        case _:
            return str(stack)
