import copy
import os
from enum import StrEnum
from typing import Any

from ruamel.yaml import YAML

from src.util import get_accelerator_name_and_path


class ModifiableParams(StrEnum):
    SRAM_BW = "sram_bw"
    SRAM_SIZE = "sram_size"
    DRAM_BW = "dram_bw"
    ARRAY_SIZE = "array_size"


class CoreModifier:
    def __init__(self, core_name_or_path: str):
        _, path = get_accelerator_name_and_path(core_name_or_path)

        self.yaml = YAML()
        with open(path, "r") as file:
            self.data: dict[str, Any] = self.yaml.load(file)  # type: ignore

    def get_value(self, parameter: ModifiableParams):
        match parameter:
            case ModifiableParams.SRAM_BW:
                return self.data["memories"]["sram"]["r_bw"]
            case ModifiableParams.SRAM_SIZE:
                return self.data["memories"]["sram"]["size"]
            case ModifiableParams.DRAM_BW:
                return self.data["memories"]["dram"]["r_bw"]
            case ModifiableParams.ARRAY_SIZE:
                return self.data["operational_array"]["sizes"][0]

    def modify_parameter(self, parameter: ModifiableParams, value: int | float):
        match parameter:
            case ModifiableParams.SRAM_BW:
                self.modify_sram_bandwidth(int(value))
            case ModifiableParams.SRAM_SIZE:
                self.modify_sram_size(int(value))
            case ModifiableParams.DRAM_BW:
                self.modify_dram_bandwidth(int(value))
            case ModifiableParams.ARRAY_SIZE:
                self.modify_operational_array_size(2 * [int(value)])

    def modify_name(self, name: str):
        self.data["name"] = name

    def modify_register_size(self, name: str, size: int):
        """Modify properties of register with given name.
        Size, r/w bandwidth are set to given size, r/w cost are linearly scaled."""
        reg_data = self.data["memories"][name]
        r_cost_per_bit = reg_data["r_cost"] / reg_data["r_bw"]
        w_cost_per_bit = reg_data["w_cost"] / reg_data["r_bw"]
        reg_data["size"] = size
        reg_data["r_bw"] = size
        reg_data["w_bw"] = size
        reg_data["r_cost"] = r_cost_per_bit * size
        reg_data["w_cost"] = w_cost_per_bit * size

    def modify_all_register_sizes(self, size: int):
        # Hardcoded
        reg_names = ["rf_I", "rf_W", "rf_O"]
        for reg_name in reg_names:
            self.modify_register_size(reg_name, size)

    def modify_sram_bandwidth(self, bandwidth: int):
        sram_data = self.data["memories"]["sram"]
        r_cost_per_bit: float = sram_data["r_cost"] / sram_data["r_bw"]
        w_cost_per_bit: float = sram_data["w_cost"] / sram_data["r_bw"]
        sram_data["r_bw"] = bandwidth
        sram_data["w_bw"] = bandwidth
        sram_data["r_cost"] = r_cost_per_bit * bandwidth
        sram_data["w_cost"] = w_cost_per_bit * bandwidth

    def modify_sram_size(self, size: int):
        sram_data = self.data["memories"]["sram"]
        sram_data["size"] = size

    def modify_dram_bandwidth(self, bandwidth: int):
        dram_data = self.data["memories"]["dram"]
        r_cost_per_bit: float = dram_data["r_cost"] / dram_data["r_bw"]
        w_cost_per_bit: float = dram_data["w_cost"] / dram_data["r_bw"]
        dram_data["r_bw"] = bandwidth
        dram_data["w_bw"] = bandwidth
        dram_data["r_cost"] = r_cost_per_bit * bandwidth
        dram_data["w_cost"] = w_cost_per_bit * bandwidth

    def modify_operational_array_size(self, sizes: list[int]):
        dimensions = [f"D{i}" for i in range(1, len(sizes) + 1)]
        oa_data = self.data["operational_array"]
        oa_data["sizes"] = sizes
        oa_data["dimensions"] = [f"D{i}" for i in range(1, len(sizes) + 1)]

        # Alter served dimensions for broadcasting memories
        for mem_name in self.data["memories"]:
            if len(self.data["memories"][mem_name]["served_dimensions"]) > 0:
                self.data["memories"][mem_name]["served_dimensions"] = copy.copy(dimensions)

    def modify_mac_energy(self, value: float):
        self.data["operational_array"]["unit_energy"] = value

    def save_modified(self, filename: str):
        assert filename.endswith("yaml")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as file:
            self.yaml.dump(self.data, file)  # type: ignore


class AcceleratorModifier:

    def __init__(self):
        self.data: dict[str, Any] = {}
        self.yaml = YAML()

    def construct(
        self,
        core_path: str,
        nb_cores: int,
        offchip_path: str,
        link_bandwidth: int = 99999999999999,
        link_energy_cost: int = 0,
    ):
        self.data["name"] = f"GEN_{nb_cores}"
        self.data["bandwidth"] = link_bandwidth
        self.data["unit_energy_cost"] = link_energy_cost
        self.data["offchip_core"] = offchip_path
        self.data["cores"] = {i: core_path for i in range(nb_cores)}
        self.data["core_connectivity"] = [", ".join(str(i) for i in range(nb_cores))] if nb_cores > 1 else []
        self.data["core_memory_sharing"] = []

    def save(self, filename: str):
        assert filename.endswith("yaml")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as file:
            self.yaml.dump(self.data, file)  # type: ignore


class MappingModifier:

    def __init__(self, base_mapping_path: str):
        self.yaml = YAML()
        with open(base_mapping_path, "r") as file:
            self.data: list[dict[str, Any]] = self.yaml.load(file)  # type: ignore

    def modify_core_allocation(self, allocation: list[int]):
        for i, _ in enumerate(self.data):
            self.data[i]["core_allocation"] = allocation

    def modify_intra_core_tiling(self, layer_dim: str, split_factor: int):
        """Modify all intra-core tiling factors greater than 1 to split_factor."""
        for layer_idx, _ in enumerate(self.data):
            intra_core_tiling: list[str] = self.data[layer_idx]["intra_core_tiling"]

            # Only replace for nodes that already have a >1 intra-core tiling
            if any(int(pair.split(" ")[1]) > 1 for pair in intra_core_tiling):
                new_tiling_str = f"{layer_dim}, {split_factor}"

                # If the layer dim already existed, replace it
                try:
                    idx = next(i for i, pair in enumerate(intra_core_tiling) if pair.startswith(layer_dim))
                    intra_core_tiling[idx] = new_tiling_str
                except StopIteration:
                    intra_core_tiling.append(new_tiling_str)

    def modify_inter_core_tiling(
        self, dim: str, split_factor: int, do_not_modify_dim: list[str] = ["proj", "conv", "norm"]
    ):
        """Modify all inter-core tilings to (dim, split_factor), except layers in do_not_modify.
        # TODO Support splits in multiple dimensions
        """
        for layer_idx, _ in enumerate(self.data):
            layer_name = self.data[layer_idx]["name"]
            if any(name.lower() in layer_name.lower() for name in do_not_modify_dim):
                keep_dim = self.data[layer_idx]["inter_core_tiling"][0].split(",")[0].strip()
                tiling_str = f"{keep_dim}, {split_factor}"
            else:
                # Assume only one pair is given
                tiling_str = f"{dim}, {split_factor}"
            self.data[layer_idx]["inter_core_tiling"] = [tiling_str]

    def save(self, filename: str):
        assert filename.endswith("yaml")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as file:
            self.yaml.dump(self.data, file)  # type: ignore
