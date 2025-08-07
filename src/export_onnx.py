import os
import sys
from typing import Any

import onnx
import torch
from onnx import NodeProto
from onnx.shape_inference import infer_shapes
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import _get_tensor_sizes

sys.path.append(os.getcwd())

from src.config import (
    Mamba1ConfigSingleLayer,
    MambaConfigSingleLayer,
    ModelConfig,
    QuantConfig,
    TransformerConfigSingeLayer,
)
from src.config_library import MAMBA1_2_8B, W32A32
from src.mamba_model import Mamba1Block
from src.transformer_model import LanguageModel
from src.transformer_model_decode import LanguageModelDecode
from src.util import Stage, get_onnx_path


def export_model_to_onnx(
    config: ModelConfig,
    quant_config: QuantConfig,
    path: str = "outputs/custom_transformer.onnx",
    stage: Stage = Stage.PREFILL,
):

    match config:
        case TransformerConfigSingeLayer():
            export_transformer_to_onnx(
                config,
                path,
                stage,
            )
        case MambaConfigSingleLayer():
            export_ssm_to_onnx(
                config,
                path,
                stage,
            )
        case _:
            raise ValueError("config must be a single layer configuration")

    # Perform shape inference
    onnx_model = onnx.load(path)
    onnx_model = infer_shapes(onnx_model)

    # Add attribute with quantization info, to be used in Zigzag
    for node in onnx_model.graph.node:
        if node.op_type != "Constant":
            add_attribute_to_onnx_node(node, "weight_size", quant_config.weight_bits)
            add_attribute_to_onnx_node(node, "act_size", quant_config.act_bits)
            add_attribute_to_onnx_node(node, "output_size", quant_config.intermediate_output_bits)

    # Save the model with external data and then remove it
    # NOTE: This requires later loading it with load_external_data=False
    external_data_filename = "external.data"
    external_data_path = os.path.join(os.path.dirname(path), external_data_filename)
    onnx.save(onnx_model, path, save_as_external_data=True, location=external_data_filename)
    if os.path.exists(external_data_path):
        os.remove(external_data_path)


def export_transformer_to_onnx(
    transformer_config: TransformerConfigSingeLayer,
    path: str = "outputs/custom_transformer.onnx",
    stage: Stage = Stage.PREFILL,
):
    print(f"Generating ONNX model at {path} ({stage})")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if stage == Stage.PREFILL:
        model = LanguageModel(transformer_config)
        dummy_input = torch.randint(
            low=0, high=255, size=(transformer_config.batch_size, transformer_config.prefill_size)
        )
    else:
        model = LanguageModelDecode(transformer_config)
        dummy_input = torch.randint(low=0, high=255, size=(transformer_config.batch_size, 1))  # Single token

    torch.onnx.export(  # type: ignore
        model,
        dummy_input,
        path,
        export_params=False,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
    )


def register_custom_ssm1_op(model: Mamba1Block, dummy_input: torch.Tensor):
    def custom_ssm1(g, dA, dBx, C, h):
        # Shape logic for correct shape inference
        # For some reason the shapes of dA and dBx inputs are not inferred correctly so we get it from the dummy input
        B, L = dummy_input.shape[:2]
        ED = model.config.d_inner
        h_shape = _get_tensor_sizes(h)
        assert B and L and ED, "Batch size, sequence length and embedding dimension must be known for shape inference"
        assert all(h_shape), "Hidden state shape must be known for shape inference"
        y_type = dBx.type().with_sizes((B, L, ED))
        h_out_type = h.type().with_sizes(h_shape)
        # Call op on graph with correct inputs and number of outputs
        y, h_out = g.op("mylibrary::SSM", dA, dBx, C, h, outputs=2)
        y.setType(y_type)
        h_out.setType(h_out_type)
        return y, h_out

    register_custom_op_symbolic("mylibrary::ssm1_op", custom_ssm1, 1)


def export_ssm_to_onnx(
    ssm_config: MambaConfigSingleLayer,
    path: str = "outputs/custom_ssm.onnx",
    stage: Stage = Stage.PREFILL,
):
    print(f"Generating {ssm_config.name} ONNX model at {path} ({stage})")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    match stage:
        case Stage.PREFILL:
            dummy_input = torch.empty((ssm_config.batch_size, ssm_config.prefill_size, ssm_config.d_model))
        case Stage.DECODE:
            dummy_input = torch.empty((ssm_config.batch_size, 1, ssm_config.d_model))

    match ssm_config:
        case Mamba1ConfigSingleLayer():
            model = Mamba1Block(ssm_config, stage)
            register_custom_ssm1_op(model, dummy_input)
        case _:
            raise NotImplementedError

    torch.onnx.export(
        model,
        dummy_input,
        f=path,
        export_params=False,
        custom_opsets={"mylibrary": 1},
    )


def add_attribute_to_onnx_node(node: NodeProto, key: str, val: Any):
    attr = onnx.helper.make_attribute(key, val)
    node.attribute.extend([attr])


if __name__ == "__main__":
    config = MAMBA1_2_8B
    config.batch_size = 1
    config.prefill_size = 1
    quant_config = W32A32
    stage = Stage.PREFILL
    config = config.to_single_layer_config()

    path = get_onnx_path(config, stage, quant_config)
    export_model_to_onnx(
        config,
        quant_config,
        stage=stage,
        path=path,
    )
