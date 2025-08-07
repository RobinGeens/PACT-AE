import math
from abc import ABCMeta, abstractmethod
from math import log
from typing import Literal

BATCH_SIZE = 1


class ModelConfig(metaclass=ABCMeta):

    num_layer: int
    name: str

    @abstractmethod
    def to_single_layer_config(self) -> "ModelConfig": ...

    @property
    def prefill_size(self) -> int: ...

    @property
    def decode_size(self) -> int: ...

    @property
    def parameterized_name(self) -> str: ...

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class TransformerConfig(ModelConfig):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        dim_ff: int,
        num_head: int,
        num_layer: int,
        batch_size: int = 1,
        vocab_size: int = 1000,
        name: str = "",
        # Automatically calculated
        head_size: int | None = None,
        prefill_size: int | None = None,
        decode_size: int | None = None,
    ):
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.dim_ff = dim_ff
        self.num_head = num_head
        self.num_layer = num_layer
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.name = name

        # Defaults
        self.head_size = head_size if head_size is not None else embedding_dim // num_head
        # Simulate prefill with half of the context window.
        self.__prefill_size = prefill_size if prefill_size is not None else seq_len // 2
        self.__decode_size = decode_size if decode_size is not None else seq_len // 2
        self.__decode_idx = self.__compute_decode_idx()

    def __compute_decode_idx(self):
        """Take the token halfway the decode sequence, as multiple of 2"""
        decode_idx = self.__prefill_size + self.__decode_size // 2
        rounded_to_two = 1 << int(log(decode_idx, 2))
        return rounded_to_two

    @property
    def prefill_size(self):
        return self.__prefill_size

    @prefill_size.setter
    def prefill_size(self, value: int):
        self.__prefill_size = value
        self.__decode_idx = self.__compute_decode_idx()

    @property
    def decode_size(self):
        return self.__decode_size

    @decode_size.setter
    def decode_size(self, value: int):
        self.__decode_size = value
        self.__decode_idx = self.__compute_decode_idx()

    @property
    def decode_idx(self):
        """To simulate the model in decode phase, only a single run (for a single) token is executed"""
        return self.__decode_idx

    @decode_idx.setter
    def decode_idx(self, value: int):
        """Manually override the to-be simulated token in the decode sequence"""
        self.__decode_idx = value

    @property
    def parameterized_name(self):
        return f"{self.name.replace('.', '_')}_B={self.batch_size}_FULL"

    @property
    def has_gate_layer(self):
        return "opt" in self.name.lower() or "llama" in self.name.lower()

    def to_single_layer_config(self):
        """Return a new TransformerConfig instance with only a single laker to make the simulation go faster. The results
        can then be multiplied to get the actual energy/latency values"""
        return TransformerConfigSingeLayer(self)


class TransformerConfigSingeLayer(TransformerConfig):
    """Configuration with only a single layer and a all heads"""

    def __init__(self, full_config: TransformerConfig):
        assert full_config.num_layer > 1
        super().__init__(
            num_layer=1,
            seq_len=full_config.seq_len,
            embedding_dim=full_config.embedding_dim,
            dim_ff=full_config.dim_ff,
            num_head=full_config.num_head,
            batch_size=full_config.batch_size,
            vocab_size=full_config.vocab_size,
            name=full_config.name,
            head_size=full_config.head_size,
            prefill_size=full_config.prefill_size,
            decode_size=full_config.decode_size,
        )
        self.num_layer_full = full_config.num_layer

    def to_single_layer_config(self):
        raise Exception("This already is a single layer configuration")

    @property
    def parameterized_name(self):
        return super().parameterized_name.replace("FULL", "SINGLELAYER")

    def get_post_simulation_multiplier(self, layer_name: str, amortize_within_batch: bool = True) -> float:
        """The model is simulated with reduced parameters i.e. only one layer. This function returns the factor with
        which the results for the given layer have to be multiplied in order to come to the result for the full model
        Moreover, the results are normalized to a single inference instead of a full batch
        @param amortize_within_batch if true, return the results for a single"""
        batch_factor = 1 / self.batch_size if amortize_within_batch else 1

        def name_contains(x: list[str]):
            return any([v in layer_name for v in x])

        # Special case: gate layer in Llama models
        if name_contains(["key_proj"]):
            return 4 * self.num_layer_full * batch_factor
        if name_contains(["query_proj", "value_proj", "out_proj"]):
            raise ValueError(
                "Only `key_proj` should be passed as argument, others are included in the factor of `key_proj`"
            )
        if name_contains(["up_proj"]) and self.has_gate_layer:
            return 2 * self.num_layer_full * batch_factor
        # For pre- and post-processing
        if name_contains(["embed", "final"]):
            return 1

        return self.num_layer_full * batch_factor


class QuantConfig:
    def __init__(self, weight_bits: int, act_bits: int, output_bits: int | None = None):
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.intermediate_output_bits = output_bits if output_bits is not None else 2 * act_bits

    @property
    def name(self):
        return f"W{self.weight_bits}A{self.act_bits}"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class MambaConfig(ModelConfig, metaclass=ABCMeta):
    def __init__(
        self,
        # Model specific
        name: str,
        d_model: int,  # D
        n_head: int = 1,
        num_layer: int = 1,
        d_state: int = 64,  # N in paper/comments
        expand_factor: int = 2,  # E in paper/comments
        d_conv: int = 4,
        n_groups: int = 1,
        vocab_size: int = 52000,
        # Inference params
        prefill_size: int = 1024,
        decode_size: int = 1024,
        batch_size: int = BATCH_SIZE,
        # To clean up
        do_rms_norm: bool = True,
        activation: Literal["swish", "silu"] = "silu",
        do_out_proj_bias: bool = False,
        do_in_proj_bias: bool = False,
        do_conv_bias: bool = True,
        use_mup: bool = True,
    ):

        self.name = name
        self.expand_factor = expand_factor
        self.n_head = n_head
        self.d_state = d_state
        self.d_conv = d_conv
        self.n_groups = n_groups
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Inference
        self.prefill_size = prefill_size
        self.decode_size = decode_size
        self.batch_size = batch_size

        # Configure
        self.do_out_proj_bias = do_out_proj_bias
        self.do_rms_norm = do_rms_norm
        self.activation = activation
        self.do_in_proj_bias = do_in_proj_bias
        self.do_conv_bias = do_conv_bias
        self.use_mup = use_mup

        self.d_head = self.d_inner // self.n_head
        assert self.d_inner % self.d_head == 0

    @property
    def d_model(self):
        return self.__d_model

    @d_model.setter
    def d_model(self, value: int):
        self.__d_model = value
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED
        self.dt_rank = math.ceil(self.d_model / 16)

    @property
    def num_layer(self):
        if isinstance(self, MambaConfigSingleLayer):
            return 1

        # Infer num_layer from d_model
        match self.d_model:
            case 2560:
                return 64
            case 2048:
                return 48
            case 1536:
                return 48
            case 1024:
                return 48
            case 768:
                return 24
            case _:
                raise ValueError(f"Unknown d_model: {self.d_model}")

    @property
    def prefill_size(self):
        return self.__prefill_size

    @prefill_size.setter
    def prefill_size(self, value: int):
        """Prefill size must always be odd to ensure that conv1D has same output shape as input shape"""
        self.__prefill_size = value

    @property
    def decode_size(self):
        return self.__decode_size

    @decode_size.setter
    def decode_size(self, value: int):
        self.__decode_size = value

    @property
    def decode_idx(self):
        """To simulate the model in decode phase, only a single run (for a single) token is executed"""
        return self.__decode_idx

    @decode_idx.setter
    def decode_idx(self, value: int):
        """Manually override the to-be simulated token in the decode sequence"""
        self.__decode_idx = value

    @property
    def parameterized_name(self):
        return f"{self.name.replace('.', '_')}_B={self.batch_size}_D={self.d_model}_FULL"

    def to_single_layer_config(self) -> "MambaConfigSingleLayer":
        """Return a new TransformerConfig instance with only a single laker to make the simulation go faster. The results
        can then be multiplied to get the actual energy/latency values"""
        assert self.num_layer > 1
        return MambaConfigSingleLayer(self)


class MambaConfigSingleLayer(MambaConfig, metaclass=ABCMeta):
    def __init__(self, full_config: MambaConfig):
        assert full_config.num_layer > 1

        super().__init__(
            name=full_config.name,
            d_model=full_config.d_model,
            num_layer=1,
            n_head=full_config.n_head,
            d_state=full_config.d_state,
            expand_factor=full_config.expand_factor,
            d_conv=full_config.d_conv,
            n_groups=full_config.n_groups,
            vocab_size=full_config.vocab_size,
            prefill_size=full_config.prefill_size,
            decode_size=full_config.decode_size,
            batch_size=full_config.batch_size,
            do_out_proj_bias=full_config.do_out_proj_bias,
            do_rms_norm=full_config.do_rms_norm,
            activation=full_config.activation,
            do_in_proj_bias=full_config.do_in_proj_bias,
            do_conv_bias=full_config.do_conv_bias,
            use_mup=full_config.use_mup,
        )
        self.num_layer_full = full_config.num_layer

    def to_single_layer_config(self):
        raise Exception("This already is a single layer configuration")

    @property
    def parameterized_name(self):
        return super().parameterized_name.replace("FULL", "SL")  # SL = single layer


class Mamba1Config(MambaConfig):
    """"""

    def to_single_layer_config(self) -> "Mamba1ConfigSingleLayer":
        assert self.num_layer > 1
        return Mamba1ConfigSingleLayer(self)


class Mamba2Config(MambaConfig):
    """"""

    def to_single_layer_config(self) -> "Mamba2ConfigSingleLayer":
        assert self.num_layer > 1
        return Mamba2ConfigSingleLayer(self)


class Mamba1ConfigSingleLayer(MambaConfigSingleLayer, Mamba1Config):
    """"""


class Mamba2ConfigSingleLayer(MambaConfigSingleLayer, Mamba2Config):
    """"""
