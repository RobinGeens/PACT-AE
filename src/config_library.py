from src.config import BATCH_SIZE, Mamba1Config, Mamba2Config, QuantConfig, TransformerConfig

W1A8 = QuantConfig(1, 8, 16)
W4A8 = QuantConfig(4, 8, 16)
W8A8 = QuantConfig(8, 8, 16)
W4A16 = QuantConfig(4, 16, 16)
W16A16 = QuantConfig(16, 16, 16)
W1A32 = QuantConfig(1, 32, 32)
W16A32 = QuantConfig(16, 32, 32)
W32A32 = QuantConfig(32, 32, 32)

TEST_MODEL = TransformerConfig(
    batch_size=2,
    seq_len=32,
    embedding_dim=32,
    dim_ff=128,
    num_head=4,
    num_layer=3,
    vocab_size=10000,
    name="Test-model",
)

LLAMA1_7B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=4096,
    dim_ff=11_008,
    num_head=32,
    num_layer=32,
    vocab_size=32_000,
    name="Llama1-7B",
)

LLAMA1_13B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=5120,
    dim_ff=13_824,
    num_head=40,
    num_layer=40,
    vocab_size=32_000,
    name="Llama1-13B",
)

LLAMA1_30B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=6_656,
    dim_ff=17_920,
    num_head=52,
    num_layer=60,
    vocab_size=32_000,
    name="Llama1-30B",
)

LLAMA2_7B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=4096,
    embedding_dim=4096,
    dim_ff=11_008,
    num_head=32,
    num_layer=32,
    vocab_size=32_000,
    name="Llama2-7B",
)

LLAMA2_13B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=4096,
    embedding_dim=5120,
    dim_ff=13_824,
    num_head=40,
    num_layer=40,
    vocab_size=32_000,
    name="Llama2-13B",
)


OPT_125M = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=768,
    dim_ff=3072,
    num_head=12,
    num_layer=12,
    vocab_size=50_272,
    name="OPT-125M",
)


OPT_1_3B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=2048,
    dim_ff=8_192,
    num_head=32,
    num_layer=24,
    vocab_size=50_272,
    name="OPT-1.3B",
)

OPT_2_7B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=2560,
    dim_ff=10240,
    num_head=32,
    num_layer=32,
    vocab_size=50_272,
    name="OPT-2.7B",
)

OPT_6_7B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=4096,
    dim_ff=16384,
    num_head=32,
    num_layer=32,
    vocab_size=50_272,
    name="OPT-6.7B",
)

OPT_13B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=5120,
    dim_ff=20480,
    num_head=40,
    num_layer=40,
    vocab_size=50_272,
    name="OPT-13B",
)

OPT_30B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=7_168,
    dim_ff=28_672,
    num_head=56,
    num_layer=48,
    vocab_size=50_272,
    name="OPT-30B",
)

GPT3_175B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=12_288,
    dim_ff=49_152,
    num_head=96,
    num_layer=96,
    vocab_size=50257,
    name="GPT3-175B",
)


LLAMA3_8B = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=4096,
    dim_ff=14336,
    num_head=32,
    num_layer=32,
    vocab_size=128_256,
    name="LLAMA3-8B",
)


# -------- State Space Models --------
MAMBA2_2_7B = Mamba2Config(
    d_model=2560,  # D
    n_head=32,  # H
    d_state=64,  # N
    expand_factor=2,  # E
    d_conv=4,
    n_groups=1,
    vocab_size=50277,
    do_out_proj_bias=False,
    name="Mamba2-2.7B",
)


MAMBA1_2_8B = Mamba1Config(
    d_model=2560,  # D
    d_state=64,  # N
    expand_factor=2,  # E
    d_conv=4,
    vocab_size=50277,
    do_out_proj_bias=False,
    name="Mamba1-2.7B",
)
