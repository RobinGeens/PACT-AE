import pandas as pd


def convert_to_ms(latency: float, frequency: int) -> float:
    return latency / frequency * 1e3


def convert_df_to_ms(df: pd.DataFrame, all_op_types: list, frequency: int) -> pd.DataFrame:
    """Convert the latency values to milliseconds"""
    df_ms = df.copy()
    for op_type in all_op_types:
        df_ms[op_type] = df_ms[op_type].apply(lambda latency: convert_to_ms(latency, frequency))
    return df_ms


def get_roofline_performance(ai, peak_ops_per_cycle, peak_memory_bandwidth):
    crossover_ai = peak_ops_per_cycle / peak_memory_bandwidth
    if ai >= crossover_ai:
        # compute bound
        performance = peak_ops_per_cycle
    else:
        # memory bound
        performance = ai * peak_memory_bandwidth
        # performance = PEAK_OPS_CYCLE  # for infinite bandwidth simulation
    return performance


def get_roofline_latency(ops, ai, peak_ops_per_cycle, peak_memory_bandwidth):
    performance = get_roofline_performance(ai, peak_ops_per_cycle, peak_memory_bandwidth)
    latency = ops / performance
    return latency
