
# Figure 1 (ops_and_mem_accesses.png)
# Requires: simple_latency_analysis.py
python scripts/op_distribution.py 

# Figure 4 (right) (latency.png)
# Generated with simple_latency_analysis.py

# Figure 4 (left and middle) (roofline.png, op_distribution.png)
# Requires: simple_latency_analysis.py
python scripts/roofline_per_op_type.py

# Figure 9 (layer_stacks_sweep.png)
# Requires: sweep_layer_stacks.py
python scripts/results_layer_stacks.py

# Figure 10 (tensor_lifetimes.png)
python scripts/plot_fusion_memory_usage.py

# Figure 11 (memory_sweep_merged.png)
# Requires: sweep_mem_split_D.py, sweep_mem_no_split.py
python scripts/results_memory_merged.py

# Figure 12 (contour_plot_L=1.png, contour_plot_L=64.png, contour_plot_L=1024.png)
# Requires: sweep_contour.py
python scripts/results_contour.py


