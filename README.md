This repository provides the complete codebase and inputs used to run all experiments in [Fine-Grained Fusion: The Missing Piece in Area-Efficient State Space Model Acceleration
](https://arxiv.org/abs/2504.17333). It is currently under evaluation as ACM Artifact.

### How to run
1. Run `source setup.sh`
2. Run `./run_experiments.sh` as a whole, or each command separately
    - `experiment_log.log` gives overview of started and finished experiments
    - Terminal output logs per experiment can be found in outputs
    - To drastically reduce the runtime, the evaluation space can be reduced. In the `sweep_*.py` scripts, change the `seq_lengths` variable to exclude `1024` and above.
3. Run `./generate_figures.sh` as a whole, or each command separately
    - The `results_*.py` files can be run continuously to generate partial figures and print progress

### Expected results
The main results are the generated figures, which should look identical to those in the paper.
- Figure 1: `ops_and_mem_accesses.png`
- FIgure 4: (`roofline.png`, `op_distribution.png`, `latency.png`)
- Figure 9: `layer_stacks_sweep.png`
- FIgure 10: `tensor_lifetimes.png`
- Figure 11: `memory_sweep_merged.png`
- FIgure 12: (`contour_plot_L=1.png`, `contour_plot_L=64.png`, `contour_plot_L=1024.png`)

### Info

Python version: 3.11.9

Corresponding [Stream](https://github.com/KULeuven-MICAS/stream.git) commit: `ae6a778894a2e1d4237e4f7892d46b14ad28bf49`


