# RL Experiments: SAC and Pre-collected Data (RLPD)

This repository contains experiments using **Soft Actor-Critic (SAC)** and **pre-collected datasets for reinforcement learning (RLPD)**. We evaluate multiple environments, hyperparameters, seeds, and buffer sizes to study the effects of pre-filled replay buffers and training variations.

---

## Environments

- Hopper-v4
- Ant-v5
- Optional: CartPole-v1 for demo or quick tests

---

## Repository Structure

- train_baseline_sac.py – Train SAC from scratch
- train_with_prior_data.py – Train SAC with pre-collected buffer data
- collect_from_checkpoint.py – Collect rollout dataset from a trained checkpoint
- sac_utils.py – SAC agent and replay buffer implementation
- evaluate_sweep.py – Evaluate multiple checkpoints over seeds
- plot_results.py – Plot learning curves, buffer & batch comparisons
- plot_seed_variance.py – Plot seed variation / variance
- generate_result_table.py – Generate CSV table with mean/std rewards
- auto_run.sh – Run full experiment sweep automatically
- baseline_rewards_*.csv – Episode reward logs
- checkpoint_final_*.pt – Saved model checkpoints
- evaluation_results.csv – Checkpoint evaluation metrics
- plots/ – Generated plots (PNG)

---

## Dependencies

Required Python packages:

- torch
- gymnasium
- numpy
- pandas
- matplotlib

---

## How to Run Experiments

### 1. Train SAC from scratch

Specify environment, total training steps, checkpoints to save, seed, buffer size, and batch size.

### 2. Collect rollouts from a checkpoint

Use a trained checkpoint to generate pre-collected rollout datasets for replay buffer training.

### 3. Train with pre-collected data

Train SAC using datasets collected from previous checkpoints. Specify dataset path, total steps, buffer capacity, batch size, and seed.

### 4. Evaluate multiple checkpoints

Evaluate trained checkpoints across multiple seeds to compute mean and standard deviation of rewards. This generates `evaluation_results.csv`.

### 5. Generate Plots

- Learning curves, buffer & batch comparisons
- Seed variance comparison
- Evaluation tables (mean ± std rewards)

---

## Outputs

**Plots (inside `plots/` folder):**

- hopper_buffer_comparison.png
- hopper_batch_comparison.png
- hopper_seed_variation.png
- ant_buffer_comparison.png
- ant_batch_comparison.png
- ant_seed_variation.png
- combined_seed_variance.png
- ant_vs_hopper_eval.png
- eval_bar_summary.png

**CSV files:**

- baseline_rewards_*.csv
- evaluation_results.csv
- evaluation_table.csv

**Checkpoints:**

- checkpoint_final_*.pt

---

## Notes / Reproducibility

- Set the seed to reproduce experiments.
- Buffer sizes and batch sizes can be varied.
- Use the automation script for batch execution of multiple seeds/environments.
