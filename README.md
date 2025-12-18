# RL Experiments: SAC and Pre-collected Data (RLPD)

This repository contains experiments using **Soft Actor-Critic (SAC)** combined with
**Reinforcement Learning with Pre-Collected Datasets (RLPD)**.
We study how incorporating offline experience into the replay buffer affects learning
stability, sample efficiency, and sensitivity to hyperparameters across continuous
control environments.

---

## Objectives

- Evaluate whether pre-collected data improves early-stage SAC training
- Analyze sensitivity to replay buffer size, batch size, and random seeds
- Compare learning stability across environments of increasing complexity

---

## Environments

- **Hopper-v4**: Fast and stable single-legged hopping task
- **Ant-v5**: High-dimensional quadruped locomotion task
- **Walker2D**: Two-legged walking agent (placeholder)
- **AntMaze**: Navigation task requiring exploration and planning (placeholder)

---

## Repository Structure

- checkpoints/ – Saved SAC model checkpoints
- data/ – Collected datasets and CSV logs
- figures/ – Generated plots (PNG)
- antmaze graphs/ – AntMaze-specific plots
- rlpd/ – RLPD-related utilities
- auto_run.sh – Automated experiment sweep
- train_baseline_sac.py – Train SAC from scratch
- train_sac.py – SAC training with prefill options
- train_with_prior_data.py – Train SAC with pre-collected data
- train_rlpd_from_dataset.py – RLPD training from offline datasets
- collect_from_checkpoint.py – Collect rollouts from checkpoints
- evaluate_sweep.py – Multi-seed evaluation
- generate_result_table.py – Mean/std reward tables
- plot_results.py – Learning curve and ablation plots
- plot_results_ant.py – Ant-specific plots
- plot_seed_variance.py – Seed variance visualization
- sac_utils.py – SAC agent and replay buffer
- README.md – Project overview and instruction

---

## Dependencies

Main Python dependencies:

- torch
- gymnasium (0.21.0 for Antmaze)
- numpy
- pandas
- matplotlib
- d4rl
- mujoco
- jax flax optax
- wandb

Dependencies can be installed manually using pip or conda.

---

## How to Run Experiments

### 1. Train SAC from Scratch

```bash
python train_baseline_sac.py \
  --env Hopper-v4 \
  --total_steps 200000 \
  --seed 0 \
  --buffer_size 500000 \
  --batch_size 256
```

**2. Collect Rollouts from a Checkpoint**

```bash
python collect_from_checkpoint.py \
  --env Hopper-v4 \
  --ckpt checkpoints/checkpoint_50000.pt \
  --out data/dataset_from_ckpt_50k_seed0.pkl \
  --rollout_steps 50000 \
  --seed 0
```

---

**3. Train SAC with Pre-collected Data**

```bash
python train_with_prior_data.py \
  --env Hopper-v4 \
  --dataset data/dataset_from_ckpt_50k_seed0.pkl \
  --total_steps 200000 \
  --buffer_size 500000 \
  --batch_size 256 \
  --seed 0
```

---

**4. Evaluate Multiple Checkpoints**

```bash
python evaluate_sweep.py
```

This generates:

- evaluation_results.csv
- Mean and standard deviation of episodic rewards across seeds

---

**5. Generate Plots**

```bash
python plot_results.py
python plot_seed_variance.py
```

---

**6. Install and Run Antmaze**

conda create -n rlpd python=3.9 # If you use conda.

conda activate rlpd

conda install patchelf # If you use conda.

pip install -r requirements.txt

conda deactivate

conda activate rlpd

XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py --env_name=antmaze-umaze-v2 \
 --utd_ratio=20 \
 --start_training 5000 \
 --max_steps 300000 \
 --config=configs/rlpd_config.py \
 --config.backup_entropy=False \
 --config.hidden_dims="(256, 256, 256)" \
 --config.num_min_qs=1 \
 --project_name=rlpd_antmaze

**Edit Configs:**

nano configs/rlpd_config.py

- Design Choice 1: Offline ratio = 0.5
- Design Choice 2: Layer norm = true
- Design Choice 3: backup entropy = true

## Outputs

**Plots (figures/)**

- hopper_buffer_comparison.png
- hopper_batch_comparison.png
- hopper_seed_variance.png
- ant_buffer_comparison.png
- ant_batch_comparison.png
- ant_seed_variance.png
- combined_seed_variance.png

**CSV Files**

- baseline*rewards*\*.csv
- evaluation_results.csv
- evaluation_table.csv

**Model Checkpoints**

- checkpoint*final*\*.pt

## Reproducibility Notes

- All experiments are fully seed-controlled
- Hyperparameters are configurable via command-line arguments
- Results can be regenerated end-to-end using auto_run.sh

## Limitations

- Computational constraints limited seed count for Ant-v5
- AntMaze and Walker2D results are placeholders
- Offline data quality depends on checkpoint quality

## References

- Ball et al. (2023). Efficient Online Reinforcement Learning with Offline Data
  https://arxiv.org/abs/2302.02948
- RLPD GitHub: https://github.com/ikostrikov/rlpd
