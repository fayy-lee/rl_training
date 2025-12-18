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

```
rl_training/
├── scripts/                          # Training, evaluation, plotting scripts
│   ├── train.py                      # Wrapper for SAC training
│   ├── eval.py                       # Wrapper for evaluation
│   ├── collect.py                    # Wrapper for data collection
│   ├── plot.py                       # Wrapper for plotting
│   ├── train_baseline_sac.py         # Train SAC from scratch
│   ├── train_sac.py                  # SAC with prefill options
│   ├── train_cartpole.py             # CartPole experiments
│   ├── train_rlpd_from_dataset.py    # RLPD offline training
│   ├── collect_from_checkpoint.py    # Collect rollouts from policy
│   ├── evaluate_sac.py               # Single checkpoint evaluation
│   ├── evaluate_sweep.py             # Multi-seed evaluation
│   ├── evaluate_and_generate_dataset.py  # Dataset generation
│   ├── plot_results.py               # Learning curves
│   ├── plot_results_ant.py           # Ant-specific plots
│   ├── plot_results_walker.py        # Walker-specific plots
│   ├── plot_seed_variance.py         # Seed variance plots
│   ├── plot_sac_prefill_comparison.py  # Prefill ablations
│   ├── compare_learning_curves.py    # Cross-environment comparison
│   ├── compare_sac_ppo.py            # SAC vs PPO comparison
│   ├── generate_result_table.py      # Summary statistics tables
│   ├── sac_utils.py                  # SAC agent implementation
│   └── slurm_scripts/                # Cluster job scripts
├── policies/                         # Trained model checkpoints
│   ├── checkpoint_final_*.pt         # Final trained policies
│   ├── checkpoint_50000.pt           # Intermediate checkpoints
│   ├── prior_buffer*.pt              # Pre-collected replay buffers
│   └── checkpoints_t/                # Timestamped checkpoints
├── results/                          # Experimental results
│   ├── figures/                      # Generated plots (PNG)
│   ├── csvs/                         # Summary CSV files
│   ├── 1M/                           # 1M step experiments
│   └── data_t/                       # Timestamped data
├── media/                            # Videos and visualizations
│   └── videos/                       # Environment rollout videos
│       ├── Ant-v4/
│       ├── Hopper-v4/
│       └── Walker2d-v4/
├── rlpd/                             # RLPD implementation (offline RL)
│   ├── train_finetuning.py           # RLPD training script
│   ├── configs/                      # RLPD hyperparameter configs
│   ├── rlpd/                         # RLPD agent implementation
│   └── README.md                     # RLPD-specific instructions
├── docs/                             # Documentation
│   └── README.md                     # Additional documentation
├── environment.yml                   # Conda environment specification
└── README.md                         # This file
```

---

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd rl_training
```

### 2. Create Environment

```bash
conda env create -f environment.yml
conda activate rl-training
```

### Dependencies

Main Python dependencies (specified in environment.yml):

- torch
- gymnasium
- numpy
- pandas
- matplotlib
- mujoco
- tensorboard
- stable-baselines3
- ml-collections
- jax, flax
- tensorflow-probability
- d4rl, dmcgym (for RLPD offline datasets)
- wandb, tqdm

---

## How to Run Experiments

### 1. Train SAC from Scratch

```bash
python scripts/train_baseline_sac.py \
  --env Hopper-v4 \
  --total_steps 200000 \
  --seed 0 \
  --buffer_size 500000 \
  --batch_size 256
```

Or use the wrapper:

```bash
python scripts/train.py --env Hopper-v4 --seed 0
```

### 2. Collect Rollouts from a Checkpoint

```bash
python scripts/collect_from_checkpoint.py \
  --env Hopper-v4 \
  --ckpt policies/checkpoint_50000.pt \
  --out results/dataset_from_ckpt_50k_seed0.pkl \
  --rollout_steps 50000 \
  --seed 0
```

---

### 3. Train SAC with Pre-collected Data

```bash
python scripts/train_with_prior_data.py \
  --env Hopper-v4 \
  --dataset results/dataset_from_ckpt_50k_seed0.pkl \
  --total_steps 200000 \
  --buffer_size 500000 \
  --batch_size 256 \
  --seed 0
```

---

### 4. Evaluate Multiple Checkpoints

```bash
cd /path/to/rl_training
python scripts/evaluate_sweep.py
```

Or use the wrapper:

```bash
python scripts/eval.py --checkpoint policies/checkpoint_final_Hopper-v4_seed0_buf500000_batch256.pt --env Hopper-v4
```

This generates:

- results/evaluation_results.csv
- Mean and standard deviation of episodic rewards across seeds

---

### 5. Generate Plots

```bash
python scripts/plot_results.py
python scripts/plot_seed_variance.py
```

Or use the wrapper:

```bash
python scripts/plot.py
```

---

### 6. RLPD Training (Offline RL)

For RLPD-specific experiments with D4RL datasets:

```bash
cd rlpd
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py \
  --env_name=halfcheetah-expert-v0 \
  --utd_ratio=20 \
  --start_training 5000 \
  --max_steps 250000 \
  --config=configs/rlpd_config.py \
  --project_name=rlpd_locomotion
```

See [rlpd/README.md](rlpd/README.md) for detailed RLPD instructions.

---

## Outputs

**Plots (results/figures/)**

- hopper_buffer_comparison.png
- hopper_batch_comparison.png
- hopper_seed_variance.png
- ant_buffer_comparison.png
- ant_batch_comparison.png
- ant_seed_variance.png
- combined_seed_variance.png
- walker_variant_comparison.png

**CSV Files (results/)**

- baseline_rewards_*.csv
- evaluation_results.csv
- evaluation_table.csv

**Model Checkpoints (policies/)**

- checkpoint_final_*.pt
- checkpoint_50000.pt, checkpoint_100000.pt
- prior_buffer.pt, temp_*.pt

## Reproducibility Checklist

| Component | Location |
|-----------|----------|
| Training scripts | [scripts/](scripts/) |
| Trained models | [policies/](policies/) |
| Results & metrics | [results/](results/) |
| Plots & figures | [results/figures/](results/figures/) |
| Videos | [media/videos/](media/videos/) |
| Environment spec | [environment.yml](environment.yml) |
| RLPD configs | [rlpd/configs/](rlpd/configs/) |
| Instructions | This README |

**Notes:**
- All experiments are fully seed-controlled
- Hyperparameters are configurable via command-line arguments
- Results can be regenerated end-to-end
- RLPD has separate configurations in rlpd/configs/

## Limitations

- Computational constraints limited seed count for Ant-v5
- Offline data quality depends on checkpoint quality
- RLPD features require additional setup (see rlpd/README.md)

## References

- Ball et al. (2023). Efficient Online Reinforcement Learning with Offline Data
  https://arxiv.org/abs/2302.02948
- RLPD GitHub: https://github.com/ikostrikov/rlpd
