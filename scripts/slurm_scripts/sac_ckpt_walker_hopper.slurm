#!/bin/bash
#SBATCH --job-name=sac_ckpt_wh
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --array=0-29
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/sac_ckpt_wh_%A_%a.out
#SBATCH --error=slurm_logs/sac_ckpt_wh_%A_%a.err

echo "SLURM job started on $(hostname) at $(date)"
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"

# ------------------------
# Conda environment
# ------------------------
module load miniconda 2>/dev/null || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

cd ~/rl_training

# ------------------------
# Experiment configuration
# ------------------------
# ENVS: Hopper + Walker2d
# PREFILL: A, B, C
# SEEDS: 5, 6, 7, 8, 9
# Total jobs: 2 * 3 * 5 = 30 (array 0-29)

ENVS=("Walker2d-v4" "Hopper-v4")
PREFILLS=("A" "B" "C")
SEEDS=(5 6 7 8 9)

NUM_ENVS=${#ENVS[@]}          # 2
NUM_PREFILLS=${#PREFILLS[@]}  # 3
NUM_SEEDS=${#SEEDS[@]}        # 5

IDX=${SLURM_ARRAY_TASK_ID}

# Map flat index → (env, prefill, seed)
ENV_INDEX=$(( IDX / (NUM_PREFILLS * NUM_SEEDS) ))   # 0 or 1
REM=$(( IDX % (NUM_PREFILLS * NUM_SEEDS) ))         # 0..14
PREFILL_INDEX=$(( REM / NUM_SEEDS ))                # 0..2
SEED_INDEX=$(( REM % NUM_SEEDS ))                   # 0..4

ENV=${ENVS[$ENV_INDEX]}
PREFILL=${PREFILLS[$PREFILL_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running ENV=${ENV} | PREFILL=${PREFILL} | SEED=${SEED} | BATCH=256 WITH checkpoints"

# ------------------------
# Run training
# ------------------------
srun python train_sac.py \
  --env "${ENV}" \
  --prefill "${PREFILL}" \
  --seed "${SEED}" \
  --batch 256 \
  --save_checkpoints

echo "FINISHED ENV=${ENV} | PREFILL=${PREFILL} | SEED=${SEED} at $(date)"
