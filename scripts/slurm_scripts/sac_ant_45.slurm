#!/bin/bash
#SBATCH --job-name=sac_ant_45
#SBATCH --partition=small
#SBATCH --time=03:00:00
#SBATCH --array=0-44
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/sac_ant_45_%A_%a.out
#SBATCH --error=slurm_logs/sac_ant_45_%A_%a.err

echo "SLURM job started on $(hostname) at $(date)"
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"

# ------------------------
# Activate virtualenv (NO conda)
# ------------------------
source ~/rl_training/myenv/bin/activate

if [ -z "$VIRTUAL_ENV" ]; then
  echo "Failed to activate myenv virtualenv. Exiting."
  exit 1
fi

cd ~/rl_training

# ------------------------
# Parameter configurations
# ------------------------

BATCHES=(128 256 512)
PREFILLS=("A" "B" "C")
SEEDS=(0 1 2 3 4)

# Mapping:
# batch_index  = (task_id / 15)
# prefill_index = (task_id % 15) / 5
# seed_index    = (task_id % 5)

BATCH_INDEX=$(( SLURM_ARRAY_TASK_ID / 15 ))
PREFILL_INDEX=$(( (SLURM_ARRAY_TASK_ID % 15) / 5 ))
SEED_INDEX=$(( SLURM_ARRAY_TASK_ID % 5 ))

BATCH=${BATCHES[$BATCH_INDEX]}
PREFILL=${PREFILLS[$PREFILL_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running Ant-v4 | BATCH=${BATCH} | PREFILL=${PREFILL} | SEED=${SEED}"

python train_sac.py \
  --env Ant-v4 \
  --batch ${BATCH} \
  --prefill ${PREFILL} \
  --seed ${SEED}

echo "Job ${SLURM_ARRAY_TASK_ID} finished at $(date)"
