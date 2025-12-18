#!/bin/bash
#SBATCH --job-name=sac_ant_array
#SBATCH --partition=small
#SBATCH --account=small
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-44
#SBATCH --output=slurm_logs/sac_ant-%A_%a.out
#SBATCH --error=slurm_logs/sac_ant-%A_%a.err

cd "$HOME/rl_training"
source myenv/bin/activate

ENV="Ant-v4"

# ------------------------
# Parameter grid (same pattern as Hopper)
# ------------------------
BATCHES=(128 256 512)
PREFILLS=(A B C)
SEEDS=(0 1 2 3 4)

IDX=${SLURM_ARRAY_TASK_ID}

NBATCH=${#BATCHES[@]}    # 3
NPREFILL=${#PREFILLS[@]} # 3
NSEED=${#SEEDS[@]}       # 5

batch_idx=$(( IDX / (NPREFILL * NSEED) ))
prefill_idx=$(( (IDX / NSEED) % NPREFILL ))
seed_idx=$(( IDX % NSEED ))

BATCH=${BATCHES[$batch_idx]}
PREFILL=${PREFILLS[$prefill_idx]}
SEED=${SEEDS[$seed_idx]}

echo "Task $IDX: ENV=$ENV BATCH=$BATCH PREFILL=$PREFILL SEED=$SEED"
echo "Python: $(which python)"

python train_sac.py \
  --env "$ENV" \
  --prefill "$PREFILL" \
  --seed "$SEED" \
  --batch "$BATCH"

echo "Task $IDX finished at $(date)"
