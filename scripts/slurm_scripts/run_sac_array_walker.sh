#!/bin/bash
#SBATCH --job-name=sac_array_walker 
#SBATCH --partition=small
#SBATCH --account=small
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-44
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

cd $HOME/rl_training
source myenv/bin/activate

ENV="Walker2d-v4"

# Define the grid
BATCHES=(128 256 512)
PREFILLS=(A B C)
SEEDS=(0 1 2 3 4)

# Decode this task's index (0..44) into (batch, prefill, seed)
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

echo "Task $IDX: BATCH=$BATCH PREFILL=$PREFILL SEED=$SEED"

python train_sac.py \
  --env "$ENV" \
  --prefill "$PREFILL" \
  --seed "$SEED" \
  --batch "$BATCH"
