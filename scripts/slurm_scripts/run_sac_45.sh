#!/bin/bash
#SBATCH --job-name=sac45
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=small
#SBATCH --account=small
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

cd $HOME/rl_training
source myenv/bin/activate

ENV="Hopper-v4"

for BATCH in 128 256 512; do
  for PREFILL in A B C; do
    for SEED in 0 1 2 3 4; do
      echo "Running BATCH=$BATCH PREFILL=$PREFILL SEED=$SEED"
      python train_sac.py --env $ENV --prefill $PREFILL --seed $SEED --batch $BATCH
    done
  done
done
