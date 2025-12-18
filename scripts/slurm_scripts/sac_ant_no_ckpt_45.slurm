#!/bin/bash
#SBATCH --job-name=sac_ant_no_ckpt_45
#SBATCH --partition=small
#SBATCH --time=03:00:00
#SBATCH --array=0-44
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/sac_ant_no_ckpt_45_%A_%a.out
#SBATCH --error=slurm_logs/sac_ant_no_ckpt_45_%A_%a.err

echo "SLURM job started on $(hostname) at $(date)"
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"

# ------------------------
# Conda
# ------------------------
module load miniconda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

cd ~/rl_training

# ------------------------
# Experiment configuration
# ------------------------
ENV="Ant-v4"
PREFILLS=("A" "B" "C")
SEEDS=(0 1 2 3 4)
BATCHES=(128 256 512)

NUM_PREFILLS=${#PREFILLS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_BATCHES=${#BATCHES[@]}

IDX=${SLURM_ARRAY_TASK_ID}

# ------------------------
# Map index → (prefill, seed, batch)
# ------------------------
PREFILL_INDEX=$(( IDX / (NUM_SEEDS * NUM_BATCHES) ))
REM=$(( IDX % (NUM_SEEDS * NUM_BATCHES) ))

SEED_INDEX=$(( REM / NUM_BATCHES ))
BATCH_INDEX=$(( REM % NUM_BATCHES ))

PREFILL=${PREFILLS[$PREFILL_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
BATCH=${BATCHES[$BATCH_INDEX]}

echo "Running ANT | PREFILL=${PREFILL} | SEED=${SEED} | BATCH=${BATCH} (NO CKPT)"

# ------------------------
# Run
# ------------------------
srun python train_sac.py \
  --env "${ENV}" \
  --prefill "${PREFILL}" \
  --seed "${SEED}" \
  --batch "${BATCH}"

echo "FINISHED Ant-v4 | PREFILL=${PREFILL} | SEED=${SEED} | BATCH=${BATCH} at $(date)"


