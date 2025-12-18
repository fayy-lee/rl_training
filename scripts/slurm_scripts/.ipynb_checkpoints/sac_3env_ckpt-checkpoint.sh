#!/bin/bash
#SBATCH --job-name=sac_3env_ckpt
#SBATCH --partition=small
#SBATCH --account=small
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-44
#SBATCH --output=slurm_logs/sac_3env_ckpt-%A_%a.out
#SBATCH --error=slurm_logs/sac_3env_ckpt-%A_%a.err

echo "Host: $(hostname)"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"

# Go to project directory
cd "$HOME/rl_training" || { echo "Failed to cd to rl_training"; exit 1; }

# Make sure logs dir exists (just in case)
mkdir -p slurm_logs

# Activate your existing venv (same one you used for Hopper)
source myenv/bin/activate

if [ -z "$VIRTUAL_ENV" ]; then
  echo "Failed to activate myenv. Exiting."
  exit 1
fi

echo "Using Python: $(which python)"

# --------------------------
# Experiment grid
# --------------------------

# Environments
ENVS=("Ant-v4" "Walker2d-v4" "Hopper-v4")

# Prefill types
PREFILLS=("A" "B" "C")

# Seeds 5–9
SEEDS=(5 6 7 8 9)

BATCH=256

IDX=${SLURM_ARRAY_TASK_ID}

NENV=${#ENVS[@]}        # 3
NPREFILL=${#PREFILLS[@]} # 3
NSEED=${#SEEDS[@]}       # 5

# Decode IDX -> (env, prefill, seed)
seed_idx=$(( IDX % NSEED ))
prefill_idx=$(( (IDX / NSEED) % NPREFILL ))
env_idx=$(( IDX / (NSEED * NPREFILL) ))

ENV=${ENVS[$env_idx]}
PREFILL=${PREFILLS[$prefill_idx]}
SEED=${SEEDS[$seed_idx]}

echo "Config for task $IDX:"
echo "  ENV     = $ENV"
echo "  PREFILL = $PREFILL"
echo "  SEED    = $SEED"
echo "  BATCH   = $BATCH"

# --------------------------
# Run training with checkpoints
# --------------------------
python train_sac.py \
  --env "$ENV" \
  --batch "$BATCH" \
  --prefill "$PREFILL" \
  --seed "$SEED" \
  --save_checkpoints

status=$?
echo "Task $IDX finished at $(date) with exit code $status"
exit $status
