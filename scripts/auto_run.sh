#!/usr/bin/env bash

run_if_missing() {
    ENV_NAME=$1
    SEED=$2
    BUFFER=$3
    BATCH=$4

    FINAL_CKPT="checkpoint_final_${ENV_NAME}_seed${SEED}_buf${BUFFER}_batch${BATCH}.pt"

    if [ -f "$FINAL_CKPT" ]; then
        echo "✔ SKIP: $FINAL_CKPT already exists"
    else
        echo "➡ RUNNING: env=$ENV_NAME seed=$SEED buffer=$BUFFER batch=$BATCH"
        python train_baseline_sac.py \
            --env "$ENV_NAME" \
            --total_steps 200000 \
            --save_at 50000,100000 \
            --seed "$SEED" \
            --buffer_size "$BUFFER" \
            --batch_size "$BATCH"
    fi
}

echo "=============================="
echo "   STARTING FULL EXPERIMENTS  "
echo "=============================="

###############################################
# A) Hopper-v4 — buffer 500k, batch 256, 5 seeds
###############################################
echo "=== A) Hopper 500k buffer, batch 256, seeds 0–4 ==="
for SEED in 0 1 2 3 4; do
    run_if_missing "Hopper-v4" $SEED 500000 256
done

###############################################
# B) Hopper-v4 — buffer 300k, batch 256, 3 seeds
###############################################
echo "=== B) Hopper 300k buffer, batch 256, seeds 0–2 ==="
for SEED in 0 1 2; do
    run_if_missing "Hopper-v4" $SEED 300000 256
done

###############################################
# C) Hopper-v4 — batch size sweep (128 and 512), buffer 500k, 3 seeds
###############################################
echo "=== C) Hopper 500k buffer, batch {128, 512}, seeds 0–2 ==="
for SEED in 0 1 2; do
    run_if_missing "Hopper-v4" $SEED 500000 128
    run_if_missing "Hopper-v4" $SEED 500000 512
done

###############################################
# D) Ant-v5 — buffer 500k, batch 256, 3 seeds
###############################################
echo "=== D) Ant-v5 500k buffer, batch 256, seeds 0–2 ==="
for SEED in 0 1 2; do
    run_if_missing "Ant-v5" $SEED 500000 256
done

echo "=============================="
echo "       ALL DONE (or skipped)   "
echo "=============================="
