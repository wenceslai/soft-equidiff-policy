#!/bin/bash
# Rotation robustness evaluation.
# Tests how much performance drops when observations are rotated at test time.
# Run from the project root: bash eval_rotation_robustness.sh

set -e  # stop on error

BASE_CKPT="outputs/base_diffusion_rerun/policy_step0150000.pt"
SOFT_CKPT="outputs/soft_constant_c4_rotaug_v2/policy_step0150000.pt"
EXACT_CKPT="outputs/equi_exact_c4_rotaug_v2/policy_step0125000.pt"

EPISODES=50
DEVICE=cuda

echo "======================================================"
echo " Rotation robustness eval  (${EPISODES} episodes each)"
echo "======================================================"

for ROT in 0 1 2 3; do
    DEG=$(( ROT * 90 ))
    echo ""
    echo "------ rotation ${ROT} × 90° = ${DEG}° ------"

    echo "[baseline]"
    python -m soft_equidiff.eval_success_rate \
        --checkpoint "$BASE_CKPT" \
        --n_episodes $EPISODES --device $DEVICE \
        --test_rotation $ROT --no_gif --no_wandb

    echo "[soft equivariant]"
    python -m soft_equidiff.eval_success_rate \
        --checkpoint "$SOFT_CKPT" \
        --n_episodes $EPISODES --device $DEVICE \
        --test_rotation $ROT --no_gif --no_wandb

    echo "[exact equivariant]"
    python -m soft_equidiff.eval_success_rate \
        --checkpoint "$EXACT_CKPT" \
        --n_episodes $EPISODES --device $DEVICE \
        --test_rotation $ROT --no_gif --no_wandb
done

echo ""
echo "======================================================"
echo " Done."
echo "======================================================"
