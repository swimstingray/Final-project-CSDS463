#!/bin/bash
#SBATCH --job-name=abmil_cont
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/abmil_cont_%j.out
#SBATCH --error=logs/abmil_cont_%j.err

set -euo pipefail
PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject
cd "$PROJECT/karthikraj_baseline"
module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"

SEED="${SEED:?Missing SEED}"
METHOD="${METHOD:?Missing METHOD}"
case "$SEED" in 1|2|3) ;; *) echo "Invalid seed"; exit 1 ;; esac
ROOT="weights/abmil/stability_extension/seed_${SEED}"
case "$METHOD" in
    standalone) CHECKPOINT="$ROOT/merges/standalone_candidate.pth" ;;
    baseline) CHECKPOINT="$ROOT/merges/merged_lambda_0.0.pth" ;;
    stability) CHECKPOINT="$ROOT/merges/merged_lambda_0.5.pth" ;;
    *) echo "Invalid method: $METHOD"; exit 1 ;;
esac

OUTPUT="$ROOT/continuation/$METHOD"
test -f "$CHECKPOINT"
if [[ -e "$OUTPUT" ]]; then
    echo "Refusing to overwrite existing continuation: $OUTPUT" >&2
    exit 1
fi

echo "Architecture: ABMIL"
echo "Seed: $SEED"
echo "Method: $METHOD"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

python -u train_fusion_abmil.py \
    --merged_checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT" \
    --seed "$SEED" \
    --model abmil \
    --dataset camelyon16 \
    --num_classes 1 \
    --feats_size 512 \
    --num_epochs 3 \
    --lr 0.0002 \
    --weight_decay 0.005 \
    --gpu_index 0

echo "ABMIL CONTINUATION COMPLETE: seed=$SEED method=$METHOD"
