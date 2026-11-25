#!/bin/bash
#SBATCH --job-name=abmil_candidates
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/abmil_candidates_%j.out
#SBATCH --error=logs/abmil_candidates_%j.err

set -euo pipefail

PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject
cd "$PROJECT/karthikraj_baseline"
module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"

SEED="${SEED:?Submit with SEED=1, 2, or 3}"
case "$SEED" in 1|2|3) ;; *) echo "Invalid seed: $SEED"; exit 1 ;; esac
OUTPUT="weights/abmil/stability_extension/seed_${SEED}"
if [[ -e "$OUTPUT" ]]; then
    echo "Refusing to overwrite existing ABMIL run: $OUTPUT" >&2
    exit 1
fi

echo "Architecture: ABMIL"
echo "Seed: $SEED"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Output: $OUTPUT"

python -c "import ctypes, numpy, pandas, torch; assert torch.cuda.is_available(); print('Python/CUDA environment OK:', torch.cuda.get_device_name(0))"
python verify_abmil_model.py

python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("datasets/camelyon16")
expected = {"camelyon16.csv": 397, "train.csv": 215, "val.csv": 54, "test.csv": 128}
for name, count in expected.items():
    table = pd.read_csv(root / name)
    assert len(table) == count, (name, len(table), count)
    assert table["path"].map(lambda value: Path(value).is_file()).all(), name
print("CAMELYON16 split and feature-file checks passed")
PY

python -u train_candidates_clam.py \
    --model abmil \
    --dataset camelyon16 \
    --num_classes 1 \
    --feats_size 512 \
    --nb_models 5 \
    --nb_epochs 5 \
    --num_epochs 3 \
    --lr 0.0002 \
    --weight_decay 0.005 \
    --init_seed "$SEED" \
    --gpu_index 0 \
    --candidate_only \
    --output_dir "$OUTPUT"

echo "ABMIL CANDIDATE GENERATION COMPLETE: seed=$SEED"
