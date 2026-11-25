#!/bin/bash
#SBATCH --job-name=abmil_eval
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/abmil_eval_%j.out
#SBATCH --error=logs/abmil_eval_%j.err

set -euo pipefail
PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject
cd "$PROJECT/karthikraj_baseline"
module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"

python -c "import ctypes, numpy, pandas, sklearn, torch; assert torch.cuda.is_available(); print('ABMIL evaluation environment OK:', torch.cuda.get_device_name(0))"
python -u evaluate_abmil_heldout.py
python -u summarize_abmil.py
python -u verify_abmil.py
echo "ABMIL HELD-OUT EVALUATION AND VERIFICATION COMPLETE"
