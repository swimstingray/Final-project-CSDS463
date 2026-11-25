#!/usr/bin/env bash
set -euo pipefail
PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject
cd "$PROJECT/karthikraj_baseline"
mkdir -p logs
for seed in 1 2 3; do
    job=$(sbatch --parsable --export=ALL,SEED="$seed" abmil_candidates.slurm)
    echo "Submitted ABMIL candidate generation seed=$seed job=$job"
done
echo
echo "Check progress with: squeue -u $USER"
echo "Do not run the merge stage until all three jobs complete successfully."
