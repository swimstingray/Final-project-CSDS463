#!/usr/bin/env bash
set -euo pipefail
PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject
cd "$PROJECT/karthikraj_baseline"
mkdir -p logs

for seed in 1 2 3; do
    for method in standalone baseline stability; do
        case "$method" in
            standalone) checkpoint="weights/abmil/stability_extension/seed_${seed}/merges/standalone_candidate.pth" ;;
            baseline) checkpoint="weights/abmil/stability_extension/seed_${seed}/merges/merged_lambda_0.0.pth" ;;
            stability) checkpoint="weights/abmil/stability_extension/seed_${seed}/merges/merged_lambda_0.5.pth" ;;
        esac
        test -f "$checkpoint" || {
            echo "Missing checkpoint: $checkpoint" >&2
            echo "Run python make_abmil_merges.py after all candidate jobs complete." >&2
            exit 1
        }
    done
done

ids=()
for seed in 1 2 3; do
    for method in standalone baseline stability; do
        job=$(sbatch --parsable --export=ALL,SEED="$seed",METHOD="$method" abmil_continuation.slurm)
        echo "Submitted ABMIL seed=$seed method=$method job=$job"
        ids+=("$job")
    done
done

dependency=$(IFS=:; echo "${ids[*]}")
evaluation=$(sbatch --parsable --dependency="afterok:$dependency" abmil_evaluation.slurm)
echo
echo "Submitted ABMIL held-out evaluation and summary job=$evaluation"
echo "Evaluation waits for all nine continuation jobs to complete successfully."
echo "Check progress with: squeue -u $USER"
