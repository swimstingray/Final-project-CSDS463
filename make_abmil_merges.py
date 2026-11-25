"""Create ABMIL standalone and TIES-merged checkpoints from completed candidates.

This stage uses training/validation artifacts only. It never opens the held-out
test split. The stability penalty is fixed at lambda=0.5 before test evaluation,
matching the CLAM comparison scale; it is exploratory, not tuned on test data.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch

ROOT = Path("weights/abmil/stability_extension")
SEEDS = (1, 2, 3)
STABILITY_LAMBDA = 0.5
N_CANDIDATES = 5
N_EARLY_EPOCHS = 5

def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Invalid state dictionary: {path}")
    return {key: value.detach().cpu() for key, value in state.items()}

def ties_merge(selected_states, initial_state):
    """Match the repository's TIES-style merge with k=1.0 and lambda=1.0."""
    if len(selected_states) != 3:
        raise ValueError("Exactly three candidate state dictionaries are required")
    merged = {}
    for key, initial in initial_state.items():
        values = [state[key] for state in selected_states]
        if any(value.shape != initial.shape for value in values):
            raise ValueError(f"State shape mismatch for {key}")
        if not torch.is_floating_point(initial):
            if any(not torch.equal(value, initial) for value in values):
                raise ValueError(f"Unexpected non-floating state variation: {key}")
            merged[key] = initial.clone()
            continue
        task_vectors = [value - initial for value in values]
        trimmed = []
        for vector in task_vectors:
            threshold = torch.quantile(vector.reshape(-1).abs(), 0.0)
            trimmed.append(torch.where(vector.abs() >= threshold, vector, torch.zeros_like(vector)))
        stack = torch.stack(trimmed, dim=0)
        elected_sign = torch.sign(stack.sum(dim=0))
        sign_matches = torch.stack([torch.sign(vector) == elected_sign for vector in trimmed], dim=0)
        selected_values = torch.where(sign_matches, stack, torch.zeros_like(stack))
        count = sign_matches.to(stack.dtype).sum(dim=0)
        average = selected_values.sum(dim=0) / count.clamp_min(1.0)
        average = torch.where(count > 0, average, torch.zeros_like(average))
        merged[key] = initial + average
    return merged

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-lambda", type=float, default=STABILITY_LAMBDA)
    args = parser.parse_args()
    if args.stability_lambda < 0:
        parser.error("--stability-lambda must be nonnegative")
    for seed in SEEDS:
        run_dir = ROOT / f"seed_{seed}"
        history_path = run_dir / "candidate_histories.npz"
        initial_path = run_dir / "initial_weights.pth"
        candidate_dir = run_dir / "candidates"
        merge_dir = run_dir / "merges"
        for required in (history_path, initial_path, candidate_dir):
            if not required.exists():
                raise FileNotFoundError(required)
        if merge_dir.exists():
            raise FileExistsError(f"Refusing to overwrite completed merge directory: {merge_dir}")
        with np.load(history_path, allow_pickle=False) as archive:
            early = np.asarray(archive["early_aucs"], dtype=float)
            recorded_seed = int(archive["seed"])
            n_models = int(archive["nb_models"])
            n_epochs = int(archive["nb_epochs"])
        if recorded_seed != seed:
            raise ValueError(f"Seed mismatch in {history_path}: {recorded_seed}")
        if early.shape != (N_CANDIDATES, N_EARLY_EPOCHS):
            raise ValueError(f"Expected 5x5 validation history, got {early.shape}")
        if n_models != N_CANDIDATES or n_epochs != N_EARLY_EPOCHS:
            raise ValueError(f"Unexpected candidate configuration in {history_path}")
        if not np.isfinite(early).all() or np.any((early < 0) | (early > 1)):
            raise ValueError(f"Invalid validation AUCs in {history_path}")
        means = early.mean(axis=1)
        stds = early.std(axis=1, ddof=0)
        baseline_scores = means
        stability_scores = means - args.stability_lambda * stds
        baseline_selected = np.argsort(baseline_scores)[-3:].tolist()
        stability_selected = np.argsort(stability_scores)[-3:].tolist()
        standalone_selected = int(np.argmax(means))
        initial = load_state(initial_path)
        candidates = [load_state(candidate_dir / f"candidate_{index}.pth") for index in range(N_CANDIDATES)]
        merge_dir.mkdir(parents=True, exist_ok=False)
        baseline_state = ties_merge([candidates[index] for index in baseline_selected], initial)
        stability_state = ties_merge([candidates[index] for index in stability_selected], initial)
        torch.save(baseline_state, merge_dir / "merged_lambda_0.0.pth")
        torch.save(stability_state, merge_dir / f"merged_lambda_{args.stability_lambda}.pth")
        shutil.copy2(candidate_dir / f"candidate_{standalone_selected}.pth", merge_dir / "standalone_candidate.pth")
        metadata = {
            "architecture": "ABMIL",
            "seed": seed,
            "candidate_count": N_CANDIDATES,
            "early_validation_epochs": N_EARLY_EPOCHS,
            "checkpoint_weights": [1, 1, 1, 1, 1],
            "stability_lambda": args.stability_lambda,
            "candidate_mean_validation_auc": means.tolist(),
            "candidate_population_std_validation_auc": stds.tolist(),
            "baseline_scores": baseline_scores.tolist(),
            "stability_scores": stability_scores.tolist(),
            "baseline_selected_candidate_indices": baseline_selected,
            "stability_selected_candidate_indices": stability_selected,
            "standalone_selected_candidate_index": standalone_selected,
            "baseline_and_stability_select_same_candidates": set(baseline_selected) == set(stability_selected),
            "heldout_test_accessed": False,
            "merge": "TIES-style sign-consensus task-vector merge, k=1.0, merge scale=1.0",
        }
        with (merge_dir / "selection.json").open("x") as file:
            json.dump(metadata, file, indent=2)
        print(f"ABMIL seed={seed}: baseline={baseline_selected}, stability={stability_selected}, standalone={standalone_selected}", flush=True)
        print("MERGES:", merge_dir, flush=True)
    print("ALL ABMIL MERGES CREATED")

if __name__ == "__main__":
    main()
