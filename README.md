# Stability-Aware Model Selection and Merging for Multiple Instance Learning in Digital Pathology

An experimental extension of an existing multiple instance learning (MIL) project, investigating whether selecting models based on both validation performance and stability can improve the consistency of merged models in digital pathology.

The experiments compare standard validation-based candidate selection with stability-aware candidate selection for **CLAM** and **DSMIL** using the **CAMELYON16** dataset.

**Main finding:** Stability-aware selection reduced the observed standard deviation of CLAM's held-out test AUC across three seeds, but also reduced its mean test AUC. DSMIL showed a very small improvement in mean test AUC with essentially unchanged variability. These results do not establish a consistent performance benefit from stability-aware selection.

## Current repository map (November 27, 2026)

The main discussion below documents the original CLAM/DSMIL stability-aware comparison. Additional completed experiments are available and should **not** be confused with the original matched test comparison:

| Workstream | Code | Results | Evaluation scope |
|---|---|---|---|
| Original standalone/baseline/stability CLAM and DSMIL | `standalone_comparison.py`, `evaluate_heldout.py` | [Three-way comparison](final_results/three_way_summary.csv), [test records](final_results/all_results.csv) | Held-out test |
| Matched ABMIL standalone/baseline/stability | `evaluate_abmil_heldout.py`, `prepare_abmil_fusions.py` | [ABMIL summary](final_results/abmil_three_way_summary.csv), [test records](heldout_evaluation/abmil_test_auc_results.csv) | Held-out test (separate protocol) |
| Experiment A — cross-architecture probability averaging | `experiment_a_ensemble.py` | [Validation measurements](experiment_a_outputs/validation_auc_results.csv), [summary](experiment_a_outputs/validation_summary.csv) | Validation only |
| Experiment B — greedy model soups | `experiment_b_greedy_soup.py` | [Validation summary](final_results/experiment_b_summary.csv), [selection manifests](experiment_b_outputs/) | Validation only |
| AUC/stability-weighted parameter merging | `prepare_abmil_weighted_fusions.py`, `prepare_other_weighted_fusions.py` | Completed training jobs; consolidated comparison still needed | No consolidated performance claim |

See the [six paper figures and captions](paper_figures/README.md), [manuscript draft](report/Project2_Manuscript_Draft.md), [project submission checklist](PROJECT2_COMPLETION.md), and [repository audit](REPOSITORY_AUDIT.md).

**Important:** Results from a common validation set used for model selection are exploratory. Three-seed held-out measurements are descriptive, not statistical evidence of a generalizable improvement. Feature data and checkpoint weights are intentionally excluded from version control.

## 1. Background and Project Attribution

Multiple instance learning is commonly used in digital pathology to classify whole-slide images using collections of patch-level features. Predictions can vary across training runs because of initialization, optimization, and model selection.

This project builds on an existing repository that reproduced MIL baselines associated with the MICCAI 2025 paper *Reducing Variability of Multiple Instance Learning Methods for Digital Pathology*.

**Original and related repositories:**

- [Original course project — karthikraj2103/Final-project-CSDS463](https://github.com/karthikraj2103/Final-project-CSDS463)
- [Underlying MIL merging implementation — mammadov7/mil_merging](https://github.com/mammadov7/mil_merging)

The existing project contains baseline implementations and experiments involving ABMIL, CLAM, and DSMIL. This extension focuses specifically on **CLAM and DSMIL**, adding stability-aware candidate selection, repeatable merging workflows, multiple-seed comparisons, and a consolidated held-out evaluation.

The original baseline results and the results of this extension use different experiment configurations and should not be treated as directly interchangeable.

## 2. Research Question

**Does penalizing variation in early validation AUC during candidate selection improve the reliability or predictive performance of merged MIL models?**

The experiment compares two candidate-selection strategies:

1. **Baseline selection:** Rank candidates using their weighted average early validation AUC.
2. **Stability-aware selection:** Rank candidates using the same average AUC, with an additional penalty for variation across selected validation checkpoints.

Both conditions follow the same overall merging and post-merge training approach.

## 3. Dataset

The experiments use **CAMELYON16**, a digital pathology dataset for detecting lymph node metastases.

The workflow uses pre-extracted, 512-dimensional patch features, organized into slide-level feature files.

| Dataset split | Slides |
|---|---:|
| Training | 215 |
| Validation | 54 |
| Held-out test | 128 |
| **Total** | **397** |

The repository's existing preprocessing and verification scripts include:

- `feature_extract.py` — download and inspect pre-extracted features.
- `check_slide.py` — inspect slide structure and labels.
- `con_H5-cv.py` — convert H5 features to slide-level CSV files.
- `Dataset_verify.py` — inspect datasets and splits.

Training and validation splits are used for model development. The final evaluation script checks that test filenames do not overlap with training or validation filenames.

**Dataset availability:** Feature CSVs and split files are not included in this GitHub repository. They must be prepared separately before training or evaluation.

## 4. Methodology

### 4.1 Candidate model training

For each architecture and experimental seed, the workflow trains five candidate MIL models and records early validation AUC measurements.

The three experimental seeds are:

`1`, `2`, and `3`

The candidate generation configuration includes:

| Parameter | Value |
|---|---|
| Candidate models per run | 5 |
| Candidate validation checkpoints | 5 |
| Feature dimensions | 512 |
| Learning rate | 0.0002 |
| Weight decay | 0.005 |
| Model architectures | CLAM, DSMIL |
| Dataset | CAMELYON16 |

The candidate-generation scripts retain model checkpoints and validation histories for subsequent selection and merging.

### 4.2 Stability-aware candidate scoring

Each candidate receives the following selection score:

```text
S_i = μ_i − λ σ_i
```

where:

- `μ_i` is the weighted mean validation AUC across early checkpoints.
- `σ_i` is the corresponding weighted standard deviation.
- `λ` determines the strength of the stability penalty.

A larger `λ` favors candidates with less variation in their recorded validation AUC, even when their average performance is similar.

The implementation uses architecture-specific checkpoint weights:

| Architecture | Checkpoint weighting |
|---|---|
| CLAM | Equal weighting across all five checkpoints |
| DSMIL | Only the final two of five checkpoints receive nonzero weight |

The baseline condition sets `λ=0`. The stability-aware condition uses the following experimentally selected comparison values:

| Architecture | Baseline λ | Stability-aware λ |
|---|---:|---:|
| CLAM | 0.0 | 0.5 |
| DSMIL | 0.0 | 5.0 |

These parameter values are specific to this experiment and are not established as universally optimal.

### 4.3 Candidate selection and model merging

For each condition, the three highest-scoring candidates are selected and combined using a TIES-style weight-merging implementation.

The procedure operates on task vectors defined relative to shared initial model parameters, applies parameter trimming, elects signs, and combines compatible updates.

The repository's CLAM and DSMIL merging implementations have different default trimming fractions, so they should not be assumed to perform identical parameter operations.

Following merging, the combined checkpoint is used to initialize a short post-merge training stage.

### 4.4 Post-merge training

The post-merge scripts are:

- `train_fusion_clam.py`
- `train_fusion_dsmil.py`

Both use the merged checkpoint as initialization and train on the training split while monitoring validation performance.

The final experiment configuration specifies `--num_epochs 3`. The actual loop is `range(1, args.num_epochs)`, so **two training epochs** are executed.

The checkpoint with the highest validation AUC during this stage is saved as `best_validation.pth`.

Some seed/lambda comparisons select the same candidate set and reuse the same merged initialization. This is reflected in identical final measurements for several paired comparisons.

### 4.5 Held-out evaluation

The final evaluation uses `evaluate_heldout.py` and the 128-slide test split.

The script:

1. Loads the 12 final validation-selected checkpoints.
2. Verifies the expected test split size and checks for train/test and validation/test filename overlap.
3. Runs model inference on the held-out slides.
4. Calculates slide-level ROC AUC.
5. Saves the result for every architecture, seed, and lambda condition.

The 12 evaluations consist of:

- 2 architectures
- 3 seeds per architecture
- 2 selection methods per seed

Evaluation outputs are saved in `heldout_evaluation/test_auc_results.csv`.

## 5. Experimental Results

The main metric is slide-level **ROC AUC**. Values below are reported as **mean ± sample standard deviation across three experimental seeds**.

### 5.1 Validation and held-out test performance

| Architecture | Method | Validation AUC | Held-out test AUC |
|---|---|---:|---:|
| CLAM | Baseline | 0.7779 ± 0.0427 | 0.7279 ± 0.0566 |
| CLAM | Stability-aware | 0.7618 ± 0.0230 | 0.7103 ± 0.0259 |
| DSMIL | Baseline | 0.5795 ± 0.0462 | 0.6205 ± 0.0570 |
| DSMIL | Stability-aware | 0.5767 ± 0.0466 | 0.6219 ± 0.0571 |

Source: [`final_results/summary.csv`](final_results/summary.csv)

![Mean validation and held-out test ROC AUC](final_results/mean_auc_comparison.png)

*Figure 1. Mean validation and held-out test ROC AUC for each architecture and selection strategy. Error bars represent standard deviation across three seeds.*

### 5.2 Paired test differences

The average held-out test AUC difference, defined as stability-aware minus baseline, was:

| Architecture | Mean paired test AUC difference |
|---|---:|
| CLAM | −0.0176 |
| DSMIL | +0.0013 |

![Paired held-out test differences](final_results/paired_test_differences.png)

*Figure 2. Difference in held-out test AUC between stability-aware and baseline selection for each experimental seed. Positive values indicate improvement with stability-aware selection.*

The per-seed observations indicate that the effect was not uniform. Some paired conditions produced identical outcomes, while CLAM seed 1 showed a substantial decrease in AUC under stability-aware selection.

### 5.3 Interpretation

For **CLAM**, stability-aware selection reduced the observed across-seed standard deviation of held-out test AUC from approximately 0.0566 to 0.0259. However, mean test AUC decreased from 0.7279 to 0.7103.

For **DSMIL**, mean test AUC increased slightly from 0.6205 to 0.6219, but the observed standard deviation remained nearly unchanged.

Overall, the results illustrate a possible trade-off between selecting candidates with consistent early validation performance and achieving higher average predictive performance.

The experiment does **not** provide sufficient evidence that stability-aware candidate selection generally outperforms the baseline.

## 6. Repository Structure

### Main experiment scripts

| File | Purpose |
|---|---|
| `train_candidates_clam.py` | Generate CLAM candidates and their validation histories |
| `train_candidates_dsmil.py` | Generate DSMIL candidates and their validation histories |
| `train_repro_clam.py` | CLAM candidate selection, merging, and training workflow |
| `train_repro_dsmil.py` | DSMIL candidate selection, merging, and training workflow |
| `train_stability_clam.py` | CLAM stability-aware training implementation |
| `train_stability_dsmil.py` | DSMIL stability-aware training implementation |
| `train_fusion_clam.py` | CLAM post-merge training |
| `train_fusion_dsmil.py` | DSMIL post-merge training |
| `evaluate_heldout.py` | Final held-out test evaluation |
| `make_final_results.py` | Aggregate results and generate figures |
| `verify_published_results.py` | Validate committed CSV consistency without model checkpoints or extra Python packages |

### Workflow and experiment scripts

| File | Purpose |
|---|---|
| `test_candidates.slurm` | Candidate generation jobs |
| `test_stability.slurm` | Stability-aware model training jobs |
| `test_fusion_clam.slurm` | CLAM post-merge training |
| `test_fusion_dsmil.slurm` | DSMIL post-merge training |
| `reproducibility_fusion.slurm` | Additional post-merge runs for seeds 2 and 3 |
| `evaluate_heldout.slurm` | Run final held-out evaluation |
| `create_stability_scripts.py` | Script used to construct stability-aware implementations |
| `fix_training_pipeline.py` | Script used to construct revised reproducibility implementations |
| `create_candidate_only.py` | Script used to construct candidate-generation implementations |

The creation and modification scripts are preserved for development history. The generated experiment scripts already exist in the repository and should not be regenerated over them.

### Final results

```text
final_results/
├── all_results.csv
├── summary.csv
├── paired_differences.csv
├── mean_auc_comparison.png
└── paired_test_differences.png

heldout_evaluation/
└── test_auc_results.csv
```

The summary contains all four architecture/method combinations. The detailed CSV files preserve the individual three-seed measurements.

## 7. Reproducibility

### 7.1 Environment

Experiments were conducted using a GPU-enabled SLURM environment on Case Western Reserve University's Pioneer HPC system.

The SLURM scripts reference:

- Python 3.11.3
- A Python virtual environment
- A CUDA-capable GPU
- PyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib
- SciPy and other dependencies required by the original MIL implementations

The original model implementation files must also be present.

The published scripts contain absolute paths specific to the original HPC workspace. These paths and environment activation commands must be updated when running on another system.

**Note:** A fully pinned dependency environment has not been established by this extension. Exact package versions, CUDA configuration, and compatibility should be verified before attempting an independent reproduction.

### 7.2 Prepare the data

Reproduce the original repository's feature preparation steps and ensure that the following files exist:

```text
datasets/camelyon16/
├── camelyon16.csv
├── train.csv
├── val.csv
└── test.csv
```

The split tables must identify their slide-level feature paths and labels. The referenced feature files must be available at those paths.

The existing split definitions are part of the original project environment rather than the new files uploaded in this extension.

### 7.3 Generate candidate models

After updating the HPC-specific paths in `test_candidates.slurm`, example SLURM submissions are:

```bash
mkdir -p logs

sbatch --export=ALL,MODEL=clam,SEED=1 test_candidates.slurm
sbatch --export=ALL,MODEL=dsmil,SEED=1 test_candidates.slurm
```

Repeat for seeds 2 and 3.

Candidate generation saves checkpoints, initial parameters, and validation-history artifacts needed for candidate selection and model merging.

### 7.4 Run stability-aware selection

An example stability-aware SLURM submission is:

```bash
sbatch --export=ALL,MODEL=clam,STABILITY_LAMBDA=0.5,SEED=1 test_stability.slurm
```

The equivalent baseline comparison uses `STABILITY_LAMBDA=0.0`.

For DSMIL, the experiment compares `0.0` and `5.0`.

**Important:** The archived final experiments also use separately generated merged checkpoints in `offline_merges/`. Generating candidates alone is not sufficient to recreate these particular final checkpoints. The offline selection/merging artifacts and their provenance must be restored or reconstructed from the stored candidate information before repeating the final fusion runs.

### 7.5 Post-merge training

For a valid, existing merged CLAM checkpoint:

```bash
python train_fusion_clam.py \
  --merged_checkpoint PATH_TO_MERGED_CLAM_CHECKPOINT \
  --output_dir PATH_TO_NEW_OUTPUT_DIRECTORY \
  --seed 1 \
  --num_epochs 3 \
  --feats_size 512
```

For DSMIL:

```bash
python train_fusion_dsmil.py \
  --merged_checkpoint PATH_TO_MERGED_DSMIL_CHECKPOINT \
  --output_dir PATH_TO_NEW_OUTPUT_DIRECTORY \
  --seed 1 \
  --num_epochs 3 \
  --feats_size 512
```

Each run saves a validation-selected checkpoint and a `history.json` file.

The repository's archived SLURM wrappers contain specific run identifiers and should be treated as records of the original experiments rather than portable, general-purpose execution scripts.

### 7.6 Evaluate held-out checkpoints

Once the 12 expected checkpoints and their corresponding history files exist, submit:

```bash
sbatch evaluate_heldout.slurm
```

The underlying command is:

```bash
python evaluate_heldout.py
```

The evaluation script uses predefined checkpoint locations. It requires a CUDA GPU and intentionally refuses to overwrite an existing `heldout_evaluation/test_auc_results.csv`.

To evaluate different runs, update the checkpoint mapping and use an appropriate new output location. Do not overwrite existing results without preserving them.

### 7.7 Regenerate the results

With the evaluation CSV and original training histories available:

```bash
python make_final_results.py
```

This creates the summary tables and the two comparison figures.

The committed CSV files and figures can be inspected directly without rerunning model training.

### 7.8 Verify the published CSV results (no GPU required)

The repository includes a dependency-free consistency checker for the committed output tables:

```bash
python verify_published_results.py
```

It checks the 12 seed/method combinations, each architecture's three seeds, the expected stability coefficients, the published means and **sample** standard deviations, the held-out test AUCs, and the recorded 128-slide evaluation size. This verifies the internal consistency of the published CSVs only; it does not independently confirm split integrity, re-run inference, or reproduce model training.

## 8. Limitations

This investigation has several important limitations.

1. **Small number of experimental seeds.** Three runs per architecture and condition provide only a preliminary description of variability. The results do not establish statistically reliable improvements.

2. **Limited architecture coverage.** The extension evaluates CLAM and DSMIL. Its conclusions should not be generalized to ABMIL, TransMIL, or other MIL architectures.

3. **Architecture-specific configurations.** Different checkpoint weights, stability penalties, and merging parameters are used for CLAM and DSMIL. Direct architecture comparisons therefore reflect more than model architecture alone.

4. **Potentially identical candidate selections.** Baseline and stability-aware selection sometimes identify the same candidates, producing the same merged initialization and, in some cases, identical final results. These runs do not provide independent evidence of a method difference.

5. **Incomplete portability.** The original dataset, training artifacts, model checkpoints, and exact environment are not bundled with this extension. The archived scripts also contain environment-specific paths and run identifiers. Full reproduction requires additional setup and artifacts.

6. **Split verification limits.** The evaluation script checks slide-filename disjointness across the specified splits. It does not establish patient-level independence, which would require an additional audit of patient identifiers.

7. **Evaluation protocol limitations.** The final evaluation uses validation-selected checkpoints, but other inherited training scripts contain separate testing pathways. The complete training and experiment history should be audited before making a definitive claim that all model-development decisions were independent of test data.

8. **No broad hyperparameter search.** The reported stability coefficients are experiment-specific settings. Additional coefficients, datasets, and repetitions would be needed to evaluate their general effectiveness.

## 9. Conclusion

This project investigated whether stability-aware candidate selection could improve model merging for digital pathology MIL architectures.

The method penalizes variation across early validation checkpoints before selecting candidates for TIES-style merging.

On CAMELYON16, stability-aware selection reduced observed across-seed variation for CLAM but also reduced its mean held-out test AUC. DSMIL exhibited a negligible mean test AUC improvement without a meaningful reduction in observed variability.

The results do not support a general claim that the stability-aware method is better than standard selection. Instead, they suggest that prioritizing stable early validation performance can affect the balance between variability and predictive accuracy.

Future work should examine more random seeds, additional stability coefficients, patient-level split verification, alternative merging strategies, and independent datasets.

## 10. References and Acknowledgments

This repository builds on the contributions of the original course project and underlying open-source MIL merging implementation.

- **Original project:** [karthikraj2103/Final-project-CSDS463](https://github.com/karthikraj2103/Final-project-CSDS463)
- **Underlying implementation:** [mammadov7/mil_merging](https://github.com/mammadov7/mil_merging)
- **Research context:** Mammadov et al., *Reducing Variability of Multiple Instance Learning Methods for Digital Pathology*, MICCAI 2025.

The original baseline work and implementations are acknowledged separately from the stability-aware selection, reproducibility, and held-out evaluation experiments documented here.


## Project 2 completion audit and additional evaluation

The proposal-requested **minimum held-out test AUC** and **highest validation-selected AUC** are now available for each CLAM/DSMIL fusion method in [`final_results/performance_extended.csv`](final_results/performance_extended.csv). Archived SLURM per-run timings and partial runtime summaries are in [`final_results/training_job_times.csv`](final_results/training_job_times.csv) and [`final_results/timing_summary.csv`](final_results/timing_summary.csv). See [results provenance](final_results/PROVENANCE.md) for definitions, original job IDs, and limitations; these timings exclude offline merging and final inference.

The original proposal's **non-fused standalone MIL** comparator has now been trained and evaluated using [`standalone_comparison.py`](standalone_comparison.py). Six continuation jobs and one shared held-out evaluation completed on Pioneer on November 25, 2026. This comparator selects the best candidate by weighted early mean validation AUC and continues training for the same two effective epochs **without parameter merging**. The new [`heldout_evaluation/standalone_auc_results.csv`](heldout_evaluation/standalone_auc_results.csv) contains all six raw 128-slide test AUC measurements; [`final_results/three_way_summary.csv`](final_results/three_way_summary.csv) reports all three methods across three seeds.

| Architecture | Method | Mean held-out test ROC AUC ± sample SD | Minimum test ROC AUC |
|---|---|---:|---:|
| CLAM | Standalone | 0.7280 ± 0.0490 | 0.6972 |
| CLAM | Baseline fusion | 0.7279 ± 0.0566 | 0.6949 |
| CLAM | Stability-aware fusion | 0.7103 ± 0.0259 | 0.6949 |
| DSMIL | Standalone | 0.6236 ± 0.0524 | 0.5642 |
| DSMIL | Baseline fusion | 0.6205 ± 0.0570 | 0.5618 |
| DSMIL | Stability-aware fusion | 0.6219 ± 0.0571 | 0.5618 |

The standalone method achieved the highest observed mean and minimum test AUC for both architectures. With only three seeds, these are descriptive comparisons, **not** statistical proof of superiority. Archived standalone training/evaluation timings and job identifiers are recorded in [`final_results/standalone_job_times.csv`](final_results/standalone_job_times.csv). Validate all three methods with `python3 verify_three_way.py`; the original checkpoint files remain excluded from GitHub. Historical ABMIL results were not obtained under this matched protocol and are not included.

A detailed [Project 2 completion and submission checklist](PROJECT2_COMPLETION.md) tracks experimental, reporting, attribution, and Canvas requirements. Generate a source-code submission archive locally with `bash package_project2_code.sh` after synchronizing the repository with Pioneer. 


**Reproduction on Pioneer:** `bash submit_standalone_all.sh` can be used to generate a fresh standalone comparison from the original ignored candidate checkpoints, but must **not** be rerun over the completed outputs. 
