# Paper figures — CSDS463 MIL fusion project

This folder contains all six editable **SVG figures**, their final **300-dpi PNG and vector PDF exports** in [`exports/`](exports/), paper-ready captions, and a plotting script. The exports were generated on Pioneer from the committed result tables.

## Figure index

| Figure | File | Section | Scope |
|---|---|---|---|
| 1 | [Experimental workflow](figure_01_workflow.svg) | Methods | Study design |
| 2 | [Held-out methods](figure_02_heldout_methods.svg) | Results | Held-out test ROC AUC, 3 seeds |
| 3 | [Stability trade-off](figure_03_stability_tradeoff.svg) | Results/discussion | Held-out mean versus across-seed SD |
| 4 | [Paired seed effects](figure_04_seed_paired_test_changes.svg) | Results/supplement | Held-out paired seed changes |
| 5 | [Experiment A](figure_05_experiment_a_ensembles.svg) | Exploratory results | Validation only |
| 6 | [Experiment B](figure_06_experiment_b_greedy_soups.svg) | Exploratory results | Validation only |

## Provenance and interpretation

- Figures 2–4: existing `final_results/three_way_summary.csv`, `final_results/summary.csv`, and `final_results/all_results.csv`. These were already committed from previous CLAM/DSMIL evaluations. The test split has 128 slides. Error bars are **sample SD across only three seeds**, never confidence intervals.
- Figure 5: `experiment_a_outputs/validation_auc_results.csv` created on Pioneer from the completed array job `4318825`. The SVG is based on the six-decimal stdout measurements supplied in the project conversation, so it is a visualization snapshot; generate final PNG/PDF from the **on-cluster CSV** to avoid rounding differences.
- Figure 6: `final_results/experiment_b_summary.csv` created on Pioneer from all nine manifests of job `4318839`. SVG points are based on the six-decimal values supplied in the project conversation. CLAM seed 1 improved from 0.794034 to 0.796875; the other eight soups selected a single checkpoint.
- Figure 1: methods schematic, not numerical data. All architectures were not evaluated with precisely identical training/continuation protocols.

**Do not interpret A or B plots as held-out test outcomes.** Experiment B selected checkpoints on the same validation slides used in its AUC comparison. The held-out analyses have only three seeds and their sample standard deviations are descriptive.

**Outstanding figures:** Weighted TIES variants (AUC-weighted versus stability-weighted) have completed training but require a verified results table before plotting. Matched ABMIL held-out results are committed at [`../heldout_evaluation/abmil_test_auc_results.csv`](../heldout_evaluation/abmil_test_auc_results.csv) but are not included in Figure 2; do not merge them into the CLAM/DSMIL plot without clearly documenting the protocol differences.

## Generating local PNG/PDF exports

Use the tracked plotting script from the repository root on Pioneer:

```bash
python paper_figures/plot_from_pioneer.py
```

The script reads committed CSVs and writes six 300-dpi PNGs and six vector PDFs. It intentionally refuses to overwrite an existing nonempty `exports/` folder; generate into a new clean working copy when reproducing. The files in this GitHub folder are **SVG companions**, not byte-identical copies of the local package exports.

## Captions

See [FIGURE_CAPTIONS.md](FIGURE_CAPTIONS.md) for paper-ready captions and limitations.

## Repository organization

- `paper_figures/`: shareable figure assets and caption documentation.
- `final_results/`: existing summary tables and result figures.
- [`../experiment_a_outputs/`](../experiment_a_outputs/), [`../experiment_b_outputs/`](../experiment_b_outputs/): committed validation results, manifests, and selection traces. Model checkpoints and feature files remain excluded.
- [`plot_from_pioneer.py`](plot_from_pioneer.py): regenerate figures from saved CSV results.
