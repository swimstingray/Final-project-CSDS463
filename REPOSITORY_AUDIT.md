# Repository audit — November 27, 2026

Audit of the current `main` branch of [swimstingray/Final-project-CSDS463](https://github.com/swimstingray/Final-project-CSDS463), conducted using the GitHub repository tree and selected live source files. This is a **repository completeness and organization audit**, not an independent reproduction of GPU training or a patient-level data audit.

## Contents by purpose

| Purpose | Where | Notes |
|---|---|---|
| Main project overview and research question | [README.md](README.md) | Retains original CLAM/DSMIL methods and provides an index of later experiments |
| Original baseline / stability-aware comparison | [final_results/all_results.csv](final_results/all_results.csv), [summary.csv](final_results/summary.csv), [three_way_summary.csv](final_results/three_way_summary.csv) | Three seeds; distinguish validation from held-out results |
| Additional ABMIL three-way held-out study | [final_results/abmil_three_way_summary.csv](final_results/abmil_three_way_summary.csv), [heldout_evaluation/abmil_test_auc_results.csv](heldout_evaluation/abmil_test_auc_results.csv) | Separate protocol; keep distinct from original CLAM/DSMIL matched test study |
| Experiment A — heterogeneous probability ensemble | [EXPERIMENT_A_README.md](EXPERIMENT_A_README.md), [experiment_a_ensemble.py](experiment_a_ensemble.py), [experiment_a_outputs/](experiment_a_outputs/) | 18 validation comparisons; three seeds, two checkpoint families |
| Experiment B — greedy model soups | [EXPERIMENT_B_README.md](EXPERIMENT_B_README.md), [experiment_b_greedy_soup.py](experiment_b_greedy_soup.py), [experiment_b_outputs/](experiment_b_outputs/), [summary](final_results/experiment_b_summary.csv) | Nine manifest/trace pairs; no test inference in experiment |
| Weighted fusion implementations | [prepare_abmil_weighted_fusions.py](prepare_abmil_weighted_fusions.py), [prepare_other_weighted_fusions.py](prepare_other_weighted_fusions.py), corresponding SLURM wrappers | Training jobs completed on Pioneer; consolidated AUC comparison **not in repository** |
| Paper figures and code | [paper_figures/](paper_figures/), [figure guide](paper_figures/README.md), [captions](paper_figures/FIGURE_CAPTIONS.md) | Six SVGs plus six PNG/PDF pairs stored under `exports/` |
| Paper draft | [report/Project2_Manuscript_Draft.md](report/Project2_Manuscript_Draft.md) | Historical CLAM/DSMIL-focused draft; reconcile with newer work before submission |
| Course submission planning | [PROJECT2_COMPLETION.md](PROJECT2_COMPLETION.md) | Historical checklist; current experiment status added at top |
| Results provenance | [final_results/PROVENANCE.md](final_results/PROVENANCE.md) | Timing limitations and original matched-study provenance |
| Code verification | [GitHub Actions workflow](.github/workflows/project2_verify.yml) | Checks original published CSV consistency, not all later studies |
| Original project / reproduction utilities | top-level `*.py`, `*.slurm`, `*.sh`, original `Results_logs/` | Intentionally kept in existing locations: automated scripts depend on these paths |

## Verified at repository level

- Six numbered figure SVG assets and twelve exported PNG/PDF files are committed.
- Experiment A contains its validation result summary, three seed-level predictions and three manifests.
- Experiment B contains a summary and nine JSON manifest/CSV selection-trace pairs.
- ABMIL has nine held-out evaluation rows (3 methods × 3 seeds), a three-row summary, and prediction-level CSV data.
- `.gitignore` excludes `datasets/`, `weights/`, `*.pth`, `*.pt`, and `*.ckpt`. Historical `weights.zip` is tracked; its contents have **not** been audited in this check.
- The original verifier workflow uses `verify_published_results.py`, `verify_project2_additions.py`, and `verify_three_way.py`.

## Outstanding work and cautions

1. **Weighted fusion performance:** prepare/continuation scripts exist but there is no consolidated per-seed validation/test performance table for all weighted arms. Do not claim an AUC improvement or create a quantitative figure until those results are reviewed.
2. **Paper consistency:** the manuscript draft and some original historical prose predate ABMIL, Experiment A, Experiment B, and the new figures. Refresh the paper's scope, figures, tables, and methods before submission.
3. **Evaluation boundaries:** Experiment A and Experiment B report validation-only AUC. Experiment B reused the same validation data for candidate selection and performance reporting. They are not direct held-out counterparts to the earlier test comparison.
4. **Statistical uncertainty:** only three seeds for the held-out results. SD is descriptive, not a confidence interval, and stability is not a demonstrated general property.
5. **Reproducibility limits:** model weights, original features, exact pinned environment, and patient-level split independence are not independently established by GitHub. Absolute Pioneer paths embedded in some CSV records are environment-specific and should not be interpreted as portable paths.
6. **Code organization:** many scripts still live in the repository root. A physical move could break SLURM paths, imports, links, and archived workflows. Prefer this map over moving working research code just before the deadline.
7. **Submission artifacts:** a GitHub repository is not the same as submitting the finalized course PDF, presentation, or source archive. Verify deliverables separately against course instructions.

## Quick checks

From an environment with the relevant Python dependencies:

```bash
python verify_published_results.py
python verify_project2_additions.py
python verify_three_way.py
```

For a local checkout, confirm synchronization with `git status -sb` and `git log -1 --oneline`.

This index intentionally preserves original methods and results rather than rewriting or renaming historical experiment files.
