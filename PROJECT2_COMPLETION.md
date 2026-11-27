# CSDS/EBME 463 Project 2 — submission and research audit

> **Historical scope note (updated November 27, 2026):** The completion status below documents the original CLAM/DSMIL Project 2 comparison. Subsequent experiments are now committed: matched ABMIL standalone/baseline/stability evaluation ([summary](final_results/abmil_three_way_summary.csv)), [Experiment A](experiment_a_outputs/validation_auc_results.csv), [Experiment B](final_results/experiment_b_summary.csv), and [six final paper figures](paper_figures/README.md). Therefore the historical statement that ABMIL was not reevaluated applies only to the original CLAM/DSMIL protocol, not to the current overall repository. See [repository audit](REPOSITORY_AUDIT.md) for a current file map and outstanding tasks.


The course requires: a standard research paper with cover page, abstract (maximum 250 words), introduction, methods, results with figures/tables, discussion, conclusion, references, and appendices; no more than **six body pages** (cover, references, appendices excluded); a separate source-code ZIP; and a PowerPoint presentation. The PDF report is due December 1, 2026, at 2:30 PM EST, and PPTX material November 30, 2026, at 2:30 PM EST, unless Canvas announcements supersede the posted document.

## Verified completed

- CAMELYON16 classification using CLAM and DSMIL and 512-dimensional patch features; 215 training, 54 validation, and 128 held-out slides.
- Early candidate validation histories, stability-aware ranking, top-three TIES-style parameter merging, and two effective continuation training epochs.
- Three seeds per architecture across **three completed methods**: standalone MIL, baseline fusion, and stability-aware fusion, totaling **18** held-out test ROC-AUC measurements (12 fusion + six standalone).
- Mean, sample SD, minimum test AUC, highest validation-selected AUC, paired test differences, and observed CLAM/DSMIL trade-offs.
- Archived SLURM times for five-candidate generation and fused continuation runs, with original job IDs. This is *partial workflow time*, not a complete cost measurement.
- Updated README, code audit and CSV verifier. **ABMIL was neither modified nor reevaluated** in the stability-aware extension; the individual proposal required an attention-based MIL approach, not a specific ABMIL experiment.

## Experimental completion and remaining nonexperimental tasks

1. **Standalone MIL arm completed.** Six model continuation jobs (4313829–4313834) and the shared held-out evaluator (4313835) completed on Pioneer on November 25, 2026, all exit 0:0. Their results are committed at `heldout_evaluation/standalone_auc_results.csv` and `final_results/three_way_summary.csv`; GPU jobs need no resubmission. Results are descriptive, with three seeds per method.
2. **Fully measured computation time.** Offline checkpoint-merging time, queue time and per-method evaluation overhead were not captured. Present current timings only as measured candidate + continuation times.
3. **Group attribution.** The second author's name and member-by-member technical contributions must be confirmed. Leave placeholders in the draft until confirmed.
4. **Course submission.** The report and presentation drafts generated in ChatGPT need final review and Canvas submission. The report is nine pages total: cover + six body pages + references + appendix. The presentation is 14 slides.
5. **Source ZIP.** After synchronizing Pioneer with the latest main, run `bash package_project2_code.sh` to create a source-code ZIP from tracked source files; submit it separately as required.

## Scientific claims that should not be overstated

- CLAM baseline/stability-aware test mean: 0.7279/0.7103; SD: 0.0566/0.0259. The worst seed stays at 0.6949.
- DSMIL baseline/stability-aware test mean: 0.6205/0.6219; SD: 0.0570/0.0571. The worst seed stays at 0.5618.
- CLAM's smaller SD mostly reflects the drop of its best-performing seed, *not* improvement of the worst seed. With n=3 there is no general statistically established reliability benefit.
- Historical ABMIL test AUC 0.8802 is from an inherited single-run baseline; it is not comparable under the current fusion protocol.

The three-method experimental comparison is complete and published, but the entire course submission should not be called finished until other required materials and human submission checks are completed.


**Reproduction only:** `bash submit_standalone_all.sh` submits the standalone training and evaluation workflow, but the seven November 25 jobs have already completed; do not rerun or overwrite their outputs. Reproducibility checks are documented in `final_results/PROVENANCE.md`.
