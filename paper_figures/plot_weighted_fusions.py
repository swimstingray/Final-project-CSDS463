#!/usr/bin/env python3
"""Create Figure 7 and a paired validation comparison for weighted MIL fusion.

Reads completed, validation-selected metrics; performs no model inference and no
test-data access. Run on Pioneer from repository root:
    python paper_figures/plot_weighted_fusions.py
Refuses to overwrite existing exports/results.
"""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEIGHTED = ROOT / "final_results" / "weighted_fusion_validation_results.csv"
ORIGINAL = ROOT / "final_results" / "all_results.csv"
ABMIL = ROOT / "heldout_evaluation" / "abmil_test_auc_results.csv"
TABLE = ROOT / "final_results" / "weighted_fusion_vs_baseline_validation.csv"
PNG = ROOT / "paper_figures" / "exports" / "figure_07_weighted_fusion_validation.png"
PDF = PNG.with_suffix(".pdf")
ARCH = ("CLAM", "DSMIL", "ABMIL")
METHODS = ("auc_weighted", "stability_weighted")

def read(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

def auc(value):
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Invalid AUC " + str(value))
    return result

def main():
    for path in (TABLE, PNG, PDF):
        if path.exists():
            raise FileExistsError("Refusing overwrite: " + str(path))

    baseline = {}
    for source in (ORIGINAL, ABMIL):
        for row in read(source):
            if row["method"] != "Baseline":
                continue
            key = (row["architecture"].upper(), int(row["seed"]))
            if key in baseline:
                raise ValueError("Duplicate baseline: " + str(key))
            baseline[key] = auc(row["validation_auc"])

    expected = {(a, seed) for a in ARCH for seed in (1, 2, 3)}
    if set(baseline) != expected:
        raise ValueError("Expected 9 baseline records; got " + str(set(baseline)))

    weighted = {}
    for row in read(WEIGHTED):
        key = (row["architecture"].upper(), int(row["seed"]), row["method"])
        if key in weighted or (key[0], key[1]) not in expected or key[2] not in METHODS:
            raise ValueError("Unexpected or repeated weighted record: " + str(key))
        weighted[key] = auc(row["best_validation_auc"])
    expected_weighted = {(a, seed, method) for a, seed in expected for method in METHODS}
    if set(weighted) != expected_weighted:
        raise ValueError("Expected exactly 18 weighted records")

    rows = []
    data = defaultdict(lambda: defaultdict(list))
    for arch in ARCH:
        for seed in (1, 2, 3):
            b = baseline[(arch, seed)]
            for method in METHODS:
                val = weighted[(arch, seed, method)]
                delta = val - b
                rows.append(dict(architecture=arch, seed=seed, method=method,
                                 equal_weight_validation_auc=b,
                                 weighted_validation_auc=val,
                                 delta_vs_equal_weight=delta))
                data[arch]["baseline"].append(b)
                data[arch][method].append(val)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(ARCH))
    width = .24
    styles = [("baseline", "Equal-weight TIES", "#657789"),
              ("auc_weighted", "AUC-weighted TIES", "#2777a3"),
              ("stability_weighted", "Stability-weighted TIES", "#218a81")]
    for i, (method, label, color) in enumerate(styles):
        means = [statistics.mean(data[a][method]) for a in ARCH]
        stds = [statistics.stdev(data[a][method]) for a in ARCH]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=4,
               label=label, color=color)
    ax.set_xticks(x, ARCH)
    ax.set_ylabel("Validation ROC AUC")
    ax.set_ylim(0, .90)
    ax.set_title("Figure 7. Equal-weight versus weighted model merging")
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    fig.text(.5, .01,
             "Mean ± sample SD across three matched seeds. Validation only; not held-out test performance.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))

    PNG.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig.savefig(PNG, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(PDF, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    for arch in ARCH:
        for method in METHODS:
            v = [weighted[(arch, seed, method)] - baseline[(arch, seed)]
                 for seed in (1, 2, 3)]
            print(f"{arch:5} {method:18} mean_delta={statistics.mean(v):+.6f}")
    for path in (TABLE, PNG, PDF):
        print("SAVED:", path.relative_to(ROOT))
    print("PASS: 18 weighted versus equal-weight validation comparisons; no test inference")

if __name__ == "__main__":
    main()
