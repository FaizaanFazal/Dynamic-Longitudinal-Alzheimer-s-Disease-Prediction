#!/usr/bin/env python3
"""Append and execute only the Phase 5 overflow-repair notebook cells.

The six plotting cells are self-contained: each reads existing aggregate CSV
outputs and never depends on earlier notebook state. Existing cells and their
outputs are verified unchanged after the append-and-execute operation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks/Phase5/Phase5_SecondaryAnalysesAndFinalReporting.ipynb"
)
REPAIR_TAG = "phase5_text_overflow_repair_20260828"


FIGURE_01 = r'''# Fixed Figure 1: keep the title clear of cohort boxes (self-contained cell)
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

ROOT = Path.cwd().resolve()
while not (ROOT / "results" / "csvs" / "phase5").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Could not locate the project root")
    ROOT = ROOT.parent

table = pd.read_csv(ROOT / "results/csvs/phase5/cohort_characteristics.csv")
out_dir = ROOT / "results/plots/phase5"
stem = "figure_01_study_and_cohort_flow_fixed"

fig, ax = plt.subplots(figsize=(8.2, 5.6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

colors = ["#d7e9f7", "#e1f2e8", "#f7ead6"]
y_positions = [0.64, 0.36, 0.08]
for color, y, row in zip(colors, y_positions, table.itertuples()):
    box = patches.FancyBboxPatch(
        (0.06, y), 0.88, 0.20,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=color, edgecolor="#34495e", linewidth=2.0,
    )
    ax.add_patch(box)
    verification = "verified age" if bool(row.age_verified) else "age proxy unverified"
    ax.text(
        0.5, y + 0.135,
        f"{row.cohort}: {row.index_participants:,} CDR 0.5 index participants",
        ha="center", va="center", fontsize=13, fontweight="bold",
    )
    ax.text(
        0.5, y + 0.065,
        f"{row.role}  •  {verification}",
        ha="center", va="center", fontsize=10.5,
    )

ax.text(
    0.5, 0.965, "Study cohorts and analysis roles",
    ha="center", va="top", fontsize=18, fontweight="bold",
)
fig.subplots_adjust(left=0.03, right=0.97, top=0.99, bottom=0.02)
png = out_dir / f"{stem}.png"
svg = out_dir / f"{stem}.svg"
fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
fig.savefig(svg, bbox_inches="tight", pad_inches=0.08, facecolor="white")
print(f"Saved {png.relative_to(ROOT)} at 600 DPI")
plt.show()
'''


FIGURE_02 = r'''# Fixed Supplementary Figure S1: enlarge CDR nodes and contain every label
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

ROOT = Path.cwd().resolve()
while not (ROOT / "results" / "csvs" / "phase1").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Could not locate the project root")
    ROOT = ROOT.parent

transition = pd.read_csv(ROOT / "results/csvs/phase1/cdr_transition_summary_all_cohorts.csv")
transition = transition[transition["transition_label"].isin(["CDR 0.5 -> 0", "CDR 0.5 -> >=1"])]
pivot = transition.pivot(index="cohort", columns="transition_label", values="events")
out_dir = ROOT / "results/plots/phase5"
stem = "figure_02_cdr_state_transition_diagram_fixed"

fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.7))
positions = {"CDR 0": (0.14, 0.48), "CDR 0.5": (0.50, 0.48), "CDR ≥1": (0.86, 0.48)}
for ax, cohort in zip(axes, ["adni", "oasis3", "aibl"]):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    for label, (x, y) in positions.items():
        circle = patches.Circle(
            (x, y), 0.135, facecolor="#eef3f7",
            edgecolor="#2c3e50", linewidth=2.0,
        )
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=10.5, fontweight="bold")

    improvement = int(pivot.loc[cohort, "CDR 0.5 -> 0"])
    progression = int(pivot.loc[cohort, "CDR 0.5 -> >=1"])
    ax.annotate(
        "", xy=(0.285, 0.54), xytext=(0.355, 0.54),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#2a9d8f"),
    )
    ax.annotate(
        "", xy=(0.715, 0.42), xytext=(0.645, 0.42),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#d95f02"),
    )
    ax.text(0.32, 0.66, f"n={improvement}", ha="center", color="#167c6b", fontsize=10.5)
    ax.text(0.68, 0.29, f"n={progression}", ha="center", color="#a94600", fontsize=10.5)
    title = "OASIS-3" if cohort == "oasis3" else cohort.upper()
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)

fig.suptitle("Observed CDR transitions from state 0.5", fontsize=18, fontweight="bold", y=0.98)
fig.subplots_adjust(left=0.025, right=0.975, top=0.78, bottom=0.08, wspace=0.10)
png = out_dir / f"{stem}.png"
svg = out_dir / f"{stem}.svg"
fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
fig.savefig(svg, bbox_inches="tight", pad_inches=0.08, facecolor="white")
print(f"Saved {png.relative_to(ROOT)} at 600 DPI")
plt.show()
'''


FIGURE_03 = r'''# Fixed Figure 2: two-row locked-validation workflow with contained box labels
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches

ROOT = Path.cwd().resolve()
while not (ROOT / "results" / "plots" / "phase5").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Could not locate the project root")
    ROOT = ROOT.parent

out_dir = ROOT / "results/plots/phase5"
stem = "figure_03_dynamic_external_validation_workflow_fixed"
stages = [
    "ADNI\nlandmarks",
    "Participant-grouped\nnested validation",
    "Frozen D1\nmodel",
    "Outcome-hidden\nOASIS prediction",
    "Untouched\nexternal validation",
    "Separate\nrecalibration",
]
positions = [(0.18, 0.62), (0.50, 0.62), (0.82, 0.62),
             (0.82, 0.30), (0.50, 0.30), (0.18, 0.30)]
colors = ["#d9eef2", "#d9eef2", "#d9eef2", "#f4e5cf", "#f4e5cf", "#eadcf2"]

fig, ax = plt.subplots(figsize=(12.0, 6.0))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

box_w, box_h = 0.25, 0.17
for (x, y), label, color in zip(positions, stages, colors):
    box = patches.FancyBboxPatch(
        (x - box_w / 2, y - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=color, edgecolor="#34495e", linewidth=2.0,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=11.5, linespacing=1.25)

# Sequential arrows: top row left-to-right, then down, then bottom row right-to-left.
for start, end in [(0, 1), (1, 2), (3, 4), (4, 5)]:
    x1, y1 = positions[start]
    x2, y2 = positions[end]
    direction = 1 if x2 > x1 else -1
    ax.annotate(
        "", xy=(x2 - direction * box_w / 2 - direction * 0.01, y2),
        xytext=(x1 + direction * box_w / 2 + direction * 0.01, y1),
        arrowprops=dict(arrowstyle="->", lw=2.1, color="#34495e"),
    )
ax.annotate(
    "", xy=(positions[3][0], positions[3][1] + box_h / 2 + 0.015),
    xytext=(positions[2][0], positions[2][1] - box_h / 2 - 0.015),
    arrowprops=dict(arrowstyle="->", lw=2.1, color="#34495e"),
)

ax.text(
    0.5, 0.945, "Dynamic model development and locked external-validation workflow",
    ha="center", va="top", fontsize=18, fontweight="bold",
)
ax.text(
    0.5, 0.075,
    "Untouched validation remains primary; OASIS-based recalibration is model updating",
    ha="center", va="center", fontsize=11.5, color="#6c3483",
)
fig.subplots_adjust(left=0.025, right=0.975, top=0.99, bottom=0.02)
png = out_dir / f"{stem}.png"
svg = out_dir / f"{stem}.svg"
fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
fig.savefig(svg, bbox_inches="tight", pad_inches=0.08, facecolor="white")
print(f"Saved {png.relative_to(ROOT)} at 600 DPI")
plt.show()
'''


FIGURE_08 = r'''# Fixed Supplementary Figure S3: wrap labels and reserve panel margins
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
while not (ROOT / "results" / "csvs" / "phase5").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Could not locate the project root")
    ROOT = ROOT.parent

shift = pd.read_csv(ROOT / "results/csvs/phase5/distribution_shift_synthesis.csv")
continuous = shift[shift["shift_type"].eq("continuous_predictor")].head(8).copy()
event = shift[shift["shift_type"].eq("event_rate") & shift["model_id"].eq("D1")].copy()
continuous["display_item"] = continuous["item"].str.replace("_", " ").map(
    lambda value: "\n".join(textwrap.wrap(value, width=22))
)
event["display_item"] = (
    event["horizon_months"].astype(int).astype(str)
    + "-month " + event["item"].str.replace("_", " ")
)
out_dir = ROOT / "results/plots/phase5"
stem = "figure_08_predictor_and_event_rate_shift_fixed"

fig, axes = plt.subplots(
    1, 2, figsize=(12.0, 6.5),
    gridspec_kw={"width_ratios": [1.35, 1.0]},
)
axes[0].barh(
    continuous["display_item"], continuous["difference_metric"],
    color=np.where(continuous["difference_metric"].ge(0), "#3182bd", "#e6550d"),
)
axes[0].axvline(0, color="#555555", linewidth=1.2)
axes[0].invert_yaxis()
axes[0].set_xlabel("Standardized mean difference\n(OASIS-3 − ADNI)", fontsize=11.5)
axes[0].set_title("Predictor shift", fontsize=14, pad=10)

axes[1].barh(event["display_item"], event["difference_metric"], color="#756bb1")
axes[1].axvline(1, color="#555555", linestyle="--", linewidth=1.4)
axes[1].set_xlabel("OASIS-3 / ADNI\nobserved-rate ratio", fontsize=11.5)
axes[1].set_title("Event-rate shift", fontsize=14, pad=10)
axes[1].invert_yaxis()

for ax in axes:
    ax.tick_params(axis="both", labelsize=9.5)
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Development-to-external dataset shift", fontsize=18, fontweight="bold", y=0.975)
fig.subplots_adjust(left=0.22, right=0.97, top=0.82, bottom=0.18, wspace=0.72)
png = out_dir / f"{stem}.png"
svg = out_dir / f"{stem}.svg"
fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.10, facecolor="white")
fig.savefig(svg, bbox_inches="tight", pad_inches=0.10, facecolor="white")
print(f"Saved {png.relative_to(ROOT)} at 600 DPI")
plt.show()
'''


FIGURE_10 = r'''# Fixed Supplementary Figure S4: separate suptitle, panel titles, and legend
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
while not (ROOT / "results" / "csvs" / "phase5").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Could not locate the project root")
    ROOT = ROOT.parent

trajectories = pd.read_csv(ROOT / "results/csvs/phase5/representative_dynamic_risk_trajectories.csv")
out_dir = ROOT / "results/plots/phase5"
stem = "figure_10_representative_dynamic_risk_trajectories_fixed"
order = ["improvement-risk-increase", "progression-risk-increase", "stable-risk-example"]
titles = {
    "improvement-risk-increase": "Improvement-risk\nincrease",
    "progression-risk-increase": "Progression-risk\nincrease",
    "stable-risk-example": "Stable-risk\nexample",
}

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.9), sharey=True)
for panel, (ax, label) in enumerate(zip(axes, order)):
    subset = trajectories[trajectories["trajectory_label"].eq(label)].sort_values("elapsed_years_from_index")
    ax.plot(
        subset["elapsed_years_from_index"], subset["probability_improvement"],
        "o-", label="Improvement", color="#2a9d8f", linewidth=2, markersize=5,
    )
    ax.plot(
        subset["elapsed_years_from_index"], subset["probability_progression"],
        "s-", label="Progression", color="#d95f02", linewidth=2, markersize=5,
    )
    ax.set_title(titles[label], fontsize=12.5, pad=10, linespacing=1.15)
    ax.set_xlabel("Years from index", fontsize=10.5)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.96, chr(65 + panel), transform=ax.transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold")

axes[0].set_ylabel("Frozen D1 36-month probability", fontsize=11)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.81),
           ncol=2, frameon=False, fontsize=10)
fig.suptitle("Representative out-of-fold dynamic risk trajectories",
             fontsize=18, fontweight="bold", y=0.98)
fig.subplots_adjust(left=0.08, right=0.98, top=0.69, bottom=0.17, wspace=0.16)
png = out_dir / f"{stem}.png"
svg = out_dir / f"{stem}.svg"
fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
fig.savefig(svg, bbox_inches="tight", pad_inches=0.08, facecolor="white")
print(f"Saved {png.relative_to(ROOT)} at 600 DPI")
plt.show()
'''


FIGURE_11 = r'''# Fixed Supplementary Figure S5: reserve the right margin for the secondary axis
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
while not (ROOT / "results" / "csvs" / "phase5").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Could not locate the project root")
    ROOT = ROOT.parent

timing = pd.read_csv(ROOT / "results/csvs/phase5/reversion_timing_and_relapse.csv")
definitions = ["apparent_reversion", "confirmed_reversion", "sustained_reversion", "strongest_reversion"]
subset = timing.set_index("definition").loc[definitions].reset_index()
labels = ["Apparent", "Confirmed", "Sustained", "Strongest"]
out_dir = ROOT / "results/plots/phase5"
stem = "figure_11_reversion_trajectory_and_relapse_fixed"

fig, ax1 = plt.subplots(figsize=(9.2, 5.8))
x = np.arange(len(subset))
bars = ax1.bar(
    x, subset["participants"], width=0.72,
    color=["#b3cde3", "#8c96c6", "#8856a7", "#810f7c"],
)
ax1.set_xticks(x, labels)
ax1.set_ylabel("Participants", fontsize=12.5, labelpad=8)
ax1.set_ylim(0, 126)
ax1.tick_params(axis="both", labelsize=10.5)
ax1.grid(axis="y", alpha=0.18)
ax1.spines[["top", "right"]].set_visible(False)

for bar, count, relapse in zip(
    bars, subset["participants"], subset["later_relapse_or_lower_state"]
):
    center = bar.get_x() + bar.get_width() / 2
    ax1.text(center, bar.get_height() + 3, str(int(count)),
             ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.text(center, 7, f"later relapse: {int(relapse)}",
             ha="center", va="bottom", rotation=90, fontsize=8.5, color="#333333")

ax2 = ax1.twinx()
ax2.plot(
    x, subset["time_median_months"], "o-",
    color="#e6550d", linewidth=2.2, markersize=7,
    label="Median qualifying time",
)
ax2.set_ylim(32, 65)
ax2.set_ylabel("Median qualifying time (months)", fontsize=12.5, labelpad=12)
ax2.tick_params(axis="y", labelsize=10.5)
ax2.spines["top"].set_visible(False)
ax2.legend(loc="upper right", frameon=False, fontsize=9.5)

ax1.set_title("Diagnosis-reversion confirmation and relapse",
              fontsize=18, fontweight="bold", pad=18)
fig.subplots_adjust(left=0.11, right=0.84, top=0.84, bottom=0.17)
png = out_dir / f"{stem}.png"
svg = out_dir / f"{stem}.svg"
fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.10, facecolor="white")
fig.savefig(svg, bbox_inches="tight", pad_inches=0.10, facecolor="white")
print(f"Saved {png.relative_to(ROOT)} at 600 DPI")
plt.show()
'''


def stable_cell_state(cell: nbformat.NotebookNode) -> str:
    """Serialize the parts that must remain unchanged for an existing cell."""
    return json.dumps(
        {
            "cell_type": cell.get("cell_type"),
            "source": cell.get("source", ""),
            "execution_count": cell.get("execution_count"),
            "outputs": cell.get("outputs", []),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    if any(REPAIR_TAG in cell.get("metadata", {}).get("tags", []) for cell in notebook.cells):
        raise RuntimeError(
            "Repair cells already exist; refusing to duplicate or rerun them."
        )

    original_count = len(notebook.cells)
    original_states = [stable_cell_state(copy.deepcopy(cell)) for cell in notebook.cells]

    note = nbformat.v4.new_markdown_cell(
        "## Overflow-safe manuscript figure variants (added 2026-08-28)\n\n"
        "The following self-contained cells regenerate only the figures that failed "
        "the visual text-overflow audit. They read existing aggregate result CSVs, "
        "do not rerun any analysis, preserve every original figure, and save new "
        "`*_fixed.png` files at 600 DPI (plus matching SVG files). Figures 04, 05, "
        "06, 07, and 09 passed the audit and are intentionally not regenerated."
    )
    note.metadata["tags"] = [REPAIR_TAG, "documentation"]
    notebook.cells.append(note)

    code_cells = [
        ("figure_01", FIGURE_01),
        ("figure_02", FIGURE_02),
        ("figure_03", FIGURE_03),
        ("figure_08", FIGURE_08),
        ("figure_10", FIGURE_10),
        ("figure_11", FIGURE_11),
    ]
    repair_indices: list[int] = []
    for name, source in code_cells:
        cell = nbformat.v4.new_code_cell(source)
        cell.metadata["tags"] = [REPAIR_TAG, name, "fixed_figure_only"]
        notebook.cells.append(cell)
        repair_indices.append(len(notebook.cells) - 1)

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="clinical_env",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    with client.setup_kernel():
        for cell_index in repair_indices:
            client.execute_cell(notebook.cells[cell_index], cell_index)

    after_states = [stable_cell_state(cell) for cell in notebook.cells[:original_count]]
    if original_states != after_states:
        raise AssertionError("An original notebook cell or output changed unexpectedly")

    nbformat.write(notebook, NOTEBOOK)
    reloaded = nbformat.read(NOTEBOOK, as_version=4)
    reloaded_states = [stable_cell_state(cell) for cell in reloaded.cells[:original_count]]
    if original_states != reloaded_states:
        raise AssertionError("Original outputs changed during notebook serialization")

    executed = [reloaded.cells[index].get("execution_count") for index in repair_indices]
    print(
        json.dumps(
            {
                "notebook": str(NOTEBOOK.relative_to(ROOT)),
                "original_cells_preserved": original_count,
                "new_markdown_cells": 1,
                "new_code_cells_executed_only": len(repair_indices),
                "new_execution_counts": executed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
