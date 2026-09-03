#!/usr/bin/env python3
"""Append and execute only the enlarged-text Figure S5 annotation-QC cell."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Phase5/Phase5_SecondaryAnalysesAndFinalReporting.ipynb"
CELL_TAG = "phase5_figure11_enlarged_annotation_qc_20260828"


FIGURE_11_QC = r'''# Figure S5 enlarged-text QC: separate count labels from the time trajectory
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
definitions = [
    "apparent_reversion", "confirmed_reversion",
    "sustained_reversion", "strongest_reversion",
]
subset = timing.set_index("definition").loc[definitions].reset_index()
labels = ["Apparent", "Confirmed", "Sustained", "Strongest"]
out_dir = ROOT / "results/plots/phase5"
stem = "figure_11_reversion_trajectory_and_relapse_fixed"

fig, ax1 = plt.subplots(figsize=(10.6, 6.6))
x = np.arange(len(subset))
bars = ax1.bar(
    x, subset["participants"], width=0.72,
    color=["#b3cde3", "#8c96c6", "#8856a7", "#810f7c"],
    zorder=2,
)
ax1.set_xticks(x, labels, fontsize=12.5)
ax1.set_ylabel("Participants", fontsize=15, labelpad=10)
ax1.set_ylim(0, 126)
ax1.tick_params(axis="y", labelsize=12.5)
ax1.grid(axis="y", alpha=0.18, zorder=0)
ax1.spines[["top", "right"]].set_visible(False)

for bar, count, relapse in zip(
    bars, subset["participants"], subset["later_relapse_or_lower_state"]
):
    center = bar.get_x() + bar.get_width() / 2
    # Keep counts just inside each bar so the orange trajectory cannot cross them.
    ax1.text(
        center, bar.get_height() - 2.5, str(int(count)),
        ha="center", va="top", fontsize=13.5, fontweight="bold", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2),
    )
    ax1.text(
        center, 7, f"later relapse: {int(relapse)}",
        ha="center", va="bottom", rotation=90, fontsize=10.5,
        color="#333333", zorder=5,
    )

ax2 = ax1.twinx()
line = ax2.plot(
    x, subset["time_median_months"], "o-",
    color="#e6550d", linewidth=2.5, markersize=8,
    label="Median qualifying time", zorder=4,
)
ax2.set_ylim(32, 65)
ax2.set_ylabel("Median qualifying time (months)", fontsize=15, labelpad=14)
ax2.tick_params(axis="y", labelsize=12.5)
ax2.spines["top"].set_visible(False)

# The approved title remains exactly 18 points; only non-title typography is enlarged.
fig.suptitle(
    "Diagnosis-reversion confirmation and relapse",
    fontsize=18, fontweight="bold", y=0.965,
)
fig.legend(
    line, ["Median qualifying time"], loc="upper center",
    bbox_to_anchor=(0.5, 0.87), frameon=False, fontsize=12.5,
    handlelength=2.7,
)
fig.subplots_adjust(left=0.12, right=0.80, top=0.70, bottom=0.17)

png = out_dir / f"{stem}.png"
svg = out_dir / f"{stem}.svg"
fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.10, facecolor="white")
fig.savefig(svg, bbox_inches="tight", pad_inches=0.10, facecolor="white")
print(f"Saved {png.relative_to(ROOT)} at 600 DPI after enlarged-text annotation QC")
plt.show()
'''


def stable_cell_state(cell: nbformat.NotebookNode) -> str:
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
    if any(CELL_TAG in cell.get("metadata", {}).get("tags", []) for cell in notebook.cells):
        raise RuntimeError("Figure S5 annotation-QC cell already exists; refusing to rerun it")

    existing_count = len(notebook.cells)
    existing_states = [stable_cell_state(copy.deepcopy(cell)) for cell in notebook.cells]

    cell = nbformat.v4.new_code_cell(FIGURE_11_QC)
    cell.metadata["tags"] = [CELL_TAG, "figure_11", "fixed_figure_only"]
    notebook.cells.append(cell)
    cell_index = len(notebook.cells) - 1

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="clinical_env",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    with client.setup_kernel():
        client.execute_cell(notebook.cells[cell_index], cell_index)

    after_states = [stable_cell_state(cell) for cell in notebook.cells[:existing_count]]
    if existing_states != after_states:
        raise AssertionError("An existing notebook cell or stored output changed unexpectedly")

    nbformat.write(notebook, NOTEBOOK)
    reloaded = nbformat.read(NOTEBOOK, as_version=4)
    reloaded_states = [stable_cell_state(cell) for cell in reloaded.cells[:existing_count]]
    if existing_states != reloaded_states:
        raise AssertionError("Existing outputs changed during notebook serialization")

    print(f"existing_cells_preserved: {existing_count}")
    print("new_code_cells_executed_only: 1")
    print(f"new_execution_count: {reloaded.cells[cell_index].get('execution_count')}")
    print(f"new_outputs: {len(reloaded.cells[cell_index].get('outputs', []))}")


if __name__ == "__main__":
    main()
