#!/usr/bin/env python3
"""Append and execute only the second-pass Figure S4 layout-repair cell."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Phase5/Phase5_SecondaryAnalysesAndFinalReporting.ipynb"
CELL_TAG = "phase5_figure10_second_pass_20260828"


FIGURE_10_SECOND_PASS = r'''# Figure S4 second-pass QC: isolate title, legend, and panel headings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
while not (ROOT / "results" / "csvs" / "phase5").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Could not locate the project root")
    ROOT = ROOT.parent

trajectories = pd.read_csv(
    ROOT / "results/csvs/phase5/representative_dynamic_risk_trajectories.csv"
)
out_dir = ROOT / "results/plots/phase5"
stem = "figure_10_representative_dynamic_risk_trajectories_fixed"
order = ["improvement-risk-increase", "progression-risk-increase", "stable-risk-example"]
titles = {
    "improvement-risk-increase": "Improvement risk increases",
    "progression-risk-increase": "Progression risk increases",
    "stable-risk-example": "Stable-risk example",
}

fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.0), sharey=True)
for panel, (ax, label) in enumerate(zip(axes, order)):
    subset = trajectories.loc[
        trajectories["trajectory_label"].eq(label)
    ].sort_values("elapsed_years_from_index")
    ax.plot(
        subset["elapsed_years_from_index"], subset["probability_improvement"],
        "o-", label="Improvement", color="#2a9d8f", linewidth=2, markersize=5,
    )
    ax.plot(
        subset["elapsed_years_from_index"], subset["probability_progression"],
        "s-", label="Progression", color="#d95f02", linewidth=2, markersize=5,
    )
    ax.text(
        0.5, 1.035, titles[label], transform=ax.transAxes,
        ha="center", va="bottom", fontsize=11, fontweight="semibold",
    )
    ax.set_xlabel("Years from index", fontsize=10.5, labelpad=6)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.02, 0.96, chr(65 + panel), transform=ax.transAxes,
        ha="left", va="top", fontsize=11, fontweight="bold",
    )

axes[0].set_ylabel("Frozen D1 36-month probability", fontsize=10.5, labelpad=8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.845),
    ncol=2, columnspacing=2.4, handlelength=2.6, frameon=False, fontsize=10,
)
fig.suptitle(
    "Representative out-of-fold dynamic risk trajectories",
    fontsize=17, fontweight="bold", y=0.965,
)
fig.subplots_adjust(left=0.085, right=0.985, top=0.70, bottom=0.16, wspace=0.16)

png = out_dir / f"{stem}.png"
svg = out_dir / f"{stem}.svg"
fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.10, facecolor="white")
fig.savefig(svg, bbox_inches="tight", pad_inches=0.10, facecolor="white")
print(f"Saved {png.relative_to(ROOT)} at 600 DPI after second-pass visual QC")
plt.show()
'''


def stable_cell_state(cell: nbformat.NotebookNode) -> str:
    """Serialize all existing cell content and output that must not change."""
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
        raise RuntimeError("Second-pass Figure S4 cell already exists; refusing to rerun it")

    existing_count = len(notebook.cells)
    existing_states = [stable_cell_state(copy.deepcopy(cell)) for cell in notebook.cells]

    cell = nbformat.v4.new_code_cell(FIGURE_10_SECOND_PASS)
    cell.metadata["tags"] = [CELL_TAG, "figure_10", "fixed_figure_only"]
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
