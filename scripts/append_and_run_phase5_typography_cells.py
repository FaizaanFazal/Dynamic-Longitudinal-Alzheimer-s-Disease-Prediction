#!/usr/bin/env python3
"""Append and execute only the six enlarged-non-title-text notebook cells."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Phase5/Phase5_SecondaryAnalysesAndFinalReporting.ipynb"
CELL_TAG = "phase5_enlarged_non_title_text_20260828"


def stable_cell_state(cell: nbformat.NotebookNode) -> str:
    """Serialize all existing cell content and output that must remain unchanged."""
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
        raise RuntimeError("Typography cells already exist; refusing to duplicate or rerun them")

    existing_count = len(notebook.cells)
    existing_states = [stable_cell_state(copy.deepcopy(cell)) for cell in notebook.cells]

    note = nbformat.v4.new_markdown_cell(
        "## Enlarged non-title typography for fixed manuscript figures\n\n"
        "These cells overwrite only the six existing `*_fixed` variants at 600 DPI. "
        "Legend, axis, tick, annotation, node, and box-label text is enlarged. Global "
        "and panel-title point sizes remain exactly unchanged from the visually approved "
        "fixed versions."
    )
    note.metadata["tags"] = [CELL_TAG, "documentation"]
    notebook.cells.append(note)

    figures = ["01", "02", "03", "08", "10", "11"]
    cell_indices: list[int] = []
    for figure in figures:
        source = (
            f"# Enlarge non-title typography for fixed Figure {figure}; title sizes unchanged\n"
            f"from scripts.phase5_fixed_figure_typography import render_figure_{figure}\n\n"
            f"render_figure_{figure}()"
        )
        cell = nbformat.v4.new_code_cell(source)
        cell.metadata["tags"] = [CELL_TAG, f"figure_{figure}", "fixed_figure_only"]
        notebook.cells.append(cell)
        cell_indices.append(len(notebook.cells) - 1)

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="clinical_env",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    with client.setup_kernel():
        for cell_index in cell_indices:
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
    print("new_markdown_cells: 1")
    print(f"new_code_cells_executed_only: {len(cell_indices)}")
    print(
        "new_execution_counts:",
        [reloaded.cells[index].get("execution_count") for index in cell_indices],
    )
    print(
        "new_output_counts:",
        [len(reloaded.cells[index].get("outputs", [])) for index in cell_indices],
    )


if __name__ == "__main__":
    main()
