"""Generate the five executed-review notebook sources.

Run this script with the project's Python 3.10 environment, then execute the
created notebooks to embed their outputs.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"

PHASES = {
    1: {
        "folder": "Phase1",
        "filename": "Phase1_DataFoundationAndEndpointValidation.ipynb",
        "title": "Phase 1 — Data Foundation and Endpoint Validation",
        "summary": (
            "Read-only, executed review of the canonical participant–visit manifests, "
            "CDR state construction, index cohort, transition audit, endpoint support, "
            "quality control, reports, tables, and figures."
        ),
    },
    2: {
        "folder": "Phase2",
        "filename": "Phase2_LandmarksMissingnessAndHarmonization.ipynb",
        "title": "Phase 2 — Landmark Datasets, Predictors, Missingness, and Harmonization",
        "summary": (
            "Read-only, executed review of the baseline and dynamic landmark datasets, "
            "predictor windows, clinical-core schema, missingness handling, MRI readiness, "
            "fold design, quality control, reports, tables, and figures."
        ),
    },
    3: {
        "folder": "Phase3",
        "filename": "Phase3_ADNIModelDevelopmentAndInternalValidation.ipynb",
        "title": "Phase 3 — ADNI Model Development and Internal Validation",
        "summary": (
            "Read-only, executed review of nested participant-grouped validation, "
            "competing-transition models, calibration, diagnostics, sensitivities, "
            "model selection, frozen Phase 4 artifacts, quality control, and reports."
        ),
    },
    4: {
        "folder": "Phase4",
        "filename": "Phase4_LockedOASIS3ExternalValidation.ipynb",
        "title": "Phase 4 — Locked OASIS-3 External Validation",
        "summary": (
            "Authoritative Stage A lock-audit notebook. It verifies the immutable Phase 3 "
            "package and enforces the stop rule before any OASIS-3 record-level feature "
            "or outcome access."
        ),
    },
    5: {
        "folder": "Phase5",
        "filename": "Phase5_SecondaryAnalysesAndFinalReporting.ipynb",
        "title": "Phase 5 — Secondary Analyses and Final Reporting",
        "summary": (
            "Executed status notebook. Phase 5 is not started: it records the planned "
            "secondary analyses and guardrails without performing new analyses."
        ),
    },
}


SETUP_CELL = """from pathlib import Path
import sys
import pandas as pd
from IPython.display import Markdown, display

cursor = Path.cwd().resolve()
PROJECT_ROOT = next(
    candidate
    for candidate in [cursor, *cursor.parents]
    if (candidate / "scripts").is_dir()
    and (candidate / "notebooks").is_dir()
    and (candidate / "ProcessedData").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from notebook_phase_review import (
    display_all_result_tables,
    display_artifact_inventory,
    display_future_phase_plan,
    display_key_results,
    display_phase4_lock,
    display_plot_gallery,
    display_qc_summary,
    display_reports,
    phase_heading,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)
print(f"Project root: {PROJECT_ROOT}")
print(f"Review helpers: {PROJECT_ROOT / 'scripts' / 'notebook_phase_review.py'}")
"""


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code_cell(source: str):
    return nbf.v4.new_code_cell(source.strip())


def phase_notebook(phase: int, details: dict) -> nbf.NotebookNode:
    status_note = (
        "This notebook summarizes existing saved artifacts; it does not rerun the phase "
        "pipeline. Large row-level tables are represented by exact dimensions and bounded "
        "previews, while every file is listed in the complete inventory."
        if phase <= 3
        else "This is a status notebook. Planned analyses are not executed."
    )
    cells = [
        markdown_cell(
            f"""
# {details['title']}

{details['summary']}

{status_note}

**Execution environment:** project `clinical_env`, Python 3.10.
"""
        ),
        code_cell(SETUP_CELL),
        code_cell(f"phase_heading({phase}, PROJECT_ROOT)"),
    ]

    if phase <= 3:
        cells.extend(
            [
                code_cell(
                    f"phase{phase}_inventory = display_artifact_inventory(PROJECT_ROOT, {phase})"
                ),
                code_cell(f"display_key_results(PROJECT_ROOT, {phase})"),
                code_cell(f"display_qc_summary(PROJECT_ROOT, {phase})"),
                code_cell(f"display_reports(PROJECT_ROOT, {phase})"),
                code_cell(f"display_all_result_tables(PROJECT_ROOT, {phase}, max_rows=20)"),
                code_cell(f"display_plot_gallery(PROJECT_ROOT, {phase}, plots_per_page=6)"),
                markdown_cell(
                    f"""
## Notebook conclusion

The Phase {phase} review is complete. Cell outputs contain the execution context,
complete file inventory, decision-relevant results, all QC checks, every saved
Markdown report, a representation of every saved result table, and every unique
PNG figure. Matching SVG figures and large intermediate datasets are listed in
the inventory.
"""
                ),
            ]
        )
    elif phase == 4:
        cells.extend(
            [
                code_cell(
                    """
import subprocess

commands = [
    [sys.executable, str(PROJECT_ROOT / "scripts" / "phase4_lock_audit.py")],
    [sys.executable, str(PROJECT_ROOT / "scripts" / "phase4_qc.py")],
]
for command in commands:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"$ {' '.join(command)}")
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Phase 4 Stage A command failed with code {completed.returncode}"
        )
"""
                ),
                code_cell(
                    """
phase4_metadata = pd.read_json(
    PROJECT_ROOT / "results" / "csvs" / "phase4" / "phase4_run_metadata.json",
    typ="series",
)
display(Markdown("## Stage A decision"))
display(phase4_metadata.to_frame("value"))

stage_a_checks = pd.read_csv(
    PROJECT_ROOT / "results" / "csvs" / "phase4" / "phase4_stage_a_qc_results.csv"
)
display(Markdown("## Critical lock checks"))
display(stage_a_checks)

package_completeness = pd.read_csv(
    PROJECT_ROOT / "results" / "csvs" / "phase4" / "frozen_package_completeness.csv"
)
display(Markdown("## Frozen-package completeness"))
display(package_completeness)
"""
                ),
                code_cell("display_qc_summary(PROJECT_ROOT, 4)"),
                code_cell("display_reports(PROJECT_ROOT, 4)"),
                code_cell("display_all_result_tables(PROJECT_ROOT, 4, max_rows=50)"),
                code_cell("display_plot_gallery(PROJECT_ROOT, 4, plots_per_page=6)"),
                code_cell("phase4_inventory = display_artifact_inventory(PROJECT_ROOT, 4)"),
                markdown_cell(
                    """
## Notebook conclusion

The attached Phase 4 specification cannot proceed beyond Stage A because the
immutable Phase 3 package does not contain the mandatory final D0 comparator,
B3 no-APOE sensitivity, or D3 strictly lagged-MMSE sensitivity bundles. All
existing frozen hashes pass, and independent QC confirms that the hold was
safely enforced.

No OASIS-3 record-level feature or outcome file was opened, no external
prediction or performance estimate was created, and Phase 5 was not started.
The recorded decision is **HOLD FOR VALIDATION AUDIT** pending a separately
versioned, outcome-blind Phase 3 lock repair.
"""
                ),
            ]
        )
    else:
        cells.extend(
            [
                code_cell("phase5_inventory = display_artifact_inventory(PROJECT_ROOT, 5)"),
                code_cell("display_future_phase_plan(5)"),
                markdown_cell(
                    """
## Notebook conclusion

Phase 5 remains unstarted. No secondary analysis was run and no result is implied
by this status notebook.
"""
                ),
            ]
        )

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3 (clinical_env)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
            "project_phase": phase,
            "review_mode": "read-only",
        }
    )
    return notebook


def main() -> None:
    for phase, details in PHASES.items():
        output_dir = NOTEBOOK_ROOT / details["folder"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / details["filename"]
        nbf.write(phase_notebook(phase, details), output_path)
        print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
