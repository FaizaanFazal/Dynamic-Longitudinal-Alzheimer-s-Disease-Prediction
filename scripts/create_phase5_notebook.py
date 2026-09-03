#!/usr/bin/env python3
"""Create the Phase 5 executed-results notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "notebooks"
    / "Phase5"
    / "Phase5_SecondaryAnalysesAndFinalReporting.ipynb"
)


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "clinical_env (Python 3.10)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
        "phase": 5,
        "analysis_type": (
            "secondary analyses, final synthesis, and manuscript reporting"
        ),
    }
    notebook["cells"] = [
        markdown(
            """
# Phase 5 — Secondary Analyses, Final Synthesis, and Manuscript Reporting

This notebook displays the completed Phase 5 machine-readable results and
figures. It does not refit any Phase 3 model. B1, D0, D1, B3, and D3 remain
frozen in `models/phase3/phase4_lock_v2`.

The secondary ADNI diagnosis-reversion models below are explicitly
low-dimensional **association analyses**, not prediction models and not
causal analyses. OASIS recalibration is local model updating and is kept
separate from untouched external validation. AIBL is descriptive only.
"""
        ),
        code(
            """
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "results" / "csvs" / "phase5").exists():
    ROOT = ROOT.parent

TABLES = ROOT / "results" / "csvs" / "phase5"
PLOTS = ROOT / "results" / "plots" / "phase5"
REPORTS = ROOT / "reports" / "phase5"
PROCESSED = ROOT / "ProcessedData" / "adni" / "phase5"
LOCK = ROOT / "models" / "phase3" / "phase4_lock_v2"

pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 120)
pd.set_option("display.width", 180)

metadata = json.loads((TABLES / "phase5_run_metadata.json").read_text())
print(f"Project root: {ROOT}")
print(f"Frozen lock preserved: {LOCK}")
print(f"Frozen models refit/modified: {metadata['frozen_models_refit_or_modified']}")
print(f"Transportability: {metadata['transportability_classification']}")
print(f"AIBL role: {metadata['aibl_role']}")
print(f"Current decision: {metadata['decision']}")
"""
        ),
        markdown(
            """
## 1. ADNI diagnosis-reversion population

The five mutually exclusive trajectory groups use longitudinal clinician
diagnoses. The strongest-confirmation group is nested within sustained
reversion.
"""
        ),
        code(
            """
counts = pd.read_csv(TABLES / "reversion_group_counts.csv")
characteristics = pd.read_csv(TABLES / "reversion_group_characteristics.csv")
display(counts)
display(characteristics[characteristics["variable"].isin([
    "baseline_age", "female_sex", "education_years", "apoe_e4_dose",
    "baseline_mmse", "follow_up_months", "later_diagnosis_assessments",
])])

expected = {
    "sustained_reversion": 60,
    "progression": 399,
    "persistent_mci": 746,
    "unstable_or_single_cn": 45,
    "no_later_assessment": 350,
    "strongest_reversion_nested": 50,
}
observed = counts.set_index("trajectory_group")["participants"].to_dict()
assert all(int(observed[key]) == value for key, value in expected.items())
print("All prespecified diagnosis-trajectory counts reproduced exactly.")
"""
        ),
        code(
            """
timing = pd.read_csv(TABLES / "reversion_timing_and_relapse.csv")
fixed = pd.read_csv(TABLES / "diagnosis_fixed_horizon_summary.csv")
opportunity = pd.read_csv(TABLES / "reversion_followup_opportunity.csv")
display(timing)
display(fixed)
display(opportunity)
print(
    "Sustained labels require future confirmation; longer follow-up therefore "
    "creates ascertainment and guarantee-time sensitivity."
)
"""
        ),
        markdown(
            """
## 2. Prespecified association analyses

Five clinical-core variables were fitted with a fixed ridge penalty and
participant bootstrap intervals. No tuning or high-dimensional exploration
was performed.
"""
        ),
        code(
            """
associations = pd.read_csv(TABLES / "reversion_adjusted_associations.csv")
observation = pd.read_csv(
    TABLES / "reversion_observation_process_associations.csv"
)
missingness = pd.read_csv(TABLES / "reversion_clinical_core_missingness.csv")
display(associations)
display(observation)
display(missingness)
assert associations["bootstrap_replicates"].eq(1000).all()
print(
    "Interpretation boundary: associations are not internally validated "
    "predictions and do not establish causality or biological recovery."
)
"""
        ),
        markdown(
            """
## 3. Diagnosis reversion versus CDR improvement

CDR 0.5 and clinician-diagnosed MCI are related but non-identical index
states. Agreement is therefore reported as an endpoint-definition comparison.
"""
        ),
        code(
            """
agreement = pd.read_csv(TABLES / "cdr_diagnosis_agreement_summary.csv")
disagreement = pd.read_csv(
    TABLES / "cdr_diagnosis_disagreement_categories.csv"
)
agreement_timing = pd.read_csv(
    TABLES / "cdr_diagnosis_timing_comparison.csv"
)
display(agreement)
display(disagreement)
display(agreement_timing.describe(include="all"))
assert int(agreement.loc[0, "overlap_participants"]) == 47
assert round(float(agreement.loc[0, "jaccard_similarity"]), 6) == 0.254054
"""
        ),
        markdown(
            """
## 4. Untouched OASIS-3 external validation and D1 value

These results come directly from completed Phase 4 and remain the primary
transportability evidence.
"""
        ),
        code(
            """
external = pd.read_csv(TABLES / "oasis_untouched_performance_synthesis.csv")
external_primary = external[
    external["model_id"].isin(["B1", "D0", "D1"])
    & external["horizon_months"].isin([12, 36])
]
display(external_primary[[
    "model_id", "horizon_months", "rows", "participants",
    "improvement_events", "progression_events",
    "auc_improvement", "auc_progression",
    "c_index_improvement", "c_index_progression",
    "multistate_brier", "calibration_slope_improvement",
    "observed_expected_improvement", "calibration_slope_progression",
    "observed_expected_progression",
]])

d1d0 = pd.read_csv(
    ROOT / "results" / "csvs" / "phase4"
    / "d1_vs_d0_paired_bootstrap_ci.csv"
)
display(d1d0)
print("Bootstrap unit:", d1d0["bootstrap_unit"].unique().tolist())
"""
        ),
        markdown(
            """
## 5. Calibration, local updating, and transport classification

The untouched rows are external validation. Cross-fitted OASIS intercept and
slope rows are model updating; original predictor coefficients were never
refit.
"""
        ),
        code(
            """
recal = pd.read_csv(TABLES / "calibration_recalibration_synthesis.csv")
params = pd.read_csv(TABLES / "recalibration_parameter_summary.csv")
framework = pd.read_csv(
    TABLES / "transportability_classification_framework.csv"
)
display(recal[[
    "model_id", "horizon_months", "method", "validation_design",
    "auc_improvement", "auc_progression", "multistate_brier",
    "calibration_slope_improvement", "observed_expected_improvement",
    "calibration_slope_progression", "observed_expected_progression",
]])
display(params)
display(framework)
assert not params["predictor_coefficients_refit"].any()
"""
        ),
        markdown(
            """
## 6. Distribution shift, endpoint sensitivity, and AIBL scope
"""
        ),
        code(
            """
shift = pd.read_csv(TABLES / "distribution_shift_synthesis.csv")
endpoint = pd.read_csv(TABLES / "endpoint_sensitivity_synthesis.csv")
aibl = pd.read_csv(TABLES / "aibl_descriptive_summary.csv")
display(shift)
display(endpoint)
display(aibl)
print(
    "AIBL is descriptive only: no discrimination, calibration, or model "
    "performance estimate was calculated."
)
"""
        ),
        markdown(
            """
## 7. Publication figures

PNG files are publication-resolution exports and every figure has an SVG
companion. Thumbnails are displayed here to keep this executed notebook
usable.
"""
        ),
        code(
            """
def show_figure_grid(first, last, columns=2, max_width=1000):
    paths = [
        PLOTS / f"figure_{number:02d}_"
        for number in range(first, last + 1)
    ]
    all_pngs = sorted(PLOTS.glob("figure_*.png"))
    selected = [
        path for path in all_pngs
        if first <= int(path.name.split("_")[1]) <= last
    ]
    rows = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 5.8 * rows))
    axes = getattr(axes, "flat", [axes])
    for ax, path in zip(axes, selected):
        image = PILImage.open(path)
        image.thumbnail((max_width, max_width))
        ax.imshow(image)
        ax.set_title(path.stem.replace("_", " "), fontsize=10)
        ax.axis("off")
    for ax in list(axes)[len(selected):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

show_figure_grid(1, 4)
"""
        ),
        code(
            """
show_figure_grid(5, 8)
"""
        ),
        code(
            """
show_figure_grid(9, 12)
"""
        ),
        markdown(
            """
## 8. Manuscript package, reports, and reproducibility
"""
        ),
        code(
            """
manuscript_tables = sorted((TABLES / "manuscript_tables").glob("table_*.csv"))
reports = sorted(REPORTS.glob("*.md"))
figures = sorted(PLOTS.glob("figure_*.png"))
svg_figures = sorted(PLOTS.glob("figure_*.svg"))

inventory = pd.DataFrame([
    {"category": "manuscript_table", "file": p.name, "bytes": p.stat().st_size}
    for p in manuscript_tables
] + [
    {"category": "report", "file": p.name, "bytes": p.stat().st_size}
    for p in reports
] + [
    {"category": "png_figure", "file": p.name, "bytes": p.stat().st_size}
    for p in figures
] + [
    {"category": "svg_figure", "file": p.name, "bytes": p.stat().st_size}
    for p in svg_figures
])
display(inventory)
print(f"Primary manuscript tables: {len(manuscript_tables)}")
print(f"Publication figures: {len(figures)} PNG + {len(svg_figures)} SVG")

qc_path = TABLES / "phase5_qc_results.csv"
if qc_path.exists():
    qc = pd.read_csv(qc_path)
    display(qc[["check_id", "section", "status", "details"]])
    print(f"Independent QC: {qc['status'].eq('PASS').sum()}/{len(qc)} PASS")
else:
    print("Independent QC is run after notebook generation; see reports/phase5.")
"""
        ),
        markdown(
            """
## Notebook conclusion

- D1 retained useful untouched OASIS-3 discrimination and added value over D0.
- Improvement risk was materially underpredicted; transportability is
  therefore **with recalibration**, not without updating.
- OASIS recalibration is explicitly local model updating.
- Reconstructed CDR did not reverse the primary Phase 4 conclusion.
- ADNI sustained diagnosis reversion involved 60 participants, with 50 under
  the strongest definition; it only partly overlapped CDR improvement.
- AIBL remains descriptive, MRI remains outside the primary model, and no
  observed reversion is interpreted as biological recovery.
- The computational manuscript package is complete; author, ethics,
  data-sharing, funding, and journal-formatting tasks remain manual.
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
