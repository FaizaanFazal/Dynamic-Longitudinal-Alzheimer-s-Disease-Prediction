#!/usr/bin/env python3
"""Create the executed-results notebook for completed Phase 4 validation."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "notebooks"
    / "Phase4"
    / "Phase4_LockedOASIS3ExternalValidation.ipynb"
)


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "clinical_env",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
        "phase": 4,
        "analysis_type": "locked external validation",
        "external_cohort": "OASIS-3",
    }
    notebook["cells"] = [
        markdown(
            """
# Phase 4 — Locked OASIS-3 External Validation

This notebook displays the completed, saved Phase 4 results. The analysis used
the repaired immutable lock at `models/phase3/phase4_lock_v2`.

Execution order was enforced: lock verification → outcome-hidden feature
construction → frozen predictions → independent prediction verification →
outcome merge → untouched validation → endpoint sensitivity → cross-fitted
recalibration → independent final QC.

The notebook does not refit or retune models. It reads the machine-readable
saved outputs so every result remains traceable to the frozen analysis.
"""
        ),
        code(
            """
from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "results" / "csvs" / "phase4").exists():
    ROOT = ROOT.parent

TABLES = ROOT / "results" / "csvs" / "phase4"
PLOTS = ROOT / "results" / "plots" / "phase4"
REPORTS = ROOT / "reports" / "phase4"
LOCK = ROOT / "models" / "phase3" / "phase4_lock_v2"

pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 180)

metadata = json.loads((TABLES / "phase4_run_metadata.json").read_text())
print(f"Project root: {ROOT}")
print(f"Frozen lock: {metadata['lock_path']}")
print(f"Lock version: {metadata['lock_package_version_id']}")
print(f"Phase 4 complete: {metadata['phase4_complete']}")
print(f"Phase 5 started: {metadata['phase5_started']}")
print(f"Final decision: {metadata['decision']}")
"""
        ),
        markdown(
            """
## 1. Lock verification and execution-order gates

No OASIS participant-level outcome was opened until all outcome-hidden
predictions had been frozen and independently regenerated.
"""
        ),
        code(
            """
stage_a = pd.read_csv(TABLES / "phase4_stage_a_qc_results.csv")
final_qc = pd.read_csv(TABLES / "phase4_qc_results.csv")
display(stage_a)
display(final_qc[["check_id", "section", "check", "status", "details"]])
assert stage_a["status"].eq("PASS").all()
assert final_qc["status"].eq("PASS").all()
print(f"Stage A: {len(stage_a)}/{len(stage_a)} PASS")
print(f"Independent final QC: {len(final_qc)}/{len(final_qc)} PASS")
"""
        ),
        code(
            """
access = pd.read_csv(TABLES / "phase4_access_audit.csv")
display(access[[
    "sequence", "timestamp_utc", "stage", "action",
    "oasis_record_level_features_accessed",
    "oasis_record_level_outcomes_accessed",
    "outcome_hidden_predictions_created",
]])

freeze = json.loads((TABLES / "prediction_freeze_metadata.json").read_text())
print("Prediction freeze:", freeze["freeze_timestamp_utc"])
print("Independent prediction QC:", freeze["independent_prediction_qc_completed_at_utc"])
print("First outcome access:", freeze["outcomes_first_accessed_at_utc"])
"""
        ),
        code(
            """
display(Image(filename=str(PLOTS / "phase4_stage_gate_workflow.png"), width=1200))
"""
        ),
        markdown(
            """
## 2. Frozen models, external populations, and deterministic predictions

B1 is the baseline clinical model; D0 is the dynamic comparator; D1 is the
selected dynamic model; B3 removes APOE; D3 restricts MMSE to strictly
pre-landmark observations.
"""
        ),
        code(
            """
inventory = pd.read_csv(LOCK / "model_inventory.csv")
display(inventory[[
    "model_id", "family", "horizon_months", "model_origin",
    "training_rows", "training_participants", "improvement_alpha",
    "progression_alpha", "external_outcomes_accessed", "sha256",
]].sort_values(["model_id", "horizon_months"]))

population = pd.read_csv(TABLES / "outcome_hidden_population_verification.csv")
display(population)
assert population["status"].eq("PASS").all()
"""
        ),
        code(
            """
prediction_manifest = pd.read_csv(TABLES / "outcome_hidden_prediction_manifest.csv")
prediction_qc = pd.read_csv(TABLES / "independent_prediction_regeneration.csv")
coherence = pd.read_csv(TABLES / "outcome_hidden_probability_coherence.csv")

display(prediction_manifest[[
    "model_id", "horizon_months", "file_role", "rows", "participants",
    "outcome_columns_present", "status", "sha256",
]])
display(prediction_qc[[
    "model_id", "horizon_months", "rows", "participants",
    "maximum_point_prediction_difference",
    "maximum_curve_prediction_difference", "status",
]])
display(coherence)
print("Maximum independent regeneration difference:",
      max(prediction_qc["maximum_point_prediction_difference"].max(),
          prediction_qc["maximum_curve_prediction_difference"].max()))
"""
        ),
        markdown(
            """
## 3. Primary untouched OASIS-3 results

The provisional primary endpoint is OASIS `CDRTOT`, treated as a de facto
global CDR. B1 12-month progression has only 18 events, so its estimate is
explicitly low precision.
"""
        ),
        code(
            """
events = pd.read_csv(TABLES / "baseline_event_count_verification.csv")
display(events)
assert events["status"].eq("PASS").all()

metrics = pd.read_csv(TABLES / "untouched_external_metrics.csv")
primary = metrics[
    metrics["model_id"].isin(["B1", "D0", "D1"])
    & metrics["horizon_months"].isin([12, 36])
].copy()
primary_columns = [
    "model_id", "horizon_months", "rows", "participants",
    "improvement_events", "progression_events",
    "auc_improvement", "auc_progression",
    "c_index_improvement", "c_index_progression",
    "brier_improvement", "brier_progression", "multistate_brier",
    "calibration_intercept_improvement", "calibration_slope_improvement",
    "observed_expected_improvement",
    "calibration_intercept_progression", "calibration_slope_progression",
    "observed_expected_progression",
]
display(primary[primary_columns].sort_values(["model_id", "horizon_months"]))
"""
        ),
        code(
            """
ci = pd.read_csv(TABLES / "participant_bootstrap_confidence_intervals.csv")
auc_ci = ci[ci["metric"].isin([
    "auc_improvement", "auc_progression",
    "c_index_improvement", "c_index_progression",
    "multistate_brier", "observed_expected_improvement",
    "observed_expected_progression",
])].copy()
display(auc_ci.sort_values(["model_id", "horizon_months", "metric"]))
print("Bootstrap unit:", auc_ci["bootstrap_unit"].unique().tolist())
print("Replicates per estimate:", auc_ci["bootstrap_replicates"].unique().tolist())
"""
        ),
        code(
            """
integrated = pd.read_csv(TABLES / "untouched_integrated_brier_scores.csv")
remaining = primary[[
    "model_id", "horizon_months",
    "calibration_intercept_remaining", "calibration_slope_remaining",
    "observed_expected_remaining", "observed_rate_remaining",
    "expected_rate_remaining",
]]
display(integrated)
display(remaining)
"""
        ),
        code(
            """
display(Image(filename=str(PLOTS / "untouched_oasis_calibration_panels.png"), width=1050))
"""
        ),
        markdown(
            """
## 4. D1 value over D0 on matched landmarks

All D1-versus-D0 estimates use the same 845 landmark rows from 437
participants, with paired participant-level bootstrap confidence intervals.
"""
        ),
        code(
            """
comparisons = pd.read_csv(TABLES / "untouched_model_comparisons.csv")
d1d0 = comparisons[comparisons["comparison"].eq("D1_vs_D0_matched_rows")]
paired_ci = pd.read_csv(TABLES / "d1_vs_d0_paired_bootstrap_ci.csv")
display(d1d0)
display(paired_ci)
"""
        ),
        code(
            """
display(Image(filename=str(PLOTS / "d1_vs_d0_transport_value.png"), width=1050))
"""
        ),
        markdown(
            """
## 5. Internal-to-external transport and dataset shift
"""
        ),
        code(
            """
transport = pd.read_csv(TABLES / "adni_oasis_performance_comparison.csv")
selected_transport = transport[
    transport["model_id"].isin(["B1", "D0", "D1"])
    & transport["horizon_months"].isin([12, 36])
    & transport["metric"].isin([
        "auc_improvement", "auc_progression",
        "c_index_improvement", "c_index_progression",
        "multistate_brier", "calibration_slope_improvement",
        "calibration_slope_progression",
        "observed_expected_improvement",
        "observed_expected_progression",
    ])
]
display(selected_transport)
display(pd.read_csv(TABLES / "transportability_classification.csv"))
"""
        ),
        code(
            """
display(Image(filename=str(PLOTS / "adni_oasis_auc_transport.png"), width=1050))
"""
        ),
        code(
            """
continuous_shift = pd.read_csv(TABLES / "predictor_distribution_shift.csv")
categorical_shift = pd.read_csv(TABLES / "categorical_distribution_shift.csv")
event_shift = pd.read_csv(TABLES / "event_rate_shift.csv")
display(continuous_shift)
display(categorical_shift)
display(event_shift)
"""
        ),
        code(
            """
display(Image(filename=str(PLOTS / "oasis_adni_distribution_shift_heatmap.png"), width=1000))
"""
        ),
        markdown(
            """
## 6. B3, D3, clinical-MCI, and extended-horizon sensitivities
"""
        ),
        code(
            """
sensitivities = metrics[
    metrics["model_id"].isin(["B3", "D3"])
    & metrics["horizon_months"].isin([12, 36])
]
display(sensitivities[primary_columns].sort_values(["model_id", "horizon_months"]))

extended = metrics[
    metrics["model_id"].isin(["B1", "D1"])
    & metrics["horizon_months"].isin([24, 60])
]
display(extended[primary_columns].sort_values(["model_id", "horizon_months"]))

clinical_mci = pd.read_csv(TABLES / "clinical_mci_subgroup_results.csv")
display(clinical_mci)
"""
        ),
        markdown(
            """
## 7. Reconstructed six-box global CDR sensitivity

The same frozen predictions are evaluated against the reconstructed endpoint;
no model was refit.
"""
        ),
        code(
            """
endpoint_counts = pd.read_csv(TABLES / "reconstructed_cdr_eligibility_event_counts.csv")
endpoint_delta = pd.read_csv(TABLES / "reconstructed_cdr_metric_comparison.csv")
display(endpoint_counts)
display(endpoint_delta[
    endpoint_delta["model_id"].isin(["B1", "D0", "D1"])
    & endpoint_delta["horizon_months"].isin([12, 36])
    & endpoint_delta["metric"].isin([
        "auc_improvement", "auc_progression",
        "c_index_improvement", "c_index_progression",
        "multistate_brier",
    ])
])
"""
        ),
        code(
            """
display(Image(filename=str(PLOTS / "reconstructed_cdr_sensitivity.png"), width=1050))
"""
        ),
        markdown(
            """
## 8. Participant-grouped cross-fitted recalibration

These are model-updating results, not untouched external validation. The
recalibration refits only calibration intercepts/slopes and never predictor
coefficients.
"""
        ),
        code(
            """
recal_params = pd.read_csv(TABLES / "cross_fitted_recalibration_parameters.csv")
recal_metrics = pd.read_csv(TABLES / "cross_fitted_recalibration_metrics.csv")
recal_effect = pd.read_csv(TABLES / "recalibration_effect_comparison.csv")

display(recal_params.groupby(
    ["model_id", "horizon_months", "method", "cause"], as_index=False
).agg(
    folds=("fold", "nunique"),
    total_training_events=("training_events", "sum"),
    predictors_refit=("predictor_coefficients_refit", "max"),
))
display(recal_metrics)
display(recal_effect[[
    "model_id", "horizon_months", "method",
    "multistate_brier", "calibration_slope_improvement",
    "observed_expected_improvement", "calibration_slope_progression",
    "observed_expected_progression",
]])
"""
        ),
        code(
            """
display(Image(filename=str(PLOTS / "cross_fitted_recalibration_effect.png"), width=1050))
"""
        ),
        markdown(
            """
## 9. Final decision and saved deliverables
"""
        ),
        code(
            """
metadata = json.loads((TABLES / "phase4_run_metadata.json").read_text())
outputs = pd.read_csv(TABLES / "phase4_required_output_status.csv")
figure_qc = pd.read_csv(TABLES / "phase4_figure_verification.csv")

display(outputs)
display(figure_qc)
print("Transportability:", metadata["transportability_classification"])
print("Independent QC pass:", metadata["independent_final_qc_pass"])
print("Phase 4 complete:", metadata["phase4_complete"])
print("Phase 5 started:", metadata["phase5_started"])
print("\\nFINAL DECISION:", metadata["decision"])
"""
        ),
        code(
            """
report_paths = sorted(REPORTS.glob("phase4_*.md"))
for path in report_paths:
    print(f"{path.name}: {path.stat().st_size:,} bytes")

display(Markdown((REPORTS / "phase4_phase5_handoff.md").read_text()))
"""
        ),
        markdown(
            """
### Interpretation

The frozen ADNI models retain useful discrimination in OASIS-3, with D1
showing added progression discrimination over D0 on identical landmarks.
Untouched calibration is not consistently transportable, especially for the
improvement transition, so cross-fitted recalibration is retained as explicit
model updating. Endpoint reconstruction does not materially reverse the main
conclusions.

**Final decision: `GO TO PHASE 5 WITH RESTRICTIONS`.**

Phase 5 has not been started.
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
