#!/usr/bin/env python3
"""Matched-row nested references for lagged-MMSE and slope sensitivities."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import phase3_adni_model_development as p


D1_CONTINUOUS = (
    "age_at_landmark",
    "education_years",
    "apoe_e4_dose",
    "current_mmse",
    "mmse_change_from_index",
    "elapsed_years_from_index",
)


def remove_models(frame: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    return frame[~frame["model_id"].isin(ids)].copy()


def main() -> None:
    previous_access = pd.read_csv(
        p.TABLE_ROOT / "phase3_file_access_audit.csv"
    )
    episode = p.AUDIT.read_csv(
        p.ROOT
        / "results"
        / "csvs"
        / "phase1"
        / "adni"
        / "primary_episode_time_to_event.csv",
        "matched dynamic sensitivity endpoint times",
    )
    inner = p.AUDIT.read_csv(
        p.ROOT
        / "results"
        / "csvs"
        / "phase2"
        / "adni_nested_inner_fold_assignments.csv",
        "matched dynamic sensitivity inner folds",
    )
    specs = {
        "D1LagRef": p.ModelSpec(
            "D1LagRef",
            "dynamic",
            "strictly_lagged_mmse",
            D1_CONTINUOUS,
            analysis_role="matched_sensitivity_reference",
            description="D1 current/change on lag-eligible rows",
        ),
        "D1SlopeRef": p.ModelSpec(
            "D1SlopeRef",
            "dynamic",
            "slope_eligible",
            D1_CONTINUOUS,
            analysis_role="matched_sensitivity_reference",
            description="D1 current/change on slope-eligible rows",
        ),
    }
    prediction_parts = []
    curve_parts = []
    tuning_rows = []
    coefficient_rows = []
    preprocessing_rows = []
    for horizon in p.PRIMARY_HORIZONS:
        for spec in specs.values():
            data = p.load_analysis_data(spec, horizon, episode, {})
            for outer_fold in range(5):
                (
                    prediction,
                    curves,
                    tuning,
                    coefficients,
                    preprocessing,
                ) = p.fit_outer_ridge(
                    data, spec, horizon, outer_fold, inner
                )
                prediction_parts.append(prediction)
                curve_parts.append(curves)
                tuning_rows.extend(tuning)
                coefficient_rows.extend(coefficients)
                preprocessing_rows.extend(preprocessing)
                print(
                    f"[matched] {spec.model_id} {horizon}m outer {outer_fold}",
                    flush=True,
                )
    new_predictions = pd.concat(prediction_parts, ignore_index=True)
    new_curves = pd.concat(curve_parts, ignore_index=True)
    ids = set(specs)

    baseline_predictions = pd.read_csv(
        p.PREDICTION_ROOT / "out_of_fold_baseline_predictions.csv"
    )
    dynamic_predictions = pd.read_csv(
        p.PREDICTION_ROOT / "out_of_fold_dynamic_predictions.csv"
    )
    dynamic_predictions = pd.concat(
        [remove_models(dynamic_predictions, ids), new_predictions],
        ignore_index=True,
    )
    p.save_table(
        dynamic_predictions,
        p.PREDICTION_ROOT / "out_of_fold_dynamic_predictions.csv",
    )
    all_predictions = pd.concat(
        [baseline_predictions, dynamic_predictions], ignore_index=True
    )
    curves = pd.read_parquet(
        p.PREDICTION_ROOT / "out_of_fold_probability_curves.parquet"
    )
    curves = pd.concat(
        [remove_models(curves, ids), new_curves], ignore_index=True
    )
    curves.to_parquet(
        p.PREDICTION_ROOT / "out_of_fold_probability_curves.parquet",
        index=False,
    )
    p.save_table(
        new_predictions,
        p.PREDICTION_ROOT / "matched_dynamic_sensitivity_predictions.csv",
    )

    update_tables = [
        (
            "fold_specific_ridge_penalty_tuning.csv",
            pd.DataFrame(tuning_rows),
        ),
        (
            "fold_specific_coefficients.csv",
            pd.DataFrame(coefficient_rows),
        ),
        (
            "fold_specific_preprocessing_parameters.csv",
            pd.DataFrame(preprocessing_rows),
        ),
    ]
    for name, addition in update_tables:
        existing = pd.read_csv(p.TABLE_ROOT / name)
        p.save_table(
            pd.concat(
                [remove_models(existing, ids), addition], ignore_index=True
            ),
            p.TABLE_ROOT / name,
        )
    tuning = pd.read_csv(
        p.TABLE_ROOT / "fold_specific_ridge_penalty_tuning.csv"
    )
    selected = tuning[
        [
            "model_id",
            "family",
            "horizon_months",
            "outer_fold",
            "cause",
            "selected_alpha",
            "selection_rule",
        ]
    ].drop_duplicates()
    p.save_table(
        selected, p.TABLE_ROOT / "fold_specific_ridge_penalties.csv"
    )

    (
        metrics,
        calibration,
        coherence,
        ibs,
        time_brier,
    ) = p.calculate_metric_tables(all_predictions, curves)
    p.save_table(
        metrics, p.TABLE_ROOT / "discrimination_and_horizon_metrics.csv"
    )
    p.save_table(calibration, p.TABLE_ROOT / "calibration_tables.csv")
    p.save_table(
        coherence, p.TABLE_ROOT / "probability_coherence_checks.csv"
    )
    p.save_table(ibs, p.TABLE_ROOT / "integrated_brier_scores.csv")
    p.save_table(
        time_brier, p.TABLE_ROOT / "time_specific_brier_scores.csv"
    )
    p.save_table(
        metrics[
            [
                "model_id",
                "family",
                "horizon_months",
                "brier_improvement",
                "brier_progression",
                "brier_remaining",
                "multistate_brier",
            ]
        ],
        p.TABLE_ROOT / "brier_score_table.csv",
    )
    comparisons = p.model_comparisons(metrics, ibs)
    p.save_table(comparisons, p.TABLE_ROOT / "model_comparison_table.csv")
    matched = metrics[
        metrics["model_id"].isin(
            ["D1LagRef", "D3", "D1SlopeRef", "D4"]
        )
    ].merge(
        ibs,
        on=["model_id", "family", "horizon_months"],
        how="left",
    )
    p.save_table(
        matched, p.TABLE_ROOT / "matched_dynamic_sensitivity_metrics.csv"
    )

    event_info = pd.read_csv(
        p.TABLE_ROOT / "participant_level_event_information.csv"
    )
    fold_events = pd.read_csv(
        p.TABLE_ROOT / "outer_fold_event_information.csv"
    )
    fold_coefficients = pd.read_csv(
        p.TABLE_ROOT / "fold_specific_coefficients.csv"
    )
    phase_metrics = pd.read_csv(
        p.TABLE_ROOT / "phase_stratified_performance.csv"
    )
    ph = pd.read_csv(p.TABLE_ROOT / "ph_diagnostic_tests.csv")
    residuals = pd.read_csv(p.TABLE_ROOT / "schoenfeld_residuals.csv")
    figure_manifest = p.create_figures(
        event_info,
        fold_events,
        tuning,
        fold_coefficients,
        metrics,
        ibs,
        calibration,
        coherence,
        phase_metrics,
        ph,
        residuals,
        all_predictions,
    )
    p.save_table(
        figure_manifest, p.TABLE_ROOT / "figure_manifest.csv"
    )

    metadata_path = p.TABLE_ROOT / "phase3_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["matched_dynamic_sensitivities"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": sorted(ids),
        "purpose": (
            "isolate lag and slope feature value from analysis-row retention"
        ),
        "metrics_table": (
            "results/csvs/phase3/matched_dynamic_sensitivity_metrics.csv"
        ),
    }
    p.write_json(metadata, metadata_path)
    p.write_reports(
        reference_lengths={str(i): 1 for i in range(11)},
        integrity=pd.read_csv(
            p.TABLE_ROOT / "phase3_input_integrity_audit.csv"
        ),
        event_info=event_info,
        d1_resolution=pd.read_csv(
            p.TABLE_ROOT / "d1_collinearity_resolution.csv"
        ),
        metrics=metrics,
        ibs=ibs,
        bootstrap_ci=pd.read_csv(
            p.TABLE_ROOT / "bootstrap_confidence_intervals.csv"
        ),
        comparisons=comparisons,
        selection=pd.read_csv(
            p.TABLE_ROOT / "final_model_selection_summary.csv"
        ),
        selected_dynamic=metadata["selected_dynamic_model"],
        final_artifacts=pd.read_csv(
            p.TABLE_ROOT / "frozen_model_artifacts.csv"
        ),
        final_coefficients=pd.read_csv(
            p.TABLE_ROOT / "final_model_coefficients.csv"
        ),
        calibration=calibration,
        coherence=coherence,
        ph=ph,
        functional=pd.read_csv(
            p.TABLE_ROOT / "functional_form_diagnostics.csv"
        ),
        influence=pd.read_csv(
            p.TABLE_ROOT / "influential_participants.csv"
        ),
        phase_metrics=phase_metrics,
        temporal=pd.read_csv(
            p.TABLE_ROOT / "temporal_validation_performance.csv"
        ),
        decision_curve=pd.read_csv(
            p.TABLE_ROOT / "progression_decision_curve.csv"
        ),
        model_version_id=metadata["model_version_id"],
        figure_manifest=figure_manifest,
    )
    refreshed_access = pd.DataFrame(p.AUDIT.rows)
    combined_access = (
        pd.concat([previous_access, refreshed_access], ignore_index=True)
        .drop_duplicates(["path", "sha256", "purpose"])
        .reset_index(drop=True)
    )
    p.save_table(
        combined_access, p.TABLE_ROOT / "phase3_file_access_audit.csv"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "prediction_rows": len(new_predictions),
                "models": sorted(ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
