#!/usr/bin/env python3
"""Refresh Phase 3 benchmark-derived outputs after an implementation audit.

The final ridge models are not refit. This script reruns only the B1 RSF/GBSA
nested benchmarks, replaces their OOF curves, recalculates derived tables, and
regenerates reports/figures. It preserves and extends the Phase 3 access log.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import phase3_adni_model_development as p


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(p.TABLE_ROOT / name)


def main() -> None:
    previous_access = pd.read_csv(
        p.TABLE_ROOT / "phase3_file_access_audit.csv"
    )
    reference_lengths = p.read_frozen_references()
    integrity, inputs = p.build_input_integrity()
    d1_resolution = read("d1_collinearity_resolution.csv")
    selected_name = d1_resolution["selected_candidate"].iloc[0]
    pair = (
        ("current_mmse", "mmse_change_from_index")
        if selected_name == "current_plus_change"
        else ("index_mmse", "mmse_change_from_index")
    )
    specs = p.model_specs(pair)
    benchmark_predictions, benchmark_curves, benchmark_tuning = (
        p.run_baseline_benchmarks(specs["B1"], inputs)
    )
    p.save_table(
        benchmark_tuning, p.TABLE_ROOT / "benchmark_inner_tuning.csv"
    )
    p.save_table(
        benchmark_predictions,
        p.PREDICTION_ROOT / "benchmark_predictions.csv",
    )

    previous_baseline = pd.read_csv(
        p.PREDICTION_ROOT / "out_of_fold_baseline_predictions.csv"
    )
    ridge_baseline = previous_baseline[
        ~previous_baseline["model_id"].isin(["RSF_B1", "GBSA_B1"])
    ].copy()
    ridge_dynamic = pd.read_csv(
        p.PREDICTION_ROOT / "out_of_fold_dynamic_predictions.csv"
    )
    predictions = pd.concat(
        [ridge_baseline, ridge_dynamic, benchmark_predictions],
        ignore_index=True,
    )
    p.save_table(
        pd.concat([ridge_baseline, benchmark_predictions], ignore_index=True),
        p.PREDICTION_ROOT / "out_of_fold_baseline_predictions.csv",
    )
    previous_curves = pd.read_parquet(
        p.PREDICTION_ROOT / "out_of_fold_probability_curves.parquet"
    )
    ridge_curves = previous_curves[
        ~previous_curves["model_id"].isin(["RSF_B1", "GBSA_B1"])
    ].copy()
    curves = pd.concat([ridge_curves, benchmark_curves], ignore_index=True)
    curves.to_parquet(
        p.PREDICTION_ROOT / "out_of_fold_probability_curves.parquet",
        index=False,
    )
    (
        metrics,
        calibration,
        coherence,
        ibs,
        time_brier,
    ) = p.calculate_metric_tables(predictions, curves)
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
    selected_dynamic, selection = p.select_final_dynamic_model(metrics, ibs)
    p.save_table(
        selection, p.TABLE_ROOT / "final_model_selection_summary.csv"
    )
    missingness = metrics[
        metrics["model_id"].isin(["B1", "B2", "B3", "B4", "D1", "D2"])
    ].merge(
        ibs,
        on=["model_id", "family", "horizon_months"],
        how="left",
    )
    p.save_table(
        missingness, p.TABLE_ROOT / "missingness_sensitivity_results.csv"
    )

    event_info = read("participant_level_event_information.csv")
    fold_events = read("outer_fold_event_information.csv")
    ridge_tuning = read("fold_specific_ridge_penalty_tuning.csv")
    fold_coefficients = read("fold_specific_coefficients.csv")
    bootstrap_ci = read("bootstrap_confidence_intervals.csv")
    phase_metrics = read("phase_stratified_performance.csv")
    temporal = read("temporal_validation_performance.csv")
    decision_curve = read("progression_decision_curve.csv")
    final_artifacts = read("frozen_model_artifacts.csv")
    final_coefficients = read("final_model_coefficients.csv")
    ph = read("ph_diagnostic_tests.csv")
    functional = read("functional_form_diagnostics.csv")
    influence = read("influential_participants.csv")
    residuals = read("schoenfeld_residuals.csv")
    metadata = json.loads(
        (p.TABLE_ROOT / "phase3_run_metadata.json").read_text(encoding="utf-8")
    )
    figure_manifest = p.create_figures(
        event_info,
        fold_events,
        ridge_tuning,
        fold_coefficients,
        metrics,
        ibs,
        calibration,
        coherence,
        phase_metrics,
        ph,
        residuals,
        predictions,
    )
    p.save_table(
        figure_manifest, p.TABLE_ROOT / "figure_manifest.csv"
    )
    p.write_reports(
        reference_lengths=reference_lengths,
        integrity=integrity,
        event_info=event_info,
        d1_resolution=d1_resolution,
        metrics=metrics,
        ibs=ibs,
        bootstrap_ci=bootstrap_ci,
        comparisons=comparisons,
        selection=selection,
        selected_dynamic=selected_dynamic,
        final_artifacts=final_artifacts,
        final_coefficients=final_coefficients,
        calibration=calibration,
        coherence=coherence,
        ph=ph,
        functional=functional,
        influence=influence,
        phase_metrics=phase_metrics,
        temporal=temporal,
        decision_curve=decision_curve,
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
    metadata["benchmark_probability_conversion_fix"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "issue": "GBSA StepFunction risk multiplier a was omitted in initial benchmark CIF conversion",
        "resolution": "recomputed benchmark CIFs using a*y+b; ridge models and frozen package unchanged",
        "affected_outputs_refreshed": [
            "benchmark predictions",
            "probability curves",
            "performance/calibration/Brier/coherence tables",
            "reports",
            "figures",
        ],
    }
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    p.write_json(metadata, p.TABLE_ROOT / "phase3_run_metadata.json")
    print(
        json.dumps(
            {
                "status": "refreshed",
                "selected_dynamic_model": selected_dynamic,
                "benchmark_rows": len(benchmark_predictions),
                "figures": figure_manifest["figure"].nunique(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
