#!/usr/bin/env python3
"""Independent on-disk verification of Phase 2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # trusted, locally generated 1200-DPI figures


ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = ROOT / "results" / "csvs" / "phase2"
PROCESSED_ROOT = ROOT / "ProcessedData"
PLOT_ROOT = ROOT / "results" / "plots" / "phase2"
REPORT_ROOT = ROOT / "reports" / "phase2"
COHORTS = ("adni", "oasis3", "aibl")
HORIZONS = (12, 24, 36, 60)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_saved_artifact_qc() -> None:
    expected_index = {"adni": 2447, "oasis3": 491, "aibl": 241}
    expected_reports = [
        "phase2_model_ready_data_harmonization_report.md",
        "oasis3_cdr_field_resolution_report.md",
        "phase2_clinical_core_data_dictionary.md",
        "phase2_missingness_and_preprocessing_specification.md",
        "phase2_mri_readiness_report.md",
        "phase2_phase3_handoff.md",
    ]
    for report in expected_reports:
        path = REPORT_ROOT / report
        require(path.exists() and path.stat().st_size > 1000, f"Missing report: {path}")

    fold_map = pd.read_csv(TABLE_ROOT / "adni_outer_fold_assignments.csv")
    require(fold_map["participant_id"].is_unique, "ADNI fold participant duplicated")
    require(len(fold_map) == 2447, "ADNI fold denominator mismatch")
    require(set(fold_map["outer_fold"]) == {0, 1, 2, 3, 4}, "Outer folds incomplete")

    baseline_core = {
        "participant_id",
        "horizon_status",
        "outcome_code",
        "age_at_landmark",
        "sex",
        "education_years",
        "apoe_e4_dose",
        "current_mmse",
        "elapsed_years_from_index",
    }
    dynamic_core = baseline_core | {
        "landmark_id",
        "landmark_time_days",
        "current_mmse_time_days",
        "previous_mmse_time_days",
        "slope_max_measurement_time_days",
        "primary_episode_id",
    }
    for cohort in COHORTS:
        for horizon in HORIZONS:
            baseline_path = (
                PROCESSED_ROOT
                / cohort
                / "phase2"
                / "baseline"
                / f"baseline_{horizon}m_clinical_core.parquet"
            )
            dynamic_path = (
                PROCESSED_ROOT
                / cohort
                / "phase2"
                / "dynamic"
                / f"dynamic_{horizon}m_clinical_core.parquet"
            )
            require(baseline_path.exists(), f"Missing {baseline_path}")
            require(dynamic_path.exists(), f"Missing {dynamic_path}")
            baseline = pd.read_parquet(baseline_path)
            dynamic = pd.read_parquet(dynamic_path)
            require(len(baseline) == expected_index[cohort], f"{cohort} index mismatch")
            require(baseline["participant_id"].is_unique, f"{cohort} baseline duplicate")
            require(baseline_core.issubset(baseline.columns), f"{cohort} baseline schema")
            require(dynamic_core.issubset(dynamic.columns), f"{cohort} dynamic schema")
            require(
                not dynamic.duplicated(["participant_id", "landmark_id"]).any(),
                f"{cohort} dynamic duplicate",
            )
            require(
                (
                    dynamic["current_mmse_time_days"].dropna()
                    <= dynamic.loc[
                        dynamic["current_mmse_time_days"].notna(), "landmark_time_days"
                    ]
                ).all(),
                f"{cohort} future current MMSE",
            )
            require(
                (
                    dynamic["slope_max_measurement_time_days"].dropna()
                    <= dynamic.loc[
                        dynamic["slope_max_measurement_time_days"].notna(),
                        "landmark_time_days",
                    ]
                ).all(),
                f"{cohort} future slope MMSE",
            )
            require(
                dynamic["transition_time_from_landmark_days"].dropna().gt(0).all(),
                f"{cohort} nonpositive outcome time",
            )
            require(
                not (
                    baseline["censored_before_horizon_flag"].fillna(False)
                    & baseline["remain_cdr_0_5_flag"].fillna(False)
                ).any(),
                f"{cohort} early censoring labelled persistent",
            )
            if cohort == "adni":
                require(
                    dynamic.groupby("participant_id")["outer_fold"].nunique().le(1).all(),
                    "ADNI participant split across dynamic folds",
                )

            lagged_path = (
                PROCESSED_ROOT
                / cohort
                / "phase2"
                / "sensitivity"
                / f"dynamic_{horizon}m_strictly_lagged_mmse.parquet"
            )
            lagged = pd.read_parquet(lagged_path)
            if not lagged.empty:
                require(
                    (
                        lagged["current_mmse_time_days"]
                        < lagged["landmark_time_days"]
                    ).all(),
                    f"{cohort} non-lagged MMSE in strict dataset",
                )
                require(
                    not lagged["same_day_mmse_flag"].fillna(False).any(),
                    f"{cohort} same-day MMSE in strict dataset",
                )

    agreement = pd.read_csv(TABLE_ROOT / "oasis3_cdr_agreement_statistics.csv")
    overall = agreement[agreement["form_family"].eq("ALL")].iloc[0]
    require(overall["exact_agreement_percent"] >= 99, "OASIS CDR agreement too low")
    comparison = pd.read_csv(TABLE_ROOT / "oasis3_cdr_field_comparison.csv")
    cdrglob = comparison[comparison["source_column"].eq("CDRGLOB")].iloc[0]
    require(cdrglob["observed_nonmissing_records"] == 0, "Unexpected CDRGLOB records")

    qc = pd.read_csv(TABLE_ROOT / "qc_results.csv")
    require(
        not ((qc["severity"].eq("critical")) & qc["status"].eq("FAIL")).any(),
        "Saved Phase 2 QC contains critical failure",
    )
    metadata = json.loads((TABLE_ROOT / "phase2_run_metadata.json").read_text())
    require(metadata["predictive_models_fitted"] is False, "Predictive model fitted")
    require(metadata["critical_qc_failures"] == 0, "Metadata records QC failure")

    pngs = sorted(PLOT_ROOT.glob("*.png"))
    require(len(pngs) >= 12, "Too few Phase 2 figures")
    for png in pngs:
        svg = png.with_suffix(".svg")
        require(svg.exists(), f"Missing vector counterpart: {svg}")
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
        require(min(dpi) >= 599, f"Raster DPI below 600: {png} ({dpi})")

    print("All saved Phase 2 artifact QC checks passed.")


if __name__ == "__main__":
    run_saved_artifact_qc()
