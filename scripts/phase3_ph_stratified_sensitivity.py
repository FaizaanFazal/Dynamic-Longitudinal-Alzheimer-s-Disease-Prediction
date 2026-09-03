#!/usr/bin/env python3
"""Parsimonious elapsed-time stratification sensitivity for D1 PH signals.

This is diagnostic only. It reuses the D1 features, frozen outer folds, and
already inner-selected penalties. Within each outer-training partition it fits
separate cause-specific Cox pairs for landmarks before 2 years and at/after 2
years from index. It never alters the frozen primary D1 package.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_preprocessing import ClinicalCorePreprocessor
from phase3_adni_model_development import (
    CONFIG,
    PRIMARY_HORIZONS,
    ROOT,
    SEED,
    TABLE_ROOT,
    add_survival_endpoints,
    save_table,
)
from phase3_survival_core import (
    KaplanMeierCensoring,
    RidgeCauseSpecificCox,
    fixed_horizon_ipcw,
    horizon_metric_record,
    paired_cumulative_incidence,
)


STRATIFICATION_THRESHOLD_YEARS = 2.0
FEATURES = (
    "age_at_landmark",
    "education_years",
    "apoe_e4_dose",
    "current_mmse",
    "mmse_change_from_index",
    "elapsed_years_from_index",
)
CATEGORICAL = ("sex",)


def calibration_deviation(row: pd.Series) -> float:
    return float(
        abs(row["calibration_intercept_improvement"])
        + abs(row["calibration_intercept_progression"])
        + abs(row["calibration_slope_improvement"] - 1)
        + abs(row["calibration_slope_progression"] - 1)
        + abs(np.log(max(row["observed_expected_improvement"], 1e-6)))
        + abs(np.log(max(row["observed_expected_progression"], 1e-6)))
    )


def main() -> None:
    episode = pd.read_csv(
        ROOT
        / "results"
        / "csvs"
        / "phase1"
        / "adni"
        / "primary_episode_time_to_event.csv"
    )
    penalties = pd.read_csv(
        TABLE_ROOT / "fold_specific_ridge_penalties.csv"
    )
    prediction_parts: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    for horizon in PRIMARY_HORIZONS:
        raw = pd.read_csv(
            ROOT
            / "ProcessedData"
            / "adni"
            / "phase2"
            / "dynamic"
            / f"dynamic_{horizon}m_clinical_core.csv"
        )
        data = add_survival_endpoints(raw, "dynamic", horizon, episode)
        data["_analysis_weight"] = 1.0
        data["sex"] = data["sex"].astype("string").fillna("Missing")
        data["_elapsed_stratum"] = np.where(
            data["elapsed_years_from_index"].lt(
                STRATIFICATION_THRESHOLD_YEARS
            ),
            "early_lt_2y",
            "late_ge_2y",
        )
        horizon_days = float(horizon * CONFIG["days_per_month"])
        for outer_fold in range(5):
            train = data[data["outer_fold"].ne(outer_fold)].copy()
            validation = data[data["outer_fold"].eq(outer_fold)].copy()
            preprocessor = ClinicalCorePreprocessor(
                FEATURES, CATEGORICAL
            ).fit(
                train,
                cohort="adni",
                partition_label=(
                    f"phase3_ph_stratified_D1_{horizon}m_outer"
                    f"{outer_fold}_training"
                ),
            )
            x_train = preprocessor.transform(train)
            x_validation = preprocessor.transform(validation)
            pooled_models: dict[str, RidgeCauseSpecificCox] = {}
            stratum_models: dict[
                tuple[str, str], RidgeCauseSpecificCox
            ] = {}
            for cause, event_col in [
                ("improvement", "_event_improvement"),
                ("progression", "_event_progression"),
            ]:
                alpha = float(
                    penalties[
                        penalties["model_id"].eq("D1")
                        & penalties["horizon_months"].eq(horizon)
                        & penalties["outer_fold"].eq(outer_fold)
                        & penalties["cause"].eq(cause)
                    ]["selected_alpha"].iloc[0]
                )
                pooled_models[cause] = RidgeCauseSpecificCox(alpha).fit(
                    x_train.to_numpy(),
                    train["_time"].to_numpy(),
                    train[event_col].to_numpy(),
                )
                for stratum in ["early_lt_2y", "late_ge_2y"]:
                    mask = train["_elapsed_stratum"].eq(stratum)
                    events = int(train.loc[mask, event_col].sum())
                    if events >= 10:
                        model = RidgeCauseSpecificCox(alpha).fit(
                            x_train.loc[mask].to_numpy(),
                            train.loc[mask, "_time"].to_numpy(),
                            train.loc[mask, event_col].to_numpy(),
                        )
                        fallback = False
                    else:
                        model = pooled_models[cause]
                        fallback = True
                    stratum_models[(stratum, cause)] = model
                    fit_rows.append(
                        {
                            "model_id": "D1_elapsed_stratified",
                            "horizon_months": horizon,
                            "outer_fold": outer_fold,
                            "elapsed_stratum": stratum,
                            "cause": cause,
                            "training_rows": int(mask.sum()),
                            "training_participants": train.loc[
                                mask, "participant_id"
                            ].nunique(),
                            "training_events": events,
                            "alpha_reused_from_inner_selected_D1": alpha,
                            "pooled_fallback": fallback,
                            "converged": model.converged_,
                        }
                    )
            fold_parts: list[pd.DataFrame] = []
            for stratum in ["early_lt_2y", "late_ge_2y"]:
                mask = validation["_elapsed_stratum"].eq(stratum)
                if not mask.any():
                    continue
                pi, pp, pr = paired_cumulative_incidence(
                    stratum_models[(stratum, "improvement")],
                    stratum_models[(stratum, "progression")],
                    x_validation.loc[mask].to_numpy(),
                    np.array([horizon_days]),
                )
                fold_parts.append(
                    pd.DataFrame(
                        {
                            "_index": validation.index[mask],
                            "participant_id": validation.loc[
                                mask, "participant_id"
                            ].to_numpy(),
                            "row_key": validation.loc[
                                mask, "landmark_id"
                            ].astype(str).to_numpy(),
                            "outer_fold": outer_fold,
                            "elapsed_stratum": stratum,
                            "analysis_time_days": validation.loc[
                                mask, "_time"
                            ].to_numpy(),
                            "event_type": validation.loc[
                                mask, "_event_type"
                            ].to_numpy(),
                            "probability_improvement": pi[:, 0],
                            "probability_progression": pp[:, 0],
                            "probability_remaining": pr[:, 0],
                        }
                    )
                )
            fold_prediction = (
                pd.concat(fold_parts, ignore_index=True)
                .sort_values("_index")
                .drop(columns="_index")
            )
            censor = KaplanMeierCensoring.fit(
                train["_time"].to_numpy(),
                train["_censoring_event"].to_numpy(),
            )
            fold_prediction["ipcw_weight"] = fixed_horizon_ipcw(
                fold_prediction["analysis_time_days"].to_numpy(),
                fold_prediction["event_type"].to_numpy(),
                horizon_days,
                censor,
            )
            fold_prediction["model_id"] = "D1_elapsed_stratified"
            fold_prediction["horizon_months"] = horizon
            prediction_parts.append(fold_prediction)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    save_table(
        predictions,
        TABLE_ROOT
        / "predictions"
        / "ph_elapsed_stratified_sensitivity_predictions.csv",
    )
    fit_audit = pd.DataFrame(fit_rows)
    save_table(
        fit_audit, TABLE_ROOT / "ph_elapsed_stratified_fit_audit.csv"
    )
    metric_rows = []
    for horizon, group in predictions.groupby("horizon_months"):
        metrics = horizon_metric_record(
            probability_improvement=group[
                "probability_improvement"
            ].to_numpy(),
            probability_progression=group[
                "probability_progression"
            ].to_numpy(),
            probability_remaining=group[
                "probability_remaining"
            ].to_numpy(),
            time=group["analysis_time_days"].to_numpy(),
            event_type=group["event_type"].to_numpy(),
            horizon=float(horizon * CONFIG["days_per_month"]),
            ipcw=group["ipcw_weight"].to_numpy(),
        )
        metric_rows.append(
            {
                "model_id": "D1_elapsed_stratified",
                "horizon_months": horizon,
                "rows": len(group),
                "participants": group["participant_id"].nunique(),
                **metrics,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    primary = pd.read_csv(
        TABLE_ROOT / "discrimination_and_horizon_metrics.csv"
    )
    primary = primary[
        primary["model_id"].eq("D1")
        & primary["horizon_months"].isin(PRIMARY_HORIZONS)
    ]
    comparison_rows = []
    for horizon in PRIMARY_HORIZONS:
        simple = primary[primary["horizon_months"].eq(horizon)].iloc[0]
        stratified = metrics[
            metrics["horizon_months"].eq(horizon)
        ].iloc[0]
        comparison_rows.append(
            {
                "horizon_months": horizon,
                "simple_multistate_brier": simple["multistate_brier"],
                "stratified_multistate_brier": stratified[
                    "multistate_brier"
                ],
                "stratified_minus_simple_brier": stratified[
                    "multistate_brier"
                ]
                - simple["multistate_brier"],
                "simple_calibration_deviation": calibration_deviation(simple),
                "stratified_calibration_deviation": calibration_deviation(
                    stratified
                ),
                "simple_auc_improvement": simple["auc_improvement"],
                "stratified_auc_improvement": stratified[
                    "auc_improvement"
                ],
                "simple_auc_progression": simple["auc_progression"],
                "stratified_auc_progression": stratified[
                    "auc_progression"
                ],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    retain_simple = not (
        comparison["stratified_minus_simple_brier"].le(-0.002).all()
        and (
            comparison["stratified_calibration_deviation"]
            <= comparison["simple_calibration_deviation"] + 0.25
        ).all()
    )
    comparison["selection"] = (
        "retain simple D1"
        if retain_simple
        else "elapsed-stratified sensitivity merits model revision"
    )
    comparison["rule"] = (
        "replace simple D1 only if stratified multi-state Brier improves by "
        "at least 0.002 at both primary horizons without calibration-deviation "
        "worsening by more than 0.25"
    )
    save_table(
        metrics, TABLE_ROOT / "ph_elapsed_stratified_sensitivity_metrics.csv"
    )
    save_table(
        comparison,
        TABLE_ROOT / "ph_elapsed_stratified_sensitivity_comparison.csv",
    )
    report_path = (
        ROOT / "reports" / "phase3" / "phase3_model_diagnostics.md"
    )
    report = report_path.read_text(encoding="utf-8")
    section = f"""

## Parsimonious elapsed-time stratification sensitivity

Because elapsed time showed nominal Schoenfeld signals, D1 was refitted inside
each outer-training fold with separate `<2 years` and `≥2 years` landmark
strata. The already inner-selected D1 penalties and identical feature set were
reused; no outer-validation choice or OASIS information entered the fit.

{comparison.round(4).to_markdown(index=False)}

Decision: **{comparison["selection"].iloc[0]}**. The frozen simple D1 package
was not altered.
"""
    marker = "\n## Parsimonious elapsed-time stratification sensitivity"
    if marker in report:
        report = report.split(marker)[0].rstrip() + "\n"
    report_path.write_text(report.rstrip() + section, encoding="utf-8")
    metadata_path = TABLE_ROOT / "phase3_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["ph_elapsed_stratified_sensitivity"] = {
        "threshold_years": STRATIFICATION_THRESHOLD_YEARS,
        "all_fits_converged": bool(fit_audit["converged"].all()),
        "selection": comparison["selection"].iloc[0],
        "comparison_table": (
            "results/csvs/phase3/"
            "ph_elapsed_stratified_sensitivity_comparison.csv"
        ),
        "frozen_model_changed": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "selection": comparison["selection"].iloc[0],
                "rows": len(predictions),
                "all_converged": bool(fit_audit["converged"].all()),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
