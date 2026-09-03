#!/usr/bin/env python3
"""Independent outcome-hidden prediction verifier for Phase 4.

This script does not import the prediction-generation script. It verifies all
frozen hashes, reloads each predictor-only extract, independently regenerates
point and time-grid predictions, and stops without opening outcomes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "models" / "phase3" / "phase4_lock_v2"
RUNTIME = LOCK / "inference_runtime"
TABLE_ROOT = ROOT / "results" / "csvs" / "phase4"
REPORT_ROOT = ROOT / "reports" / "phase4"
FEATURE_ROOT = (
    ROOT
    / "ProcessedData"
    / "oasis3"
    / "phase4"
    / "outcome_hidden_features"
)
PREDICTION_ROOT = (
    ROOT
    / "ProcessedData"
    / "oasis3"
    / "phase4"
    / "frozen_predictions"
)
GENERATED_AT = datetime.now(timezone.utc).isoformat()
TOLERANCE = 1e-10

sys.dont_write_bytecode = True
sys.path.insert(0, str(RUNTIME))

from phase3_inference import predict_from_frozen
from phase3_survival_core import paired_cumulative_incidence


OUTCOME_TOKENS = {
    "outcome",
    "event_type",
    "analysis_time_days",
    "horizon_status",
    "fixed_horizon_class",
    "outcome_code",
    "first_transition",
    "transition_time_days",
    "transition_time_months",
    "event_by_horizon_flag",
    "transition_to_cdr_0_flag",
    "transition_to_cdr_ge1_flag",
    "remain_cdr_0_5_flag",
    "censored_before_horizon_flag",
    "first_transition_after_landmark",
    "transition_time_from_landmark_days",
    "censoring_time_from_landmark_days",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def maximum_difference(left: pd.DataFrame, right: pd.DataFrame) -> float:
    columns = [
        "improvement_probability",
        "progression_probability",
        "remaining_cdr_0_5_probability",
        "probability_sum",
    ]
    return float(
        np.max(
            np.abs(
                left[columns].to_numpy(dtype=float)
                - right[columns].to_numpy(dtype=float)
            )
        )
    )


def main() -> None:
    freeze_path = TABLE_ROOT / "prediction_freeze_metadata.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / freeze["prediction_manifest_path"]
    manifest_hash_ok = sha256(manifest_path) == freeze[
        "prediction_manifest_sha256"
    ]
    manifest = pd.read_csv(manifest_path)
    inventory = pd.read_csv(LOCK / "model_inventory.csv")
    feature_manifest = pd.read_csv(
        TABLE_ROOT / "outcome_hidden_feature_manifest.csv"
    )

    file_rows: list[dict[str, Any]] = []
    for record in manifest.itertuples(index=False):
        path = ROOT / str(record.relative_path)
        exists = path.exists()
        hash_ok = exists and sha256(path) == str(record.sha256)
        bytes_ok = exists and path.stat().st_size == int(record.bytes)
        columns = pd.read_csv(path, nrows=0).columns.tolist() if exists else []
        forbidden = sorted(OUTCOME_TOKENS.intersection(columns))
        file_rows.append(
            {
                "model_id": record.model_id,
                "horizon_months": int(record.horizon_months),
                "file_role": record.file_role,
                "relative_path": record.relative_path,
                "hash_match": hash_ok,
                "byte_count_match": bytes_ok,
                "forbidden_outcome_columns": "|".join(forbidden),
                "outcome_columns_absent": not forbidden,
                "status": (
                    "PASS"
                    if hash_ok and bytes_ok and not forbidden
                    else "FAIL"
                ),
            }
        )
    file_verification = pd.DataFrame(file_rows)

    model_rows: list[dict[str, Any]] = []
    for record in inventory.sort_values(
        ["model_id", "horizon_months"]
    ).itertuples(index=False):
        model_id = str(record.model_id)
        horizon = int(record.horizon_months)
        artifact = ROOT / str(record.artifact_path)
        bundle = joblib.load(artifact)
        feature_record = feature_manifest[
            feature_manifest["model_id"].eq(model_id)
            & feature_manifest["horizon_months"].eq(horizon)
        ].iloc[0]
        feature_path = ROOT / str(feature_record.relative_path)
        feature = pd.read_csv(feature_path)
        feature_hash_ok = sha256(feature_path) == str(feature_record.sha256)
        forbidden_feature_columns = sorted(
            OUTCOME_TOKENS.intersection(feature.columns)
        )
        required = list(bundle["continuous_features"]) + list(
            bundle["categorical_features"]
        )
        required_exact = (
            set(required).issubset(feature.columns)
            and feature[required].columns.tolist() == required
        )

        point_record = manifest[
            manifest["model_id"].eq(model_id)
            & manifest["horizon_months"].eq(horizon)
            & manifest["file_role"].eq("point_prediction")
        ].iloc[0]
        curve_record = manifest[
            manifest["model_id"].eq(model_id)
            & manifest["horizon_months"].eq(horizon)
            & manifest["file_role"].eq("time_grid_prediction")
        ].iloc[0]
        saved_point = pd.read_csv(ROOT / str(point_record.relative_path))
        saved_curve = pd.read_csv(ROOT / str(curve_record.relative_path))

        regenerated = predict_from_frozen(feature[required], artifact)
        point_difference = maximum_difference(regenerated, saved_point)
        point_metadata_exact = (
            regenerated[
                [
                    "prediction_horizon_months",
                    "model_id",
                    "model_version_id",
                    "quality_flags",
                ]
            ]
            .reset_index(drop=True)
            .equals(
                saved_point[
                    [
                        "prediction_horizon_months",
                        "model_id",
                        "model_version_id",
                        "quality_flags",
                    ]
                ].reset_index(drop=True)
            )
        )

        transformed = bundle["preprocessor"].transform(feature[required])
        evaluation_times = np.sort(
            saved_curve["evaluation_time_days"].unique()
        )
        pi, pp, pr = paired_cumulative_incidence(
            bundle["improvement_model"],
            bundle["progression_model"],
            transformed.to_numpy(),
            evaluation_times,
        )
        regenerated_parts = []
        base = feature[["phase4_row_key", "participant_id"]].reset_index(drop=True)
        for index, evaluation_time in enumerate(evaluation_times):
            regenerated_parts.append(
                base.assign(
                    evaluation_time_days=evaluation_time,
                    improvement_probability=pi[:, index],
                    progression_probability=pp[:, index],
                    remaining_cdr_0_5_probability=pr[:, index],
                    probability_sum=pi[:, index] + pp[:, index] + pr[:, index],
                )
            )
        regenerated_curve = pd.concat(regenerated_parts, ignore_index=True)
        curve_difference = maximum_difference(
            regenerated_curve,
            saved_curve,
        )
        curve_ids_exact = regenerated_curve[
            ["phase4_row_key", "participant_id", "evaluation_time_days"]
        ].equals(
            saved_curve[
                ["phase4_row_key", "participant_id", "evaluation_time_days"]
            ]
        )
        sums_ok = np.allclose(
            regenerated["probability_sum"].to_numpy(dtype=float),
            1.0,
            atol=TOLERANCE,
            rtol=0,
        )
        probability_values = regenerated[
            [
                "improvement_probability",
                "progression_probability",
                "remaining_cdr_0_5_probability",
            ]
        ].to_numpy(dtype=float)
        bounded = (probability_values >= -TOLERANCE).all() and (
            probability_values <= 1 + TOLERANCE
        ).all()
        passed = (
            feature_hash_ok
            and not forbidden_feature_columns
            and required_exact
            and point_difference <= TOLERANCE
            and curve_difference <= TOLERANCE
            and point_metadata_exact
            and curve_ids_exact
            and sums_ok
            and bounded
        )
        model_rows.append(
            {
                "model_id": model_id,
                "family": record.family,
                "horizon_months": horizon,
                "rows": len(feature),
                "participants": feature["participant_id"].nunique(),
                "feature_hash_match": feature_hash_ok,
                "outcome_hidden_feature_columns": not forbidden_feature_columns,
                "required_feature_order_exact": required_exact,
                "maximum_point_prediction_difference": point_difference,
                "point_metadata_exact": point_metadata_exact,
                "maximum_curve_prediction_difference": curve_difference,
                "curve_identifiers_exact": curve_ids_exact,
                "probabilities_bounded": bounded,
                "probabilities_sum_to_one": sums_ok,
                "status": "PASS" if passed else "FAIL",
            }
        )

    model_verification = pd.DataFrame(model_rows)
    all_pass = bool(
        manifest_hash_ok
        and file_verification["status"].eq("PASS").all()
        and model_verification["status"].eq("PASS").all()
        and not bool(freeze["oasis_outcomes_accessed_before_freeze"])
    )
    save_csv(
        file_verification,
        TABLE_ROOT / "independent_prediction_file_verification.csv",
    )
    save_csv(
        model_verification,
        TABLE_ROOT / "independent_prediction_regeneration.csv",
    )
    summary = pd.DataFrame(
        [
            {
                "check": "prediction manifest hash matches pre-outcome freeze",
                "status": "PASS" if manifest_hash_ok else "FAIL",
                "details": f"manifest={rel(manifest_path)}",
            },
            {
                "check": "all frozen prediction file hashes and schemas pass",
                "status": (
                    "PASS"
                    if file_verification["status"].eq("PASS").all()
                    else "FAIL"
                ),
                "details": (
                    f"files={len(file_verification)}; "
                    f"failures={int(file_verification['status'].ne('PASS').sum())}"
                ),
            },
            {
                "check": "all point and curve predictions regenerate exactly",
                "status": (
                    "PASS"
                    if model_verification["status"].eq("PASS").all()
                    else "FAIL"
                ),
                "details": (
                    f"models={len(model_verification)}; "
                    f"max_point={model_verification['maximum_point_prediction_difference'].max():.3g}; "
                    f"max_curve={model_verification['maximum_curve_prediction_difference'].max():.3g}"
                ),
            },
            {
                "check": "no outcome field is present or was accessed",
                "status": (
                    "PASS"
                    if file_verification["outcome_columns_absent"].all()
                    and model_verification[
                        "outcome_hidden_feature_columns"
                    ].all()
                    and not bool(
                        freeze["oasis_outcomes_accessed_before_freeze"]
                    )
                    else "FAIL"
                ),
                "details": "predictor-only extracts and frozen predictions inspected",
            },
        ]
    )
    save_csv(
        summary,
        TABLE_ROOT / "phase4_independent_prediction_qc_results.csv",
    )

    access_path = TABLE_ROOT / "phase4_access_audit.csv"
    access = pd.read_csv(access_path)
    access = pd.concat(
        [
            access,
            pd.DataFrame(
                [
                    {
                        "sequence": len(access) + 1,
                        "timestamp_utc": GENERATED_AT,
                        "stage": "C",
                        "action": "independently regenerate every outcome-hidden point and curve prediction",
                        "resource_scope": (
                            "frozen predictor extracts, phase4_lock_v2, and "
                            "frozen prediction files"
                        ),
                        "loaded_columns": "identifiers, frozen predictors, predictions",
                        "oasis_record_level_features_accessed": True,
                        "oasis_record_level_outcomes_accessed": False,
                        "outcome_hidden_predictions_created": True,
                        "outcomes_first_accessed_at_utc": "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    save_csv(access, access_path)

    freeze["independent_prediction_qc_completed_at_utc"] = GENERATED_AT
    freeze["independent_prediction_qc_pending"] = False
    freeze["independent_prediction_qc_pass"] = all_pass
    freeze["phase4_outcomes_merged"] = False
    save_json(freeze, freeze_path)
    run_path = TABLE_ROOT / "phase4_run_metadata.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["stage_c_independent_prediction_qc_pass"] = all_pass
    run["oasis_record_level_outcomes_accessed"] = False
    run["decision"] = (
        "HOLD FOR VALIDATION AUDIT" if all_pass else "HOLD FOR LOCK REPAIR"
    )
    save_json(run, run_path)

    report = f"""# Phase 4 Independent Outcome-Hidden Prediction QC

Generated: {GENERATED_AT}

## Decision

**{'PASS — OUTCOME GATE MAY OPEN' if all_pass else 'FAIL — KEEP OUTCOMES CLOSED'}**

{summary.to_markdown(index=False)}

## Independent model regeneration

{model_verification.to_markdown(index=False)}

No OASIS outcome column was opened by this verifier.
"""
    (REPORT_ROOT / "phase4_prediction_independent_qc_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "prediction_files_verified": len(file_verification),
                "model_horizons_regenerated": len(model_verification),
                "maximum_point_difference": float(
                    model_verification[
                        "maximum_point_prediction_difference"
                    ].max()
                ),
                "maximum_curve_difference": float(
                    model_verification[
                        "maximum_curve_prediction_difference"
                    ].max()
                ),
                "outcomes_accessed": False,
                "outcome_gate": "OPEN" if all_pass else "CLOSED",
            },
            indent=2,
        )
    )
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
