#!/usr/bin/env python3
"""Stage A/B of locked OASIS-3 external validation.

This one-shot script:

1. re-verifies the repaired Phase 3 lock;
2. reads only explicitly selected OASIS predictor/identifier columns;
3. applies frozen ADNI preprocessing and models without fitting;
4. freezes point and time-grid predictions with hashes;
5. stops before opening any outcome column.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
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
CONFIG_PATH = ROOT / "config" / "phase4_analysis_parameters.json"
TABLE_ROOT = ROOT / "results" / "csvs" / "phase4"
REPORT_ROOT = ROOT / "reports" / "phase4"
PHASE3_QC_ROOT = ROOT / "results" / "csvs" / "phase3_lock_repair"
PHASE4_DATA_ROOT = ROOT / "ProcessedData" / "oasis3" / "phase4"
FEATURE_ROOT = PHASE4_DATA_ROOT / "outcome_hidden_features"
PREDICTION_ROOT = PHASE4_DATA_ROOT / "frozen_predictions"
FEATURE_BUILD = PHASE4_DATA_ROOT / ".outcome_hidden_features.building"
PREDICTION_BUILD = PHASE4_DATA_ROOT / ".frozen_predictions.building"
GENERATED_AT = datetime.now(timezone.utc).isoformat()

TABLE_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
PHASE4_DATA_ROOT.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUNTIME))

from phase3_inference import predict_from_frozen
from phase3_survival_core import paired_cumulative_incidence


CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
TOLERANCE = float(CONFIG["probability_sum_tolerance"])
GRID_POINTS = int(CONFIG["integrated_brier_grid_points"])

OUTCOME_FIELD_NAMES = {
    "eligible_for_fixed_horizon_analysis",
    "censoring_status",
    "horizon_status",
    "fixed_horizon_class",
    "outcome_code",
    "first_transition",
    "transition_time_days",
    "transition_time_months",
    "observed_through_horizon_flag",
    "event_by_horizon_flag",
    "transition_to_cdr_0_flag",
    "transition_to_cdr_ge1_flag",
    "remain_cdr_0_5_flag",
    "censored_before_horizon_flag",
    "first_transition_after_landmark",
    "transition_manifest_key",
    "transition_time_from_landmark_days",
    "transition_time_from_landmark_months",
    "last_classifiable_time_days",
    "censoring_time_from_landmark_days",
    "outcome_assessment_mmse_same_day_flag",
    "external_outcome",
    "event_type",
    "analysis_time_days",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame, sort_columns: list[str]) -> str:
    work = frame.sort_values(sort_columns, kind="mergesort").copy()
    return hashlib.sha256(
        work.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def save_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def save_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def frozen_parameter_hash(
    table: pd.DataFrame,
    model_id: str,
    horizon: int,
    sort_columns: list[str],
) -> str:
    subset = table[
        table["model_id"].eq(model_id)
        & table["horizon_months"].eq(horizon)
    ]
    if subset.empty:
        raise RuntimeError(f"Frozen parameter table missing {model_id} {horizon}m.")
    return dataframe_sha256(subset, sort_columns)


def source_path(model_id: str, family: str, horizon: int) -> Path:
    if model_id == "D3":
        return (
            ROOT
            / "ProcessedData"
            / "oasis3"
            / "phase2"
            / "sensitivity"
            / f"dynamic_{horizon}m_strictly_lagged_mmse.csv"
        )
    return (
        ROOT
        / "ProcessedData"
        / "oasis3"
        / "phase2"
        / family
        / f"{family}_{horizon}m_clinical_core.csv"
    )


def identifier_columns(header: list[str], family: str) -> list[str]:
    baseline = [
        "participant_id",
        "primary_episode_id",
        "index_time_days",
    ]
    dynamic = baseline + [
        "landmark_id",
        "landmark_manifest_key",
        "landmark_number",
        "landmark_time_days",
    ]
    return [column for column in (baseline if family == "baseline" else dynamic) if column in header]


def expected_population(model_id: str) -> tuple[int, int]:
    return {
        "B1": (491, 491),
        "B3": (491, 491),
        "D0": (858, 437),
        "D1": (845, 437),
        "D3": (421, 207),
    }[model_id]


def lock_stage_a() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    phase3_qc = pd.read_csv(PHASE3_QC_ROOT / "independent_qc_results.csv")
    phase3_meta = json.loads(
        (PHASE3_QC_ROOT / "phase3_lock_repair_run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    checks: list[dict[str, Any]] = []

    def add(check_id: str, description: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "stage": "A",
                "description": description,
                "severity": "critical",
                "status": "PASS" if passed else "FAIL",
                "details": details,
            }
        )

    add(
        "A01",
        "independent repaired-lock QC is current and fully passing",
        bool(phase3_meta["independent_qc_pass"])
        and phase3_qc["status"].eq("PASS").all(),
        (
            f"checks={len(phase3_qc)}; "
            f"failures={int(phase3_qc['status'].ne('PASS').sum())}"
        ),
    )

    external_manifest = json.loads(
        (LOCK / "external_validation_lock_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    index_path = LOCK / "frozen_model_index.json"
    manifest_path = LOCK / "checksum_manifest.csv"
    add(
        "A02",
        "external manifest authenticates the frozen index and checksum manifest",
        sha256(index_path) == external_manifest["model_index_sha256"]
        and sha256(manifest_path)
        == external_manifest["checksum_manifest_sha256"],
        (
            f"index={sha256(index_path)}; "
            f"manifest={sha256(manifest_path)}"
        ),
    )

    manifest = pd.read_csv(manifest_path)
    hash_rows: list[dict[str, Any]] = []
    for record in manifest.itertuples(index=False):
        path = LOCK / str(record.package_relative_path)
        exists = path.exists()
        observed_hash = sha256(path) if exists else ""
        observed_bytes = path.stat().st_size if exists else -1
        passed = (
            exists
            and observed_hash == str(record.sha256)
            and observed_bytes == int(record.bytes)
        )
        hash_rows.append(
            {
                "artifact_path": rel(path) if exists else str(path),
                "expected_sha256": record.sha256,
                "observed_sha256": observed_hash,
                "expected_bytes": int(record.bytes),
                "observed_bytes": observed_bytes,
                "status": "PASS" if passed else "FAIL",
            }
        )
    hashes = pd.DataFrame(hash_rows)
    add(
        "A03",
        "all phase4_lock_v2 checksums and byte sizes pass",
        hashes["status"].eq("PASS").all(),
        f"{hashes['status'].eq('PASS').sum()}/{len(hashes)} PASS",
    )

    inventory = pd.read_csv(LOCK / "model_inventory.csv")
    expected = {
        ("B1", 12), ("B1", 24), ("B1", 36), ("B1", 60),
        ("D0", 12), ("D0", 36),
        ("D1", 12), ("D1", 24), ("D1", 36), ("D1", 60),
        ("B3", 12), ("B3", 36),
        ("D3", 12), ("D3", 36),
    }
    observed = {
        (str(row.model_id), int(row.horizon_months))
        for row in inventory.itertuples(index=False)
    }
    add(
        "A04",
        "all Phase 4 required model/horizon bundles are present",
        observed == expected,
        f"present={sorted(observed)}; missing={sorted(expected - observed)}",
    )

    artifact_qc = pd.read_csv(
        PHASE3_QC_ROOT / "independent_model_artifact_verification.csv"
    )
    add(
        "A05",
        "schemas, preprocessing, coefficients, penalties, hazards and counts pass",
        len(artifact_qc) == 14 and artifact_qc["status"].eq("PASS").all(),
        f"bundles={len(artifact_qc)}; failures={int(artifact_qc['status'].ne('PASS').sum())}",
    )
    fixture_qc = pd.read_csv(
        PHASE3_QC_ROOT / "independent_inference_fixture_verification.csv"
    )
    equivalence_qc = pd.read_csv(
        PHASE3_QC_ROOT / "independent_b1_d1_equivalence.csv"
    )
    add(
        "A06",
        "inference compatibility and original-copy equivalence pass",
        fixture_qc["status"].eq("PASS").all()
        and equivalence_qc["status"].eq("PASS").all(),
        (
            f"new_fixtures={len(fixture_qc)}; "
            f"copy_comparisons={len(equivalence_qc)}"
        ),
    )
    add(
        "A07",
        "lock development cohort is ADNI and external outcomes were unused",
        (~inventory["external_outcomes_accessed"].map(bool_value)).all(),
        "all 14 bundles external_outcomes_accessed=False",
    )
    add(
        "A08",
        "Stage A accessed no OASIS participant-level feature or outcome file",
        True,
        "Stage A read only the repaired lock and its independent QC artifacts",
    )
    stage_a = pd.DataFrame(checks)
    if not stage_a["status"].eq("PASS").all():
        save_csv(stage_a, TABLE_ROOT / "phase4_stage_a_qc_results.csv")
        raise RuntimeError("Critical Phase 4 Stage A lock verification failed.")
    return stage_a, hashes, phase3_meta


def main() -> None:
    if FEATURE_ROOT.exists() or PREDICTION_ROOT.exists():
        raise FileExistsError(
            "Outcome-hidden Phase 4 files already exist; refusing to overwrite "
            "frozen predictions."
        )
    for building in [FEATURE_BUILD, PREDICTION_BUILD]:
        if building.exists():
            shutil.rmtree(building)
        building.mkdir(parents=True)

    stage_a, hashes, phase3_meta = lock_stage_a()
    save_csv(stage_a, TABLE_ROOT / "phase4_stage_a_qc_results.csv")
    save_csv(hashes, TABLE_ROOT / "lock_checksum_verification.csv")
    shutil.copy2(
        PHASE3_QC_ROOT / "independent_model_artifact_verification.csv",
        TABLE_ROOT / "frozen_model_component_verification.csv",
    )
    shutil.copy2(
        PHASE3_QC_ROOT / "independent_inference_fixture_verification.csv",
        TABLE_ROOT / "deterministic_inference_verification.csv",
    )

    access_rows: list[dict[str, Any]] = [
        {
            "sequence": 1,
            "timestamp_utc": GENERATED_AT,
            "stage": "A",
            "action": "re-run complete phase4_lock_v2 verification",
            "resource_scope": rel(LOCK),
            "loaded_columns": "lock/QC artifacts only",
            "oasis_record_level_features_accessed": False,
            "oasis_record_level_outcomes_accessed": False,
            "outcome_hidden_predictions_created": False,
            "outcomes_first_accessed_at_utc": "",
        },
        {
            "sequence": 2,
            "timestamp_utc": GENERATED_AT,
            "stage": "B_PREFLIGHT",
            "action": "predictor-only population/schema preflight after Stage A PASS",
            "resource_scope": "OASIS Phase 2 files via identifier/predictor usecols",
            "loaded_columns": "identifiers and frozen predictor columns only",
            "oasis_record_level_features_accessed": True,
            "oasis_record_level_outcomes_accessed": False,
            "outcome_hidden_predictions_created": False,
            "outcomes_first_accessed_at_utc": "",
        },
    ]

    inventory = pd.read_csv(LOCK / "model_inventory.csv").sort_values(
        ["model_id", "horizon_months"]
    )
    preprocessing_table = pd.read_csv(
        LOCK / "frozen_preprocessing_parameters.csv"
    )
    schema_table = pd.read_csv(LOCK / "frozen_feature_schema.csv")
    feature_rows: list[dict[str, Any]] = []
    prediction_manifest_rows: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    coherence_rows: list[dict[str, Any]] = []

    for sequence, record in enumerate(inventory.itertuples(index=False), start=1):
        model_id = str(record.model_id)
        horizon = int(record.horizon_months)
        family = str(record.family)
        artifact = ROOT / str(record.artifact_path)
        bundle = joblib.load(artifact)
        source = source_path(model_id, family, horizon)
        header = pd.read_csv(source, nrows=0).columns.tolist()
        identifiers = identifier_columns(header, family)
        features = list(bundle["continuous_features"]) + list(
            bundle["categorical_features"]
        )
        loaded_columns = identifiers + features
        if OUTCOME_FIELD_NAMES.intersection(loaded_columns):
            raise RuntimeError("Outcome field requested during predictor-only stage.")
        frame = pd.read_csv(source, usecols=loaded_columns)
        if model_id == "D1" and CONFIG[
            "d1_requires_observed_current_and_change_mmse"
        ]:
            frame = frame.dropna(
                subset=["current_mmse", "mmse_change_from_index"]
            ).copy()
        key_column = (
            "participant_id"
            if family == "baseline"
            else "landmark_manifest_key"
        )
        frame.insert(0, "phase4_row_key", frame[key_column].astype(str))
        if frame["phase4_row_key"].duplicated().any():
            raise RuntimeError(f"Duplicate prediction row key for {model_id}.")
        expected_rows, expected_participants = expected_population(model_id)
        observed_population = (
            len(frame),
            frame["participant_id"].nunique(),
        )
        population_ok = observed_population == (
            expected_rows,
            expected_participants,
        )
        population_rows.append(
            {
                "model_id": model_id,
                "family": family,
                "horizon_months": horizon,
                "expected_rows": expected_rows,
                "observed_rows": len(frame),
                "expected_participants": expected_participants,
                "observed_participants": frame["participant_id"].nunique(),
                "d1_observed_mmse_eligibility_applied": model_id == "D1",
                "status": "PASS" if population_ok else "FAIL",
            }
        )
        if not population_ok:
            raise RuntimeError(
                f"Unexpected OASIS predictor population for {model_id}: "
                f"{observed_population}."
            )

        feature_name = f"{model_id.lower()}_{horizon}m_outcome_hidden_features.csv"
        feature_path = FEATURE_BUILD / feature_name
        save_csv(frame, feature_path)
        feature_hash = sha256(feature_path)
        preprocessing_hash = frozen_parameter_hash(
            preprocessing_table,
            model_id,
            horizon,
            ["feature"],
        )
        schema_hash = frozen_parameter_hash(
            schema_table,
            model_id,
            horizon,
            ["feature"],
        )

        predictor_input = frame[features].copy()
        first = predict_from_frozen(predictor_input, artifact)
        second = predict_from_frozen(predictor_input, artifact)
        probability_columns = [
            "improvement_probability",
            "progression_probability",
            "remaining_cdr_0_5_probability",
            "probability_sum",
        ]
        repeat_difference = float(
            np.max(
                np.abs(
                    first[probability_columns].to_numpy(dtype=float)
                    - second[probability_columns].to_numpy(dtype=float)
                )
            )
        )
        probabilities = first[
            [
                "improvement_probability",
                "progression_probability",
                "remaining_cdr_0_5_probability",
            ]
        ].to_numpy(dtype=float)
        bounded = bool(
            (probabilities >= -TOLERANCE).all()
            and (probabilities <= 1 + TOLERANCE).all()
        )
        coherent = bool(
            np.allclose(
                first["probability_sum"].to_numpy(dtype=float),
                1.0,
                atol=TOLERANCE,
                rtol=0,
            )
        )
        if not (repeat_difference <= TOLERANCE and bounded and coherent):
            raise RuntimeError(f"Prediction verification failed for {model_id}.")

        metadata_columns = identifiers.copy()
        point = frame[["phase4_row_key"] + metadata_columns].reset_index(drop=True)
        point = pd.concat([point, first.reset_index(drop=True)], axis=1)
        point.insert(2, "family", family)
        point["model_artifact_sha256"] = sha256(artifact)
        point["preprocessing_parameters_sha256"] = preprocessing_hash
        point["feature_schema_sha256"] = schema_hash
        point["input_feature_file_sha256"] = feature_hash
        point["outcome_columns_present"] = False
        prediction_name = f"{model_id.lower()}_{horizon}m_predictions.csv"
        prediction_path = PREDICTION_BUILD / prediction_name
        save_csv(point, prediction_path)

        transformed = bundle["preprocessor"].transform(predictor_input)
        if transformed.columns.tolist() != list(
            bundle["transformed_feature_order"]
        ):
            raise RuntimeError("Frozen transformed feature order changed.")
        horizon_days = float(bundle["horizon_days"])
        evaluation_times = np.linspace(
            horizon_days / GRID_POINTS,
            horizon_days,
            GRID_POINTS,
        )
        pi, pp, pr = paired_cumulative_incidence(
            bundle["improvement_model"],
            bundle["progression_model"],
            transformed.to_numpy(),
            evaluation_times,
        )
        base = frame[["phase4_row_key", "participant_id"]].reset_index(drop=True)
        curve_parts = []
        for index, evaluation_time in enumerate(evaluation_times):
            curve_parts.append(
                base.assign(
                    model_id=model_id,
                    family=family,
                    horizon_months=horizon,
                    model_version_id=bundle["model_version_id"],
                    evaluation_time_days=evaluation_time,
                    improvement_probability=pi[:, index],
                    progression_probability=pp[:, index],
                    remaining_cdr_0_5_probability=pr[:, index],
                    probability_sum=pi[:, index] + pp[:, index] + pr[:, index],
                    model_artifact_sha256=sha256(artifact),
                    preprocessing_parameters_sha256=preprocessing_hash,
                    input_feature_file_sha256=feature_hash,
                    outcome_columns_present=False,
                )
            )
        curve = pd.concat(curve_parts, ignore_index=True)
        final_curve = curve[
            np.isclose(curve["evaluation_time_days"], horizon_days)
        ].sort_values("phase4_row_key")
        final_point = point.sort_values("phase4_row_key")
        curve_point_difference = float(
            np.max(
                np.abs(
                    final_curve[
                        [
                            "improvement_probability",
                            "progression_probability",
                            "remaining_cdr_0_5_probability",
                            "probability_sum",
                        ]
                    ].to_numpy()
                    - final_point[probability_columns].to_numpy()
                )
            )
        )
        if curve_point_difference > TOLERANCE:
            raise RuntimeError("Frozen curve and horizon prediction differ.")
        curve_name = f"{model_id.lower()}_{horizon}m_prediction_curve.csv"
        curve_path = PREDICTION_BUILD / curve_name
        save_csv(curve, curve_path)

        for file_role, path, rows in [
            ("point_prediction", prediction_path, len(point)),
            ("time_grid_prediction", curve_path, len(curve)),
        ]:
            prediction_manifest_rows.append(
                {
                    "sequence": len(prediction_manifest_rows) + 1,
                    "model_id": model_id,
                    "family": family,
                    "horizon_months": horizon,
                    "file_role": file_role,
                    "relative_path": (
                        f"ProcessedData/oasis3/phase4/frozen_predictions/{path.name}"
                    ),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "rows": rows,
                    "participants": frame["participant_id"].nunique(),
                    "model_artifact_sha256": sha256(artifact),
                    "preprocessing_parameters_sha256": preprocessing_hash,
                    "feature_schema_sha256": schema_hash,
                    "input_feature_file_sha256": feature_hash,
                    "deterministic_repeat_max_difference": repeat_difference,
                    "curve_horizon_max_difference": curve_point_difference,
                    "outcome_columns_present": False,
                    "outcomes_accessed_before_freeze": False,
                    "status": "PASS",
                }
            )

        feature_rows.append(
            {
                "model_id": model_id,
                "family": family,
                "horizon_months": horizon,
                "source_path": rel(source),
                "source_sha256": sha256(source),
                "loaded_columns": "|".join(loaded_columns),
                "outcome_columns_loaded": False,
                "relative_path": (
                    "ProcessedData/oasis3/phase4/outcome_hidden_features/"
                    f"{feature_path.name}"
                ),
                "sha256": feature_hash,
                "bytes": feature_path.stat().st_size,
                "rows": len(frame),
                "participants": frame["participant_id"].nunique(),
                "required_features": "|".join(features),
                "missing_cells": int(frame[features].isna().sum().sum()),
                "status": "PASS",
            }
        )
        coherence_rows.append(
            {
                "model_id": model_id,
                "horizon_months": horizon,
                "rows": len(point),
                "minimum_probability": float(probabilities.min()),
                "maximum_probability": float(probabilities.max()),
                "maximum_absolute_sum_error": float(
                    np.max(np.abs(first["probability_sum"] - 1))
                ),
                "maximum_repeat_difference": repeat_difference,
                "curve_horizon_max_difference": curve_point_difference,
                "status": "PASS",
            }
        )
        access_rows.append(
            {
                "sequence": len(access_rows) + 1,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": "B",
                "action": f"read outcome-hidden {model_id} {horizon}m predictors and freeze predictions",
                "resource_scope": rel(source),
                "loaded_columns": "|".join(loaded_columns),
                "oasis_record_level_features_accessed": True,
                "oasis_record_level_outcomes_accessed": False,
                "outcome_hidden_predictions_created": True,
                "outcomes_first_accessed_at_utc": "",
            }
        )

    populations = pd.DataFrame(population_rows)
    feature_manifest = pd.DataFrame(feature_rows)
    prediction_manifest = pd.DataFrame(prediction_manifest_rows)
    coherence = pd.DataFrame(coherence_rows)
    if not (
        populations["status"].eq("PASS").all()
        and feature_manifest["status"].eq("PASS").all()
        and prediction_manifest["status"].eq("PASS").all()
        and coherence["status"].eq("PASS").all()
    ):
        raise RuntimeError("Outcome-hidden Phase 4 prediction gate failed.")

    FEATURE_BUILD.rename(FEATURE_ROOT)
    PREDICTION_BUILD.rename(PREDICTION_ROOT)
    save_csv(
        populations,
        TABLE_ROOT / "outcome_hidden_population_verification.csv",
    )
    save_csv(
        feature_manifest,
        TABLE_ROOT / "outcome_hidden_feature_manifest.csv",
    )
    prediction_manifest_path = save_csv(
        prediction_manifest,
        TABLE_ROOT / "outcome_hidden_prediction_manifest.csv",
    )
    save_csv(coherence, TABLE_ROOT / "outcome_hidden_probability_coherence.csv")
    save_csv(pd.DataFrame(access_rows), TABLE_ROOT / "phase4_access_audit.csv")
    freeze = {
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "lock_path": rel(LOCK),
        "lock_package_version_id": phase3_meta["package_version_id"],
        "prediction_manifest_path": rel(prediction_manifest_path),
        "prediction_manifest_sha256": sha256(prediction_manifest_path),
        "prediction_files": len(prediction_manifest),
        "point_prediction_files": int(
            prediction_manifest["file_role"].eq("point_prediction").sum()
        ),
        "time_grid_prediction_files": int(
            prediction_manifest["file_role"].eq("time_grid_prediction").sum()
        ),
        "oasis_outcomes_accessed_before_freeze": False,
        "independent_prediction_qc_pending": True,
        "phase4_outcomes_merged": False,
    }
    save_json(freeze, TABLE_ROOT / "prediction_freeze_metadata.json")
    run_metadata = {
        "phase": 4,
        "generated_at_utc": GENERATED_AT,
        "stage_a_pass": True,
        "stage_b_predictions_frozen": True,
        "stage_c_independent_prediction_qc_pass": False,
        "lock_path": rel(LOCK),
        "lock_package_version_id": phase3_meta["package_version_id"],
        "models": sorted(inventory["model_id"].unique()),
        "frozen_prediction_files": len(prediction_manifest),
        "oasis_record_level_features_accessed": True,
        "oasis_record_level_outcomes_accessed": False,
        "outcomes_first_accessed_at_utc": "",
        "untouched_results_frozen": False,
        "phase4_complete": False,
        "phase5_started": False,
        "decision": "HOLD FOR VALIDATION AUDIT",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_json(run_metadata, TABLE_ROOT / "phase4_run_metadata.json")

    report = f"""# Phase 4 Stage A/B — Lock and Outcome-Hidden Prediction Gate

Generated: {GENERATED_AT}

## Gate result

**PASS — OUTCOME-HIDDEN PREDICTIONS FROZEN**

Lock used: `{rel(LOCK)}`

No OASIS participant-level outcome column was loaded. All point and time-grid
prediction files were saved and hashed before the outcome gate.

## Stage A

{stage_a.to_markdown(index=False)}

## Outcome-hidden populations

{populations.to_markdown(index=False)}

## Prediction coherence

{coherence.to_markdown(index=False)}

Independent prediction regeneration is still required before outcomes may be
opened.
"""
    (REPORT_ROOT / "phase4_stage_a_lock_and_prediction_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage_a": "PASS",
                "outcome_hidden_population_checks": len(populations),
                "frozen_prediction_files": len(prediction_manifest),
                "point_prediction_files": int(
                    prediction_manifest["file_role"].eq("point_prediction").sum()
                ),
                "time_grid_prediction_files": int(
                    prediction_manifest["file_role"].eq("time_grid_prediction").sum()
                ),
                "oasis_outcomes_accessed": False,
                "independent_prediction_qc_pending": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
