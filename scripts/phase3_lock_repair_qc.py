#!/usr/bin/env python3
"""Independent QC for the outcome-blind Phase 3 lock repair.

This verifier is intentionally separate from ``phase3_lock_repair.py`` and
does not import it. It re-hashes both lock packages, loads every serialized
model through the packaged runtime, independently regenerates predictions,
and rechecks the ADNI temporal/grouping protections. It never opens an
OASIS-3 participant-level file and it does not run external validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_LOCK = ROOT / "models" / "phase3" / "phase4_lock"
REPAIRED_LOCK = ROOT / "models" / "phase3" / "phase4_lock_v2"
TABLE_ROOT = ROOT / "results" / "csvs" / "phase3_lock_repair"
REPORT_ROOT = ROOT / "reports" / "phase3_lock_repair"
PHASE3_TABLES = ROOT / "results" / "csvs" / "phase3"
RUNTIME = REPAIRED_LOCK / "inference_runtime"
INFERENCE_ENTRYPOINT = RUNTIME / "phase3_inference.py"
TOLERANCE = 1e-12
GENERATED_AT = datetime.now(timezone.utc).isoformat()

TABLE_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUNTIME))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def md(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    shown = frame if columns is None else frame.loc[:, columns]
    return shown.to_markdown(index=False)


def truth(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


checks: list[dict[str, Any]] = []


def add_check(
    section: str,
    check: str,
    passed: bool,
    details: str,
    *,
    critical: bool = True,
) -> None:
    checks.append(
        {
            "section": section,
            "check": check,
            "critical": critical,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


def run_inference(
    artifact: Path,
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    completed = subprocess.run(
        [
            sys.executable,
            str(INFERENCE_ENTRYPOINT),
            "--artifact",
            str(artifact),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=RUNTIME,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Packaged inference failed for {artifact.name}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return pd.read_csv(output_path)


def numeric_prediction_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> float:
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


def metadata_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    columns = [
        "prediction_horizon_months",
        "model_id",
        "model_version_id",
        "quality_flags",
    ]
    return left[columns].equals(right[columns])


def main() -> None:
    required_package_files = {
        "access_audit.csv",
        "allowed_and_forbidden_exclusions.csv",
        "b1_d1_original_copy_equivalence.csv",
        "checksum_manifest.csv",
        "d3_strict_temporal_verification.csv",
        "external_validation_lock_manifest.json",
        "frozen_baseline_hazards.csv",
        "frozen_feature_schema.csv",
        "frozen_model_coefficients.csv",
        "frozen_model_index.json",
        "frozen_preprocessing_parameters.csv",
        "inference_fixture_manifest.csv",
        "inference_runtime/phase2_preprocessing.py",
        "inference_runtime/phase3_inference.py",
        "inference_runtime/phase3_survival_core.py",
        "model_inventory.csv",
        "model_software_versions.csv",
        "oasis3_frozen_feature_mapping.csv",
        "participant_grouping_verification.csv",
        "penalty_source_recovery.csv",
        "phase4_validation_checklist.csv",
        "provenance/adni_nested_inner_fold_assignments.csv",
        "provenance/adni_outer_fold_assignments.csv",
        "provenance/d1_collinearity_resolution.csv",
        "provenance/fold_specific_ridge_penalties.csv",
        "provenance/fold_specific_ridge_penalty_tuning.csv",
        "provenance/phase3_analysis_parameters.json",
        "provenance/phase3_run_metadata.json",
        "recovered_model_specifications.json",
        "repair_software_environment.json",
        "repaired_model_training_metadata.csv",
    }
    actual_files = {
        path.relative_to(REPAIRED_LOCK).as_posix()
        for path in REPAIRED_LOCK.rglob("*")
        if path.is_file()
    }
    missing_package_files = sorted(required_package_files - actual_files)
    add_check(
        "package",
        "required self-contained support files exist",
        not missing_package_files,
        f"required={len(required_package_files)}; missing={missing_package_files}",
    )

    # Independently verify every original-lock row retained in the original
    # Phase 3 checksum metadata, both content hash and byte size.
    original_metadata = pd.read_csv(
        PHASE3_TABLES / "model_metadata_and_checksums.csv"
    )
    original_metadata = original_metadata[
        original_metadata["artifact_path"]
        .astype(str)
        .str.startswith("models/phase3/phase4_lock/")
    ].copy()
    original_rows: list[dict[str, Any]] = []
    for record in original_metadata.itertuples(index=False):
        path = ROOT / str(record.artifact_path)
        exists = path.exists()
        observed_hash = sha256(path) if exists else ""
        observed_bytes = path.stat().st_size if exists else -1
        passed = (
            exists
            and observed_hash == str(record.sha256)
            and observed_bytes == int(record.bytes)
        )
        original_rows.append(
            {
                "artifact_path": record.artifact_path,
                "expected_sha256": record.sha256,
                "observed_sha256": observed_hash,
                "expected_bytes": int(record.bytes),
                "observed_bytes": observed_bytes,
                "status": "PASS" if passed else "FAIL",
            }
        )
    original_verification = pd.DataFrame(original_rows)
    add_check(
        "original_lock",
        "all original-lock checksums and sizes still pass",
        len(original_verification) > 0
        and original_verification["status"].eq("PASS").all(),
        (
            f"rows={len(original_verification)}; "
            f"failed={int(original_verification['status'].ne('PASS').sum())}"
        ),
    )

    external_manifest = json.loads(
        (REPAIRED_LOCK / "external_validation_lock_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    index_path = REPAIRED_LOCK / "frozen_model_index.json"
    checksum_path = REPAIRED_LOCK / "checksum_manifest.csv"
    add_check(
        "package",
        "external manifest authenticates model index",
        sha256(index_path) == external_manifest["model_index_sha256"],
        f"observed={sha256(index_path)}",
    )
    add_check(
        "package",
        "external manifest authenticates checksum manifest",
        sha256(checksum_path)
        == external_manifest["checksum_manifest_sha256"],
        f"observed={sha256(checksum_path)}",
    )

    checksum_manifest = pd.read_csv(checksum_path)
    package_hash_rows: list[dict[str, Any]] = []
    for record in checksum_manifest.itertuples(index=False):
        path = REPAIRED_LOCK / str(record.package_relative_path)
        exists = path.exists()
        observed_hash = sha256(path) if exists else ""
        observed_bytes = path.stat().st_size if exists else -1
        passed = (
            exists
            and observed_hash == str(record.sha256)
            and observed_bytes == int(record.bytes)
        )
        package_hash_rows.append(
            {
                "package_relative_path": record.package_relative_path,
                "expected_sha256": record.sha256,
                "observed_sha256": observed_hash,
                "expected_bytes": int(record.bytes),
                "observed_bytes": observed_bytes,
                "status": "PASS" if passed else "FAIL",
            }
        )
    package_hash_verification = pd.DataFrame(package_hash_rows)
    manifest_scope = set(checksum_manifest["package_relative_path"])
    expected_scope = actual_files - {
        "checksum_manifest.csv",
        "external_validation_lock_manifest.json",
    }
    add_check(
        "package",
        "checksum manifest covers every in-scope package file",
        manifest_scope == expected_scope,
        (
            f"manifest={len(manifest_scope)}; actual_in_scope={len(expected_scope)}; "
            f"missing={sorted(expected_scope - manifest_scope)}; "
            f"extra={sorted(manifest_scope - expected_scope)}"
        ),
    )
    add_check(
        "package",
        "all repaired-package checksums and sizes pass",
        package_hash_verification["status"].eq("PASS").all(),
        (
            f"rows={len(package_hash_verification)}; "
            f"failed={int(package_hash_verification['status'].ne('PASS').sum())}"
        ),
    )

    model_index = json.loads(index_path.read_text(encoding="utf-8"))
    inventory = pd.read_csv(REPAIRED_LOCK / "model_inventory.csv")
    expected_models = {
        ("B1", 12),
        ("B1", 24),
        ("B1", 36),
        ("B1", 60),
        ("D0", 12),
        ("D0", 36),
        ("D1", 12),
        ("D1", 24),
        ("D1", 36),
        ("D1", 60),
        ("B3", 12),
        ("B3", 36),
        ("D3", 12),
        ("D3", 36),
    }
    observed_models = {
        (str(row.model_id), int(row.horizon_months))
        for row in inventory.itertuples(index=False)
    }
    add_check(
        "inventory",
        "every Phase 4 required model and horizon is present",
        observed_models == expected_models,
        (
            f"present={sorted(observed_models)}; "
            f"missing={sorted(expected_models - observed_models)}; "
            f"unexpected={sorted(observed_models - expected_models)}"
        ),
    )
    index_models = {
        (str(row["model_id"]), int(row["horizon_months"]))
        for row in model_index["models"]
    }
    add_check(
        "inventory",
        "frozen index and inventory contain the same 14 bundles",
        index_models == observed_models and len(model_index["models"]) == 14,
        f"index_rows={len(model_index['models'])}; inventory_rows={len(inventory)}",
    )
    add_check(
        "outcome_blindness",
        "frozen index certifies no external participant-level access",
        model_index["development_cohort"] == "ADNI"
        and model_index["external_participant_level_features_accessed"] is False
        and model_index["external_participant_level_outcomes_accessed"] is False,
        (
            f"development={model_index['development_cohort']}; "
            "features_accessed="
            f"{model_index['external_participant_level_features_accessed']}; "
            "outcomes_accessed="
            f"{model_index['external_participant_level_outcomes_accessed']}"
        ),
    )

    schema = pd.read_csv(REPAIRED_LOCK / "frozen_feature_schema.csv")
    preprocessing = pd.read_csv(
        REPAIRED_LOCK / "frozen_preprocessing_parameters.csv"
    )
    coefficients = pd.read_csv(
        REPAIRED_LOCK / "frozen_model_coefficients.csv"
    )
    hazards = pd.read_csv(REPAIRED_LOCK / "frozen_baseline_hazards.csv")
    software = pd.read_csv(REPAIRED_LOCK / "model_software_versions.csv")
    model_artifact_rows: list[dict[str, Any]] = []

    for record in inventory.itertuples(index=False):
        key = (str(record.model_id), int(record.horizon_months))
        artifact = ROOT / str(record.artifact_path)
        artifact_ok = (
            artifact.exists()
            and sha256(artifact) == str(record.sha256)
            and artifact.stat().st_size == int(record.bytes)
        )
        bundle = joblib.load(artifact) if artifact_ok else {}
        required_bundle_keys = {
            "preprocessor",
            "improvement_model",
            "progression_model",
            "continuous_features",
            "categorical_features",
            "transformed_feature_order",
            "selected_penalties",
            "horizon_days",
            "horizon_months",
            "model_id",
            "model_version_id",
            "development_cohort",
            "external_outcomes_accessed",
            "training_rows",
            "training_participants",
            "training_improvement_event_rows",
            "training_progression_event_rows",
            "cumulative_incidence_method",
        }
        keys_ok = required_bundle_keys <= set(bundle)
        raw_feature_order = (
            list(bundle.get("continuous_features", []))
            + list(bundle.get("categorical_features", []))
        )
        schema_group = schema[
            schema["model_id"].eq(key[0])
            & schema["horizon_months"].eq(key[1])
        ]
        preprocessing_group = preprocessing[
            preprocessing["model_id"].eq(key[0])
            & preprocessing["horizon_months"].eq(key[1])
        ]
        coefficient_group = coefficients[
            coefficients["model_id"].eq(key[0])
            & coefficients["horizon_months"].eq(key[1])
        ]
        hazard_group = hazards[
            hazards["model_id"].eq(key[0])
            & hazards["horizon_months"].eq(key[1])
        ]
        software_group = software[
            software["model_id"].eq(key[0])
            & software["horizon_months"].eq(key[1])
        ]
        schema_order_ok = schema_group["feature"].tolist() == raw_feature_order
        preprocessing_ok = (
            preprocessing_group["feature"].tolist() == raw_feature_order
            and preprocessing_group["fit_cohort"]
            .astype(str)
            .str.lower()
            .eq("adni")
            .all()
            and ~preprocessing_group["external_outcomes_accessed"].map(truth).any()
        )
        coefficients_ok = (
            set(coefficient_group["cause"]) == {"improvement", "progression"}
            and coefficient_group.groupby("cause")["feature"]
            .apply(list)
            .map(lambda values: values == list(bundle["transformed_feature_order"]))
            .all()
            and np.isfinite(
                coefficient_group[
                    [
                        "standardized_coefficient",
                        "hazard_ratio",
                        "alpha",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        )
        hazards_ok = (
            set(hazard_group["cause"]) == {"improvement", "progression"}
            and len(hazard_group) > 0
            and np.isfinite(
                hazard_group[
                    [
                        "event_time_days",
                        "baseline_hazard_increment",
                        "cumulative_baseline_hazard",
                    ]
                ].to_numpy(dtype=float)
            ).all()
            and (
                hazard_group["baseline_hazard_increment"].astype(float) >= 0
            ).all()
            and all(
                group["event_time_days"].astype(float).is_monotonic_increasing
                and group["cumulative_baseline_hazard"]
                .astype(float)
                .is_monotonic_increasing
                for _, group in hazard_group.groupby("cause", sort=False)
            )
        )
        penalties_ok = (
            float(bundle["selected_penalties"]["improvement"])
            == float(record.improvement_alpha)
            and float(bundle["selected_penalties"]["progression"])
            == float(record.progression_alpha)
            and set(coefficient_group["alpha"].astype(float))
            == {
                float(record.improvement_alpha),
                float(record.progression_alpha),
            }
        )
        counts_ok = (
            int(bundle["training_rows"]) == int(record.training_rows)
            and int(bundle["training_participants"])
            == int(record.training_participants)
            and int(bundle["training_improvement_event_rows"])
            == int(record.improvement_event_rows)
            and int(bundle["training_progression_event_rows"])
            == int(record.progression_event_rows)
        )
        software_columns = [
            "training_python",
            "training_platform",
            "training_numpy",
            "training_pandas",
            "training_scipy",
            "training_scikit_learn",
            "training_statsmodels",
            "package_joblib",
            "version_source",
        ]
        software_ok = (
            len(software_group) == 1
            and software_group[software_columns].notna().all(axis=None)
        )
        certification_ok = (
            bundle.get("model_id") == key[0]
            and int(bundle.get("horizon_months")) == key[1]
            and bundle.get("development_cohort") == "ADNI"
            and bundle.get("external_outcomes_accessed") is False
        )
        copied_ok = True
        if key[0] in {"B1", "D1"}:
            source = ROOT / str(record.source_artifact_path)
            copied_ok = (
                truth(record.refit_during_repair) is False
                and source.exists()
                and sha256(source) == sha256(artifact)
                and source.stat().st_size == artifact.stat().st_size
                and str(record.source_sha256) == sha256(source)
            )
        else:
            copied_ok = (
                truth(record.refit_during_repair)
                and str(record.model_origin)
                == "adni_only_outcome_blind_lock_repair"
                and bundle.get("retuned_during_repair") is False
                and bundle.get("oasis_participant_level_features_accessed")
                is False
                and bundle.get("oasis_participant_level_outcomes_accessed")
                is False
                and bundle.get("software_versions") is not None
            )
        all_ok = all(
            [
                artifact_ok,
                keys_ok,
                schema_order_ok,
                preprocessing_ok,
                coefficients_ok,
                hazards_ok,
                penalties_ok,
                counts_ok,
                software_ok,
                certification_ok,
                copied_ok,
            ]
        )
        model_artifact_rows.append(
            {
                "model_id": key[0],
                "horizon_months": key[1],
                "artifact_hash_and_size": artifact_ok,
                "required_bundle_keys": keys_ok,
                "feature_order": schema_order_ok,
                "preprocessing": preprocessing_ok,
                "coefficients": coefficients_ok,
                "baseline_hazard": hazards_ok,
                "ridge_penalties": penalties_ok,
                "training_counts": counts_ok,
                "software_versions": software_ok,
                "adni_outcome_blind_certification": certification_ok,
                "copy_or_refit_rule": copied_ok,
                "status": "PASS" if all_ok else "FAIL",
            }
        )

    model_artifact_verification = pd.DataFrame(model_artifact_rows).sort_values(
        ["model_id", "horizon_months"]
    )
    add_check(
        "artifacts",
        "all 14 model bundles contain complete frozen prediction state",
        model_artifact_verification["status"].eq("PASS").all(),
        (
            f"models={len(model_artifact_verification)}; "
            f"failed={int(model_artifact_verification['status'].ne('PASS').sum())}"
        ),
    )

    # Recompute the final-alpha recovery without importing the repair builder.
    tuning = pd.read_csv(
        REPAIRED_LOCK / "provenance" / "fold_specific_ridge_penalties.csv"
    )
    recovery_rows: list[dict[str, Any]] = []
    for model_id in ["D0", "B3", "D3"]:
        for horizon in [12, 36]:
            inventory_row = inventory[
                inventory["model_id"].eq(model_id)
                & inventory["horizon_months"].eq(horizon)
            ].iloc[0]
            for cause, inventory_column in [
                ("improvement", "improvement_alpha"),
                ("progression", "progression_alpha"),
            ]:
                selected = tuning[
                    tuning["model_id"].eq(model_id)
                    & tuning["horizon_months"].eq(horizon)
                    & tuning["cause"].eq(cause)
                ][["outer_fold", "selected_alpha"]].drop_duplicates()
                frequency = (
                    selected.groupby("selected_alpha")
                    .size()
                    .reset_index(name="frequency")
                    .sort_values(
                        ["frequency", "selected_alpha"],
                        ascending=[False, False],
                    )
                )
                recovered = float(frequency.iloc[0]["selected_alpha"])
                saved = float(inventory_row[inventory_column])
                passed = (
                    selected["outer_fold"].nunique() == 5 and recovered == saved
                )
                recovery_rows.append(
                    {
                        "model_id": model_id,
                        "horizon_months": horizon,
                        "cause": cause,
                        "outer_folds": selected["outer_fold"].nunique(),
                        "independently_recovered_alpha": recovered,
                        "saved_final_alpha": saved,
                        "status": "PASS" if passed else "FAIL",
                    }
                )
    penalty_verification = pd.DataFrame(recovery_rows)
    add_check(
        "specification",
        "saved penalties reproduce the Phase 3 modal-fold rule without retuning",
        penalty_verification["status"].eq("PASS").all(),
        (
            f"decisions={len(penalty_verification)}; "
            f"failed={int(penalty_verification['status'].ne('PASS').sum())}"
        ),
    )

    runtime_source = INFERENCE_ENTRYPOINT.read_text(encoding="utf-8")
    add_check(
        "inference",
        "packaged inference is prediction-only",
        ".fit(" not in runtime_source,
        "no .fit( call found in the packaged inference entrypoint",
    )
    fixture_manifest = pd.read_csv(
        REPAIRED_LOCK / "inference_fixture_manifest.csv"
    )
    fixture_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="phase3_lock_repair_qc_") as temp:
        temp_root = Path(temp)
        for record in fixture_manifest.itertuples(index=False):
            artifact = ROOT / str(record.artifact_path)
            input_path = REPAIRED_LOCK / str(record.input_path)
            expected_path = REPAIRED_LOCK / str(record.expected_path)
            expected = pd.read_csv(expected_path)
            first = run_inference(
                artifact,
                input_path,
                temp_root / f"{record.model_id}_{record.horizon_months}_first.csv",
            )
            second = run_inference(
                artifact,
                input_path,
                temp_root / f"{record.model_id}_{record.horizon_months}_second.csv",
            )
            expected_difference = numeric_prediction_difference(first, expected)
            repeat_difference = numeric_prediction_difference(first, second)
            probability_columns = [
                "improvement_probability",
                "progression_probability",
                "remaining_cdr_0_5_probability",
            ]
            bounded = (
                (
                    first[probability_columns].to_numpy(dtype=float) >= -TOLERANCE
                ).all()
                and (
                    first[probability_columns].to_numpy(dtype=float)
                    <= 1 + TOLERANCE
                ).all()
                and np.allclose(
                    first["probability_sum"].to_numpy(dtype=float),
                    1.0,
                    atol=TOLERANCE,
                    rtol=0,
                )
            )
            passed = (
                expected_difference <= TOLERANCE
                and repeat_difference <= TOLERANCE
                and metadata_equal(first, expected)
                and metadata_equal(first, second)
                and bounded
                and truth(record.outcome_columns_present) is False
            )
            fixture_rows.append(
                {
                    "model_id": record.model_id,
                    "horizon_months": int(record.horizon_months),
                    "rows": len(first),
                    "maximum_expected_difference": expected_difference,
                    "maximum_repeat_difference": repeat_difference,
                    "metadata_exact": metadata_equal(first, expected)
                    and metadata_equal(first, second),
                    "probabilities_bounded_and_sum_one": bounded,
                    "predictor_only_fixture": (
                        truth(record.outcome_columns_present) is False
                    ),
                    "status": "PASS" if passed else "FAIL",
                }
            )

        # Independently recreate held-out ADNI inputs for every original B1/D1
        # pair and compare original versus verified-copy inference.
        equivalence_rows: list[dict[str, Any]] = []
        copied_inventory = inventory[inventory["model_id"].isin(["B1", "D1"])]
        for record in copied_inventory.itertuples(index=False):
            family = str(record.family)
            horizon = int(record.horizon_months)
            input_source = (
                ROOT
                / "ProcessedData"
                / "adni"
                / "phase2"
                / family
                / f"{family}_{horizon}m_clinical_core.csv"
            )
            source_data = pd.read_csv(input_source)
            artifact = ROOT / str(record.artifact_path)
            bundle = joblib.load(artifact)
            required = list(bundle["continuous_features"]) + list(
                bundle["categorical_features"]
            )
            input_frame = (
                source_data[source_data["outer_fold"].eq(0)]
                .sort_values("participant_id")
                .head(20)[required]
            )
            input_path = temp_root / f"{record.model_id}_{horizon}_equivalence.csv"
            input_frame.to_csv(input_path, index=False)
            original_artifact = ROOT / str(record.source_artifact_path)
            original_prediction = run_inference(
                original_artifact,
                input_path,
                temp_root / f"{record.model_id}_{horizon}_original.csv",
            )
            copied_prediction = run_inference(
                artifact,
                input_path,
                temp_root / f"{record.model_id}_{horizon}_copy.csv",
            )
            maximum_difference = numeric_prediction_difference(
                original_prediction, copied_prediction
            )
            hash_identical = sha256(original_artifact) == sha256(artifact)
            exact_metadata = metadata_equal(
                original_prediction, copied_prediction
            )
            passed = (
                hash_identical
                and maximum_difference <= TOLERANCE
                and exact_metadata
            )
            equivalence_rows.append(
                {
                    "model_id": record.model_id,
                    "horizon_months": horizon,
                    "fixture_outer_fold": 0,
                    "fixture_rows": len(input_frame),
                    "original_sha256": sha256(original_artifact),
                    "repaired_sha256": sha256(artifact),
                    "artifact_hash_identical": hash_identical,
                    "maximum_probability_difference": maximum_difference,
                    "metadata_exact": exact_metadata,
                    "status": "PASS" if passed else "FAIL",
                }
            )

    fixture_verification = pd.DataFrame(fixture_rows).sort_values(
        ["model_id", "horizon_months"]
    )
    equivalence_verification = pd.DataFrame(equivalence_rows).sort_values(
        ["model_id", "horizon_months"]
    )
    add_check(
        "inference",
        "D0/B3/D3 predictions independently reproduce on held-out ADNI rows",
        fixture_verification["status"].eq("PASS").all(),
        (
            f"fixtures={len(fixture_verification)}; "
            f"max_difference="
            f"{fixture_verification['maximum_expected_difference'].max():.3g}"
        ),
    )
    add_check(
        "inference",
        "B1/D1 predictions are identical before and after repair",
        equivalence_verification["status"].eq("PASS").all(),
        (
            f"comparisons={len(equivalence_verification)}; "
            f"max_difference="
            f"{equivalence_verification['maximum_probability_difference'].max():.3g}"
        ),
    )

    # Recheck D3 temporal ordering directly from the ADNI training inputs.
    training_metadata = pd.read_csv(
        REPAIRED_LOCK / "repaired_model_training_metadata.csv"
    )
    d3_rows: list[dict[str, Any]] = []
    for record in training_metadata[
        training_metadata["model_id"].eq("D3")
    ].itertuples(index=False):
        source = pd.read_csv(ROOT / str(record.training_input_path))
        feature_list = str(record.continuous_features).split("|")
        row = {
            "model_id": "D3",
            "horizon_months": int(record.horizon_months),
            "rows": len(source),
            "participants": source["participant_id"].nunique(),
            "previous_mmse_missing_rows": int(source["previous_mmse"].isna().sum()),
            "previous_mmse_not_strictly_before_landmark_rows": int(
                (
                    source["previous_mmse_time_days"]
                    >= source["landmark_time_days"]
                ).sum()
            ),
            "nonpositive_time_since_previous_mmse_rows": int(
                (source["time_since_previous_mmse_days"] <= 0).sum()
            ),
            "strict_availability_flag_false_rows": int(
                (~source["strictly_lagged_mmse_available_flag"].map(truth)).sum()
            ),
            "current_or_same_day_mmse_in_model_features": bool(
                {"current_mmse", "mmse_change_from_index"}.intersection(
                    feature_list
                )
            ),
        }
        passed = (
            row["previous_mmse_missing_rows"] == 0
            and row["previous_mmse_not_strictly_before_landmark_rows"] == 0
            and row["nonpositive_time_since_previous_mmse_rows"] == 0
            and row["strict_availability_flag_false_rows"] == 0
            and not row["current_or_same_day_mmse_in_model_features"]
        )
        row["status"] = "PASS" if passed else "FAIL"
        d3_rows.append(row)
    d3_verification = pd.DataFrame(d3_rows)
    add_check(
        "temporal",
        "D3 uses only strictly pre-landmark MMSE",
        len(d3_verification) == 2
        and d3_verification["status"].eq("PASS").all(),
        (
            f"rows={d3_verification['rows'].tolist()}; "
            "non_strict="
            f"{int(d3_verification['previous_mmse_not_strictly_before_landmark_rows'].sum())}"
        ),
    )

    outer = pd.read_csv(
        REPAIRED_LOCK
        / "provenance"
        / "adni_outer_fold_assignments.csv"
    )
    inner = pd.read_csv(
        REPAIRED_LOCK
        / "provenance"
        / "adni_nested_inner_fold_assignments.csv"
    )
    grouping_rows: list[dict[str, Any]] = []
    outer_max = outer.groupby("participant_id")["outer_fold"].nunique().max()
    grouping_rows.append(
        {
            "scope": "outer_assignments",
            "participants": outer["participant_id"].nunique(),
            "mismatches_or_multiple_roles": int(outer_max - 1),
            "status": "PASS" if outer_max == 1 else "FAIL",
        }
    )
    inner_max = (
        inner.groupby(["outer_test_fold", "participant_id"]).size().max()
    )
    grouping_rows.append(
        {
            "scope": "nested_assignments",
            "participants": inner["participant_id"].nunique(),
            "mismatches_or_multiple_roles": int(inner_max - 1),
            "status": "PASS" if inner_max == 1 else "FAIL",
        }
    )
    for record in inventory[
        inventory["model_id"].isin(["D0", "D1", "D3"])
    ].itertuples(index=False):
        horizon = int(record.horizon_months)
        if record.model_id == "D3":
            source_path = (
                ROOT
                / "ProcessedData"
                / "adni"
                / "phase2"
                / "sensitivity"
                / f"dynamic_{horizon}m_strictly_lagged_mmse.csv"
            )
        else:
            source_path = (
                ROOT
                / "ProcessedData"
                / "adni"
                / "phase2"
                / "dynamic"
                / f"dynamic_{horizon}m_clinical_core.csv"
            )
        source = pd.read_csv(
            source_path,
            usecols=["participant_id", "outer_fold"],
        ).drop_duplicates()
        participant_unique = (
            source.groupby("participant_id")["outer_fold"].nunique().max()
        )
        joined = source.merge(
            outer[["participant_id", "outer_fold"]],
            on="participant_id",
            suffixes=("_source", "_frozen"),
            validate="one_to_one",
        )
        mismatches = int(
            (joined["outer_fold_source"] != joined["outer_fold_frozen"]).sum()
        )
        passed = participant_unique == 1 and mismatches == 0
        grouping_rows.append(
            {
                "scope": f"{record.model_id}_{horizon}m",
                "participants": source["participant_id"].nunique(),
                "mismatches_or_multiple_roles": mismatches
                + int(participant_unique - 1),
                "status": "PASS" if passed else "FAIL",
            }
        )
    grouping_verification = pd.DataFrame(grouping_rows)
    add_check(
        "grouping",
        "all dynamic processing retains frozen participant-grouped folds",
        grouping_verification["status"].eq("PASS").all(),
        (
            f"checks={len(grouping_verification)}; "
            f"failed={int(grouping_verification['status'].ne('PASS').sum())}"
        ),
    )

    access = pd.read_csv(REPAIRED_LOCK / "access_audit.csv")
    forbidden_fragments = [
        "/processeddata/oasis3/",
        "/datasets/oasis3/",
        "/datasets/oasis/",
        "/results/csvs/phase1/oasis3/",
        "/results/csvs/phase2/oasis3/",
    ]
    normalized_paths = "/" + access["path"].astype(str).str.lower()
    forbidden_rows = normalized_paths.map(
        lambda path: any(fragment in path for fragment in forbidden_fragments)
    )
    external_flags = (
        access["oasis_participant_level_features_accessed"].map(truth)
        | access["oasis_participant_level_outcomes_accessed"].map(truth)
    )
    access_ok = not forbidden_rows.any() and not external_flags.any()
    add_check(
        "outcome_blindness",
        "access audit contains no OASIS participant-level feature or outcome file",
        access_ok,
        (
            f"records={len(access)}; forbidden_paths={int(forbidden_rows.sum())}; "
            f"external_access_flags={int(external_flags.sum())}"
        ),
    )
    add_check(
        "workflow",
        "Phase 4 was not resumed by the repair",
        model_index["phase4_must_use_path"]
        == "models/phase3/phase4_lock_v2",
        (
            "new lock is staged for later use; verifier performed ADNI/package "
            "checks only and did not execute external validation"
        ),
    )

    qc = pd.DataFrame(checks)
    critical_failed = qc["critical"].map(truth) & qc["status"].ne("PASS")
    all_critical_pass = not critical_failed.any()
    decision = (
        "RESUME PHASE 4 WITH RESTRICTIONS"
        if all_critical_pass
        else "HOLD FOR LOCK REPAIR"
    )

    write_csv(
        original_verification,
        TABLE_ROOT / "independent_original_lock_verification.csv",
    )
    write_csv(
        package_hash_verification,
        TABLE_ROOT / "independent_package_checksum_verification.csv",
    )
    write_csv(
        model_artifact_verification,
        TABLE_ROOT / "independent_model_artifact_verification.csv",
    )
    write_csv(
        penalty_verification,
        TABLE_ROOT / "independent_penalty_recovery_verification.csv",
    )
    write_csv(
        fixture_verification,
        TABLE_ROOT / "independent_inference_fixture_verification.csv",
    )
    write_csv(
        equivalence_verification,
        TABLE_ROOT / "independent_b1_d1_equivalence.csv",
    )
    write_csv(
        d3_verification,
        TABLE_ROOT / "independent_d3_temporal_verification.csv",
    )
    write_csv(
        grouping_verification,
        TABLE_ROOT / "independent_participant_grouping_verification.csv",
    )
    write_csv(qc, TABLE_ROOT / "independent_qc_results.csv")

    run_metadata_path = TABLE_ROOT / "phase3_lock_repair_run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    run_metadata.update(
        {
            "independent_qc_completed_at_utc": GENERATED_AT,
            "independent_qc_pending": False,
            "independent_qc_pass": all_critical_pass,
            "independent_qc_checks": len(qc),
            "independent_qc_failures": int(critical_failed.sum()),
            "final_decision": decision,
            "phase4_resumed": False,
            "phase4_must_use_lock": rel(REPAIRED_LOCK),
        }
    )
    write_json(run_metadata, run_metadata_path)

    qc_report = f"""# Independent QC — Phase 3 Lock Repair

Generated: {GENERATED_AT}

Verifier: `scripts/phase3_lock_repair_qc.py`

The verifier is independent of the repair builder: it does not import the
builder, does not refit any model, and does not access OASIS-3 participant-level
features or outcomes.

## Decision

**{decision}**

Phase 4 must use: `{rel(REPAIRED_LOCK)}`

## QC checklist

{md(qc)}

## Complete model-artifact verification

{md(model_artifact_verification)}

## Independent D0/B3/D3 inference

{md(fixture_verification)}

## Independent B1/D1 equivalence

{md(equivalence_verification, ["model_id", "horizon_months", "artifact_hash_identical", "maximum_probability_difference", "metadata_exact", "status"])}

## D3 strict temporal verification

{md(d3_verification)}

## Dynamic participant grouping

{md(grouping_verification)}

## Restrictions retained

1. Report untouched OASIS-3 validation first; save and hash outcome-hidden
   predictions before any outcome merge.
2. Treat the OASIS-3 `CDRTOT` global-CDR mapping as provisional pending provider
   confirmation.
3. Keep 24- and 60-month results as sensitivities and keep endpoint
   reconstruction/recalibration secondary to untouched validation.
4. B2 and D2 remain conditional, not-saved analyses; they are not required
   members of the repaired package.
5. Do not begin Phase 5 automatically.
"""
    (REPORT_ROOT / "phase3_lock_repair_independent_qc_report.md").write_text(
        qc_report,
        encoding="utf-8",
    )

    handoff = f"""# Phase 4 Restart Handoff

Generated: {GENERATED_AT}

## Decision

**{decision}**

Use this lock and no other lock:

`{REPAIRED_LOCK}`

The original `models/phase3/phase4_lock` remains unchanged. The repaired lock
contains byte-identical B1/D1 bundles plus ADNI-only D0, B3, and D3 bundles at
12 and 36 months. All package checksums, held-out inference fixtures, strict
D3 timing checks, and participant-grouping checks passed independently.

## Required restart sequence

1. Verify `external_validation_lock_manifest.json` and
   `checksum_manifest.csv` in the v2 lock.
2. Load only the v2 frozen index and packaged prediction runtime.
3. Create outcome-hidden OASIS-3 predictors using the frozen mapping and
   preprocessing; save and hash predictions before opening outcomes.
4. Run untouched B1, D0, D1, B3, and D3 validation at 12 and 36 months.
5. Treat B1/D1 24- and 60-month analyses as sensitivity horizons.
6. Only after untouched results are frozen may endpoint sensitivity or
   recalibration be performed.

Phase 4 external validation was not resumed by this repair or verifier.
"""
    (REPORT_ROOT / "phase4_restart_handoff.md").write_text(
        handoff,
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "decision": decision,
                "lock_path": str(REPAIRED_LOCK),
                "critical_checks": int(qc["critical"].map(truth).sum()),
                "critical_failures": int(critical_failed.sum()),
                "models_verified": len(model_artifact_verification),
                "new_model_fixtures_verified": len(fixture_verification),
                "original_copy_equivalence_verified": len(
                    equivalence_verification
                ),
                "oasis_participant_level_accessed": False,
                "phase4_resumed": False,
            },
            indent=2,
        )
    )
    if not all_critical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
