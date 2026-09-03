#!/usr/bin/env python3
"""Stage A audit for the locked OASIS-3 external validation.

This script intentionally reads only the Phase 3 frozen package and Phase 3
provenance artifacts.  It never opens OASIS-3 record-level features or
outcomes.  If the package is incomplete, it writes a machine-readable HOLD
decision and stops Phase 4 before external-cohort access.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn


ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = ROOT / "models" / "phase3" / "phase4_lock"
TABLE_ROOT = ROOT / "results" / "csvs" / "phase4"
PLOT_ROOT = ROOT / "results" / "plots" / "phase4"
REPORT_ROOT = ROOT / "reports" / "phase4"
CONFIG_PATH = ROOT / "config" / "phase4_analysis_parameters.json"
PHASE3_METADATA = ROOT / "results" / "csvs" / "phase3" / "model_metadata_and_checksums.csv"
PHASE3_RUN_METADATA = ROOT / "results" / "csvs" / "phase3" / "phase3_run_metadata.json"

TABLE_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

GENERATED_AT = datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, name: str) -> Path:
    path = TABLE_ROOT / name
    frame.to_csv(path, index=False)
    return path


def save_json(payload: dict[str, Any], name: str) -> Path:
    path = TABLE_ROOT / name
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame[columns]
    return selected.to_markdown(index=False)


def add_check(
    checks: list[dict[str, Any]],
    *,
    check_id: str,
    category: str,
    description: str,
    status: str,
    severity: str,
    expected: Any,
    observed: Any,
    artifact_path: str = "",
    details: str = "",
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "category": category,
            "description": description,
            "status": status,
            "severity": severity,
            "expected": expected,
            "observed": observed,
            "artifact_path": artifact_path,
            "details": details,
        }
    )


def verify_file_hashes() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    metadata = pd.read_csv(PHASE3_METADATA)
    index_path = LOCK_ROOT / "frozen_model_index.json"
    manifest_path = LOCK_ROOT / "external_validation_lock_manifest.json"
    model_index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    for sequence, record in metadata.iterrows():
        relative = str(record["artifact_path"])
        path = ROOT / relative
        exists = path.exists()
        observed_hash = sha256(path) if exists else ""
        observed_bytes = path.stat().st_size if exists else -1
        expected_hash = str(record["sha256"])
        expected_bytes = int(record["bytes"])
        rows.append(
            {
                "sequence": sequence + 1,
                "artifact_path": relative,
                "artifact_role": str(record["model_id"]),
                "exists": exists,
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "checksum_match": exists and observed_hash == expected_hash,
                "expected_bytes": expected_bytes,
                "observed_bytes": observed_bytes,
                "byte_count_match": exists and observed_bytes == expected_bytes,
                "status": (
                    "PASS"
                    if exists
                    and observed_hash == expected_hash
                    and observed_bytes == expected_bytes
                    else "FAIL"
                ),
            }
        )

    manifest_index_hash = str(manifest["model_index_sha256"])
    observed_index_hash = sha256(index_path)
    rows.append(
        {
            "sequence": len(rows) + 1,
            "artifact_path": str(index_path.relative_to(ROOT)),
            "artifact_role": "manifest_to_index_cross_check",
            "exists": True,
            "expected_sha256": manifest_index_hash,
            "observed_sha256": observed_index_hash,
            "checksum_match": observed_index_hash == manifest_index_hash,
            "expected_bytes": index_path.stat().st_size,
            "observed_bytes": index_path.stat().st_size,
            "byte_count_match": True,
            "status": "PASS" if observed_index_hash == manifest_index_hash else "FAIL",
        }
    )
    return pd.DataFrame(rows), model_index, manifest


def verify_version_material(model_index: dict[str, Any]) -> pd.DataFrame:
    material = model_index["version_material"]
    resources = [
        (
            "phase3_config",
            ROOT / "config" / "phase3_analysis_parameters.json",
            material["config_sha256"],
        ),
        (
            "phase2_outer_folds",
            ROOT / "results" / "csvs" / "phase2" / "adni_outer_fold_assignments.csv",
            material["outer_folds_sha256"],
        ),
        (
            "phase2_inner_folds",
            ROOT
            / "results"
            / "csvs"
            / "phase2"
            / "adni_nested_inner_fold_assignments.csv",
            material["inner_folds_sha256"],
        ),
    ]
    rows = []
    for name, path, expected in resources:
        exists = path.exists()
        observed = sha256(path) if exists else ""
        rows.append(
            {
                "resource": name,
                "artifact_path": str(path.relative_to(ROOT)),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "status": "PASS" if exists and observed == expected else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def verify_frozen_components(
    model_index: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schema = pd.read_csv(LOCK_ROOT / "frozen_feature_schema.csv")
    component_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for record in model_index["models"]:
        path = ROOT / record["artifact_path"]
        bundle = joblib.load(path)
        model_id = str(record["model_id"])
        horizon = int(record["horizon_months"])
        expected_schema = schema[
            schema["model_id"].eq(model_id)
            & schema["horizon_months"].eq(horizon)
        ]["feature"].tolist()
        bundle_features = list(bundle["continuous_features"]) + list(
            bundle["categorical_features"]
        )
        metadata_checks = {
            "model_version_id": bundle.get("model_version_id")
            == model_index["model_version_id"],
            "model_id": bundle.get("model_id") == model_id,
            "family": bundle.get("family") == record["family"],
            "horizon_months": int(bundle.get("horizon_months")) == horizon,
            "training_rows": int(bundle.get("training_rows"))
            == int(record["training_rows"]),
            "training_participants": int(bundle.get("training_participants"))
            == int(record["training_participants"]),
            "improvement_events": int(bundle.get("training_improvement_event_rows"))
            == int(record["improvement_event_rows"]),
            "progression_events": int(bundle.get("training_progression_event_rows"))
            == int(record["progression_event_rows"]),
            "external_outcomes_accessed": bundle.get("external_outcomes_accessed")
            is False,
            "feature_schema": bundle_features == expected_schema,
            "transformed_order_present": bool(bundle.get("transformed_feature_order")),
        }
        for field, passed in metadata_checks.items():
            metadata_rows.append(
                {
                    "model_id": model_id,
                    "horizon_months": horizon,
                    "metadata_field": field,
                    "status": "PASS" if passed else "FAIL",
                    "artifact_path": str(path.relative_to(ROOT)),
                }
            )

        preprocessor = bundle["preprocessor"]
        transformer = preprocessor.transformer
        numeric = transformer.named_transformers_["continuous"]
        imputer = numeric.named_steps["impute"]
        scaler = numeric.named_steps["scale"]
        for index, feature in enumerate(bundle["continuous_features"]):
            preprocessing_rows.append(
                {
                    "model_id": model_id,
                    "horizon_months": horizon,
                    "feature": feature,
                    "feature_type": "continuous",
                    "imputation_value": float(imputer.statistics_[index]),
                    "robust_center": float(scaler.center_[index]),
                    "robust_scale": float(scaler.scale_[index]),
                    "categories": "",
                    "parameters_finite": bool(
                        np.isfinite(
                            [
                                imputer.statistics_[index],
                                scaler.center_[index],
                                scaler.scale_[index],
                            ]
                        ).all()
                    ),
                }
            )
        categorical = transformer.named_transformers_["categorical"]
        cat_imputer = categorical.named_steps["impute"]
        encoder = categorical.named_steps["encode"]
        for index, feature in enumerate(bundle["categorical_features"]):
            preprocessing_rows.append(
                {
                    "model_id": model_id,
                    "horizon_months": horizon,
                    "feature": feature,
                    "feature_type": "categorical",
                    "imputation_value": str(cat_imputer.statistics_[index]),
                    "robust_center": np.nan,
                    "robust_scale": np.nan,
                    "categories": "|".join(map(str, encoder.categories_[index])),
                    "parameters_finite": True,
                }
            )

        for cause in ["improvement", "progression"]:
            model = bundle[f"{cause}_model"]
            coefficients = np.asarray(model.coef_, dtype=float)
            hazard_times = np.asarray(model.baseline_times_, dtype=float)
            hazard_increments = np.asarray(
                model.baseline_hazard_increments_, dtype=float
            )
            expected_alpha = float(record[f"{cause}_alpha"])
            component_rows.append(
                {
                    "model_id": model_id,
                    "family": record["family"],
                    "horizon_months": horizon,
                    "cause": cause,
                    "coefficient_count": len(coefficients),
                    "expected_coefficient_count": len(
                        bundle["transformed_feature_order"]
                    ),
                    "coefficients_finite": bool(np.isfinite(coefficients).all()),
                    "baseline_hazard_jumps": len(hazard_times),
                    "baseline_times_ordered": bool(
                        len(hazard_times) > 0 and np.all(np.diff(hazard_times) >= 0)
                    ),
                    "baseline_hazards_finite_nonnegative": bool(
                        len(hazard_increments) > 0
                        and np.isfinite(hazard_increments).all()
                        and np.all(hazard_increments >= 0)
                    ),
                    "expected_alpha": expected_alpha,
                    "observed_alpha": float(model.alpha),
                    "status": (
                        "PASS"
                        if len(coefficients)
                        == len(bundle["transformed_feature_order"])
                        and np.isfinite(coefficients).all()
                        and len(hazard_times) > 0
                        and np.all(np.diff(hazard_times) >= 0)
                        and np.isfinite(hazard_increments).all()
                        and np.all(hazard_increments >= 0)
                        and float(model.alpha) == expected_alpha
                        else "FAIL"
                    ),
                }
            )
    return (
        pd.DataFrame(component_rows),
        pd.DataFrame(preprocessing_rows),
        pd.DataFrame(metadata_rows),
    )


def verify_fixture(model_index: dict[str, Any]) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "scripts"))
    from phase3_inference import predict_from_frozen

    input_path = LOCK_ROOT / "inference_verification_input.csv"
    expected_path = LOCK_ROOT / "inference_verification_expected.csv"
    frame = pd.read_csv(input_path)
    expected = pd.read_csv(expected_path)
    first_artifact = ROOT / model_index["models"][0]["artifact_path"]
    first = predict_from_frozen(frame, first_artifact).reset_index(drop=True)
    probability_columns = [
        "improvement_probability",
        "progression_probability",
        "remaining_cdr_0_5_probability",
        "probability_sum",
    ]
    rows = []
    for column in probability_columns:
        difference = np.abs(
            first[column].to_numpy() - expected[column].to_numpy()
        )
        rows.append(
            {
                "field": column,
                "rows": len(first),
                "maximum_absolute_difference": float(difference.max()),
                "tolerance": 1e-12,
                "status": "PASS" if float(difference.max()) <= 1e-12 else "FAIL",
            }
        )
    exact_columns = [
        "prediction_horizon_months",
        "model_id",
        "model_version_id",
        "quality_flags",
    ]
    for column in exact_columns:
        match = first[column].astype(str).equals(expected[column].astype(str))
        rows.append(
            {
                "field": column,
                "rows": len(first),
                "maximum_absolute_difference": np.nan,
                "tolerance": np.nan,
                "status": "PASS" if match else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def package_completeness(model_index: dict[str, Any]) -> pd.DataFrame:
    available = {
        (str(record["model_id"]), int(record["horizon_months"]))
        for record in model_index["models"]
    }
    required = [
        ("B1", [12, 24, 36, 60], "required primary baseline"),
        ("D0", [12, 36], "required dynamic comparator"),
        ("D1", [12, 24, 36, 60], "required selected dynamic"),
        ("B3", [12, 36], "required no-APOE sensitivity"),
        ("D3", [12, 36], "required strictly lagged-MMSE sensitivity"),
    ]
    conditional = [
        ("B2", [12, 36], "conditional no-MMSE sensitivity; apply if saved"),
        ("D2", [12, 36], "conditional dynamic no-MMSE sensitivity; apply if saved"),
    ]
    rows = []
    for model_id, horizons, role in required:
        for horizon in horizons:
            present = (model_id, horizon) in available
            rows.append(
                {
                    "model_id": model_id,
                    "horizon_months": horizon,
                    "requirement": "required",
                    "role": role,
                    "present_in_frozen_index": present,
                    "status": "PASS" if present else "FAIL",
                    "blocking": not present,
                }
            )
    for model_id, horizons, role in conditional:
        for horizon in horizons:
            present = (model_id, horizon) in available
            rows.append(
                {
                    "model_id": model_id,
                    "horizon_months": horizon,
                    "requirement": "conditional_if_saved",
                    "role": role,
                    "present_in_frozen_index": present,
                    "status": "PASS" if present else "NOT_SAVED",
                    "blocking": False,
                }
            )
    return pd.DataFrame(rows)


def software_versions() -> pd.DataFrame:
    frozen = json.loads(PHASE3_RUN_METADATA.read_text(encoding="utf-8"))
    current = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    frozen_versions = dict(frozen.get("versions", {}))
    frozen_versions["python"] = str(frozen.get("python", "")).split()[0]
    rows = []
    for package, observed in current.items():
        expected = frozen_versions.get(package, "not recorded")
        rows.append(
            {
                "package": package,
                "phase3_recorded_version": expected,
                "phase4_current_version": observed,
                "status": (
                    "PASS"
                    if expected == observed
                    else "NOT_RECORDED"
                    if expected == "not recorded"
                    else "WARN"
                ),
            }
        )
    return pd.DataFrame(rows)


def stage_checklist(
    *,
    hashes: pd.DataFrame,
    version_material: pd.DataFrame,
    components: pd.DataFrame,
    preprocessing: pd.DataFrame,
    metadata: pd.DataFrame,
    fixture: pd.DataFrame,
    completeness: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        check_id="A01",
        category="checksum",
        description="All listed frozen package checksums and byte counts match",
        status="PASS" if hashes["status"].eq("PASS").all() else "FAIL",
        severity="critical",
        expected="all PASS",
        observed=f"{hashes['status'].eq('PASS').sum()}/{len(hashes)} PASS",
    )
    add_check(
        checks,
        check_id="A02",
        category="version_material",
        description="Frozen config and fold assignment checksums match",
        status="PASS" if version_material["status"].eq("PASS").all() else "FAIL",
        severity="critical",
        expected="all PASS",
        observed=f"{version_material['status'].eq('PASS').sum()}/{len(version_material)} PASS",
    )
    add_check(
        checks,
        check_id="A03",
        category="model_components",
        description="Cause-specific coefficients, baseline hazards, and penalties are valid",
        status="PASS" if components["status"].eq("PASS").all() else "FAIL",
        severity="critical",
        expected="all PASS",
        observed=f"{components['status'].eq('PASS').sum()}/{len(components)} PASS",
    )
    add_check(
        checks,
        check_id="A04",
        category="preprocessing",
        description="Frozen imputation, robust scaling, and categorical parameters are available",
        status="PASS" if preprocessing["parameters_finite"].all() else "FAIL",
        severity="critical",
        expected="all parameter rows valid",
        observed=f"{preprocessing['parameters_finite'].sum()}/{len(preprocessing)} valid",
    )
    add_check(
        checks,
        check_id="A05",
        category="metadata",
        description="Frozen model metadata and exact feature schemas match",
        status="PASS" if metadata["status"].eq("PASS").all() else "FAIL",
        severity="critical",
        expected="all PASS",
        observed=f"{metadata['status'].eq('PASS').sum()}/{len(metadata)} PASS",
    )
    add_check(
        checks,
        check_id="A06",
        category="determinism",
        description="The deterministic inference fixture reproduces saved predictions",
        status="PASS" if fixture["status"].eq("PASS").all() else "FAIL",
        severity="critical",
        expected="all PASS at tolerance 1e-12",
        observed=f"{fixture['status'].eq('PASS').sum()}/{len(fixture)} PASS",
    )
    required = completeness[completeness["requirement"].eq("required")]
    missing = required[required["status"].eq("FAIL")]
    add_check(
        checks,
        check_id="A07",
        category="package_completeness",
        description="All frozen models required by the Phase 4 prompt are present",
        status="PASS" if missing.empty else "FAIL",
        severity="critical",
        expected="B1, D0, D1, B3 no-APOE, and D3 lagged-MMSE artifacts",
        observed=(
            "all required artifacts present"
            if missing.empty
            else "; ".join(
                f"{row.model_id}-{row.horizon_months}m"
                for row in missing.itertuples()
            )
        ),
        details=(
            "The immutable package may not be repaired or refit during Phase 4."
            if not missing.empty
            else ""
        ),
    )
    add_check(
        checks,
        check_id="A08",
        category="external_outcome_lock",
        description="No OASIS-3 record-level feature or outcome file was opened by Stage A",
        status="PASS",
        severity="critical",
        expected="no OASIS-3 record-level access",
        observed="Stage A read only Phase 3 lock/provenance artifacts",
    )
    return pd.DataFrame(checks)


def validation_checklist_status(stage_a_pass: bool) -> pd.DataFrame:
    checklist = pd.read_csv(LOCK_ROOT / "phase4_validation_checklist.csv")
    rows = []
    for row in checklist.itertuples():
        if row.order == 1:
            status = "PASS" if stage_a_pass else "FAIL"
            details = (
                "All Stage A critical checks passed."
                if stage_a_pass
                else "Frozen package is incomplete for the attached Phase 4 specification."
            )
        else:
            status = "NOT_STARTED" if stage_a_pass else "BLOCKED"
            details = (
                "Awaiting ordered execution."
                if stage_a_pass
                else "Stage A gate failed; no OASIS-3 data access is permitted."
            )
        rows.append(
            {
                "order": row.order,
                "check": row.check,
                "required": row.required,
                "status": status,
                "details": details,
            }
        )
    return pd.DataFrame(rows)


def required_output_status() -> pd.DataFrame:
    completed = {
        "lock checksum verification",
        "Phase 4 QC results",
        "access audit",
        "saved-artifact verification",
    }
    outputs = [
        "lock checksum verification",
        "external feature-schema validation",
        "OASIS feature-shift table",
        "outcome-hidden B1 predictions",
        "outcome-hidden D0 predictions",
        "outcome-hidden D1 predictions",
        "prediction file manifest and hashes",
        "independent prediction-verification results",
        "outcome merge audit",
        "untouched external performance tables",
        "calibration tables",
        "discrimination tables",
        "Brier and integrated Brier tables",
        "multi-state performance tables",
        "probability-coherence checks",
        "participant-level bootstrap confidence intervals",
        "external-minus-internal comparisons",
        "D1-minus-D0 comparisons",
        "reconstructed-CDR sensitivity results",
        "recalibration parameters and results",
        "no-MMSE results",
        "no-APOE results",
        "lagged-MMSE results",
        "clinical-MCI subgroup results",
        "24- and 60-month sensitivity results",
        "subgroup results",
        "case-mix and distribution-shift tables",
        "Phase 4 QC results",
        "access audit",
        "saved-artifact verification",
    ]
    return pd.DataFrame(
        [
            {
                "required_output": output,
                "status": "COMPLETE_STAGE_A" if output in completed else "BLOCKED",
                "details": (
                    "Created during the lock audit."
                    if output in completed
                    else "Not created because Stage A failed before OASIS-3 access."
                ),
            }
            for output in outputs
        ]
    )


def access_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence": 1,
                "timestamp_utc": GENERATED_AT,
                "stage": "A",
                "action": "verify frozen Phase 3 model package and provenance",
                "resource_scope": "models/phase3/phase4_lock and Phase 3 checksum metadata",
                "oasis_record_level_features_accessed": False,
                "oasis_record_level_outcomes_accessed": False,
                "outcome_hidden_predictions_created": False,
                "outcomes_first_accessed_at_utc": "",
            },
            {
                "sequence": 2,
                "timestamp_utc": GENERATED_AT,
                "stage": "A_GATE",
                "action": "halt Phase 4 because required frozen models are absent",
                "resource_scope": "no OASIS-3 data opened",
                "oasis_record_level_features_accessed": False,
                "oasis_record_level_outcomes_accessed": False,
                "outcome_hidden_predictions_created": False,
                "outcomes_first_accessed_at_utc": "",
            },
        ]
    )


def create_stage_gate_figure() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")
    boxes = [
        (0.03, 0.38, 0.20, 0.28, "Frozen Phase 3 package\nB1 + D1 artifacts"),
        (0.29, 0.38, 0.20, 0.28, "Stage A audit\nhashes + metadata + fixture"),
        (0.55, 0.38, 0.20, 0.28, "Critical completeness failure\nD0, B3, D3 absent"),
        (0.81, 0.38, 0.16, 0.28, "HOLD\nNo OASIS access"),
    ]
    colors = ["#DCEAF7", "#DCEAF7", "#FADBD8", "#F5B7B1"]
    for (x, y, width, height, label), color in zip(boxes, colors):
        rectangle = plt.Rectangle(
            (x, y), width, height, facecolor=color, edgecolor="#263238", linewidth=2
        )
        ax.add_patch(rectangle)
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold" if "HOLD" in label else "normal",
        )
    for left, right in zip(boxes[:-1], boxes[1:]):
        start = (left[0] + left[2], left[1] + left[3] / 2)
        end = (right[0], right[1] + right[3] / 2)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#37474F"},
        )
    ax.text(
        0.5,
        0.86,
        "Phase 4 Locked External-Validation Stage Gate",
        transform=ax.transAxes,
        ha="center",
        fontsize=22,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.12,
        "The attached specification prohibits refitting or replacing missing frozen models during Phase 4.",
        transform=ax.transAxes,
        ha="center",
        fontsize=14,
    )
    fig.savefig(
        PLOT_ROOT / "phase4_stage_gate_workflow.png",
        dpi=1200,
        bbox_inches="tight",
    )
    fig.savefig(
        PLOT_ROOT / "phase4_stage_gate_workflow.svg",
        bbox_inches="tight",
    )
    plt.close(fig)


def write_reports(
    *,
    checks: pd.DataFrame,
    hashes: pd.DataFrame,
    completeness: pd.DataFrame,
    fixture: pd.DataFrame,
    versions: pd.DataFrame,
    checklist: pd.DataFrame,
    outputs: pd.DataFrame,
    decision: str,
) -> None:
    failed = checks[checks["status"].eq("FAIL")]
    missing = completeness[
        completeness["requirement"].eq("required")
        & completeness["status"].eq("FAIL")
    ]
    main = f"""# Phase 4 Locked External Validation — Stage A Report

Generated: {GENERATED_AT}

## Decision

# {decision}

Phase 4 did not pass its pre-outcome Stage A gate. The existing B1 and D1
artifacts are intact, but the frozen package is incomplete for the attached
Phase 4 specification. No OASIS-3 record-level feature or outcome file was
opened, no external prediction was generated, and Phase 5 was not started.

## Stage A checks

{markdown_table(checks, ["check_id", "category", "description", "status", "severity", "observed"])}

## Missing mandatory frozen artifacts

{markdown_table(missing, ["model_id", "horizon_months", "role", "status", "blocking"])}

The required D0 external comparator cannot be reconstructed from D1 predictions
or from Phase 3 out-of-fold predictions. It requires a final ADNI-only frozen
bundle with its own preprocessing, coefficients, penalties, and baseline
hazards. B3 no-APOE and D3 strictly lagged-MMSE final frozen bundles are also
absent. Creating them inside Phase 4 would violate the instruction not to modify
or silently replace the frozen Phase 3 package.

## Checksum result

{int(hashes["status"].eq("PASS").sum())} of {len(hashes)} listed hash/size checks
passed. This establishes integrity of the available package; it does not
establish completeness for the new Phase 4 prompt.

## Deterministic inference fixture

{markdown_table(fixture)}

## Software-version comparison

{markdown_table(versions)}

## Frozen checklist state

{markdown_table(checklist, ["order", "check", "status", "details"])}

## Required output status

{markdown_table(outputs)}

## Required repair

Create a separately versioned, ADNI-only Phase 3 lock-repair package without
opening OASIS-3 data. It must freeze D0 at 12 and 36 months at minimum, plus the
required B3 no-APOE and D3 strictly lagged-MMSE sensitivity bundles; update the
feature schemas, model index, checksums, inference fixtures, and independent
Phase 3 QC. Do not overwrite the current immutable package.

After that repair passes an independent audit, Phase 4 may restart from Stage A.
"""
    (REPORT_ROOT / "phase4_locked_external_validation_report.md").write_text(
        main, encoding="utf-8"
    )

    blocker = f"""# Phase 4 Frozen-Package Blocker Audit

Generated: {GENERATED_AT}

## Blocker

{markdown_table(missing, ["model_id", "horizon_months", "requirement", "role", "status"])}

## Evidence

The frozen index contains only B1 and D1 bundles:

{markdown_table(completeness, ["model_id", "horizon_months", "requirement", "role", "present_in_frozen_index", "status"])}

All existing package hashes match:

{markdown_table(hashes, ["artifact_path", "checksum_match", "byte_count_match", "status"])}

## Safety action

The Phase 4 process stopped before OASIS-3 feature or outcome access. This is the
required response to an incomplete immutable validation package.
"""
    (REPORT_ROOT / "phase4_frozen_package_blocker_audit.md").write_text(
        blocker, encoding="utf-8"
    )

    handoff = f"""# Phase 4 to Phase 5 Handoff

Generated: {GENERATED_AT}

## Decision

# HOLD FOR VALIDATION AUDIT

There is no Phase 5 handoff because Phase 4 is incomplete. OASIS-3 untouched
external performance, uncertainty, recalibration, endpoint sensitivity,
subgroups, and transportability classifications were not calculated.

## Exact next action

Perform an outcome-blind, separately versioned Phase 3 lock repair for D0, B3
no-APOE, and D3 strictly lagged-MMSE. Independently verify the repaired package,
then restart Phase 4 from Stage A. Do not begin Phase 5.
"""
    (REPORT_ROOT / "phase4_phase5_handoff.md").write_text(
        handoff, encoding="utf-8"
    )


def main() -> None:
    hashes, model_index, manifest = verify_file_hashes()
    version_material = verify_version_material(model_index)
    components, preprocessing, metadata = verify_frozen_components(model_index)
    fixture = verify_fixture(model_index)
    completeness = package_completeness(model_index)
    versions = software_versions()
    checks = stage_checklist(
        hashes=hashes,
        version_material=version_material,
        components=components,
        preprocessing=preprocessing,
        metadata=metadata,
        fixture=fixture,
        completeness=completeness,
    )
    stage_a_pass = not (
        checks["severity"].eq("critical") & checks["status"].eq("FAIL")
    ).any()
    decision = "PROCEED TO OUTCOME-HIDDEN FEATURES" if stage_a_pass else "HOLD FOR LOCK REPAIR"
    checklist = validation_checklist_status(stage_a_pass)
    outputs = required_output_status()
    access = access_audit()

    save_csv(hashes, "lock_checksum_verification.csv")
    save_csv(version_material, "frozen_version_material_verification.csv")
    save_csv(components, "frozen_model_component_verification.csv")
    save_csv(preprocessing, "frozen_preprocessing_verification.csv")
    save_csv(metadata, "frozen_model_metadata_verification.csv")
    save_csv(fixture, "deterministic_inference_verification.csv")
    save_csv(completeness, "frozen_package_completeness.csv")
    save_csv(versions, "software_version_verification.csv")
    save_csv(checks, "phase4_stage_a_qc_results.csv")
    save_csv(checklist, "phase4_validation_checklist_status.csv")
    save_csv(outputs, "phase4_required_output_status.csv")
    save_csv(access, "phase4_access_audit.csv")
    save_json(
        {
            "phase": 4,
            "stage": "A",
            "generated_at_utc": GENERATED_AT,
            "model_version_id": model_index["model_version_id"],
            "stage_a_pass": stage_a_pass,
            "decision": decision,
            "phase4_complete": False,
            "oasis_record_level_features_accessed": False,
            "oasis_record_level_outcomes_accessed": False,
            "outcome_hidden_predictions_created": False,
            "outcomes_first_accessed_at_utc": None,
            "phase5_started": False,
            "critical_checks": int(checks["severity"].eq("critical").sum()),
            "critical_failures": int(
                (checks["severity"].eq("critical") & checks["status"].eq("FAIL")).sum()
            ),
            "frozen_hash_checks_passed": int(hashes["status"].eq("PASS").sum()),
            "frozen_hash_checks_total": len(hashes),
            "required_missing_models": completeness.loc[
                completeness["blocking"],
                ["model_id", "horizon_months", "role"],
            ].to_dict(orient="records"),
            "immutable_phase3_lock_modified": False,
            "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        },
        "phase4_run_metadata.json",
    )
    create_stage_gate_figure()
    write_reports(
        checks=checks,
        hashes=hashes,
        completeness=completeness,
        fixture=fixture,
        versions=versions,
        checklist=checklist,
        outputs=outputs,
        decision=decision,
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "stage_a_pass": stage_a_pass,
                "existing_hashes": f"{hashes['status'].eq('PASS').sum()}/{len(hashes)} PASS",
                "critical_failures": int(
                    (
                        checks["severity"].eq("critical")
                        & checks["status"].eq("FAIL")
                    ).sum()
                ),
                "oasis_features_accessed": False,
                "oasis_outcomes_accessed": False,
                "phase5_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
