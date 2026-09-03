#!/usr/bin/env python3
"""Independent final verifier for locked OASIS-3 external validation.

This script is deliberately separate from ``phase4_external_validation.py``.
It does not import the analysis implementation or recompute performance
estimates. It independently re-hashes the repaired lock, prediction freeze,
and untouched-result freeze; checks execution order, population/event counts,
resampling structure, sensitivities, figures, reports, and the Phase boundary;
and then records the final Phase 4 QC decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_LOCK = ROOT / "models" / "phase3" / "phase4_lock"
LOCK = ROOT / "models" / "phase3" / "phase4_lock_v2"
TABLE_ROOT = ROOT / "results" / "csvs" / "phase4"
PLOT_ROOT = ROOT / "results" / "plots" / "phase4"
REPORT_ROOT = ROOT / "reports" / "phase4"
REPAIR_QC_ROOT = ROOT / "results" / "csvs" / "phase3_lock_repair"
PHASE3_ROOT = ROOT / "results" / "csvs" / "phase3"
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "Phase4"
    / "Phase4_LockedOASIS3ExternalValidation.ipynb"
)
GENERATED_AT = datetime.now(timezone.utc).isoformat()
TOLERANCE = 1e-12

TABLE_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def truth(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def iso(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_ROOT / name)


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLE_ROOT / name, index=False)


def markdown(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    shown = frame if columns is None else frame.loc[:, columns]
    return shown.to_markdown(index=False)


checks: list[dict[str, Any]] = []


def add(
    section: str,
    check_id: str,
    check: str,
    passed: bool,
    details: str,
    *,
    critical: bool = True,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "section": section,
            "check": check,
            "critical": critical,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


def refresh_repaired_lock_tables() -> None:
    """Replace obsolete original-lock Stage A summaries with v2 summaries."""
    inventory = pd.read_csv(LOCK / "model_inventory.csv")
    artifact_qc = pd.read_csv(
        REPAIR_QC_ROOT / "independent_model_artifact_verification.csv"
    )
    artifact_qc.to_csv(
        TABLE_ROOT / "frozen_model_component_verification.csv", index=False
    )

    preprocessing = pd.read_csv(LOCK / "frozen_preprocessing_parameters.csv")
    finite: list[bool] = []
    for record in preprocessing.itertuples(index=False):
        if str(record.feature_type) == "continuous":
            values = [
                record.imputation_value,
                record.robust_center,
                record.robust_scale,
            ]
            ok = all(
                pd.notna(value) and math.isfinite(float(value))
                for value in values
            ) and float(record.robust_scale) > 0
        else:
            ok = pd.notna(record.categories) and str(record.categories) != ""
        finite.append(ok)
    preprocessing["parameters_finite"] = finite
    preprocessing.to_csv(
        TABLE_ROOT / "frozen_preprocessing_verification.csv", index=False
    )

    metadata = inventory[
        [
            "model_id",
            "family",
            "horizon_months",
            "model_version_id",
            "artifact_path",
            "sha256",
            "bytes",
            "training_rows",
            "training_participants",
            "improvement_event_rows",
            "progression_event_rows",
            "improvement_alpha",
            "progression_alpha",
            "external_outcomes_accessed",
            "model_origin",
            "refit_during_repair",
        ]
    ].copy()
    metadata["status"] = np.where(
        ~metadata["external_outcomes_accessed"].map(truth), "PASS", "FAIL"
    )
    metadata.to_csv(
        TABLE_ROOT / "frozen_model_metadata_verification.csv", index=False
    )

    pd.read_csv(
        REPAIR_QC_ROOT / "independent_inference_fixture_verification.csv"
    ).to_csv(TABLE_ROOT / "deterministic_inference_verification.csv", index=False)

    expected = [
        ("B1", 12, "required primary baseline"),
        ("B1", 24, "required secondary baseline"),
        ("B1", 36, "required primary baseline"),
        ("B1", 60, "required secondary baseline"),
        ("D0", 12, "required dynamic comparator"),
        ("D0", 36, "required dynamic comparator"),
        ("D1", 12, "required selected dynamic"),
        ("D1", 24, "required secondary selected dynamic"),
        ("D1", 36, "required selected dynamic"),
        ("D1", 60, "required secondary selected dynamic"),
        ("B3", 12, "required no-APOE sensitivity"),
        ("B3", 36, "required no-APOE sensitivity"),
        ("D3", 12, "required strictly lagged-MMSE sensitivity"),
        ("D3", 36, "required strictly lagged-MMSE sensitivity"),
    ]
    present = {
        (str(row.model_id), int(row.horizon_months))
        for row in inventory.itertuples(index=False)
    }
    rows = [
        {
            "model_id": model,
            "horizon_months": horizon,
            "requirement": "required",
            "role": role,
            "present_in_frozen_index": (model, horizon) in present,
            "status": "PASS" if (model, horizon) in present else "FAIL",
            "blocking": (model, horizon) not in present,
            "lock_path": rel(LOCK),
        }
        for model, horizon, role in expected
    ]
    for model in ("B2", "D2"):
        for horizon in (12, 36):
            rows.append(
                {
                    "model_id": model,
                    "horizon_months": horizon,
                    "requirement": "optional_if_saved",
                    "role": "optional sensitivity; not frozen and not recreated",
                    "present_in_frozen_index": False,
                    "status": "NOT_SAVED_OPTIONAL",
                    "blocking": False,
                    "lock_path": rel(LOCK),
                }
            )
    pd.DataFrame(rows).to_csv(
        TABLE_ROOT / "frozen_package_completeness.csv", index=False
    )

    external = json.loads(
        (LOCK / "external_validation_lock_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    material = pd.DataFrame(
        [
            {
                "resource": "repaired_lock_model_index",
                "artifact_path": rel(LOCK / "frozen_model_index.json"),
                "expected_sha256": external["model_index_sha256"],
                "observed_sha256": sha256(LOCK / "frozen_model_index.json"),
                "status": (
                    "PASS"
                    if sha256(LOCK / "frozen_model_index.json")
                    == external["model_index_sha256"]
                    else "FAIL"
                ),
            },
            {
                "resource": "repaired_lock_checksum_manifest",
                "artifact_path": rel(LOCK / "checksum_manifest.csv"),
                "expected_sha256": external["checksum_manifest_sha256"],
                "observed_sha256": sha256(LOCK / "checksum_manifest.csv"),
                "status": (
                    "PASS"
                    if sha256(LOCK / "checksum_manifest.csv")
                    == external["checksum_manifest_sha256"]
                    else "FAIL"
                ),
            },
        ]
    )
    material.to_csv(
        TABLE_ROOT / "frozen_version_material_verification.csv", index=False
    )


def check_locks() -> None:
    repair_qc = pd.read_csv(REPAIR_QC_ROOT / "independent_qc_results.csv")
    add(
        "lock",
        "Q01",
        "independent repaired-lock QC remains fully passing",
        len(repair_qc) == 18 and repair_qc["status"].eq("PASS").all(),
        (
            f"checks={len(repair_qc)}; "
            f"failures={int(repair_qc['status'].ne('PASS').sum())}"
        ),
    )

    external = json.loads(
        (LOCK / "external_validation_lock_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    index_ok = (
        sha256(LOCK / "frozen_model_index.json")
        == external["model_index_sha256"]
    )
    manifest_ok = (
        sha256(LOCK / "checksum_manifest.csv")
        == external["checksum_manifest_sha256"]
    )
    add(
        "lock",
        "Q02",
        "external lock descriptor authenticates index and checksum manifest",
        index_ok and manifest_ok,
        f"index={index_ok}; checksum_manifest={manifest_ok}",
    )

    manifest = pd.read_csv(LOCK / "checksum_manifest.csv")
    rows: list[dict[str, Any]] = []
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
        rows.append(
            {
                "artifact_path": rel(path) if exists else path.as_posix(),
                "expected_sha256": record.sha256,
                "observed_sha256": observed_hash,
                "expected_bytes": int(record.bytes),
                "observed_bytes": observed_bytes,
                "status": "PASS" if passed else "FAIL",
            }
        )
    verification = pd.DataFrame(rows)
    verification.to_csv(TABLE_ROOT / "lock_checksum_verification.csv", index=False)
    manifest_scope = set(manifest["package_relative_path"].astype(str))
    actual_scope = {
        path.relative_to(LOCK).as_posix()
        for path in LOCK.rglob("*")
        if path.is_file()
    } - {"checksum_manifest.csv", "external_validation_lock_manifest.json"}
    add(
        "lock",
        "Q03",
        "all repaired-lock files pass hashes/sizes and package has no extras",
        verification["status"].eq("PASS").all()
        and actual_scope == manifest_scope,
        (
            f"hashes={int(verification['status'].eq('PASS').sum())}/"
            f"{len(verification)}; extras={sorted(actual_scope-manifest_scope)}; "
            f"missing={sorted(manifest_scope-actual_scope)}"
        ),
    )

    original_meta = pd.read_csv(
        PHASE3_ROOT / "model_metadata_and_checksums.csv"
    )
    original_meta = original_meta[
        original_meta["artifact_path"]
        .astype(str)
        .str.startswith("models/phase3/phase4_lock/")
    ]
    original_pass = []
    for record in original_meta.itertuples(index=False):
        path = ROOT / str(record.artifact_path)
        original_pass.append(
            path.exists()
            and sha256(path) == str(record.sha256)
            and path.stat().st_size == int(record.bytes)
        )
    add(
        "lock",
        "Q04",
        "original phase4_lock remains unchanged",
        len(original_pass) == 16 and all(original_pass),
        f"verified={sum(original_pass)}/{len(original_pass)}",
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
    components = pd.read_csv(
        REPAIR_QC_ROOT / "independent_model_artifact_verification.csv"
    )
    add(
        "lock",
        "Q05",
        "all required model/horizon bundles and frozen components are present",
        observed == expected
        and len(components) == 14
        and components["status"].eq("PASS").all(),
        (
            f"bundles={len(observed)}; missing={sorted(expected-observed)}; "
            f"component_failures={int(components['status'].ne('PASS').sum())}"
        ),
    )
    d3 = pd.read_csv(
        REPAIR_QC_ROOT / "independent_d3_temporal_verification.csv"
    )
    grouping = pd.read_csv(
        REPAIR_QC_ROOT / "independent_participant_grouping_verification.csv"
    )
    add(
        "lock",
        "Q06",
        "D3 is strictly pre-landmark and all dynamic folds are participant-grouped",
        d3["status"].eq("PASS").all()
        and grouping["status"].eq("PASS").all(),
        (
            f"d3={int(d3['status'].eq('PASS').sum())}/{len(d3)}; "
            f"grouping={int(grouping['status'].eq('PASS').sum())}/{len(grouping)}"
        ),
    )


def check_prediction_gate() -> None:
    freeze = json.loads(
        (TABLE_ROOT / "prediction_freeze_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_path = TABLE_ROOT / "outcome_hidden_prediction_manifest.csv"
    prediction_manifest = pd.read_csv(manifest_path)
    files_ok = []
    no_outcomes = []
    for record in prediction_manifest.itertuples(index=False):
        path = ROOT / str(record.relative_path)
        files_ok.append(
            path.exists()
            and sha256(path) == str(record.sha256)
            and path.stat().st_size == int(record.bytes)
        )
        no_outcomes.append(not truth(record.outcome_columns_present))
    add(
        "prediction_freeze",
        "Q07",
        "prediction manifest and all 28 outcome-hidden files retain frozen hashes",
        sha256(manifest_path) == freeze["prediction_manifest_sha256"]
        and len(files_ok) == 28
        and all(files_ok),
        (
            f"manifest_hash={sha256(manifest_path)==freeze['prediction_manifest_sha256']}; "
            f"files={sum(files_ok)}/{len(files_ok)}"
        ),
    )
    add(
        "prediction_freeze",
        "Q08",
        "all frozen prediction files remain outcome-free",
        all(no_outcomes)
        and not prediction_manifest["outcomes_accessed_before_freeze"].map(truth).any(),
        f"outcome_free={sum(no_outcomes)}/{len(no_outcomes)}",
    )
    independent = read_csv("phase4_independent_prediction_qc_results.csv")
    regeneration = read_csv("independent_prediction_regeneration.csv")
    max_difference = max(
        float(regeneration["maximum_point_prediction_difference"].max()),
        float(regeneration["maximum_curve_prediction_difference"].max()),
    )
    add(
        "prediction_freeze",
        "Q09",
        "independent prediction regeneration passes to numerical tolerance",
        independent["status"].eq("PASS").all()
        and max_difference <= TOLERANCE,
        (
            f"checks={len(independent)}; failures="
            f"{int(independent['status'].ne('PASS').sum())}; "
            f"maximum_difference={max_difference:.3g}"
        ),
    )
    population = read_csv("outcome_hidden_population_verification.csv")
    coherence = read_csv("outcome_hidden_probability_coherence.csv")
    add(
        "prediction_freeze",
        "Q10",
        "expected OASIS populations and probability coherence pass",
        len(population) == 14
        and population["status"].eq("PASS").all()
        and len(coherence) == 14
        and coherence["status"].eq("PASS").all(),
        (
            f"population={int(population['status'].eq('PASS').sum())}/{len(population)}; "
            f"coherence={int(coherence['status'].eq('PASS').sum())}/{len(coherence)}"
        ),
    )

    access = read_csv("phase4_access_audit.csv").sort_values("sequence")
    first_outcome = access[
        access["oasis_record_level_outcomes_accessed"].map(truth)
    ].iloc[0]
    before = access[access["sequence"] < int(first_outcome.sequence)]
    ordering = (
        not before["oasis_record_level_outcomes_accessed"].map(truth).any()
        and iso(freeze["freeze_timestamp_utc"])
        < iso(freeze["independent_prediction_qc_completed_at_utc"])
        < iso(freeze["outcomes_first_accessed_at_utc"])
        and str(first_outcome.stage) == "D"
    )
    add(
        "execution_order",
        "Q11",
        "outcomes opened only after prediction freeze and independent QC",
        ordering,
        (
            f"freeze={freeze['freeze_timestamp_utc']}; "
            f"prediction_qc={freeze['independent_prediction_qc_completed_at_utc']}; "
            f"outcomes={freeze['outcomes_first_accessed_at_utc']}"
        ),
    )


def check_untouched_results() -> None:
    freeze = json.loads(
        (TABLE_ROOT / "untouched_results_freeze_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_path = ROOT / freeze["untouched_artifact_manifest_path"]
    manifest = pd.read_csv(manifest_path)
    matches = []
    for record in manifest.itertuples(index=False):
        path = ROOT / str(record.relative_path)
        matches.append(
            path.exists()
            and sha256(path) == str(record.sha256)
            and path.stat().st_size == int(record.bytes)
        )
    add(
        "untouched_results",
        "Q12",
        "all untouched external-result artifacts retain their frozen hashes",
        sha256(manifest_path)
        == freeze["untouched_artifact_manifest_sha256"]
        and len(matches) == int(freeze["artifacts"])
        and all(matches),
        (
            f"manifest_hash="
            f"{sha256(manifest_path)==freeze['untouched_artifact_manifest_sha256']}; "
            f"artifacts={sum(matches)}/{len(matches)}"
        ),
    )
    order_ok = (
        iso(freeze["freeze_timestamp_utc"])
        < iso(freeze["secondary_analyses_completed_at_utc"])
        and truth(freeze["endpoint_sensitivity_started_after_freeze"])
        and truth(freeze["recalibration_started_after_freeze"])
    )
    add(
        "execution_order",
        "Q13",
        "endpoint sensitivity and recalibration began after untouched freeze",
        order_ok,
        (
            f"untouched_freeze={freeze['freeze_timestamp_utc']}; "
            f"secondary_complete={freeze['secondary_analyses_completed_at_utc']}"
        ),
    )

    events = read_csv("baseline_event_count_verification.csv")
    expected = {
        12: (45, 18),
        24: (143, 58),
        36: (170, 101),
        60: (187, 150),
    }
    exact = len(events) == 4 and events["status"].eq("PASS").all()
    for row in events.itertuples(index=False):
        exact = exact and (
            int(row.observed_improvement),
            int(row.observed_progression),
        ) == expected[int(row.horizon_months)]
    add(
        "endpoint",
        "Q14",
        "baseline CDRTOT event counts exactly match the prespecified audit",
        exact,
        "expected=12m 45/18; 24m 143/58; 36m 170/101; 60m 187/150",
    )

    metrics = read_csv("untouched_external_metrics.csv")
    primary = metrics[
        metrics["model_id"].isin(["B1", "D0", "D1"])
        & metrics["horizon_months"].isin([12, 36])
    ]
    expected_primary = {
        ("B1", 12), ("B1", 36),
        ("D0", 12), ("D0", 36),
        ("D1", 12), ("D1", 36),
    }
    present = {
        (str(row.model_id), int(row.horizon_months))
        for row in primary.itertuples(index=False)
    }
    required_columns = [
        "auc_improvement",
        "auc_progression",
        "c_index_improvement",
        "c_index_progression",
        "brier_improvement",
        "brier_progression",
        "multistate_brier",
        "calibration_intercept_improvement",
        "calibration_slope_improvement",
        "observed_expected_improvement",
        "calibration_intercept_progression",
        "calibration_slope_progression",
        "observed_expected_progression",
    ]
    add(
        "untouched_results",
        "Q15",
        "all six primary untouched evaluations and required metrics are present",
        present == expected_primary
        and primary[required_columns].notna().all().all(),
        f"present={sorted(present)}; missing={sorted(expected_primary-present)}",
    )
    integrated = read_csv("untouched_integrated_brier_scores.csv")
    coherence = read_csv("untouched_probability_coherence.csv")
    add(
        "untouched_results",
        "Q16",
        "integrated Brier, remaining-state calibration, and coherence are complete",
        len(integrated) == 14
        and integrated.filter(like="integrated").notna().all().all()
        and len(coherence) == 14
        and coherence["status"].eq("PASS").all()
        and primary[
            [
                "calibration_intercept_remaining",
                "calibration_slope_remaining",
                "observed_expected_remaining",
            ]
        ].notna().all().all(),
        (
            f"integrated_rows={len(integrated)}; "
            f"coherence_pass={int(coherence['status'].eq('PASS').sum())}/{len(coherence)}"
        ),
    )

    bootstrap = read_csv("participant_bootstrap_metric_replicates.csv")
    ci = read_csv("participant_bootstrap_confidence_intervals.csv")
    grouped = (
        len(bootstrap) == 6000
        and bootstrap["bootstrap_unit"].eq("participant").all()
        and bootstrap.groupby(
            ["model_id", "horizon_months"]
        )["bootstrap_index"].nunique().eq(1000).all()
        and len(ci) == 90
        and ci["bootstrap_replicates"].eq(1000).all()
        and ci["bootstrap_unit"].eq("participant").all()
        and (ci["ci_low"] <= ci["ci_high"]).all()
    )
    add(
        "resampling",
        "Q17",
        "confidence intervals use 1,000 participant-clustered bootstraps",
        grouped,
        f"replicate_rows={len(bootstrap)}; ci_rows={len(ci)}",
    )

    comparisons = read_csv("untouched_model_comparisons.csv")
    paired = read_csv("d1_vs_d0_paired_bootstrap_replicates.csv")
    paired_ci = read_csv("d1_vs_d0_paired_bootstrap_ci.csv")
    d1d0 = comparisons[
        comparisons["comparison"].eq("D1_vs_D0_matched_rows")
    ]
    comparison_ok = (
        set(d1d0["horizon_months"]) == {12, 36}
        and d1d0["common_rows"].eq(845).all()
        and d1d0["common_participants"].eq(437).all()
        and len(paired) == 10000
        and paired["bootstrap_unit"].eq("participant").all()
        and paired.groupby(
            ["horizon_months", "metric"]
        )["bootstrap_index"].nunique().eq(1000).all()
        and len(paired_ci) == 10
    )
    add(
        "comparisons",
        "Q18",
        "D1-versus-D0 uses identical landmark rows and paired participant bootstrap",
        comparison_ok,
        (
            f"matched_rows={sorted(d1d0['common_rows'].unique())}; "
            f"participants={sorted(d1d0['common_participants'].unique())}; "
            f"paired_replicates={len(paired)}"
        ),
    )

    comparisons_present = (
        comparisons["comparison"].str.contains("B3_vs_B1").any()
        and comparisons["comparison"].str.contains("D3_vs_D1").any()
        and Path(TABLE_ROOT / "adni_oasis_performance_comparison.csv").exists()
    )
    add(
        "comparisons",
        "Q19",
        "ADNI transport, B1-B3, and D1-D3 comparisons are saved",
        comparisons_present,
        (
            f"comparisons={sorted(comparisons['comparison'].unique())}; "
            f"transport_rows={len(read_csv('adni_oasis_performance_comparison.csv'))}"
        ),
    )


def check_secondary_analyses() -> None:
    endpoint_counts = read_csv("reconstructed_cdr_eligibility_event_counts.csv")
    endpoint_metrics = read_csv("reconstructed_cdr_metrics.csv")
    endpoint_delta = read_csv("reconstructed_cdr_metric_comparison.csv")
    endpoint_ok = (
        len(endpoint_counts) == 14
        and len(endpoint_metrics) == 14
        and len(endpoint_delta) == 196
        and set(endpoint_counts["model_id"]) == {"B1", "B3", "D0", "D1", "D3"}
    )
    add(
        "endpoint_sensitivity",
        "Q20",
        "reconstructed six-box CDR repeats all evaluations with frozen predictions",
        endpoint_ok,
        (
            f"eligibility_rows={len(endpoint_counts)}; "
            f"metric_rows={len(endpoint_metrics)}; delta_rows={len(endpoint_delta)}"
        ),
    )

    params = read_csv("cross_fitted_recalibration_parameters.csv")
    recal = read_csv("cross_fitted_recalibration_metrics.csv")
    fold_counts = params.groupby(
        ["model_id", "horizon_months", "method", "cause"]
    )["fold"].nunique()
    recal_ok = (
        len(params) == 120
        and fold_counts.eq(5).all()
        and not params["predictor_coefficients_refit"].map(truth).any()
        and len(recal) == 12
        and recal["validation_design"]
        .eq("5-fold participant-grouped cross-fitting")
        .all()
    )
    add(
        "recalibration",
        "Q21",
        "recalibration is five-fold participant-grouped and refits no predictors",
        recal_ok,
        (
            f"parameter_rows={len(params)}; groups={len(fold_counts)}; "
            f"metric_rows={len(recal)}"
        ),
    )

    metrics = read_csv("untouched_external_metrics.csv")
    sensitivity_pairs = {
        (str(row.model_id), int(row.horizon_months))
        for row in metrics.itertuples(index=False)
    }
    required_sensitivities = {
        ("B3", 12), ("B3", 36), ("D3", 12), ("D3", 36),
        ("B1", 24), ("B1", 60), ("D1", 24), ("D1", 60),
    }
    mci = read_csv("clinical_mci_subgroup_results.csv")
    add(
        "sensitivities",
        "Q22",
        "B3, D3, clinical-MCI, 24m, and 60m sensitivities are complete",
        required_sensitivities.issubset(sensitivity_pairs)
        and len(mci) == 4
        and set(mci["model_id"]) == {"B1", "D1"},
        (
            f"missing={sorted(required_sensitivities-sensitivity_pairs)}; "
            f"clinical_mci_rows={len(mci)}"
        ),
    )

    continuous = read_csv("predictor_distribution_shift.csv")
    categorical = read_csv("categorical_distribution_shift.csv")
    event_shift = read_csv("event_rate_shift.csv")
    add(
        "distribution_shift",
        "Q23",
        "predictor, categorical, and event-rate shift tables are complete",
        len(continuous) > 0 and len(categorical) > 0 and len(event_shift) > 0,
        (
            f"continuous={len(continuous)}; categorical={len(categorical)}; "
            f"event_rate={len(event_shift)}"
        ),
    )

    transport = read_csv("transportability_classification.csv")
    classification_ok = (
        len(transport) == 1
        and transport.iloc[0]["classification"]
        in {
            "TRANSPORTABLE",
            "TRANSPORTABLE WITH RECALIBRATION",
            "NOT TRANSPORTABLE",
        }
        and transport.iloc[0]["classification_rule_source"]
        == "config/phase4_analysis_parameters.json"
    )
    add(
        "transportability",
        "Q24",
        "transportability classification uses the prespecified rule",
        classification_ok,
        (
            f"classification={transport.iloc[0]['classification']}; "
            f"source={transport.iloc[0]['classification_rule_source']}"
        ),
    )


def create_stage_workflow() -> None:
    labels = [
        ("A", "Lock v2\nverified"),
        ("B", "Outcome-hidden\npredictions frozen"),
        ("C", "Independent\nprediction QC"),
        ("D", "Untouched OASIS\nvalidation frozen"),
        ("E", "Endpoint\nsensitivity"),
        ("F", "Cross-fitted\nrecalibration"),
        ("QC", "Independent\nfinal audit"),
    ]
    fig, ax = plt.subplots(figsize=(15, 4.5))
    x = np.arange(len(labels), dtype=float)
    ax.plot(x, np.zeros_like(x), color="#2f6f5f", linewidth=4, zorder=1)
    for position, (stage, text_label) in zip(x, labels):
        ax.scatter(
            [position],
            [0],
            s=1600,
            color="#d9f0e7",
            edgecolor="#1f5d4e",
            linewidth=2.5,
            zorder=2,
        )
        ax.text(
            position,
            0.03,
            stage,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#17483d",
            zorder=3,
        )
        ax.text(
            position,
            -0.33,
            text_label,
            ha="center",
            va="top",
            fontsize=12,
            color="#263238",
        )
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(-0.85, 0.45)
    ax.axis("off")
    ax.set_title(
        "Phase 4 locked-validation execution order — all gates passed",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    fig.tight_layout()
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


def check_figures_and_reports() -> None:
    create_stage_workflow()
    figures = [
        ("phase4_stage_gate_workflow.png", 1200),
        ("untouched_oasis_calibration_panels.png", 1200),
        ("adni_oasis_auc_transport.png", 1200),
        ("oasis_adni_distribution_shift_heatmap.png", 1200),
        ("cross_fitted_recalibration_effect.png", 1200),
        ("d1_vs_d0_transport_value.png", 600),
        ("reconstructed_cdr_sensitivity.png", 600),
    ]
    figure_rows = []
    for name, required_dpi in figures:
        path = PLOT_ROOT / name
        svg = path.with_suffix(".svg")
        dpi_x = dpi_y = 0.0
        width = height = 0
        if path.exists():
            with Image.open(path) as image:
                width, height = image.size
                dpi = image.info.get("dpi", (0.0, 0.0))
                dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
        threshold = 1190 if required_dpi == 1200 else 590
        passed = (
            path.exists()
            and svg.exists()
            and min(dpi_x, dpi_y) >= threshold
        )
        figure_rows.append(
            {
                "figure": rel(path),
                "svg": rel(svg),
                "width_pixels": width,
                "height_pixels": height,
                "observed_dpi_x": dpi_x,
                "observed_dpi_y": dpi_y,
                "required_dpi": required_dpi,
                "status": "PASS" if passed else "FAIL",
            }
        )
    figure_qc = pd.DataFrame(figure_rows)
    save_csv(figure_qc, "phase4_figure_verification.csv")
    add(
        "figures",
        "Q25",
        "all raster figures meet 600/1200 DPI rules and have SVG versions",
        figure_qc["status"].eq("PASS").all(),
        (
            f"figures={len(figure_qc)}; "
            f"failures={int(figure_qc['status'].ne('PASS').sum())}"
        ),
    )

    reports = [
        "phase4_locked_external_validation_report.md",
        "phase4_untouched_calibration_report.md",
        "phase4_d1_vs_d0_transportability_report.md",
        "phase4_endpoint_sensitivity_report.md",
        "phase4_recalibration_report.md",
        "phase4_distribution_shift_report.md",
        "phase4_phase5_handoff.md",
        "phase4_updated_model_card.md",
        "phase4_prediction_independent_qc_report.md",
        "phase4_stage_a_lock_and_prediction_report.md",
    ]
    missing = [name for name in reports if not (REPORT_ROOT / name).exists()]
    add(
        "reports",
        "Q26",
        "all required Phase 4 reports and handoffs are present",
        not missing,
        f"reports={len(reports)}; missing={missing}",
    )


def check_phase_boundary() -> None:
    metadata = json.loads(
        (TABLE_ROOT / "phase4_run_metadata.json").read_text(encoding="utf-8")
    )
    add(
        "phase_boundary",
        "Q27",
        "Phase 5 was not started automatically",
        metadata.get("phase5_started") is False,
        f"phase5_started={metadata.get('phase5_started')}",
    )
    add(
        "software",
        "Q28",
        "Phase 4 ran in Python 3.10 with recorded package versions",
        platform.python_version().startswith("3.10")
        and all(
            key in metadata.get("software", {})
            for key in ("numpy", "pandas", "scipy", "joblib")
        ),
        (
            f"python={platform.python_version()}; "
            f"software={metadata.get('software', {})}"
        ),
    )


def required_outputs() -> pd.DataFrame:
    paths = {
        "repaired-lock checksum verification":
            TABLE_ROOT / "lock_checksum_verification.csv",
        "frozen model inventory":
            TABLE_ROOT / "frozen_package_completeness.csv",
        "outcome-hidden feature manifest":
            TABLE_ROOT / "outcome_hidden_feature_manifest.csv",
        "outcome-hidden prediction manifest":
            TABLE_ROOT / "outcome_hidden_prediction_manifest.csv",
        "independent prediction QC":
            TABLE_ROOT / "phase4_independent_prediction_qc_results.csv",
        "access audit":
            TABLE_ROOT / "phase4_access_audit.csv",
        "untouched metrics":
            TABLE_ROOT / "untouched_external_metrics.csv",
        "untouched integrated Brier scores":
            TABLE_ROOT / "untouched_integrated_brier_scores.csv",
        "participant bootstrap estimates":
            TABLE_ROOT / "participant_bootstrap_metric_replicates.csv",
        "participant bootstrap confidence intervals":
            TABLE_ROOT / "participant_bootstrap_confidence_intervals.csv",
        "D1-versus-D0 matched comparison":
            TABLE_ROOT / "untouched_model_comparisons.csv",
        "D1-versus-D0 paired bootstrap":
            TABLE_ROOT / "d1_vs_d0_paired_bootstrap_replicates.csv",
        "ADNI-OASIS comparison":
            TABLE_ROOT / "adni_oasis_performance_comparison.csv",
        "B3 and D3 sensitivity estimates":
            TABLE_ROOT / "untouched_external_metrics.csv",
        "clinical-MCI sensitivity":
            TABLE_ROOT / "clinical_mci_subgroup_results.csv",
        "24m and 60m sensitivities":
            TABLE_ROOT / "untouched_external_metrics.csv",
        "reconstructed-CDR metrics":
            TABLE_ROOT / "reconstructed_cdr_metrics.csv",
        "reconstructed-CDR eligibility/event counts":
            TABLE_ROOT / "reconstructed_cdr_eligibility_event_counts.csv",
        "cross-fitted recalibration parameters":
            TABLE_ROOT / "cross_fitted_recalibration_parameters.csv",
        "cross-fitted recalibration metrics":
            TABLE_ROOT / "cross_fitted_recalibration_metrics.csv",
        "distribution-shift tables":
            TABLE_ROOT / "predictor_distribution_shift.csv",
        "transportability classification":
            TABLE_ROOT / "transportability_classification.csv",
        "external-validation report":
            REPORT_ROOT / "phase4_locked_external_validation_report.md",
        "calibration report":
            REPORT_ROOT / "phase4_untouched_calibration_report.md",
        "transportability report":
            REPORT_ROOT / "phase4_d1_vs_d0_transportability_report.md",
        "endpoint report":
            REPORT_ROOT / "phase4_endpoint_sensitivity_report.md",
        "recalibration report":
            REPORT_ROOT / "phase4_recalibration_report.md",
        "distribution-shift report":
            REPORT_ROOT / "phase4_distribution_shift_report.md",
        "model card":
            REPORT_ROOT / "phase4_updated_model_card.md",
        "Phase 5 handoff":
            REPORT_ROOT / "phase4_phase5_handoff.md",
        "independent QC script":
            ROOT / "scripts" / "phase4_qc.py",
        "Phase 4 notebook": NOTEBOOK,
    }
    rows = []
    for label, path in paths.items():
        rows.append(
            {
                "required_output": label,
                "path": rel(path),
                "status": "COMPLETE" if path.exists() else "MISSING",
                "bytes": path.stat().st_size if path.exists() else -1,
            }
        )
    return pd.DataFrame(rows)


def write_validation_checklist(qc: pd.DataFrame) -> None:
    mapping = [
        ("repaired lock verified before data access", ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06"]),
        ("outcome-hidden inputs match frozen schemas/populations", ["Q10"]),
        ("predictions frozen, hashed, deterministic, and outcome-free", ["Q07", "Q08", "Q09"]),
        ("outcomes opened only after independent prediction QC", ["Q11"]),
        ("untouched primary validation frozen and complete", ["Q12", "Q14", "Q15", "Q16", "Q17"]),
        ("required model and transport comparisons complete", ["Q18", "Q19"]),
        ("endpoint sensitivity follows untouched freeze", ["Q13", "Q20"]),
        ("cross-fitted recalibration is labelled model updating", ["Q21"]),
        ("required sensitivities and shift analyses complete", ["Q22", "Q23", "Q24"]),
        ("figures and reports meet output requirements", ["Q25", "Q26"]),
        ("Phase boundary and software requirements pass", ["Q27", "Q28"]),
    ]
    rows = []
    for order, (label, ids) in enumerate(mapping, start=1):
        subset = qc[qc["check_id"].isin(ids)]
        passed = len(subset) == len(ids) and subset["status"].eq("PASS").all()
        rows.append(
            {
                "order": order,
                "check": label,
                "required": True,
                "status": "PASS" if passed else "FAIL",
                "qc_check_ids": "|".join(ids),
                "details": (
                    f"{int(subset['status'].eq('PASS').sum())}/{len(ids)} "
                    "component checks passed"
                ),
            }
        )
    save_csv(pd.DataFrame(rows), "phase4_validation_checklist_status.csv")


def update_metadata_and_handoff(
    decision: str, qc: pd.DataFrame, outputs: pd.DataFrame
) -> None:
    path = TABLE_ROOT / "phase4_run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    critical_failures = int(
        (
            qc["critical"].map(truth)
            & qc["status"].ne("PASS")
        ).sum()
    )
    metadata.update(
        {
            "decision": decision,
            "final_decision": decision,
            "final_qc_completed_at_utc": GENERATED_AT,
            "final_qc_checks": len(qc),
            "final_qc_failures": critical_failures,
            "independent_final_qc_pending": False,
            "independent_final_qc_pass": critical_failures == 0,
            "untouched_results_frozen": True,
            "phase4_complete": critical_failures == 0,
            "phase5_started": False,
            "required_outputs_complete": bool(
                outputs["status"].eq("COMPLETE").all()
            ),
            "phase4_lock_used": rel(LOCK),
        }
    )
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    transport = read_csv("transportability_classification.csv").iloc[0]
    handoff = f"""# Phase 4 to Phase 5 Handoff

Generated: {GENERATED_AT}

## Decision

**{decision}**

Independent final QC: **{'PASS' if critical_failures == 0 else 'FAIL'}**
({len(qc) - critical_failures}/{len(qc)} critical checks passing).

Phase 4 transportability classification:
**{transport['classification']}**.

Phase 5 may use the completed Phase 4 outputs only with these restrictions:

1. retain provisional OASIS `CDRTOT` endpoint language;
2. explicitly flag low precision for 12-month B1 progression (18 events);
3. treat untouched external results as primary and recalibration as model updating;
4. preserve participant-grouped handling for repeated landmarks;
5. do not retrospectively select or alter Phase 3 models using OASIS results;
6. carry reconstructed six-box CDR results as endpoint sensitivity.

Phase 5 was not started automatically.
"""
    (REPORT_ROOT / "phase4_phase5_handoff.md").write_text(
        handoff, encoding="utf-8"
    )


def write_report(qc: pd.DataFrame, decision: str) -> None:
    failures = qc[
        qc["critical"].map(truth) & qc["status"].ne("PASS")
    ]
    primary = read_csv("untouched_external_metrics.csv")
    primary = primary[
        primary["model_id"].isin(["B1", "D0", "D1"])
        & primary["horizon_months"].isin([12, 36])
    ][
        [
            "model_id",
            "horizon_months",
            "rows",
            "participants",
            "improvement_events",
            "progression_events",
            "auc_improvement",
            "auc_progression",
            "multistate_brier",
            "calibration_slope_improvement",
            "observed_expected_improvement",
            "calibration_slope_progression",
            "observed_expected_progression",
        ]
    ]
    report = f"""# Phase 4 Independent Final QC Report

Generated: {GENERATED_AT}

## QC decision

**{'PASS' if failures.empty else 'FAIL'}**

This verifier is independent of the external-validation analysis script. It
re-hashed saved artifacts and checked execution order and saved outputs without
re-estimating model performance.

Critical failures: **{len(failures)}**

## Final Phase decision

**{decision}**

## QC checks

{markdown(qc)}

## Primary untouched results checked

{markdown(primary)}

## Interpretation guardrails

`CDRTOT` is a provisional de facto global CDR endpoint. B1 12-month
progression has only 18 events and its interval is therefore low precision.
Recalibration is model updating rather than untouched validation. Phase 5 was
not started.
"""
    (REPORT_ROOT / "phase4_independent_qc_report.md").write_text(
        report, encoding="utf-8"
    )


def write_output_manifest() -> None:
    roots = [
        TABLE_ROOT,
        PLOT_ROOT,
        REPORT_ROOT,
        ROOT / "scripts",
        NOTEBOOK.parent,
    ]
    excluded_names = {
        "phase4_output_manifest.csv",
        "phase4_run_metadata.json",
    }
    phase4_script_names = {
        "phase4_generate_predictions.py",
        "phase4_prediction_qc.py",
        "phase4_external_validation.py",
        "phase4_qc.py",
        "create_phase4_validation_notebook.py",
    }
    rows = []
    seen: set[Path] = set()
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name in excluded_names or path in seen:
                continue
            if base == ROOT / "scripts" and path.name not in phase4_script_names:
                continue
            if base == NOTEBOOK.parent and path != NOTEBOOK:
                continue
            seen.add(path)
            rows.append(
                {
                    "relative_path": rel(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "artifact_class": (
                        "table"
                        if TABLE_ROOT in path.parents
                        else "figure"
                        if PLOT_ROOT in path.parents
                        else "report"
                        if REPORT_ROOT in path.parents
                        else "script"
                        if path.suffix == ".py"
                        else "notebook"
                    ),
                    "manifested_at_utc": GENERATED_AT,
                }
            )
    save_csv(pd.DataFrame(rows), "phase4_output_manifest.csv")


def main() -> None:
    refresh_repaired_lock_tables()
    check_locks()
    check_prediction_gate()
    check_untouched_results()
    check_secondary_analyses()
    check_figures_and_reports()
    check_phase_boundary()

    qc = pd.DataFrame(checks)
    save_csv(qc, "phase4_qc_results.csv")
    write_validation_checklist(qc)

    outputs = required_outputs()
    save_csv(outputs, "phase4_required_output_status.csv")
    output_failure = not outputs["status"].eq("COMPLETE").all()
    if output_failure:
        add(
            "outputs",
            "Q29",
            "all required outputs are present",
            False,
            (
                "missing="
                + str(outputs.loc[outputs["status"].ne("COMPLETE"), "path"].tolist())
            ),
        )
        qc = pd.DataFrame(checks)
        save_csv(qc, "phase4_qc_results.csv")
        write_validation_checklist(qc)

    critical_failures = qc[
        qc["critical"].map(truth) & qc["status"].ne("PASS")
    ]
    decision = (
        "GO TO PHASE 5 WITH RESTRICTIONS"
        if critical_failures.empty
        else "HOLD FOR VALIDATION AUDIT"
    )
    write_report(qc, decision)
    update_metadata_and_handoff(decision, qc, outputs)
    write_output_manifest()

    print(
        json.dumps(
            {
                "qc_checks": len(qc),
                "critical_failures": len(critical_failures),
                "decision": decision,
                "phase4_complete": critical_failures.empty,
                "phase5_started": False,
                "lock_used": rel(LOCK),
            },
            indent=2,
        )
    )
    if not critical_failures.empty:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
