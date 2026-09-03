#!/usr/bin/env python3
"""Independent saved-artifact verification for Phase 3.

This script does not call the model-development runner. It verifies persisted
predictions, folds, tuning provenance, bootstrap units, model checksums, the
no-fit inference function, figure/report inventory, and the OASIS access lock.
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
from PIL import Image

from phase3_inference import predict_from_frozen


ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = ROOT / "results" / "csvs" / "phase3"
PREDICTION_ROOT = TABLE_ROOT / "predictions"
MODEL_ROOT = ROOT / "models" / "phase3" / "phase4_lock"
REPORT_ROOT = ROOT / "reports" / "phase3"
PLOT_ROOT = ROOT / "results" / "plots" / "phase3"
Image.MAX_IMAGE_PIXELS = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QC:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.access: list[dict[str, Any]] = []

    def read_csv(self, path: Path, purpose: str) -> pd.DataFrame:
        frame = pd.read_csv(path)
        self.access.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "path": str(path.relative_to(ROOT)),
                "purpose": purpose,
                "sha256": sha256(path),
                "rows": len(frame),
                "columns": len(frame.columns),
                "oasis_record_level_outcome_accessed": False,
            }
        )
        return frame

    def read_json(self, path: Path, purpose: str) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.access.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "path": str(path.relative_to(ROOT)),
                "purpose": purpose,
                "sha256": sha256(path),
                "rows": np.nan,
                "columns": np.nan,
                "oasis_record_level_outcome_accessed": False,
            }
        )
        return payload

    def check(
        self,
        scope: str,
        name: str,
        passed: bool,
        details: str,
        *,
        severity: str = "critical",
    ) -> None:
        self.rows.append(
            {
                "scope": scope,
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "severity": severity,
                "details": details,
            }
        )


def main() -> None:
    qc = QC()
    access = qc.read_csv(
        TABLE_ROOT / "phase3_file_access_audit.csv",
        "verify primary runner file-access lock",
    )
    lower_paths = access["path"].str.lower()
    forbidden = (
        lower_paths.str.contains(r"processeddata/oasis3|datasets/oasis", regex=True)
        | lower_paths.str.contains(r"results/csvs/phase[12]/oasis3/", regex=True)
    )
    qc.check(
        "external_lock",
        "no OASIS record-level file accessed",
        not forbidden.any()
        and not access["oasis_record_level_outcome_accessed"].astype(bool).any(),
        f"accessed_inputs={len(access)}; forbidden_paths={int(forbidden.sum())}",
    )

    integrity = qc.read_csv(
        TABLE_ROOT / "phase3_input_integrity_audit.csv",
        "verify independent input-integrity gate",
    )
    critical_integrity = integrity[integrity["severity"].eq("critical")]
    qc.check(
        "inputs",
        "all input-integrity critical checks pass",
        critical_integrity["status"].eq("PASS").all(),
        f"critical_checks={len(critical_integrity)}; failures={(critical_integrity['status'] != 'PASS').sum()}",
    )

    baseline = qc.read_csv(
        PREDICTION_ROOT / "out_of_fold_baseline_predictions.csv",
        "verify baseline OOF predictions",
    )
    dynamic = qc.read_csv(
        PREDICTION_ROOT / "out_of_fold_dynamic_predictions.csv",
        "verify dynamic OOF predictions",
    )
    benchmark = qc.read_csv(
        PREDICTION_ROOT / "benchmark_predictions.csv",
        "verify benchmark predictions",
    )
    predictions = pd.concat([baseline, dynamic], ignore_index=True)
    outer = qc.read_csv(
        ROOT / "results" / "csvs" / "phase2" / "adni_outer_fold_assignments.csv",
        "independently verify prediction folds against frozen assignments",
    )
    fold_map = outer.set_index("participant_id")["outer_fold"]
    mapped = predictions["participant_id"].map(fold_map)
    qc.check(
        "folds",
        "OOF prediction fold matches frozen participant fold",
        mapped.notna().all()
        and np.array_equal(
            mapped.to_numpy(dtype=int),
            predictions["outer_fold"].to_numpy(dtype=int),
        ),
        f"prediction_rows={len(predictions)}; mismatches={int((mapped.fillna(-1).astype(int) != predictions['outer_fold']).sum())}",
    )
    participant_fold_counts = predictions.groupby(
        ["model_id", "horizon_months", "participant_id"]
    )["outer_fold"].nunique()
    qc.check(
        "folds",
        "no participant leakage across outer folds",
        participant_fold_counts.max() == 1,
        f"maximum_folds_per_model_horizon_participant={participant_fold_counts.max()}",
    )
    duplicate = predictions.duplicated(
        ["model_id", "horizon_months", "row_key"], keep=False
    )
    qc.check(
        "coverage",
        "no duplicate OOF prediction",
        not duplicate.any(),
        f"duplicates={int(duplicate.sum())}",
    )
    expected_rows = {
        "B0": 2447,
        "B1": 2447,
        "B2": 2447,
        "B3": 2447,
        "B4": 1770,
        "D0": 4517,
        "D1": 4517,
        "D2": 4517,
        "D3": 3134,
        "D4": 3075,
        "D5": 874,
        "D1PN": 4517,
    }
    coverage_failures = []
    coverage = (
        predictions.groupby(["model_id", "horizon_months"])
        .size()
        .reset_index(name="rows")
    )
    for record in coverage.itertuples(index=False):
        if record.model_id in expected_rows and record.rows != expected_rows[record.model_id]:
            coverage_failures.append(
                f"{record.model_id}:{record.horizon_months}={record.rows}"
            )
    qc.check(
        "coverage",
        "OOF predictions cover each eligible frozen row exactly once",
        not coverage_failures,
        f"groups={len(coverage)}; failures={coverage_failures}",
    )
    benchmark_coverage = (
        benchmark.groupby(["model_id", "horizon_months"]).size().eq(2447).all()
    )
    qc.check(
        "coverage",
        "baseline benchmark coverage complete",
        bool(benchmark_coverage),
        benchmark.groupby(["model_id", "horizon_months"]).size().to_dict().__str__(),
    )

    probability_columns = [
        "probability_improvement",
        "probability_progression",
        "probability_remaining",
    ]
    probability_values = predictions[probability_columns]
    sums = probability_values.sum(axis=1)
    qc.check(
        "probabilities",
        "all OOF probabilities within [0,1]",
        (
            probability_values.ge(-1e-12)
            & probability_values.le(1 + 1e-12)
        ).all().all(),
        f"minimum={probability_values.min().min():.12g}; maximum={probability_values.max().max():.12g}",
    )
    qc.check(
        "probabilities",
        "OOF state probabilities sum to one",
        np.max(np.abs(sums - 1)) <= 1e-10,
        f"maximum_absolute_error={np.max(np.abs(sums - 1)):.3e}",
    )

    preprocessing = qc.read_csv(
        TABLE_ROOT / "fold_specific_preprocessing_parameters.csv",
        "verify training-fold preprocessing provenance",
    )
    qc.check(
        "preprocessing",
        "all preprocessors fit only on ADNI outer training partitions",
        preprocessing["fit_cohort"].str.lower().eq("adni").all()
        and preprocessing["partition"].eq("outer_training").all(),
        f"rows={len(preprocessing)}; fit_cohorts={sorted(preprocessing['fit_cohort'].unique())}",
    )
    tuning = qc.read_csv(
        TABLE_ROOT / "fold_specific_ridge_penalty_tuning.csv",
        "verify nested ridge tuning provenance",
    )
    inner_sets = tuning.groupby(
        ["model_id", "horizon_months", "outer_fold", "cause"]
    )["inner_fold"].nunique()
    qc.check(
        "tuning",
        "ridge penalties tuned in all five inner folds only",
        inner_sets.min() == 5 and inner_sets.max() == 5,
        f"groups={len(inner_sets)}; inner_fold_range={inner_sets.min()}-{inner_sets.max()}",
    )
    qc.check(
        "tuning",
        "ridge fits converged",
        tuning.loc[tuning["error"].fillna("").eq(""), "converged"].astype(bool).all(),
        f"failed_or_nonconverged={int((~tuning['converged'].astype(bool)).sum())}",
    )

    bootstrap = qc.read_csv(
        TABLE_ROOT / "bootstrap_metric_replicates.csv",
        "verify participant bootstrap",
    )
    bootstrap_counts = bootstrap.groupby(
        ["model_id", "horizon_months"]
    )["bootstrap_index"].nunique()
    qc.check(
        "uncertainty",
        "at least 1,000 participant bootstrap replicates per primary group",
        bootstrap["bootstrap_unit"].eq("participant").all()
        and bootstrap_counts.min() >= 1000,
        f"group_replicates={bootstrap_counts.to_dict()}",
    )

    artifacts = qc.read_csv(
        TABLE_ROOT / "frozen_model_artifacts.csv",
        "verify frozen model inventory",
    )
    checksum_failures = []
    for record in artifacts.itertuples(index=False):
        path = ROOT / record.artifact_path
        if not path.exists() or sha256(path) != record.sha256:
            checksum_failures.append(record.artifact_path)
    qc.check(
        "freeze",
        "all frozen model checksums match",
        not checksum_failures,
        f"artifacts={len(artifacts)}; failures={checksum_failures}",
    )
    index = qc.read_json(
        MODEL_ROOT / "frozen_model_index.json",
        "verify frozen model index and external lock",
    )
    qc.check(
        "freeze",
        "model index certifies ADNI-only development and untouched external outcomes",
        index.get("development_cohort") == "ADNI"
        and index.get("external_outcomes_accessed_in_phase3") is False,
        f"version={index.get('model_version_id')}; selected_dynamic={index.get('selected_dynamic_model')}",
    )

    fixture_input = qc.read_csv(
        MODEL_ROOT / "inference_verification_input.csv",
        "run frozen inference fixture",
    )
    fixture_expected = qc.read_csv(
        MODEL_ROOT / "inference_verification_expected.csv",
        "verify frozen inference fixture",
    )
    artifact_path = ROOT / artifacts.iloc[0]["artifact_path"]
    regenerated = predict_from_frozen(fixture_input, artifact_path)
    numeric_columns = [
        "improvement_probability",
        "progression_probability",
        "remaining_cdr_0_5_probability",
        "probability_sum",
    ]
    identical = np.allclose(
        regenerated[numeric_columns].to_numpy(),
        fixture_expected[numeric_columns].to_numpy(),
        rtol=0,
        atol=1e-12,
    ) and regenerated[
        ["model_id", "model_version_id", "quality_flags"]
    ].reset_index(drop=True).equals(
        fixture_expected[
            ["model_id", "model_version_id", "quality_flags"]
        ].reset_index(drop=True)
    )
    qc.check(
        "inference",
        "saved model regenerates identical fixture predictions",
        identical,
        f"rows={len(regenerated)}; maximum_numeric_difference={np.max(np.abs(regenerated[numeric_columns].to_numpy() - fixture_expected[numeric_columns].to_numpy())):.3e}",
    )
    inference_source = (ROOT / "scripts" / "phase3_inference.py").read_text(
        encoding="utf-8"
    )
    qc.check(
        "inference",
        "frozen inference implementation contains no fit call",
        ".fit(" not in inference_source,
        "static source verification; inference calls transform and paired CIF only",
    )

    schema = qc.read_csv(
        MODEL_ROOT / "frozen_feature_schema.csv",
        "verify frozen predictor schema",
    )
    feature_failures = []
    for record in artifacts.itertuples(index=False):
        bundle = joblib.load(ROOT / record.artifact_path)
        expected = set(bundle["continuous_features"]) | set(
            bundle["categorical_features"]
        )
        observed = set(
            schema[
                schema["model_id"].eq(record.model_id)
                & schema["horizon_months"].eq(record.horizon_months)
            ]["feature"]
        )
        if expected != observed:
            feature_failures.append(
                f"{record.model_id}-{record.horizon_months}m"
            )
    qc.check(
        "freeze",
        "frozen bundles use exactly the frozen feature schema",
        not feature_failures,
        f"failures={feature_failures}",
    )

    required_reports = [
        "phase3_adni_model_development_report.md",
        "phase3_internal_validation_report.md",
        "phase3_calibration_report.md",
        "phase3_model_comparison_and_selection.md",
        "phase3_model_diagnostics.md",
        "phase3_external_validation_lock_specification.md",
        "phase3_phase4_handoff.md",
        "phase3_model_card.md",
    ]
    missing_reports = [
        name for name in required_reports if not (REPORT_ROOT / name).exists()
    ]
    qc.check(
        "outputs",
        "all eight required reports exist",
        not missing_reports,
        f"missing={missing_reports}",
    )
    figure_manifest = qc.read_csv(
        TABLE_ROOT / "figure_manifest.csv",
        "verify figure inventory and resolution",
    )
    png_rows = figure_manifest[figure_manifest["format"].eq("png")]
    dpi_failures = []
    for record in png_rows.itertuples(index=False):
        path = ROOT / record.path
        if not path.exists():
            dpi_failures.append(f"missing:{record.path}")
            continue
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
        if min(dpi) < 599:
            dpi_failures.append(f"{record.path}:{dpi}")
    qc.check(
        "outputs",
        "at least 18 figures saved as >=600 DPI PNG and vector SVG",
        figure_manifest["figure"].nunique() >= 18
        and png_rows["figure"].nunique() >= 18
        and figure_manifest[figure_manifest["format"].eq("svg")][
            "figure"
        ].nunique()
        >= 18
        and not dpi_failures,
        f"figures={figure_manifest['figure'].nunique()}; dpi_failures={dpi_failures}",
    )

    metadata = qc.read_json(
        TABLE_ROOT / "phase3_run_metadata.json",
        "verify deterministic seeds and package versions",
    )
    qc.check(
        "reproducibility",
        "random seed, versions, horizons, and bootstrap unit recorded",
        metadata.get("random_seed") == 20260728
        and metadata.get("bootstrap_unit") == "participant"
        and bool(metadata.get("versions"))
        and metadata.get("phase4_started") is False,
        f"version={metadata.get('model_version_id')}; seed={metadata.get('random_seed')}",
    )

    # Source-level protections complement saved-output checks.
    core_source = (ROOT / "scripts" / "phase3_survival_core.py").read_text(
        encoding="utf-8"
    )
    development_source = (
        ROOT / "scripts" / "phase3_adni_model_development.py"
    ).read_text(encoding="utf-8")
    qc.check(
        "methods",
        "paired cause-specific CIF implementation and censoring code present",
        "paired_cumulative_incidence" in core_source
        and "_event_improvement" in development_source
        and "_event_progression" in development_source
        and "_censoring_event" in development_source,
        "static method audit plus probability/censoring saved-output verification",
    )
    qc.check(
        "leakage",
        "no post-landmark predictor or censor-as-persistence discrepancy",
        integrity[
            integrity["check"].str.contains(
                "predictors not post-landmark|early censoring is not persistence|slope uses no future",
                regex=True,
            )
        ]["status"].eq("PASS").all(),
        "verified from Phase 3 input-integrity table",
    )

    result = pd.DataFrame(qc.rows)
    result.to_csv(TABLE_ROOT / "phase3_qc_results.csv", index=False)
    pd.DataFrame(qc.access).to_csv(
        TABLE_ROOT / "phase3_qc_file_access_log.csv", index=False
    )
    critical = result[result["severity"].eq("critical")]
    passed = critical["status"].eq("PASS").all()
    report = f"""# Phase 3 Independent QC Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Result

# {"PASS" if passed else "FAIL"}

This verifier ran separately from the primary model-development implementation. It checked the OASIS access lock, frozen input gate, participant folds, OOF coverage, probabilities, preprocessing/tuning provenance, participant bootstrap, checksums, deterministic no-fit inference, exact feature schema, reports, figures, and reproducibility metadata.

{result.to_markdown(index=False)}
"""
    (REPORT_ROOT / "phase3_independent_qc_report.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS" if passed else "FAIL",
                "critical_checks": len(critical),
                "critical_failures": int((critical["status"] != "PASS").sum()),
                "results": str(
                    (TABLE_ROOT / "phase3_qc_results.csv").relative_to(ROOT)
                ),
            },
            indent=2,
        )
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
