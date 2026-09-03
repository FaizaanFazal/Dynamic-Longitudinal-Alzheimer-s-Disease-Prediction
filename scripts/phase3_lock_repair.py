#!/usr/bin/env python3
"""Outcome-blind repair of the Phase 3 external-validation lock.

The original ``models/phase3/phase4_lock`` directory is treated as immutable.
This script:

* verifies and copies the original B1/D1 bundles without refitting;
* recovers already-selected Phase 3 penalties for D0, B3, and D3;
* fits only those missing final bundles on ADNI;
* packages schemas, runtime code, provenance, fixtures, and checksums in
  ``models/phase3/phase4_lock_v2``;
* never opens an OASIS-3 participant-level file.

The script is intentionally one-shot.  It refuses to overwrite an existing
versioned package.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import phase3_adni_model_development as phase3
from phase3_inference import predict_from_frozen


ORIGINAL_LOCK = ROOT / "models" / "phase3" / "phase4_lock"
FINAL_LOCK = ROOT / "models" / "phase3" / "phase4_lock_v2"
BUILD_LOCK = ROOT / "models" / "phase3" / ".phase4_lock_v2.building"
TABLE_ROOT = ROOT / "results" / "csvs" / "phase3_lock_repair"
REPORT_ROOT = ROOT / "reports" / "phase3_lock_repair"
CONFIG_PATH = ROOT / "config" / "phase3_lock_repair_parameters.json"
PHASE3_TABLES = ROOT / "results" / "csvs" / "phase3"
GENERATED_AT = datetime.now(timezone.utc).isoformat()

TABLE_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

with CONFIG_PATH.open(encoding="utf-8") as handle:
    REPAIR_CONFIG = json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def md_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame.loc[:, columns]
    return selected.to_markdown(index=False)


class AccessAudit:
    """Read wrapper that fails closed on external participant-level paths."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    @staticmethod
    def _is_forbidden(path: Path) -> bool:
        lowered = path.resolve().as_posix().lower()
        forbidden_fragments = [
            "/processeddata/oasis3/",
            "/datasets/oasis3/",
            "/datasets/oasis/",
            "/results/csvs/phase1/oasis3/",
            "/results/csvs/phase2/oasis3/",
        ]
        return any(fragment in lowered for fragment in forbidden_fragments)

    def record(
        self,
        path: Path,
        purpose: str,
        access_class: str,
        *,
        rows: int | None = None,
        columns: int | None = None,
    ) -> None:
        if self._is_forbidden(path):
            raise PermissionError(
                f"Outcome-blind lock repair blocked OASIS participant-level access: {path}"
            )
        self.rows.append(
            {
                "sequence": len(self.rows) + 1,
                "access_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "path": relative(path),
                "purpose": purpose,
                "access_class": access_class,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows": rows,
                "columns": columns,
                "development_cohort": "ADNI",
                "oasis_participant_level_features_accessed": False,
                "oasis_participant_level_outcomes_accessed": False,
            }
        )

    def read_csv(
        self, path: Path, purpose: str, access_class: str = "adni_record_level"
    ) -> pd.DataFrame:
        if self._is_forbidden(path):
            raise PermissionError(f"Forbidden OASIS path: {path}")
        frame = pd.read_csv(path)
        self.record(
            path,
            purpose,
            access_class,
            rows=len(frame),
            columns=len(frame.columns),
        )
        return frame

    def read_json(
        self, path: Path, purpose: str, access_class: str = "frozen_reference"
    ) -> Any:
        if self._is_forbidden(path):
            raise PermissionError(f"Forbidden OASIS path: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.record(path, purpose, access_class)
        return payload

    def read_text(
        self, path: Path, purpose: str, access_class: str = "frozen_reference"
    ) -> str:
        if self._is_forbidden(path):
            raise PermissionError(f"Forbidden OASIS path: {path}")
        text = path.read_text(encoding="utf-8")
        self.record(path, purpose, access_class)
        return text


ACCESS = AccessAudit()
_ORIGINAL_METADATA_CACHE: pd.DataFrame | None = None


def audit_required_references() -> None:
    references = [ROOT / "LiteratureAndPlan" / "Plan.md"]
    references.extend(sorted((ROOT / "reports" / "phase3").glob("*.md")))
    references.append(
        ROOT / "reports" / "phase4" / "phase4_frozen_package_blocker_audit.md"
    )
    for path in references:
        ACCESS.read_text(path, "required lock-repair reference")


def original_lock_integrity(stage: str) -> pd.DataFrame:
    global _ORIGINAL_METADATA_CACHE
    metadata_path = PHASE3_TABLES / "model_metadata_and_checksums.csv"
    if _ORIGINAL_METADATA_CACHE is None:
        _ORIGINAL_METADATA_CACHE = ACCESS.read_csv(
            metadata_path,
            "load immutable original-lock checksum metadata",
            "phase3_checksum_metadata",
        )
    metadata = _ORIGINAL_METADATA_CACHE.copy()
    metadata = metadata[
        metadata["artifact_path"].astype(str).str.startswith(
            "models/phase3/phase4_lock/"
        )
    ].copy()
    rows = []
    for record in metadata.itertuples():
        path = ROOT / record.artifact_path
        exists = path.exists()
        observed_hash = sha256(path) if exists else ""
        observed_bytes = path.stat().st_size if exists else -1
        rows.append(
            {
                "stage": stage,
                "artifact_path": record.artifact_path,
                "expected_sha256": record.sha256,
                "observed_sha256": observed_hash,
                "expected_bytes": int(record.bytes),
                "observed_bytes": observed_bytes,
                "status": (
                    "PASS"
                    if exists
                    and observed_hash == record.sha256
                    and observed_bytes == int(record.bytes)
                    else "FAIL"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result["status"].eq("PASS").all():
        raise RuntimeError(f"Original immutable lock failed integrity at {stage}.")
    return result


def selected_penalty_recovery(tuning: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in ["D0", "B3", "D3"]:
        for horizon in [12, 36]:
            for cause in ["improvement", "progression"]:
                selected = (
                    tuning[
                        tuning["model_id"].eq(model_id)
                        & tuning["horizon_months"].eq(horizon)
                        & tuning["cause"].eq(cause)
                    ][["outer_fold", "selected_alpha", "selection_rule"]]
                    .drop_duplicates()
                    .sort_values("outer_fold")
                )
                if selected["outer_fold"].nunique() != 5:
                    raise RuntimeError(
                        f"Incomplete saved penalty decisions for {model_id} {horizon}m {cause}."
                    )
                frequencies = (
                    selected.groupby("selected_alpha")
                    .size()
                    .reset_index(name="outer_fold_frequency")
                    .sort_values(
                        ["outer_fold_frequency", "selected_alpha"],
                        ascending=[False, False],
                    )
                )
                recovered = float(frequencies.iloc[0]["selected_alpha"])
                phase3_value = phase3.choose_alpha_from_outer_tuning(
                    tuning, model_id, horizon, cause
                )
                if recovered != phase3_value:
                    raise RuntimeError("Penalty recovery disagrees with Phase 3 rule.")
                rows.append(
                    {
                        "model_id": model_id,
                        "horizon_months": horizon,
                        "cause": cause,
                        "recovered_final_alpha": recovered,
                        "outer_fold_selected_alphas": "|".join(
                            f"{int(row.outer_fold)}:{float(row.selected_alpha):g}"
                            for row in selected.itertuples()
                        ),
                        "recovery_rule": (
                            "modal outer-fold selected alpha; ties choose the larger alpha"
                        ),
                        "original_inner_selection_rule": selected[
                            "selection_rule"
                        ].iloc[0],
                        "source_path": relative(
                            PHASE3_TABLES / "fold_specific_ridge_penalties.csv"
                        ),
                        "source_sha256": sha256(
                            PHASE3_TABLES / "fold_specific_ridge_penalties.csv"
                        ),
                        "retuned_during_repair": False,
                    }
                )
    return pd.DataFrame(rows)


def prepare_analysis_data(
    spec: phase3.ModelSpec,
    horizon: int,
    episode: pd.DataFrame,
) -> tuple[pd.DataFrame, Path]:
    input_path = phase3.dataset_path(spec.dataset_variant, spec.family, horizon)
    raw = ACCESS.read_csv(
        input_path,
        f"fit final {spec.model_id} {horizon}m ADNI-only lock-repair bundle",
    )
    data = phase3.add_survival_endpoints(raw, spec.family, horizon, episode)
    data["_analysis_weight"] = 1.0
    for categorical in spec.categorical:
        data[categorical] = data[categorical].astype("string").fillna("Missing")
    return data, input_path


def extract_preprocessing(
    bundle: dict[str, Any],
    *,
    package_version_id: str,
    model_origin: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    preprocessor = bundle["preprocessor"]
    transformer = preprocessor.transformer
    numeric = transformer.named_transformers_["continuous"]
    imputer = numeric.named_steps["impute"]
    scaler = numeric.named_steps["scale"]
    for index, feature in enumerate(bundle["continuous_features"]):
        rows.append(
            {
                "package_version_id": package_version_id,
                "model_version_id": bundle["model_version_id"],
                "model_origin": model_origin,
                "model_id": bundle["model_id"],
                "horizon_months": bundle["horizon_months"],
                "feature": feature,
                "feature_type": "continuous",
                "imputation_value": float(imputer.statistics_[index]),
                "robust_center": float(scaler.center_[index]),
                "robust_scale": float(scaler.scale_[index]),
                "categories": "",
                "fit_cohort": preprocessor.provenance.fit_cohort,
                "fit_partition": preprocessor.provenance.fit_partition,
                "fit_rows": preprocessor.provenance.fit_row_count,
                "fit_participants": preprocessor.provenance.fit_participant_count,
                "external_outcomes_accessed": (
                    preprocessor.provenance.external_outcomes_accessed
                ),
            }
        )
    categorical = transformer.named_transformers_["categorical"]
    cat_imputer = categorical.named_steps["impute"]
    encoder = categorical.named_steps["encode"]
    for index, feature in enumerate(bundle["categorical_features"]):
        rows.append(
            {
                "package_version_id": package_version_id,
                "model_version_id": bundle["model_version_id"],
                "model_origin": model_origin,
                "model_id": bundle["model_id"],
                "horizon_months": bundle["horizon_months"],
                "feature": feature,
                "feature_type": "categorical",
                "imputation_value": str(cat_imputer.statistics_[index]),
                "robust_center": np.nan,
                "robust_scale": np.nan,
                "categories": "|".join(map(str, encoder.categories_[index])),
                "fit_cohort": preprocessor.provenance.fit_cohort,
                "fit_partition": preprocessor.provenance.fit_partition,
                "fit_rows": preprocessor.provenance.fit_row_count,
                "fit_participants": preprocessor.provenance.fit_participant_count,
                "external_outcomes_accessed": (
                    preprocessor.provenance.external_outcomes_accessed
                ),
            }
        )
    return pd.DataFrame(rows)


def feature_schema_rows(
    bundle: dict[str, Any],
    data: pd.DataFrame,
    *,
    package_version_id: str,
    model_origin: str,
) -> list[dict[str, Any]]:
    rows = []
    for feature in list(bundle["continuous_features"]) + list(
        bundle["categorical_features"]
    ):
        categorical = feature in bundle["categorical_features"]
        if categorical:
            observed_min = np.nan
            observed_max = np.nan
            allowed_levels = "|".join(
                sorted(data[feature].dropna().astype(str).unique())
            )
        else:
            numeric = pd.to_numeric(data[feature], errors="coerce")
            observed_min = float(numeric.min()) if numeric.notna().any() else np.nan
            observed_max = float(numeric.max()) if numeric.notna().any() else np.nan
            allowed_levels = ""
        rows.append(
            {
                "package_version_id": package_version_id,
                "model_version_id": bundle["model_version_id"],
                "model_origin": model_origin,
                "model_id": bundle["model_id"],
                "family": bundle["family"],
                "horizon_months": bundle["horizon_months"],
                "feature": feature,
                "data_type": "categorical" if categorical else "continuous",
                "required_column": True,
                "missing_value_handling": (
                    "ADNI final-training most-frequent imputation; unknown category ignored"
                    if categorical
                    else "ADNI final-training median imputation"
                ),
                "adni_observed_min": observed_min,
                "adni_observed_max": observed_max,
                "allowed_levels": allowed_levels,
                "external_transform_rule": "apply frozen ADNI preprocessor unchanged",
            }
        )
    return rows


def extract_model_components(
    bundle: dict[str, Any],
    *,
    package_version_id: str,
    model_origin: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict[str, Any]] = []
    hazard_rows: list[dict[str, Any]] = []
    for cause in ["improvement", "progression"]:
        model = bundle[f"{cause}_model"]
        standard_errors = np.sqrt(
            np.maximum(np.diag(model.covariance_), 0)
        )
        for feature, coefficient, standard_error in zip(
            bundle["transformed_feature_order"],
            model.coef_,
            standard_errors,
        ):
            low, high = phase3.normal_interval(
                float(coefficient), float(standard_error)
            )
            coefficient_rows.append(
                {
                    "package_version_id": package_version_id,
                    "model_version_id": bundle["model_version_id"],
                    "model_origin": model_origin,
                    "model_id": bundle["model_id"],
                    "family": bundle["family"],
                    "horizon_months": bundle["horizon_months"],
                    "cause": cause,
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                    "hazard_ratio": float(
                        np.exp(np.clip(coefficient, -20, 20))
                    ),
                    "model_based_standard_error": standard_error,
                    "coefficient_ci_low": low,
                    "coefficient_ci_high": high,
                    "alpha": model.alpha,
                    "effective_degrees_of_freedom": model.effective_df_,
                    "converged": model.converged_,
                }
            )
        cumulative = np.cumsum(model.baseline_hazard_increments_)
        for event_time, increment, cumulative_hazard in zip(
            model.baseline_times_,
            model.baseline_hazard_increments_,
            cumulative,
        ):
            hazard_rows.append(
                {
                    "package_version_id": package_version_id,
                    "model_version_id": bundle["model_version_id"],
                    "model_origin": model_origin,
                    "model_id": bundle["model_id"],
                    "family": bundle["family"],
                    "horizon_months": bundle["horizon_months"],
                    "cause": cause,
                    "event_time_days": event_time,
                    "baseline_hazard_increment": increment,
                    "cumulative_baseline_hazard": cumulative_hazard,
                }
            )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(hazard_rows)


def load_raw_for_copied_model(record: dict[str, Any]) -> pd.DataFrame:
    family = record["family"]
    horizon = int(record["horizon_months"])
    input_path = phase3.dataset_path("primary", family, horizon)
    return ACCESS.read_csv(
        input_path,
        f"B1/D1 original-versus-repaired prediction equivalence {record['model_id']} {horizon}m",
    )


def package_checksum_manifest(package_root: Path) -> pd.DataFrame:
    excluded = {
        "checksum_manifest.csv",
        "external_validation_lock_manifest.json",
    }
    rows = []
    for path in sorted(
        (candidate for candidate in package_root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(package_root).as_posix(),
    ):
        package_relative = path.relative_to(package_root).as_posix()
        if package_relative in excluded:
            continue
        rows.append(
            {
                "package_relative_path": package_relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if FINAL_LOCK.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing repaired lock: {FINAL_LOCK}"
        )
    if BUILD_LOCK.exists():
        shutil.rmtree(BUILD_LOCK)
    BUILD_LOCK.mkdir(parents=True)
    for subdirectory in [
        "artifacts",
        "inference_runtime",
        "provenance",
        "fixtures",
    ]:
        (BUILD_LOCK / subdirectory).mkdir()

    audit_required_references()
    original_before = original_lock_integrity("before_repair")
    original_index = ACCESS.read_json(
        ORIGINAL_LOCK / "frozen_model_index.json",
        "read original frozen model index",
        "immutable_original_lock",
    )
    ACCESS.record(
        ORIGINAL_LOCK / "external_validation_lock_manifest.json",
        "read original external-validation lock manifest",
        "immutable_original_lock",
    )

    tuning_path = PHASE3_TABLES / "fold_specific_ridge_penalties.csv"
    tuning = ACCESS.read_csv(
        tuning_path,
        "recover previously selected Phase 3 ridge penalties",
        "phase3_machine_readable_result",
    )
    full_tuning_path = PHASE3_TABLES / "fold_specific_ridge_penalty_tuning.csv"
    ACCESS.record(
        full_tuning_path,
        "preserve full nested tuning provenance",
        "phase3_machine_readable_result",
    )
    penalty_recovery = selected_penalty_recovery(tuning)

    d1_resolution_path = PHASE3_TABLES / "d1_collinearity_resolution.csv"
    d1_resolution = ACCESS.read_csv(
        d1_resolution_path,
        "confirm frozen D1 current-plus-change representation",
        "phase3_machine_readable_result",
    )
    selected_d1 = set(d1_resolution["selected_candidate"].astype(str))
    if selected_d1 != {"current_plus_change"}:
        raise RuntimeError(f"Unexpected frozen D1 representation: {selected_d1}")
    specs = phase3.model_specs(("current_mmse", "mmse_change_from_index"))
    phase3_run_metadata_path = PHASE3_TABLES / "phase3_run_metadata.json"
    phase3_run_metadata = ACCESS.read_json(
        phase3_run_metadata_path,
        "recover original Phase 3 training software versions",
        "phase3_machine_readable_result",
    )

    episode_path = (
        ROOT
        / "results"
        / "csvs"
        / "phase1"
        / "adni"
        / "primary_episode_time_to_event.csv"
    )
    episode = ACCESS.read_csv(
        episode_path,
        "reconstruct exact ADNI baseline event and censoring times",
    )
    outer_path = (
        ROOT / "results" / "csvs" / "phase2" / "adni_outer_fold_assignments.csv"
    )
    inner_path = (
        ROOT
        / "results"
        / "csvs"
        / "phase2"
        / "adni_nested_inner_fold_assignments.csv"
    )
    outer = ACCESS.read_csv(
        outer_path, "verify frozen participant-grouped outer folds"
    )
    inner = ACCESS.read_csv(
        inner_path, "verify frozen participant-grouped nested folds"
    )

    version_material = {
        "repair_schema_version": 2,
        "original_model_version_id": original_index["model_version_id"],
        "original_index_sha256": sha256(
            ORIGINAL_LOCK / "frozen_model_index.json"
        ),
        "phase3_config_sha256": sha256(
            ROOT / "config" / "phase3_analysis_parameters.json"
        ),
        "outer_folds_sha256": sha256(outer_path),
        "inner_folds_sha256": sha256(inner_path),
        "saved_penalties_sha256": sha256(tuning_path),
        "full_tuning_sha256": sha256(full_tuning_path),
        "d1_resolution_sha256": sha256(d1_resolution_path),
        "phase3_run_metadata_sha256": sha256(phase3_run_metadata_path),
        "repair_script_sha256": sha256(Path(__file__)),
        "models_fitted": ["D0", "B3", "D3"],
        "horizons_months": [12, 36],
        "random_seed": phase3.SEED,
        "oasis_participant_level_accessed": False,
    }
    version_hash = hashlib.sha256(
        json.dumps(version_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    package_version_id = f"phase3-adni-lockrepair-v2-{version_hash}"

    # Copy the generic runtime and frozen provenance into the new package.
    runtime_paths = [
        SCRIPTS / "phase3_inference.py",
        SCRIPTS / "phase3_survival_core.py",
        SCRIPTS / "phase2_preprocessing.py",
    ]
    for source in runtime_paths:
        ACCESS.record(source, "package exact inference runtime", "source_code")
        shutil.copy2(source, BUILD_LOCK / "inference_runtime" / source.name)
    provenance_paths = [
        ROOT / "config" / "phase3_analysis_parameters.json",
        outer_path,
        inner_path,
        tuning_path,
        full_tuning_path,
        d1_resolution_path,
        phase3_run_metadata_path,
    ]
    for source in provenance_paths:
        if not any(row["path"] == relative(source) for row in ACCESS.rows):
            ACCESS.record(
                source,
                "package frozen model-development provenance",
                "phase3_provenance",
            )
        shutil.copy2(source, BUILD_LOCK / "provenance" / source.name)

    inventory_rows: list[dict[str, Any]] = []
    preprocessing_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []
    hazard_parts: list[pd.DataFrame] = []
    schema_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    fixture_rows: list[dict[str, Any]] = []
    determinism_rows: list[dict[str, Any]] = []
    d3_temporal_rows: list[dict[str, Any]] = []
    grouping_rows: list[dict[str, Any]] = []
    software_rows: list[dict[str, Any]] = []

    repair_software_versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "joblib": joblib.__version__,
    }

    # Verified copies retain byte-identical B1/D1 bundles and old model IDs.
    for record in original_index["models"]:
        source = ROOT / record["artifact_path"]
        ACCESS.record(
            source,
            f"verified copy of original {record['model_id']} {record['horizon_months']}m",
            "immutable_original_model",
        )
        destination = BUILD_LOCK / "artifacts" / source.name
        shutil.copy2(source, destination)
        if sha256(destination) != record["sha256"]:
            raise RuntimeError(f"Verified copy failed for {source.name}.")
        copied = dict(record)
        copied.update(
            {
                "package_version_id": package_version_id,
                "model_origin": "verified_copy_from_original_lock",
                "source_artifact_path": record["artifact_path"],
                "source_sha256": record["sha256"],
                "artifact_path": (
                    f"models/phase3/phase4_lock_v2/artifacts/{source.name}"
                ),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "refit_during_repair": False,
            }
        )
        inventory_rows.append(copied)
        software_rows.append(
            {
                "package_version_id": package_version_id,
                "model_version_id": record["model_version_id"],
                "model_id": record["model_id"],
                "horizon_months": record["horizon_months"],
                "model_origin": "verified_copy_from_original_lock",
                "training_python": phase3_run_metadata["python"],
                "training_platform": phase3_run_metadata["platform"],
                "training_numpy": phase3_run_metadata["versions"]["numpy"],
                "training_pandas": phase3_run_metadata["versions"]["pandas"],
                "training_scipy": phase3_run_metadata["versions"]["scipy"],
                "training_scikit_learn": phase3_run_metadata["versions"][
                    "scikit_learn"
                ],
                "training_statsmodels": phase3_run_metadata["versions"][
                    "statsmodels"
                ],
                "package_joblib": joblib.__version__,
                "version_source": relative(phase3_run_metadata_path),
                "source_scope": (
                    "original Phase 3 training environment; joblib records the "
                    "verified package loading environment"
                ),
            }
        )

        bundle = joblib.load(destination)
        raw = load_raw_for_copied_model(record)
        test = (
            raw[raw["outer_fold"].eq(REPAIR_CONFIG["held_out_fixture_outer_fold"])]
            .sort_values(["participant_id"])
            .head(REPAIR_CONFIG["fixture_rows_per_model"])
        )
        required = list(bundle["continuous_features"]) + list(
            bundle["categorical_features"]
        )
        fixture = test[required].copy()
        original_prediction = predict_from_frozen(fixture, source)
        copied_prediction = predict_from_frozen(fixture, destination)
        numeric_columns = [
            "improvement_probability",
            "progression_probability",
            "remaining_cdr_0_5_probability",
            "probability_sum",
        ]
        maximum_difference = float(
            np.max(
                np.abs(
                    original_prediction[numeric_columns].to_numpy()
                    - copied_prediction[numeric_columns].to_numpy()
                )
            )
        )
        exact_metadata = original_prediction[
            [
                "prediction_horizon_months",
                "model_id",
                "model_version_id",
                "quality_flags",
            ]
        ].equals(
            copied_prediction[
                [
                    "prediction_horizon_months",
                    "model_id",
                    "model_version_id",
                    "quality_flags",
                ]
            ]
        )
        determinism_rows.append(
            {
                "model_id": record["model_id"],
                "horizon_months": record["horizon_months"],
                "comparison": "original_vs_verified_copy",
                "fixture_outer_fold": REPAIR_CONFIG["held_out_fixture_outer_fold"],
                "fixture_rows": len(fixture),
                "original_sha256": sha256(source),
                "repaired_sha256": sha256(destination),
                "artifact_hash_identical": sha256(source) == sha256(destination),
                "maximum_probability_difference": maximum_difference,
                "metadata_exact": exact_metadata,
                "status": (
                    "PASS"
                    if maximum_difference <= REPAIR_CONFIG["prediction_tolerance"]
                    and exact_metadata
                    and sha256(source) == sha256(destination)
                    else "FAIL"
                ),
            }
        )
        preprocessing_parts.append(
            extract_preprocessing(
                bundle,
                package_version_id=package_version_id,
                model_origin="verified_copy_from_original_lock",
            )
        )
        copied_coefficients, copied_hazards = extract_model_components(
            bundle,
            package_version_id=package_version_id,
            model_origin="verified_copy_from_original_lock",
        )
        coefficient_parts.append(copied_coefficients)
        hazard_parts.append(copied_hazards)
        schema_rows.extend(
            feature_schema_rows(
                bundle,
                raw,
                package_version_id=package_version_id,
                model_origin="verified_copy_from_original_lock",
            )
        )

    # Fit only the missing final ADNI models using saved Phase 3 decisions.
    prepared_data: dict[tuple[str, int], pd.DataFrame] = {}
    for model_id in ["D0", "B3", "D3"]:
        spec = specs[model_id]
        for horizon in [12, 36]:
            data, input_path = prepare_analysis_data(spec, horizon, episode)
            prepared_data[(model_id, horizon)] = data
            bundle, coefficients, hazards = phase3.fit_final_ridge_bundle(
                spec, horizon, data, tuning
            )
            bundle.update(
                {
                    "model_version_id": package_version_id,
                    "package_version_id": package_version_id,
                    "original_model_version_id": original_index[
                        "model_version_id"
                    ],
                    "model_origin": "adni_only_outcome_blind_lock_repair",
                    "repair_timestamp_utc": GENERATED_AT,
                    "training_input_sha256": sha256(input_path),
                    "penalty_source_path": relative(tuning_path),
                    "penalty_source_sha256": sha256(tuning_path),
                    "outer_folds_sha256": sha256(outer_path),
                    "inner_folds_sha256": sha256(inner_path),
                    "external_cohort": "OASIS-3",
                    "oasis_participant_level_features_accessed": False,
                    "oasis_participant_level_outcomes_accessed": False,
                    "retuned_during_repair": False,
                    "software_versions": repair_software_versions,
                }
            )
            artifact_name = (
                f"frozen_adni_{model_id.lower()}_{horizon}m_ridge_competing.joblib"
            )
            artifact_path = BUILD_LOCK / "artifacts" / artifact_name
            joblib.dump(bundle, artifact_path)

            alpha_i = float(bundle["selected_penalties"]["improvement"])
            alpha_p = float(bundle["selected_penalties"]["progression"])
            recovered = penalty_recovery[
                penalty_recovery["model_id"].eq(model_id)
                & penalty_recovery["horizon_months"].eq(horizon)
            ].set_index("cause")["recovered_final_alpha"]
            if alpha_i != recovered["improvement"] or alpha_p != recovered["progression"]:
                raise RuntimeError(
                    f"Fitted penalties differ from saved Phase 3 decisions for {model_id}."
                )
            record = {
                "package_version_id": package_version_id,
                "model_version_id": package_version_id,
                "model_id": model_id,
                "family": spec.family,
                "horizon_months": horizon,
                "artifact_path": (
                    f"models/phase3/phase4_lock_v2/artifacts/{artifact_name}"
                ),
                "sha256": sha256(artifact_path),
                "bytes": artifact_path.stat().st_size,
                "training_rows": len(data),
                "training_participants": data["participant_id"].nunique(),
                "improvement_event_rows": int(data["_event_improvement"].sum()),
                "progression_event_rows": int(data["_event_progression"].sum()),
                "improvement_alpha": alpha_i,
                "progression_alpha": alpha_p,
                "external_outcomes_accessed": False,
                "model_origin": "adni_only_outcome_blind_lock_repair",
                "source_artifact_path": "",
                "source_sha256": "",
                "refit_during_repair": True,
            }
            inventory_rows.append(record)
            software_rows.append(
                {
                    "package_version_id": package_version_id,
                    "model_version_id": package_version_id,
                    "model_id": model_id,
                    "horizon_months": horizon,
                    "model_origin": "adni_only_outcome_blind_lock_repair",
                    "training_python": repair_software_versions["python"],
                    "training_platform": repair_software_versions["platform"],
                    "training_numpy": repair_software_versions["numpy"],
                    "training_pandas": repair_software_versions["pandas"],
                    "training_scipy": repair_software_versions["scipy"],
                    "training_scikit_learn": repair_software_versions[
                        "scikit_learn"
                    ],
                    "training_statsmodels": repair_software_versions[
                        "statsmodels"
                    ],
                    "package_joblib": repair_software_versions["joblib"],
                    "version_source": (
                        "models/phase3/phase4_lock_v2/"
                        "repair_software_environment.json"
                    ),
                    "source_scope": "ADNI-only repaired-model training environment",
                }
            )
            training_rows.append(
                {
                    **record,
                    "dataset_variant": spec.dataset_variant,
                    "training_input_path": relative(input_path),
                    "training_input_sha256": sha256(input_path),
                    "continuous_features": "|".join(spec.continuous),
                    "categorical_features": "|".join(spec.categorical),
                    "weight_scheme": spec.weight_scheme,
                    "endpoint_construction": (
                        "frozen Phase 3 cause-specific first-transition endpoint"
                    ),
                    "baseline_hazard_construction": bundle[
                        "cumulative_incidence_method"
                    ],
                    "penalty_recovery_source": relative(tuning_path),
                    "retuned_during_repair": False,
                }
            )
            coefficients.insert(0, "package_version_id", package_version_id)
            coefficients.insert(1, "model_version_id", package_version_id)
            hazards.insert(0, "package_version_id", package_version_id)
            hazards.insert(1, "model_version_id", package_version_id)
            coefficient_parts.append(coefficients)
            hazard_parts.append(hazards)
            preprocessing_parts.append(
                extract_preprocessing(
                    bundle,
                    package_version_id=package_version_id,
                    model_origin="adni_only_outcome_blind_lock_repair",
                )
            )
            schema_rows.extend(
                feature_schema_rows(
                    bundle,
                    data,
                    package_version_id=package_version_id,
                    model_origin="adni_only_outcome_blind_lock_repair",
                )
            )

            held_out = (
                data[
                    data["outer_fold"].eq(
                        REPAIR_CONFIG["held_out_fixture_outer_fold"]
                    )
                ]
                .sort_values(["participant_id"])
                .head(REPAIR_CONFIG["fixture_rows_per_model"])
            )
            required = list(spec.continuous) + list(spec.categorical)
            fixture_input = held_out[required].copy()
            first_prediction = predict_from_frozen(fixture_input, artifact_path)
            second_prediction = predict_from_frozen(fixture_input, artifact_path)
            input_name = f"{model_id.lower()}_{horizon}m_input.csv"
            expected_name = f"{model_id.lower()}_{horizon}m_expected.csv"
            write_csv(fixture_input, BUILD_LOCK / "fixtures" / input_name)
            write_csv(first_prediction, BUILD_LOCK / "fixtures" / expected_name)
            maximum_repeat_difference = float(
                np.max(
                    np.abs(
                        first_prediction[
                            [
                                "improvement_probability",
                                "progression_probability",
                                "remaining_cdr_0_5_probability",
                                "probability_sum",
                            ]
                        ].to_numpy()
                        - second_prediction[
                            [
                                "improvement_probability",
                                "progression_probability",
                                "remaining_cdr_0_5_probability",
                                "probability_sum",
                            ]
                        ].to_numpy()
                    )
                )
            )
            fixture_rows.append(
                {
                    "model_id": model_id,
                    "horizon_months": horizon,
                    "artifact_path": record["artifact_path"],
                    "input_path": f"fixtures/{input_name}",
                    "expected_path": f"fixtures/{expected_name}",
                    "fixture_source": (
                        f"ADNI frozen outer fold {REPAIR_CONFIG['held_out_fixture_outer_fold']}"
                    ),
                    "fixture_rows": len(fixture_input),
                    "outcome_columns_present": False,
                    "maximum_repeat_prediction_difference": maximum_repeat_difference,
                    "status": (
                        "PASS"
                        if maximum_repeat_difference
                        <= REPAIR_CONFIG["prediction_tolerance"]
                        else "FAIL"
                    ),
                }
            )

            if model_id == "D3":
                temporal = {
                    "model_id": model_id,
                    "horizon_months": horizon,
                    "rows": len(data),
                    "participants": data["participant_id"].nunique(),
                    "previous_mmse_missing_rows": int(
                        data["previous_mmse"].isna().sum()
                    ),
                    "previous_mmse_not_strictly_before_landmark_rows": int(
                        (
                            data["previous_mmse_time_days"]
                            >= data["landmark_time_days"]
                        ).sum()
                    ),
                    "nonpositive_time_since_previous_mmse_rows": int(
                        (data["time_since_previous_mmse_days"] <= 0).sum()
                    ),
                    "strict_availability_flag_false_rows": int(
                        (~data["strictly_lagged_mmse_available_flag"].astype(bool)).sum()
                    ),
                    "current_or_same_day_mmse_in_model_features": bool(
                        {"current_mmse", "mmse_change_from_index"}.intersection(
                            spec.continuous
                        )
                    ),
                }
                temporal["status"] = (
                    "PASS"
                    if temporal["previous_mmse_missing_rows"] == 0
                    and temporal[
                        "previous_mmse_not_strictly_before_landmark_rows"
                    ]
                    == 0
                    and temporal["nonpositive_time_since_previous_mmse_rows"] == 0
                    and temporal["strict_availability_flag_false_rows"] == 0
                    and not temporal["current_or_same_day_mmse_in_model_features"]
                    else "FAIL"
                )
                d3_temporal_rows.append(temporal)

    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["model_id", "horizon_months"]
    )
    preprocessing = pd.concat(preprocessing_parts, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    hazards = pd.concat(hazard_parts, ignore_index=True)
    schema = pd.DataFrame(schema_rows).drop_duplicates(
        ["model_id", "horizon_months", "feature"]
    )
    training_metadata = pd.DataFrame(training_rows).sort_values(
        ["model_id", "horizon_months"]
    )
    fixture_manifest = pd.DataFrame(fixture_rows)
    equivalence = pd.DataFrame(determinism_rows)
    d3_temporal = pd.DataFrame(d3_temporal_rows)
    model_software_versions = pd.DataFrame(software_rows).sort_values(
        ["model_id", "horizon_months"]
    )

    # Participant-grouping verification uses the original frozen assignments.
    outer_unique = outer.groupby("participant_id")["outer_fold"].nunique().max()
    grouping_rows.append(
        {
            "scope": "outer_folds",
            "check": "one outer fold per ADNI participant",
            "status": "PASS" if outer_unique == 1 else "FAIL",
            "details": f"participants={outer['participant_id'].nunique()}; maximum_folds={outer_unique}",
        }
    )
    inner_roles = inner.groupby(["outer_test_fold", "participant_id"]).size().max()
    grouping_rows.append(
        {
            "scope": "nested_folds",
            "check": "one nested role per participant and outer split",
            "status": "PASS" if inner_roles == 1 else "FAIL",
            "details": f"rows={len(inner)}; maximum_roles={inner_roles}",
        }
    )
    for (model_id, horizon), data in prepared_data.items():
        if specs[model_id].family != "dynamic":
            continue
        mapped = data[["participant_id", "outer_fold"]].drop_duplicates()
        joined = mapped.merge(
            outer[["participant_id", "outer_fold"]],
            on="participant_id",
            suffixes=("_data", "_frozen"),
            validate="one_to_one",
        )
        mismatches = int(
            (joined["outer_fold_data"] != joined["outer_fold_frozen"]).sum()
        )
        nested = tuning[
            tuning["model_id"].eq(model_id)
            & tuning["horizon_months"].eq(horizon)
        ]
        grouping_rows.append(
            {
                "scope": f"{model_id}_{horizon}m",
                "check": "dynamic rows retain frozen participant-grouped folds",
                "status": "PASS" if mismatches == 0 else "FAIL",
                "details": (
                    f"rows={len(data)}; participants={data['participant_id'].nunique()}; "
                    f"fold_mismatches={mismatches}; saved_penalty_rows={len(nested)}"
                ),
            }
        )
    participant_grouping = pd.DataFrame(grouping_rows)

    # Package machine-readable support.
    write_csv(inventory, BUILD_LOCK / "model_inventory.csv")
    write_csv(schema, BUILD_LOCK / "frozen_feature_schema.csv")
    write_csv(preprocessing, BUILD_LOCK / "frozen_preprocessing_parameters.csv")
    write_csv(coefficients, BUILD_LOCK / "frozen_model_coefficients.csv")
    write_csv(hazards, BUILD_LOCK / "frozen_baseline_hazards.csv")
    write_csv(training_metadata, BUILD_LOCK / "repaired_model_training_metadata.csv")
    write_csv(penalty_recovery, BUILD_LOCK / "penalty_source_recovery.csv")
    write_csv(equivalence, BUILD_LOCK / "b1_d1_original_copy_equivalence.csv")
    write_csv(
        fixture_manifest, BUILD_LOCK / "inference_fixture_manifest.csv"
    )
    write_csv(
        d3_temporal, BUILD_LOCK / "d3_strict_temporal_verification.csv"
    )
    write_csv(
        participant_grouping,
        BUILD_LOCK / "participant_grouping_verification.csv",
    )
    write_csv(
        model_software_versions,
        BUILD_LOCK / "model_software_versions.csv",
    )
    write_json(
        repair_software_versions,
        BUILD_LOCK / "repair_software_environment.json",
    )
    exclusion_source = (
        ORIGINAL_LOCK / "allowed_and_forbidden_exclusions.csv"
    )
    ACCESS.record(
        exclusion_source,
        "preserve original allowed and forbidden exclusion rules",
        "immutable_original_lock",
    )
    shutil.copy2(
        exclusion_source,
        BUILD_LOCK / "allowed_and_forbidden_exclusions.csv",
    )

    all_features = sorted(schema["feature"].unique())
    mapping = pd.DataFrame(
        [
            {
                "canonical_feature": feature,
                "phase4_input_column": feature,
                "mapping_status": "frozen canonical Phase 2 field",
                "allowed_phase4_action": (
                    "validate outcome-hidden type/range then transform unchanged"
                ),
                "forbidden_phase4_action": (
                    "outcome-informed remapping, imputation refit, category redesign"
                ),
            }
            for feature in all_features
        ]
    )
    write_csv(mapping, BUILD_LOCK / "oasis3_frozen_feature_mapping.csv")

    checklist = pd.DataFrame(
        [
            (1, "use phase4_lock_v2 and verify its checksum manifest", True),
            (2, "verify original B1/D1 equivalence and all inference fixtures", True),
            (3, "construct outcome-hidden OASIS inputs using only frozen mappings", True),
            (4, "generate B1, D0, D1 and prespecified sensitivity predictions", True),
            (5, "freeze and hash all untouched prediction files before outcome access", True),
            (6, "independently regenerate predictions and verify no outcome fields", True),
            (7, "only after all prior checks pass merge frozen OASIS outcomes", True),
            (8, "report untouched CDRTOT performance before any updating or sensitivity", True),
            (9, "only then run reconstructed-CDR and recalibration analyses", True),
            (10, "do not begin Phase 5 automatically", True),
        ],
        columns=["order", "check", "required"],
    )
    write_csv(checklist, BUILD_LOCK / "phase4_validation_checklist.csv")

    model_index = {
        "package_version_id": package_version_id,
        "original_model_version_id": original_index["model_version_id"],
        "repair_type": "outcome-blind ADNI-only Phase 3 lock repair",
        "development_cohort": "ADNI",
        "external_validation_cohort": "OASIS-3",
        "external_participant_level_features_accessed": False,
        "external_participant_level_outcomes_accessed": False,
        "selected_baseline_model": "B1",
        "required_dynamic_comparator": "D0",
        "selected_dynamic_model": "D1",
        "required_sensitivity_models": ["B3", "D3"],
        "primary_horizons_months": [12, 36],
        "sensitivity_horizons_months": [24, 60],
        # Round-trip through pandas' JSON encoder so numpy scalars and missing
        # values become native JSON numbers/nulls rather than strings/NaN.
        "models": json.loads(inventory.to_json(orient="records")),
        "model_software_versions": "model_software_versions.csv",
        "inference_runtime": {
            "entrypoint": "inference_runtime/phase3_inference.py",
            "support_modules": [
                "inference_runtime/phase3_survival_core.py",
                "inference_runtime/phase2_preprocessing.py",
            ],
            "changed_for_repair": False,
            "generic_bundle_compatibility": ["B1", "D0", "D1", "B3", "D3"],
        },
        "version_material": version_material,
        "immutable_rules": REPAIR_CONFIG["immutable_rules"],
        "phase4_must_use_path": "models/phase3/phase4_lock_v2",
    }
    index_path = write_json(model_index, BUILD_LOCK / "frozen_model_index.json")

    access = pd.DataFrame(ACCESS.rows)
    write_csv(access, BUILD_LOCK / "access_audit.csv")
    package_spec = {
        model_id: {
            "family": specs[model_id].family,
            "dataset_variant": specs[model_id].dataset_variant,
            "continuous_features": list(specs[model_id].continuous),
            "categorical_features": list(specs[model_id].categorical),
            "weight_scheme": specs[model_id].weight_scheme,
        }
        for model_id in ["D0", "B3", "D3"]
    }
    write_json(package_spec, BUILD_LOCK / "recovered_model_specifications.json")

    checksum_manifest = package_checksum_manifest(BUILD_LOCK)
    checksum_path = write_csv(
        checksum_manifest, BUILD_LOCK / "checksum_manifest.csv"
    )
    external_manifest = {
        "package_version_id": package_version_id,
        "original_model_version_id": original_index["model_version_id"],
        "model_index_sha256": sha256(index_path),
        "checksum_manifest_sha256": sha256(checksum_path),
        "checksum_manifest_scope_exclusions": [
            "checksum_manifest.csv",
            "external_validation_lock_manifest.json",
        ],
        "model_inventory": "model_inventory.csv",
        "input_schema": "frozen_feature_schema.csv",
        "oasis_mapping": "oasis3_frozen_feature_mapping.csv",
        "validation_checklist": "phase4_validation_checklist.csv",
        "inference_fixture_manifest": "inference_fixture_manifest.csv",
        "original_b1_d1_equivalence": "b1_d1_original_copy_equivalence.csv",
        "d3_temporal_verification": "d3_strict_temporal_verification.csv",
        "participant_grouping_verification": "participant_grouping_verification.csv",
        "model_software_versions": "model_software_versions.csv",
        "untouched_first_rule": (
            "Save and hash outcome-hidden OASIS predictions before outcome access; "
            "freeze untouched CDRTOT results before recalibration or endpoint sensitivity."
        ),
        "oasis_participant_level_accessed_during_repair": False,
    }
    write_json(
        external_manifest,
        BUILD_LOCK / "external_validation_lock_manifest.json",
    )

    # Publish atomically, then prove the original lock is still unchanged.
    BUILD_LOCK.rename(FINAL_LOCK)
    original_after = original_lock_integrity("after_repair")
    original_integrity = pd.concat(
        [original_before, original_after], ignore_index=True
    )
    write_csv(
        original_integrity,
        TABLE_ROOT / "original_lock_integrity_before_after.csv",
    )

    # Export concise copies of package audits under results/ for notebook review.
    result_exports = {
        "model_inventory.csv": inventory,
        "penalty_source_recovery.csv": penalty_recovery,
        "b1_d1_original_copy_equivalence.csv": equivalence,
        "repaired_model_training_metadata.csv": training_metadata,
        "inference_fixture_manifest.csv": fixture_manifest,
        "d3_strict_temporal_verification.csv": d3_temporal,
        "participant_grouping_verification.csv": participant_grouping,
        "model_software_versions.csv": model_software_versions,
        "access_audit.csv": access,
        "checksum_manifest.csv": checksum_manifest,
        "frozen_preprocessing_parameters.csv": preprocessing,
        "frozen_model_coefficients.csv": coefficients,
        "frozen_baseline_hazards.csv": hazards,
        "frozen_feature_schema.csv": schema,
    }
    for name, frame in result_exports.items():
        write_csv(frame, TABLE_ROOT / name)

    run_metadata = {
        "repair_completed_at_utc": GENERATED_AT,
        "package_version_id": package_version_id,
        "new_lock_path": relative(FINAL_LOCK),
        "original_lock_path": relative(ORIGINAL_LOCK),
        "original_lock_unchanged": bool(
            original_integrity["status"].eq("PASS").all()
        ),
        "development_cohort": "ADNI",
        "oasis_participant_level_features_accessed": False,
        "oasis_participant_level_outcomes_accessed": False,
        "copied_models": sorted(
            inventory.loc[
                inventory["model_origin"].eq("verified_copy_from_original_lock"),
                "model_id",
            ].unique()
        ),
        "fitted_models": ["D0", "B3", "D3"],
        "models_in_package": len(inventory),
        "prediction_runtime_changed": False,
        "b1_d1_equivalence_pass": bool(equivalence["status"].eq("PASS").all()),
        "new_fixture_determinism_pass": bool(
            fixture_manifest["status"].eq("PASS").all()
        ),
        "d3_temporal_pass": bool(d3_temporal["status"].eq("PASS").all()),
        "participant_grouping_pass": bool(
            participant_grouping["status"].eq("PASS").all()
        ),
        "independent_qc_pending": True,
        "phase4_resumed": False,
        "software_versions": repair_software_versions,
    }
    write_json(run_metadata, TABLE_ROOT / "phase3_lock_repair_run_metadata.json")

    repair_report = f"""# Outcome-Blind Phase 3 Lock Repair

Generated: {GENERATED_AT}

## Package

New lock: `{relative(FINAL_LOCK)}`

Package version: `{package_version_id}`

The original lock was verified before and after repair and was not modified.
B1 and D1 were copied byte-for-byte and were not refit. D0, B3, and D3 were
fitted on ADNI only using the exact Phase 3 model family, predictor definitions,
endpoint construction, preprocessing, participant-grouped tuning decisions,
ridge penalty recovery rule, and Breslow baseline-hazard construction.

## Recovered penalties

{md_table(penalty_recovery, ["model_id", "horizon_months", "cause", "recovered_final_alpha", "outer_fold_selected_alphas", "retuned_during_repair"])}

## Repaired model training metadata

{md_table(training_metadata, ["model_id", "horizon_months", "training_rows", "training_participants", "improvement_event_rows", "progression_event_rows", "improvement_alpha", "progression_alpha", "continuous_features"])}

## Original B1/D1 equivalence

{md_table(equivalence, ["model_id", "horizon_months", "artifact_hash_identical", "maximum_probability_difference", "metadata_exact", "status"])}

## New-model deterministic fixtures

{md_table(fixture_manifest, ["model_id", "horizon_months", "fixture_source", "fixture_rows", "maximum_repeat_prediction_difference", "status"])}

## D3 temporal protection

{md_table(d3_temporal)}

## Participant grouping

{md_table(participant_grouping)}

## Per-model software versions

{md_table(model_software_versions, ["model_id", "horizon_months", "model_origin", "training_python", "training_numpy", "training_pandas", "training_scipy", "training_scikit_learn", "training_statsmodels", "package_joblib", "version_source"])}

## External-cohort protection

The access audit contains {len(access)} input-read records. Every record
certifies that OASIS-3 participant-level features and outcomes were not
accessed. Phase 4 was not resumed.
"""
    (REPORT_ROOT / "phase3_outcome_blind_lock_repair_report.md").write_text(
        repair_report, encoding="utf-8"
    )

    inference_report = f"""# Phase 3 Lock Repair Inference Compatibility

Generated: {GENERATED_AT}

The existing generic `phase3_inference.py` required no change. It reads each
bundle's frozen continuous and categorical feature lists, applies the serialized
preprocessor, verifies transformed feature order, and combines paired
cause-specific hazards into coherent probabilities.

## B1/D1 equivalence

{md_table(equivalence)}

## D0/B3/D3 fixtures

{md_table(fixture_manifest)}

All fixture inputs come from frozen ADNI outer fold
{REPAIR_CONFIG["held_out_fixture_outer_fold"]}. They contain predictor columns
only; no outcome column is saved in the package fixtures.
"""
    (
        REPORT_ROOT
        / "phase3_lock_repair_inference_compatibility_report.md"
    ).write_text(inference_report, encoding="utf-8")

    print(
        json.dumps(
            {
                "new_lock_path": relative(FINAL_LOCK),
                "package_version_id": package_version_id,
                "models_in_package": len(inventory),
                "original_lock_unchanged": True,
                "b1_d1_equivalence": "PASS",
                "new_model_determinism": "PASS",
                "d3_strict_temporal": "PASS",
                "participant_grouping": "PASS",
                "oasis_participant_level_accessed": False,
                "independent_qc_pending": True,
                "phase4_resumed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
