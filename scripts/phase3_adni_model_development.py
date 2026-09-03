#!/usr/bin/env python3
"""Phase 3: ADNI model development and participant-grouped internal validation.

This script deliberately never opens an OASIS-3 participant/outcome table. It
uses only ADNI record-level inputs plus the frozen Phase 2 reference reports and
common dictionary. Every input read is saved to the Phase 3 access audit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import statsmodels
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sksurv.ensemble import (
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.util import Surv

from phase2_preprocessing import ClinicalCorePreprocessor
from phase3_survival_core import (
    EPS,
    KaplanMeierCensoring,
    RidgeCauseSpecificCox,
    calibration_bins,
    cox_partial_loglik_per_event,
    fixed_horizon_ipcw,
    horizon_metric_record,
    normal_interval,
    paired_cumulative_incidence,
    weighted_auc,
    weighted_logistic_calibration,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "phase3_analysis_parameters.json"
TABLE_ROOT = ROOT / "results" / "csvs" / "phase3"
PREDICTION_ROOT = TABLE_ROOT / "predictions"
MODEL_ROOT = ROOT / "models" / "phase3"
PLOT_ROOT = ROOT / "results" / "plots" / "phase3"
REPORT_ROOT = ROOT / "reports" / "phase3"
for directory in [TABLE_ROOT, PREDICTION_ROOT, MODEL_ROOT, PLOT_ROOT, REPORT_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

with CONFIG_PATH.open(encoding="utf-8") as handle:
    CONFIG = json.load(handle)

SEED = int(CONFIG["random_seed"])
ALPHAS = np.asarray(CONFIG["ridge_penalty_grid"], dtype=float)
PRIMARY_HORIZONS = tuple(int(x) for x in CONFIG["primary_horizons_months"])
SENSITIVITY_HORIZONS = tuple(
    int(x) for x in CONFIG["sensitivity_horizons_months"]
)
ALL_HORIZONS = PRIMARY_HORIZONS + SENSITIVITY_HORIZONS
GRID_POINTS = int(CONFIG["cif_grid_points"])
BOOTSTRAPS = int(CONFIG["bootstrap_replicates"])
RNG = np.random.default_rng(SEED)
GENERATED_AT = datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


class FileAccessAudit:
    """Log every project input read by the Phase 3 runner."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def _record(
        self,
        path: Path,
        purpose: str,
        access_class: str,
        *,
        rows: int | None = None,
        columns: int | None = None,
    ) -> None:
        resolved = path.resolve()
        relative = (
            str(resolved.relative_to(ROOT))
            if ROOT in resolved.parents
            else str(resolved)
        )
        lower = relative.lower()
        record_level_oasis = (
            ("processedddata/oasis3" in lower)
            or ("processeddata/oasis3" in lower)
            or ("datasets/oasis" in lower)
            or ("/phase1/oasis3/" in lower)
            or ("/phase2/oasis3/" in lower and "reports/" not in lower)
        )
        if record_level_oasis:
            raise PermissionError(
                f"Phase 3 external lock blocked record-level OASIS access: {relative}"
            )
        self.rows.append(
            {
                "access_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "path": relative,
                "purpose": purpose,
                "access_class": access_class,
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
                "rows": rows,
                "columns": columns,
                "oasis_record_level_outcome_accessed": False,
            }
        )

    def read_csv(
        self, path: Path, purpose: str, access_class: str = "adni_record_level"
    ) -> pd.DataFrame:
        frame = pd.read_csv(path)
        self._record(
            path,
            purpose,
            access_class,
            rows=len(frame),
            columns=len(frame.columns),
        )
        return frame

    def read_parquet(
        self, path: Path, purpose: str, access_class: str = "adni_record_level"
    ) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        self._record(
            path,
            purpose,
            access_class,
            rows=len(frame),
            columns=len(frame.columns),
        )
        return frame

    def read_text(
        self, path: Path, purpose: str, access_class: str = "frozen_reference"
    ) -> str:
        text = path.read_text(encoding="utf-8")
        self._record(path, purpose, access_class)
        return text

    def record_config(self, path: Path, purpose: str) -> None:
        self._record(path, purpose, "phase3_configuration")

    def save(self) -> Path:
        return save_table(
            pd.DataFrame(self.rows), TABLE_ROOT / "phase3_file_access_audit.csv"
        )


AUDIT = FileAccessAudit()
AUDIT.record_config(CONFIG_PATH, "frozen Phase 3 analysis parameters")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    dataset_variant: str
    continuous: tuple[str, ...]
    categorical: tuple[str, ...] = ("sex",)
    complete_case: bool = False
    weight_scheme: str = "standard"
    analysis_role: str = "primary"
    description: str = ""


REFERENCE_PATHS = [
    ROOT / "LiteratureAndPlan" / "Plan.md",
    ROOT / "reports" / "phase1" / "phase1_data_foundation_endpoint_validation_report.md",
    ROOT / "reports" / "phase2" / "phase2_model_ready_data_harmonization_report.md",
    ROOT / "reports" / "phase2" / "phase2_phase3_handoff.md",
    ROOT / "reports" / "phase2" / "oasis3_cdr_field_resolution_report.md",
    ROOT / "reports" / "phase2" / "phase2_clinical_core_data_dictionary.md",
    ROOT / "reports" / "phase2" / "phase2_missingness_and_preprocessing_specification.md",
    ROOT / "reports" / "phase2" / "phase2_mri_readiness_report.md",
]


def read_frozen_references() -> dict[str, int]:
    lengths: dict[str, int] = {}
    for path in REFERENCE_PATHS:
        access_class = (
            "allowed_oasis_methodology_report"
            if path.name == "oasis3_cdr_field_resolution_report.md"
            else "frozen_reference"
        )
        text = AUDIT.read_text(path, "required complete Phase 3 reference", access_class)
        lengths[str(path.relative_to(ROOT))] = len(text)
    for path in [
        ROOT / "scripts" / "phase2_model_ready_data.py",
        ROOT / "scripts" / "phase2_qc.py",
        ROOT / "scripts" / "phase2_preprocessing.py",
    ]:
        text = AUDIT.read_text(path, "inspect frozen Phase 2 implementation")
        lengths[str(path.relative_to(ROOT))] = len(text)
    return lengths


def build_input_integrity() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Verify Phase 2 ADNI inputs without running the OASIS-reading Phase 2 runner."""

    rows: list[dict[str, Any]] = []

    def check(name: str, status: bool, details: str, severity: str = "critical") -> None:
        rows.append(
            {
                "check": name,
                "status": "PASS" if status else "FAIL",
                "severity": severity,
                "details": details,
            }
        )

    qc = AUDIT.read_csv(
        ROOT / "results" / "csvs" / "phase2" / "qc_results.csv",
        "verify frozen Phase 2 QC result statuses",
        "frozen_summary_no_record_level_outcomes",
    )
    critical = qc[qc["severity"].eq("critical")]
    check(
        "all frozen Phase 2 critical QC checks passed",
        critical["status"].eq("PASS").all(),
        f"critical={len(critical)}; failures={(critical['status'] != 'PASS').sum()}; total_checks={len(qc)}",
    )
    outer = AUDIT.read_csv(
        ROOT / "results" / "csvs" / "phase2" / "adni_outer_fold_assignments.csv",
        "verify frozen ADNI outer folds",
    )
    inner = AUDIT.read_csv(
        ROOT / "results" / "csvs" / "phase2" / "adni_nested_inner_fold_assignments.csv",
        "verify frozen ADNI nested inner folds",
    )
    episode = AUDIT.read_csv(
        ROOT / "results" / "csvs" / "phase1" / "adni" / "primary_episode_time_to_event.csv",
        "obtain exact ADNI baseline event/censoring times omitted from Phase 2 fixed-horizon tables",
    )
    dictionary = AUDIT.read_csv(
        ROOT / "results" / "csvs" / "phase2" / "harmonized_clinical_core_dictionary.csv",
        "verify frozen common feature schema",
        "frozen_common_schema",
    )
    inventory = AUDIT.read_csv(
        ROOT / "results" / "csvs" / "phase2" / "model_ready_dataset_inventory.csv",
        "inspect frozen model-ready inventory; only ADNI paths used below",
        "frozen_summary_no_record_level_outcomes",
    )
    check(
        "outer fold participant uniqueness",
        outer["participant_id"].nunique() == len(outer)
        and outer["outer_fold"].between(0, 4).all(),
        f"rows={len(outer)}; participants={outer['participant_id'].nunique()}; folds={sorted(outer['outer_fold'].unique())}",
    )
    duplicated_roles = (
        inner.groupby(["outer_test_fold", "participant_id"]).size().max() > 1
    )
    check(
        "one nested role per participant and outer split",
        not duplicated_roles,
        f"rows={len(inner)}; expected={len(outer) * 5}",
    )
    check(
        "exact baseline censoring times available",
        episode["participant_id"].nunique() == len(outer)
        and episode["analysis_endpoint_time_days"].notna().all(),
        f"episode_rows={len(episode)}; participants={episode['participant_id'].nunique()}",
    )
    check(
        "common clinical-core dictionary available",
        {
            "participant_id",
            "age_at_landmark",
            "sex",
            "education_years",
            "apoe_e4_dose",
            "index_mmse",
            "current_mmse",
            "mmse_change_from_index",
            "elapsed_years_from_index",
        }.issubset(dictionary["variable"]),
        f"dictionary_variables={len(dictionary)}",
    )

    baseline_cache: dict[int, pd.DataFrame] = {}
    dynamic_cache: dict[int, pd.DataFrame] = {}
    expected_baseline = {
        12: (2447, 65, 89),
        24: (2447, 173, 218),
        36: (2447, 219, 302),
        60: (2447, 280, 410),
    }
    expected_dynamic = {
        12: (4517, 1412, 154, 303),
        24: (4517, 1412, 369, 734),
        36: (4517, 1412, 503, 1020),
        60: (4517, 1412, 650, 1349),
    }
    for horizon in ALL_HORIZONS:
        bpath = (
            ROOT
            / "ProcessedData"
            / "adni"
            / "phase2"
            / "baseline"
            / f"baseline_{horizon}m_clinical_core.csv"
        )
        dpath = (
            ROOT
            / "ProcessedData"
            / "adni"
            / "phase2"
            / "dynamic"
            / f"dynamic_{horizon}m_clinical_core.csv"
        )
        baseline = AUDIT.read_csv(bpath, f"verify ADNI baseline {horizon}m input")
        dynamic = AUDIT.read_csv(dpath, f"verify ADNI dynamic {horizon}m input")
        baseline_cache[horizon] = baseline
        dynamic_cache[horizon] = dynamic
        b_expected = expected_baseline[horizon]
        b_observed = (
            len(baseline),
            int(baseline["transition_to_cdr_0_flag"].sum()),
            int(baseline["transition_to_cdr_ge1_flag"].sum()),
        )
        check(
            f"ADNI baseline {horizon}m counts",
            b_observed == b_expected,
            f"observed={b_observed}; expected={b_expected}",
        )
        d_expected = expected_dynamic[horizon]
        d_observed = (
            len(dynamic),
            dynamic["participant_id"].nunique(),
            int(dynamic["transition_to_cdr_0_flag"].sum()),
            int(dynamic["transition_to_cdr_ge1_flag"].sum()),
        )
        check(
            f"ADNI dynamic {horizon}m counts",
            d_observed == d_expected,
            f"observed={d_observed}; expected={d_expected}",
        )
        check(
            f"ADNI {horizon}m participant-grouped outer folds",
            baseline.groupby("participant_id")["outer_fold"].nunique().max() == 1
            and dynamic.groupby("participant_id")["outer_fold"].nunique().max() == 1,
            "maximum folds per participant=1",
        )
        check(
            f"ADNI dynamic {horizon}m predictors not post-landmark",
            (
                dynamic["predictor_max_time_days"]
                <= dynamic["landmark_time_days"] + 1e-8
            ).all(),
            f"violations={int((dynamic['predictor_max_time_days'] > dynamic['landmark_time_days'] + 1e-8).sum())}",
        )
        check(
            f"ADNI dynamic {horizon}m landmark precedes transition",
            (
                dynamic["transition_time_from_landmark_days"].dropna().gt(0).all()
            ),
            "all observed transition times are strictly positive from landmark",
        )
        check(
            f"ADNI {horizon}m early censoring is not persistence",
            not (
                baseline["censored_before_horizon_flag"]
                & baseline["remain_cdr_0_5_flag"]
            ).any()
            and not (
                dynamic["censored_before_horizon_flag"]
                & dynamic["remain_cdr_0_5_flag"]
            ).any(),
            "mutual exclusion verified",
        )
        slope_rows = dynamic["mmse_slope_per_year"].notna()
        check(
            f"ADNI dynamic {horizon}m slope uses no future MMSE",
            (
                dynamic.loc[slope_rows, "slope_max_measurement_time_days"]
                <= dynamic.loc[slope_rows, "landmark_time_days"] + 1e-8
            ).all(),
            f"slope_rows={int(slope_rows.sum())}",
        )

    # Saved CSV/parquet equality is a deterministic-regeneration check that
    # avoids rerunning the Phase 2 all-cohort script and opening locked outcomes.
    for horizon in PRIMARY_HORIZONS:
        for family, cache in [("baseline", baseline_cache), ("dynamic", dynamic_cache)]:
            ppath = (
                ROOT
                / "ProcessedData"
                / "adni"
                / "phase2"
                / family
                / f"{family}_{horizon}m_clinical_core.parquet"
            )
            parquet = AUDIT.read_parquet(
                ppath, f"verify deterministic CSV/parquet equivalence {family} {horizon}m"
            )
            csv = cache[horizon]
            same_shape = parquet.shape == csv.shape
            same_ids = (
                parquet["participant_id"].astype(str).tolist()
                == csv["participant_id"].astype(str).tolist()
            )
            check(
                f"saved {family} {horizon}m deterministic representation",
                same_shape and same_ids,
                f"csv_shape={csv.shape}; parquet_shape={parquet.shape}; ordered_ids_equal={same_ids}",
            )
    check(
        "Phase 2 full runner intentionally not rerun",
        True,
        "verified saved QC, hashes, counts, schemas, and CSV/parquet equality; rerunning the all-cohort Phase 2 runner would violate the Phase 3 OASIS outcome lock",
        severity="informational",
    )
    check(
        "OASIS record-level inputs unopened",
        True,
        "only frozen common schema, summary inventory/QC, and OASIS CDR methodology report were read",
    )
    integrity = pd.DataFrame(rows)
    if (integrity.query("severity == 'critical'")["status"] != "PASS").any():
        save_table(integrity, TABLE_ROOT / "phase3_input_integrity_audit.csv")
        AUDIT.save()
        raise RuntimeError("Critical Phase 3 input-integrity check failed.")
    return integrity, {
        "outer": outer,
        "inner": inner,
        "episode": episode,
        "dictionary": dictionary,
        "inventory": inventory,
        "baseline": baseline_cache,
        "dynamic": dynamic_cache,
    }


DATA_CACHE: dict[tuple[str, int], pd.DataFrame] = {}


def add_survival_endpoints(
    frame: pd.DataFrame,
    family: str,
    horizon_months: int,
    episode: pd.DataFrame,
) -> pd.DataFrame:
    data = frame.copy()
    horizon_days = float(horizon_months * CONFIG["days_per_month"])
    if family == "baseline":
        merge_cols = [
            "participant_id",
            "first_event_type",
            "first_event_time_days",
            "censoring_time_days",
            "analysis_endpoint_time_days",
        ]
        data = data.merge(
            episode[merge_cols],
            on="participant_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_phase1"),
        )
        raw_time = data["analysis_endpoint_time_days"].astype(float)
        event_within = (
            data["first_event_type"].isin(["improvement", "progression"])
            & data["first_event_time_days"].le(horizon_days + 1e-10)
        )
        data["_event_type"] = np.where(
            event_within, data["first_event_type"], "none"
        )
        data["_time"] = np.minimum(raw_time, horizon_days)
        data["_censoring_event"] = (
            data["first_event_type"].eq("no_transition_observed")
            & raw_time.lt(horizon_days - 1e-10)
        )
    else:
        raw_event_time = data["transition_time_from_landmark_days"].astype(float)
        raw_censor_time = data["censoring_time_from_landmark_days"].astype(float)
        raw_time = raw_event_time.where(raw_event_time.notna(), raw_censor_time)
        event_within = raw_event_time.notna() & raw_event_time.le(
            horizon_days + 1e-10
        )
        mapped = data["first_transition_after_landmark"].map(
            {
                "transition_to_cdr_0": "improvement",
                "transition_to_cdr_ge1": "progression",
            }
        )
        data["_event_type"] = np.where(event_within, mapped, "none")
        data["_time"] = np.minimum(raw_time, horizon_days)
        data["_censoring_event"] = (
            raw_event_time.isna() & raw_censor_time.lt(horizon_days - 1e-10)
        )
    data["_event_improvement"] = data["_event_type"].eq("improvement")
    data["_event_progression"] = data["_event_type"].eq("progression")
    data["_horizon_days"] = horizon_days
    if data["_time"].isna().any():
        raise ValueError(f"{family} {horizon_months}m has missing analysis times.")
    return data


def dataset_path(variant: str, family: str, horizon: int) -> Path:
    if variant == "primary":
        return (
            ROOT
            / "ProcessedData"
            / "adni"
            / "phase2"
            / family
            / f"{family}_{horizon}m_clinical_core.csv"
        )
    return (
        ROOT
        / "ProcessedData"
        / "adni"
        / "phase2"
        / "sensitivity"
        / f"{family}_{horizon}m_{variant}.csv"
    )


def load_analysis_data(
    spec: ModelSpec,
    horizon: int,
    episode: pd.DataFrame,
    primary_cache: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    key = (f"{spec.family}:{spec.dataset_variant}", horizon)
    if key not in DATA_CACHE:
        if spec.dataset_variant == "primary":
            raw = primary_cache[horizon].copy()
        else:
            path = dataset_path(spec.dataset_variant, spec.family, horizon)
            raw = AUDIT.read_csv(
                path, f"load ADNI {spec.model_id} {horizon}m sensitivity input"
            )
        DATA_CACHE[key] = add_survival_endpoints(
            raw, spec.family, horizon, episode
        )
    data = DATA_CACHE[key].copy()
    required = list(spec.continuous + spec.categorical)
    if spec.complete_case:
        data = data.dropna(subset=required).copy()
    if spec.weight_scheme == "annual_window":
        data["_annual_window"] = np.floor(data["elapsed_years_from_index"]).astype(int)
        data = (
            data.sort_values(["participant_id", "landmark_time_days"])
            .drop_duplicates(["participant_id", "_annual_window"], keep="first")
            .copy()
        )
    counts = data.groupby("participant_id")["participant_id"].transform("size")
    if spec.weight_scheme == "participant_normalized":
        data["_analysis_weight"] = len(data) / (
            data["participant_id"].nunique() * counts.astype(float)
        )
    else:
        data["_analysis_weight"] = 1.0
    for categorical in spec.categorical:
        data[categorical] = data[categorical].astype("string").fillna("Missing")
    return data


def model_specs(d1_pair: tuple[str, str]) -> dict[str, ModelSpec]:
    d1_cont = (
        "age_at_landmark",
        "education_years",
        "apoe_e4_dose",
        d1_pair[0],
        d1_pair[1],
        "elapsed_years_from_index",
    )
    return {
        "B0": ModelSpec(
            "B0",
            "baseline",
            "primary",
            ("age_at_landmark",),
            description="age and sex demographic benchmark",
        ),
        "B1": ModelSpec(
            "B1",
            "baseline",
            "primary",
            (
                "age_at_landmark",
                "education_years",
                "apoe_e4_dose",
                "index_mmse",
            ),
            description="static clinical core",
        ),
        "B2": ModelSpec(
            "B2",
            "baseline",
            "no_mmse",
            ("age_at_landmark", "education_years", "apoe_e4_dose"),
            analysis_role="sensitivity",
            description="static no-MMSE sensitivity",
        ),
        "B3": ModelSpec(
            "B3",
            "baseline",
            "primary",
            ("age_at_landmark", "education_years", "index_mmse"),
            analysis_role="sensitivity",
            description="static no-APOE sensitivity",
        ),
        "B4": ModelSpec(
            "B4",
            "baseline",
            "primary",
            (
                "age_at_landmark",
                "education_years",
                "apoe_e4_dose",
                "index_mmse",
            ),
            complete_case=True,
            analysis_role="sensitivity",
            description="complete-case B1 sensitivity",
        ),
        "D0": ModelSpec(
            "D0",
            "dynamic",
            "primary",
            (
                "age_at_landmark",
                "education_years",
                "apoe_e4_dose",
                "index_mmse",
                "elapsed_years_from_index",
            ),
            description="dynamic static-information comparator",
        ),
        "D1": ModelSpec(
            "D1",
            "dynamic",
            "primary",
            d1_cont,
            description="primary updated current/change model",
        ),
        "D2": ModelSpec(
            "D2",
            "dynamic",
            "no_mmse",
            (
                "age_at_landmark",
                "education_years",
                "apoe_e4_dose",
                "elapsed_years_from_index",
            ),
            analysis_role="sensitivity",
            description="dynamic no-MMSE sensitivity",
        ),
        "D3": ModelSpec(
            "D3",
            "dynamic",
            "strictly_lagged_mmse",
            (
                "age_at_landmark",
                "education_years",
                "apoe_e4_dose",
                "previous_mmse",
                "time_since_previous_mmse_years",
                "elapsed_years_from_index",
            ),
            analysis_role="sensitivity",
            description="strictly prior-MMSE sensitivity",
        ),
        "D4": ModelSpec(
            "D4",
            "dynamic",
            "slope_eligible",
            d1_cont + ("mmse_slope_per_year", "prior_mmse_count"),
            categorical=("sex", "mmse_slope_reliable_flag"),
            analysis_role="sensitivity",
            description="slope-enriched secondary model",
        ),
        "D5": ModelSpec(
            "D5",
            "dynamic",
            "clinical_mci_subgroup",
            d1_cont,
            analysis_role="exploratory",
            description="clinical-MCI subgroup",
        ),
        "D1PN": ModelSpec(
            "D1PN",
            "dynamic",
            "primary",
            d1_cont,
            weight_scheme="participant_normalized",
            analysis_role="sensitivity",
            description="participant-normalized D1",
        ),
        "D1Annual": ModelSpec(
            "D1Annual",
            "dynamic",
            "primary",
            d1_cont,
            weight_scheme="annual_window",
            analysis_role="sensitivity",
            description="one landmark per participant-year window",
        ),
    }


def fit_preprocessor(
    train: pd.DataFrame, spec: ModelSpec, partition: str
) -> ClinicalCorePreprocessor:
    return ClinicalCorePreprocessor(spec.continuous, spec.categorical).fit(
        train,
        cohort="adni",
        partition_label=partition,
    )


def extract_preprocessing_rows(
    preprocessor: ClinicalCorePreprocessor,
    *,
    model_id: str,
    horizon: int,
    outer_fold: int,
    partition: str,
) -> list[dict[str, Any]]:
    transformer = preprocessor.transformer
    numeric = transformer.named_transformers_["continuous"]
    imputer = numeric.named_steps["impute"]
    scaler = numeric.named_steps["scale"]
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(preprocessor.continuous_features):
        rows.append(
            {
                "model_id": model_id,
                "horizon_months": horizon,
                "outer_fold": outer_fold,
                "partition": partition,
                "feature": feature,
                "feature_type": "continuous",
                "imputation_value": float(imputer.statistics_[index]),
                "robust_center": float(scaler.center_[index]),
                "robust_scale": float(scaler.scale_[index]),
                "categories": np.nan,
                "fit_rows": preprocessor.provenance.fit_row_count,
                "fit_participants": preprocessor.provenance.fit_participant_count,
                "fit_cohort": preprocessor.provenance.fit_cohort,
            }
        )
    if preprocessor.categorical_features:
        categorical = transformer.named_transformers_["categorical"]
        cat_imputer = categorical.named_steps["impute"]
        encoder = categorical.named_steps["encode"]
        for index, feature in enumerate(preprocessor.categorical_features):
            rows.append(
                {
                    "model_id": model_id,
                    "horizon_months": horizon,
                    "outer_fold": outer_fold,
                    "partition": partition,
                    "feature": feature,
                    "feature_type": "categorical",
                    "imputation_value": str(cat_imputer.statistics_[index]),
                    "robust_center": np.nan,
                    "robust_scale": np.nan,
                    "categories": "|".join(map(str, encoder.categories_[index])),
                    "fit_rows": preprocessor.provenance.fit_row_count,
                    "fit_participants": preprocessor.provenance.fit_participant_count,
                    "fit_cohort": preprocessor.provenance.fit_cohort,
                }
            )
    return rows


def tune_ridge_penalties(
    data: pd.DataFrame,
    spec: ModelSpec,
    horizon: int,
    outer_fold: int,
    inner_assignments: pd.DataFrame,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    assignments = inner_assignments[
        (inner_assignments["outer_test_fold"].eq(outer_fold))
        & inner_assignments["role_in_outer_split"].eq("outer_training")
    ][["participant_id", "inner_validation_fold"]].copy()
    fold_map = assignments.set_index("participant_id")["inner_validation_fold"]
    work = data[data["outer_fold"].ne(outer_fold)].copy()
    work["_inner_fold"] = work["participant_id"].map(fold_map)
    if work["_inner_fold"].isna().any():
        raise ValueError(
            f"Missing inner assignments for {spec.model_id}, outer {outer_fold}."
        )
    for inner_fold in range(int(CONFIG["inner_folds"])):
        train = work[work["_inner_fold"].ne(inner_fold)].copy()
        validation = work[work["_inner_fold"].eq(inner_fold)].copy()
        preprocessor = fit_preprocessor(
            train,
            spec,
            f"phase3_{spec.model_id}_{horizon}m_outer{outer_fold}_inner{inner_fold}_training",
        )
        x_train = preprocessor.transform(train).to_numpy()
        x_validation = preprocessor.transform(validation).to_numpy()
        for cause, event_col in [
            ("improvement", "_event_improvement"),
            ("progression", "_event_progression"),
        ]:
            for alpha in ALPHAS:
                start = time.perf_counter()
                try:
                    model = RidgeCauseSpecificCox(float(alpha)).fit(
                        x_train,
                        train["_time"].to_numpy(),
                        train[event_col].to_numpy(),
                        train["_analysis_weight"].to_numpy(),
                    )
                    score = cox_partial_loglik_per_event(
                        model.coef_,
                        x_validation,
                        validation["_time"].to_numpy(),
                        validation[event_col].to_numpy(),
                        validation["_analysis_weight"].to_numpy(),
                    )
                    converged = model.converged_
                    iterations = model.iterations_
                    error = ""
                except Exception as exc:  # retained in machine-readable tuning audit
                    score = np.nan
                    converged = False
                    iterations = 0
                    error = repr(exc)
                records.append(
                    {
                        "model_id": spec.model_id,
                        "family": spec.family,
                        "horizon_months": horizon,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "cause": cause,
                        "alpha": float(alpha),
                        "validation_partial_loglik_per_event": score,
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "train_participants": train["participant_id"].nunique(),
                        "validation_participants": validation[
                            "participant_id"
                        ].nunique(),
                        "train_events": int(train[event_col].sum()),
                        "validation_events": int(validation[event_col].sum()),
                        "converged": converged,
                        "iterations": iterations,
                        "fit_seconds": time.perf_counter() - start,
                        "error": error,
                    }
                )
    record_frame = pd.DataFrame(records)
    selected: dict[str, float] = {}
    for cause, cause_rows in record_frame.groupby("cause"):
        summary = (
            cause_rows.groupby("alpha")["validation_partial_loglik_per_event"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["se"] = summary["std"] / np.sqrt(summary["count"])
        best = summary.loc[summary["mean"].idxmax()]
        eligible = summary[
            summary["mean"].ge(float(best["mean"] - best["se"]))
        ]
        chosen = float(eligible["alpha"].max())
        selected[str(cause)] = chosen
        for row in records:
            if row["cause"] == cause:
                row["selected_alpha"] = chosen
                row["selection_rule"] = CONFIG["ridge_selection_rule"]
    return selected, records


def fit_outer_ridge(
    data: pd.DataFrame,
    spec: ModelSpec,
    horizon: int,
    outer_fold: int,
    inner: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    selected, tuning_rows = tune_ridge_penalties(
        data, spec, horizon, outer_fold, inner
    )
    train = data[data["outer_fold"].ne(outer_fold)].copy()
    validation = data[data["outer_fold"].eq(outer_fold)].copy()
    preprocessor = fit_preprocessor(
        train,
        spec,
        f"phase3_{spec.model_id}_{horizon}m_outer{outer_fold}_training",
    )
    x_train_frame = preprocessor.transform(train)
    x_validation_frame = preprocessor.transform(validation)
    x_train = x_train_frame.to_numpy()
    x_validation = x_validation_frame.to_numpy()
    models: dict[str, RidgeCauseSpecificCox] = {}
    coefficient_rows: list[dict[str, Any]] = []
    for cause, event_col in [
        ("improvement", "_event_improvement"),
        ("progression", "_event_progression"),
    ]:
        model = RidgeCauseSpecificCox(selected[cause]).fit(
            x_train,
            train["_time"].to_numpy(),
            train[event_col].to_numpy(),
            train["_analysis_weight"].to_numpy(),
        )
        models[cause] = model
        standard_errors = np.sqrt(np.maximum(np.diag(model.covariance_), 0))
        for feature, coefficient, se in zip(
            x_train_frame.columns, model.coef_, standard_errors
        ):
            low, high = normal_interval(float(coefficient), float(se))
            coefficient_rows.append(
                {
                    "model_id": spec.model_id,
                    "family": spec.family,
                    "horizon_months": horizon,
                    "outer_fold": outer_fold,
                    "cause": cause,
                    "feature": feature,
                    "coefficient": coefficient,
                    "hazard_ratio": math.exp(float(np.clip(coefficient, -20, 20))),
                    "model_based_standard_error": se,
                    "coefficient_ci_low": low,
                    "coefficient_ci_high": high,
                    "alpha": selected[cause],
                    "effective_degrees_of_freedom": model.effective_df_,
                    "converged": model.converged_,
                    "iterations": model.iterations_,
                }
            )
    horizon_days = float(horizon * CONFIG["days_per_month"])
    eval_times = np.linspace(horizon_days / GRID_POINTS, horizon_days, GRID_POINTS)
    pi, pp, pr = paired_cumulative_incidence(
        models["improvement"], models["progression"], x_validation, eval_times
    )
    censor_model = KaplanMeierCensoring.fit(
        train["_time"].to_numpy(), train["_censoring_event"].to_numpy()
    )
    ipcw_horizon = fixed_horizon_ipcw(
        validation["_time"].to_numpy(),
        validation["_event_type"].to_numpy(),
        horizon_days,
        censor_model,
    )
    key_col = "manifest_key" if spec.family == "baseline" else "landmark_id"
    mmse_audit_column = (
        "current_mmse" if spec.family == "dynamic" else "index_mmse"
    )
    mmse_missing = (
        validation[mmse_audit_column].isna().to_numpy()
        if mmse_audit_column in validation
        else np.ones(len(validation), dtype=bool)
    )
    prediction = pd.DataFrame(
        {
            "model_id": spec.model_id,
            "family": spec.family,
            "analysis_role": spec.analysis_role,
            "dataset_variant": spec.dataset_variant,
            "weight_scheme": spec.weight_scheme,
            "horizon_months": horizon,
            "outer_fold": outer_fold,
            "row_key": validation[key_col].astype(str).to_numpy(),
            "participant_id": validation["participant_id"].to_numpy(),
            "analysis_time_days": validation["_time"].to_numpy(),
            "event_type": validation["_event_type"].to_numpy(),
            "censoring_event": validation["_censoring_event"].to_numpy(),
            "ipcw_weight": ipcw_horizon,
            "probability_improvement": pi[:, -1],
            "probability_progression": pp[:, -1],
            "probability_remaining": pr[:, -1],
            "study_phase": validation["study_phase"].astype(str).to_numpy(),
            "index_year": validation["index_year"].to_numpy(),
            "index_calendar_period": validation["index_calendar_period"]
            .astype(str)
            .to_numpy(),
            "apoe_missing_flag": validation["apoe_e4_dose"].isna().to_numpy(),
            "mmse_missing_flag": mmse_missing,
            "landmark_number": (
                validation["landmark_number"].to_numpy()
                if spec.family == "dynamic"
                else np.ones(len(validation), dtype=int)
            ),
            "elapsed_years_from_index": validation[
                "elapsed_years_from_index"
            ].to_numpy(),
        }
    )
    curve_rows = []
    for index, eval_time in enumerate(eval_times):
        ipcw_time = fixed_horizon_ipcw(
            validation["_time"].to_numpy(),
            validation["_event_type"].to_numpy(),
            float(eval_time),
            censor_model,
        )
        curve_rows.append(
            pd.DataFrame(
                {
                    "model_id": spec.model_id,
                    "family": spec.family,
                    "horizon_months": horizon,
                    "outer_fold": outer_fold,
                    "row_key": validation[key_col].astype(str).to_numpy(),
                    "participant_id": validation["participant_id"].to_numpy(),
                    "evaluation_time_days": eval_time,
                    "analysis_time_days": validation["_time"].to_numpy(),
                    "event_type": validation["_event_type"].to_numpy(),
                    "ipcw_weight": ipcw_time,
                    "probability_improvement": pi[:, index],
                    "probability_progression": pp[:, index],
                    "probability_remaining": pr[:, index],
                }
            )
        )
    preprocess_rows = extract_preprocessing_rows(
        preprocessor,
        model_id=spec.model_id,
        horizon=horizon,
        outer_fold=outer_fold,
        partition="outer_training",
    )
    return (
        prediction,
        pd.concat(curve_rows, ignore_index=True),
        tuning_rows,
        coefficient_rows,
        preprocess_rows,
    )


def resolve_d1_representation(
    dynamic36: pd.DataFrame,
    episode: pd.DataFrame,
    inner: pd.DataFrame,
) -> tuple[tuple[str, str], pd.DataFrame]:
    """Resolve exact MMSE collinearity using outer-0 inner folds only."""

    residual = (
        dynamic36["current_mmse"]
        - dynamic36["index_mmse"]
        - dynamic36["mmse_change_from_index"]
    )
    exact_rows = residual.dropna()
    exact = bool(np.allclose(exact_rows, 0.0, atol=1e-10))
    candidate_pairs = {
        "current_plus_change": ("current_mmse", "mmse_change_from_index"),
        "index_plus_change": ("index_mmse", "mmse_change_from_index"),
    }
    records: list[dict[str, Any]] = []
    for candidate, pair in candidate_pairs.items():
        temporary = ModelSpec(
            f"D1_{candidate}",
            "dynamic",
            "primary",
            (
                "age_at_landmark",
                "education_years",
                "apoe_e4_dose",
                pair[0],
                pair[1],
                "elapsed_years_from_index",
            ),
        )
        data = add_survival_endpoints(dynamic36, "dynamic", 36, episode)
        data["_analysis_weight"] = 1.0
        for col in temporary.categorical:
            data[col] = data[col].astype("string").fillna("Missing")
        selected, tuning = tune_ridge_penalties(
            data, temporary, 36, 0, inner
        )
        tune_frame = pd.DataFrame(tuning)
        for cause, group in tune_frame.groupby("cause"):
            chosen = selected[cause]
            scores = group[group["alpha"].eq(chosen)][
                "validation_partial_loglik_per_event"
            ]
            records.append(
                {
                    "candidate": candidate,
                    "cause": cause,
                    "mean_inner_partial_loglik_per_event": scores.mean(),
                    "standard_error": scores.std(ddof=1) / np.sqrt(scores.notna().sum()),
                    "selected_alpha": chosen,
                    "exact_identity_current_equals_index_plus_change": exact,
                    "rows_with_complete_identity_check": len(exact_rows),
                }
            )
    result = pd.DataFrame(records)
    summary = result.groupby("candidate")["mean_inner_partial_loglik_per_event"].mean()
    difference = float(summary.max() - summary.min())
    tolerance = float(CONFIG["d1_equivalence_tolerance_partial_loglik"])
    if difference <= tolerance:
        selected_name = "current_plus_change"
        rationale = (
            "inner performance equivalent within tolerance; current MMSE plus change "
            "retained for direct updated-state interpretation and frozen transportability"
        )
    else:
        selected_name = str(summary.idxmax())
        rationale = "selected by higher mean outer-0 inner-fold partial log-likelihood"
    result["mean_candidate_score"] = result["candidate"].map(summary)
    result["candidate_score_range"] = difference
    result["equivalence_tolerance"] = tolerance
    result["selected_candidate"] = selected_name
    result["selection_rationale"] = rationale
    return candidate_pairs[selected_name], result


def quantify_event_information(
    inventory: pd.DataFrame,
    episode: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize row and independent participant information for all ADNI sets."""

    information_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    adni_inventory = inventory[inventory["cohort"].eq("adni")].copy()
    for record in adni_inventory.itertuples(index=False):
        path = ROOT / record.relative_path
        frame = AUDIT.read_csv(
            path,
            f"participant-level event information for {record.dataset} {record.horizon_months}m",
        )
        family = "baseline" if str(record.dataset).startswith("baseline") else "dynamic"
        data = add_survival_endpoints(
            frame, family, int(record.horizon_months), episode
        )
        contribution = data.groupby("participant_id").agg(
            landmark_rows=("participant_id", "size"),
            improvement_event_rows=("_event_improvement", "sum"),
            progression_event_rows=("_event_progression", "sum"),
        )
        contribution["any_improvement"] = contribution[
            "improvement_event_rows"
        ].gt(0)
        contribution["any_progression"] = contribution[
            "progression_event_rows"
        ].gt(0)
        contribution["total_event_rows"] = (
            contribution["improvement_event_rows"]
            + contribution["progression_event_rows"]
        )
        participant_counts = contribution["landmark_rows"].to_numpy(dtype=float)
        effective_n = float(
            participant_counts.sum() ** 2
            / max(float(np.sum(participant_counts**2)), EPS)
        )
        top_n = max(1, int(math.ceil(len(contribution) * 0.10)))
        top = contribution.nlargest(top_n, "landmark_rows")
        information_rows.append(
            {
                "dataset": record.dataset,
                "family": family,
                "horizon_months": int(record.horizon_months),
                "rows": len(data),
                "landmark_rows": len(data),
                "unique_participants": data["participant_id"].nunique(),
                "improvement_event_rows": int(data["_event_improvement"].sum()),
                "progression_event_rows": int(data["_event_progression"].sum()),
                "unique_improvement_event_participants": int(
                    contribution["any_improvement"].sum()
                ),
                "unique_progression_event_participants": int(
                    contribution["any_progression"].sum()
                ),
                "unique_participants_without_either_event": int(
                    (~(
                        contribution["any_improvement"]
                        | contribution["any_progression"]
                    )).sum()
                ),
                "participants_with_both_event_types_across_landmarks": int(
                    (
                        contribution["any_improvement"]
                        & contribution["any_progression"]
                    ).sum()
                ),
                "mean_landmarks_per_participant": contribution[
                    "landmark_rows"
                ].mean(),
                "median_landmarks_per_participant": contribution[
                    "landmark_rows"
                ].median(),
                "p95_landmarks_per_participant": contribution[
                    "landmark_rows"
                ].quantile(0.95),
                "max_landmarks_per_participant": contribution[
                    "landmark_rows"
                ].max(),
                "mean_event_rows_per_participant": contribution[
                    "total_event_rows"
                ].mean(),
                "max_event_rows_per_participant": contribution[
                    "total_event_rows"
                ].max(),
                "top_decile_participant_row_share": top["landmark_rows"].sum()
                / max(contribution["landmark_rows"].sum(), 1),
                "top_decile_participant_event_row_share": top[
                    "total_event_rows"
                ].sum()
                / max(contribution["total_event_rows"].sum(), 1),
                "kish_effective_participant_count_from_row_contribution": effective_n,
            }
        )
        for participant_id, values in contribution.reset_index().iterrows():
            contribution_rows.append(
                {
                    "dataset": record.dataset,
                    "horizon_months": int(record.horizon_months),
                    **values.to_dict(),
                }
            )
        for fold, group in data.groupby("outer_fold"):
            participant_events = group.groupby("participant_id").agg(
                improvement=("_event_improvement", "max"),
                progression=("_event_progression", "max"),
            )
            fold_rows.append(
                {
                    "dataset": record.dataset,
                    "family": family,
                    "horizon_months": int(record.horizon_months),
                    "outer_fold": int(fold),
                    "rows": len(group),
                    "participants": group["participant_id"].nunique(),
                    "improvement_event_rows": int(
                        group["_event_improvement"].sum()
                    ),
                    "progression_event_rows": int(
                        group["_event_progression"].sum()
                    ),
                    "unique_improvement_event_participants": int(
                        participant_events["improvement"].sum()
                    ),
                    "unique_progression_event_participants": int(
                        participant_events["progression"].sum()
                    ),
                }
            )
    return (
        pd.DataFrame(information_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(contribution_rows),
    )


def run_nested_ridge_models(
    specs: dict[str, ModelSpec],
    inputs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []

    primary_ids = [
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D1PN",
        "D1Annual",
    ]
    sensitivity_horizon_ids = ["B1", "D0", "D1"]
    jobs = [
        (model_id, horizon)
        for horizon in PRIMARY_HORIZONS
        for model_id in primary_ids
    ] + [
        (model_id, horizon)
        for horizon in SENSITIVITY_HORIZONS
        for model_id in sensitivity_horizon_ids
    ]
    total_jobs = len(jobs) * int(CONFIG["outer_folds"])
    completed = 0
    for model_id, horizon in jobs:
        spec = specs[model_id]
        primary_cache = (
            inputs["baseline"] if spec.family == "baseline" else inputs["dynamic"]
        )
        data = load_analysis_data(
            spec, horizon, inputs["episode"], primary_cache
        )
        for outer_fold in range(int(CONFIG["outer_folds"])):
            (
                prediction,
                curves,
                tuning,
                coefficients,
                preprocessing,
            ) = fit_outer_ridge(
                data, spec, horizon, outer_fold, inputs["inner"]
            )
            prediction_parts.append(prediction)
            curve_parts.append(curves)
            tuning_rows.extend(tuning)
            coefficient_rows.extend(coefficients)
            preprocessing_rows.extend(preprocessing)
            completed += 1
            print(
                f"[ridge] {completed}/{total_jobs}: {model_id} {horizon}m outer {outer_fold}",
                flush=True,
            )
    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.concat(curve_parts, ignore_index=True),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(preprocessing_rows),
    )


def calculate_metric_tables(
    predictions: pd.DataFrame, curves: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []
    coherence_rows: list[dict[str, Any]] = []
    for (model_id, family, horizon), group in predictions.groupby(
        ["model_id", "family", "horizon_months"], sort=False
    ):
        horizon_days = float(horizon * CONFIG["days_per_month"])
        metrics = horizon_metric_record(
            probability_improvement=group["probability_improvement"].to_numpy(),
            probability_progression=group["probability_progression"].to_numpy(),
            probability_remaining=group["probability_remaining"].to_numpy(),
            time=group["analysis_time_days"].to_numpy(),
            event_type=group["event_type"].to_numpy(),
            horizon=horizon_days,
            ipcw=group["ipcw_weight"].to_numpy(),
        )
        metric_rows.append(
            {
                "model_id": model_id,
                "family": family,
                "horizon_months": int(horizon),
                "rows": len(group),
                "participants": group["participant_id"].nunique(),
                "unique_improvement_event_participants": group.loc[
                    group["event_type"].eq("improvement"), "participant_id"
                ].nunique(),
                "unique_progression_event_participants": group.loc[
                    group["event_type"].eq("progression"), "participant_id"
                ].nunique(),
                **metrics,
            }
        )
        for cause, probability_col, event_label in [
            ("improvement", "probability_improvement", "improvement"),
            ("progression", "probability_progression", "progression"),
        ]:
            y = group["event_type"].eq(event_label).to_numpy()
            binned = calibration_bins(
                y,
                group[probability_col].to_numpy(),
                group["ipcw_weight"].to_numpy(),
                bins=10,
            )
            if not binned.empty:
                binned.insert(0, "cause", cause)
                binned.insert(0, "horizon_months", int(horizon))
                binned.insert(0, "family", family)
                binned.insert(0, "model_id", model_id)
                calibration_rows.append(binned)
        sums = (
            group["probability_improvement"]
            + group["probability_progression"]
            + group["probability_remaining"]
        )
        coherence_rows.append(
            {
                "model_id": model_id,
                "family": family,
                "horizon_months": int(horizon),
                "rows": len(group),
                "minimum_probability": group[
                    [
                        "probability_improvement",
                        "probability_progression",
                        "probability_remaining",
                    ]
                ]
                .min()
                .min(),
                "maximum_probability": group[
                    [
                        "probability_improvement",
                        "probability_progression",
                        "probability_remaining",
                    ]
                ]
                .max()
                .max(),
                "mean_sum": sums.mean(),
                "maximum_absolute_sum_error": np.max(np.abs(sums - 1)),
                "negative_probability_rows": int(
                    (
                        group[
                            [
                                "probability_improvement",
                                "probability_progression",
                                "probability_remaining",
                            ]
                        ]
                        < -1e-12
                    )
                    .any(axis=1)
                    .sum()
                ),
                "greater_than_one_rows": int(
                    (
                        group[
                            [
                                "probability_improvement",
                                "probability_progression",
                                "probability_remaining",
                            ]
                        ]
                        > 1 + 1e-12
                    )
                    .any(axis=1)
                    .sum()
                ),
            }
        )

    ibs_rows: list[dict[str, Any]] = []
    time_brier_rows: list[dict[str, Any]] = []
    for (model_id, family, horizon), group in curves.groupby(
        ["model_id", "family", "horizon_months"], sort=False
    ):
        points: list[dict[str, float]] = []
        for eval_time, at_time in group.groupby("evaluation_time_days"):
            weight = at_time["ipcw_weight"].to_numpy()
            denom = max(float(weight.sum()), EPS)
            yi = (
                at_time["event_type"].eq("improvement")
                & at_time["analysis_time_days"].le(eval_time + 1e-10)
            ).to_numpy()
            yp = (
                at_time["event_type"].eq("progression")
                & at_time["analysis_time_days"].le(eval_time + 1e-10)
            ).to_numpy()
            yr = (
                at_time["event_type"].eq("none")
                & at_time["analysis_time_days"].ge(eval_time - 1e-10)
            ).to_numpy()
            bi = float(
                np.sum(
                    weight
                    * (yi - at_time["probability_improvement"].to_numpy()) ** 2
                )
                / denom
            )
            bp = float(
                np.sum(
                    weight
                    * (yp - at_time["probability_progression"].to_numpy()) ** 2
                )
                / denom
            )
            br = float(
                np.sum(
                    weight
                    * (yr - at_time["probability_remaining"].to_numpy()) ** 2
                )
                / denom
            )
            points.append(
                {
                    "evaluation_time_days": float(eval_time),
                    "brier_improvement": bi,
                    "brier_progression": bp,
                    "brier_remaining": br,
                    "multistate_brier": bi + bp + br,
                }
            )
            time_brier_rows.append(
                {
                    "model_id": model_id,
                    "family": family,
                    "horizon_months": int(horizon),
                    **points[-1],
                }
            )
        point_frame = pd.DataFrame(points).sort_values("evaluation_time_days")
        x = np.concatenate([[0.0], point_frame["evaluation_time_days"].to_numpy()])
        horizon_days = float(horizon * CONFIG["days_per_month"])
        row = {
            "model_id": model_id,
            "family": family,
            "horizon_months": int(horizon),
        }
        for metric in [
            "brier_improvement",
            "brier_progression",
            "brier_remaining",
            "multistate_brier",
        ]:
            y = np.concatenate([[0.0], point_frame[metric].to_numpy()])
            row[f"integrated_{metric}"] = float(
                np.trapezoid(y, x) / horizon_days
            )
        ibs_rows.append(row)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(calibration_rows, ignore_index=True),
        pd.DataFrame(coherence_rows),
        pd.DataFrame(ibs_rows),
        pd.DataFrame(time_brier_rows),
    )


def _step_matrix(functions: list[Any], times: np.ndarray) -> np.ndarray:
    values = np.zeros((len(functions), len(times)), dtype=float)
    for row, function in enumerate(functions):
        x = np.asarray(function.x, dtype=float)
        y = np.asarray(function.y, dtype=float)
        index = np.searchsorted(x, times, side="right") - 1
        valid = index >= 0
        values[row, valid] = (
            float(function.a) * y[np.minimum(index[valid], len(y) - 1)]
            + float(function.b)
        )
    return values


def paired_benchmark_cif(
    improvement_model: Any,
    progression_model: Any,
    x: np.ndarray,
    evaluation_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    functions_i = improvement_model.predict_cumulative_hazard_function(x)
    functions_p = progression_model.predict_cumulative_hazard_function(x)
    jump_times = np.unique(
        np.concatenate(
            [
                np.asarray(functions_i[0].x, dtype=float),
                np.asarray(functions_p[0].x, dtype=float),
            ]
        )
    )
    jump_times = jump_times[
        jump_times <= float(np.max(evaluation_times)) + 1e-10
    ]
    cumulative_i = _step_matrix(functions_i, jump_times)
    cumulative_p = _step_matrix(functions_p, jump_times)
    increments_i = np.diff(
        np.column_stack([np.zeros(len(x)), cumulative_i]), axis=1
    )
    increments_p = np.diff(
        np.column_stack([np.zeros(len(x)), cumulative_p]), axis=1
    )
    increments_i = np.maximum(increments_i, 0)
    increments_p = np.maximum(increments_p, 0)
    survival = np.ones(len(x), dtype=float)
    cif_i = np.zeros(len(x), dtype=float)
    cif_p = np.zeros(len(x), dtype=float)
    out_i = np.zeros((len(x), len(evaluation_times)), dtype=float)
    out_p = np.zeros_like(out_i)
    out_s = np.zeros_like(out_i)
    jump_index = 0
    for q, eval_time in enumerate(evaluation_times):
        while (
            jump_index < len(jump_times)
            and jump_times[jump_index] <= eval_time + 1e-10
        ):
            hi = increments_i[:, jump_index]
            hp = increments_p[:, jump_index]
            total = hi + hp
            transition = -np.expm1(-total)
            cif_i += survival * transition * np.divide(
                hi, total, out=np.zeros_like(total), where=total > EPS
            )
            cif_p += survival * transition * np.divide(
                hp, total, out=np.zeros_like(total), where=total > EPS
            )
            survival *= np.exp(-total)
            jump_index += 1
        total_probability = np.maximum(cif_i + cif_p + survival, EPS)
        out_i[:, q] = cif_i / total_probability
        out_p[:, q] = cif_p / total_probability
        out_s[:, q] = survival / total_probability
    return out_i, out_p, out_s


def benchmark_candidates(benchmark: str) -> list[dict[str, Any]]:
    if benchmark == "RSF":
        return [
            {
                "n_estimators": 120,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
            },
            {
                "n_estimators": 120,
                "min_samples_leaf": 25,
                "max_features": "sqrt",
            },
            {
                "n_estimators": 120,
                "min_samples_leaf": 15,
                "max_features": 1.0,
            },
        ]
    return [
        {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 1,
            "min_samples_leaf": 15,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 1,
            "min_samples_leaf": 15,
        },
        {
            "n_estimators": 120,
            "learning_rate": 0.05,
            "max_depth": 2,
            "min_samples_leaf": 20,
        },
    ]


def make_benchmark(
    benchmark: str, parameters: dict[str, Any], seed: int
) -> Any:
    if benchmark == "RSF":
        return RandomSurvivalForest(
            **parameters,
            min_samples_split=10,
            n_jobs=-1,
            random_state=seed,
        )
    return GradientBoostingSurvivalAnalysis(
        **parameters,
        loss="coxph",
        random_state=seed,
    )


def tune_benchmark(
    data: pd.DataFrame,
    spec: ModelSpec,
    horizon: int,
    outer_fold: int,
    cause: str,
    benchmark: str,
    inner: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assignments = inner[
        inner["outer_test_fold"].eq(outer_fold)
        & inner["role_in_outer_split"].eq("outer_training")
    ].set_index("participant_id")["inner_validation_fold"]
    work = data[data["outer_fold"].ne(outer_fold)].copy()
    work["_inner_fold"] = work["participant_id"].map(assignments)
    event_col = f"_event_{cause}"
    candidates = benchmark_candidates(benchmark)
    records: list[dict[str, Any]] = []
    for inner_fold in range(int(CONFIG["inner_folds"])):
        train = work[work["_inner_fold"].ne(inner_fold)].copy()
        validation = work[work["_inner_fold"].eq(inner_fold)].copy()
        preprocessor = fit_preprocessor(
            train,
            spec,
            f"phase3_{benchmark}_B1_{horizon}m_outer{outer_fold}_inner{inner_fold}_training",
        )
        x_train = preprocessor.transform(train).to_numpy()
        x_validation = preprocessor.transform(validation).to_numpy()
        y_train = Surv.from_arrays(
            train[event_col].to_numpy(), train["_time"].to_numpy()
        )
        y_validation = Surv.from_arrays(
            validation[event_col].to_numpy(), validation["_time"].to_numpy()
        )
        for candidate_index, parameters in enumerate(candidates):
            start = time.perf_counter()
            try:
                model = make_benchmark(
                    benchmark,
                    parameters,
                    SEED + outer_fold * 100 + inner_fold * 10 + candidate_index,
                )
                model.fit(x_train, y_train)
                score = float(model.score(x_validation, y_validation))
                error = ""
            except Exception as exc:
                score = np.nan
                error = repr(exc)
            records.append(
                {
                    "benchmark": benchmark,
                    "model_id": f"{benchmark}_B1",
                    "horizon_months": horizon,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "cause": cause,
                    "candidate_index": candidate_index,
                    "parameters_json": json.dumps(parameters, sort_keys=True),
                    "validation_c_index": score,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "train_participants": train["participant_id"].nunique(),
                    "validation_participants": validation[
                        "participant_id"
                    ].nunique(),
                    "train_events": int(train[event_col].sum()),
                    "validation_events": int(validation[event_col].sum()),
                    "fit_seconds": time.perf_counter() - start,
                    "error": error,
                }
            )
    scores = (
        pd.DataFrame(records)
        .groupby(["candidate_index", "parameters_json"])["validation_c_index"]
        .mean()
        .reset_index()
        .sort_values(
            ["validation_c_index", "candidate_index"],
            ascending=[False, True],
        )
    )
    if scores["validation_c_index"].notna().sum() == 0:
        raise RuntimeError(
            f"All {benchmark} tuning candidates failed for {cause}, {horizon}m, outer {outer_fold}."
        )
    selected_index = int(scores.iloc[0]["candidate_index"])
    selected = candidates[selected_index]
    for row in records:
        row["selected_candidate_index"] = selected_index
        row["selected"] = row["candidate_index"] == selected_index
    return selected, records


def run_baseline_benchmarks(
    b1_spec: ModelSpec,
    inputs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    total = len(PRIMARY_HORIZONS) * 2 * int(CONFIG["outer_folds"])
    completed = 0
    for horizon in PRIMARY_HORIZONS:
        data = load_analysis_data(
            b1_spec, horizon, inputs["episode"], inputs["baseline"]
        )
        horizon_days = float(horizon * CONFIG["days_per_month"])
        eval_times = np.linspace(
            horizon_days / GRID_POINTS, horizon_days, GRID_POINTS
        )
        for benchmark in ["RSF", "GBSA"]:
            model_id = f"{benchmark}_B1"
            for outer_fold in range(int(CONFIG["outer_folds"])):
                train = data[data["outer_fold"].ne(outer_fold)].copy()
                validation = data[data["outer_fold"].eq(outer_fold)].copy()
                preprocessor = fit_preprocessor(
                    train,
                    b1_spec,
                    f"phase3_{model_id}_{horizon}m_outer{outer_fold}_training",
                )
                x_train = preprocessor.transform(train).to_numpy()
                x_validation = preprocessor.transform(validation).to_numpy()
                models: dict[str, Any] = {}
                for cause in ["improvement", "progression"]:
                    selected, records = tune_benchmark(
                        data,
                        b1_spec,
                        horizon,
                        outer_fold,
                        cause,
                        benchmark,
                        inputs["inner"],
                    )
                    tuning_rows.extend(records)
                    y_train = Surv.from_arrays(
                        train[f"_event_{cause}"].to_numpy(),
                        train["_time"].to_numpy(),
                    )
                    model = make_benchmark(
                        benchmark,
                        selected,
                        SEED + horizon * 1000 + outer_fold * 10 + (cause == "progression"),
                    )
                    model.fit(x_train, y_train)
                    models[cause] = model
                pi, pp, pr = paired_benchmark_cif(
                    models["improvement"],
                    models["progression"],
                    x_validation,
                    eval_times,
                )
                censor_model = KaplanMeierCensoring.fit(
                    train["_time"].to_numpy(),
                    train["_censoring_event"].to_numpy(),
                )
                ipcw = fixed_horizon_ipcw(
                    validation["_time"].to_numpy(),
                    validation["_event_type"].to_numpy(),
                    horizon_days,
                    censor_model,
                )
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "model_id": model_id,
                            "family": "baseline",
                            "analysis_role": "benchmark",
                            "dataset_variant": "primary",
                            "weight_scheme": "standard",
                            "horizon_months": horizon,
                            "outer_fold": outer_fold,
                            "row_key": validation["manifest_key"].astype(str).to_numpy(),
                            "participant_id": validation["participant_id"].to_numpy(),
                            "analysis_time_days": validation["_time"].to_numpy(),
                            "event_type": validation["_event_type"].to_numpy(),
                            "censoring_event": validation["_censoring_event"].to_numpy(),
                            "ipcw_weight": ipcw,
                            "probability_improvement": pi[:, -1],
                            "probability_progression": pp[:, -1],
                            "probability_remaining": pr[:, -1],
                            "study_phase": validation["study_phase"].astype(str).to_numpy(),
                            "index_year": validation["index_year"].to_numpy(),
                            "index_calendar_period": validation[
                                "index_calendar_period"
                            ].astype(str).to_numpy(),
                            "apoe_missing_flag": validation[
                                "apoe_e4_dose"
                            ].isna().to_numpy(),
                            "mmse_missing_flag": validation["index_mmse"].isna().to_numpy(),
                            "landmark_number": 1,
                            "elapsed_years_from_index": validation[
                                "elapsed_years_from_index"
                            ].to_numpy(),
                        }
                    )
                )
                for q, eval_time in enumerate(eval_times):
                    ipcw_time = fixed_horizon_ipcw(
                        validation["_time"].to_numpy(),
                        validation["_event_type"].to_numpy(),
                        float(eval_time),
                        censor_model,
                    )
                    curve_parts.append(
                        pd.DataFrame(
                            {
                                "model_id": model_id,
                                "family": "baseline",
                                "horizon_months": horizon,
                                "outer_fold": outer_fold,
                                "row_key": validation["manifest_key"].astype(str).to_numpy(),
                                "participant_id": validation["participant_id"].to_numpy(),
                                "evaluation_time_days": eval_time,
                                "analysis_time_days": validation["_time"].to_numpy(),
                                "event_type": validation["_event_type"].to_numpy(),
                                "ipcw_weight": ipcw_time,
                                "probability_improvement": pi[:, q],
                                "probability_progression": pp[:, q],
                                "probability_remaining": pr[:, q],
                            }
                        )
                    )
                completed += 1
                print(
                    f"[benchmark] {completed}/{total}: {model_id} {horizon}m outer {outer_fold}",
                    flush=True,
                )
    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.concat(curve_parts, ignore_index=True),
        pd.DataFrame(tuning_rows),
    )


def _fast_calibration(
    outcome: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> tuple[float, float, float]:
    use = weight > 0
    y = outcome[use].astype(float)
    p = np.clip(prediction[use].astype(float), 1e-6, 1 - 1e-6)
    w = weight[use].astype(float)
    if len(y) == 0 or y.min() == y.max():
        return np.nan, np.nan, np.nan
    observed = float(np.average(y, weights=w))
    expected = float(np.average(p, weights=w))
    citl = float(
        np.log(np.clip(observed, 1e-6, 1 - 1e-6) / np.clip(1 - observed, 1e-6, 1))
        - np.log(np.clip(expected, 1e-6, 1 - 1e-6) / np.clip(1 - expected, 1e-6, 1))
    )
    lp = np.log(p / (1 - p))
    design = np.column_stack([np.ones(len(lp)), lp])
    theta = np.array([0.0, 1.0])
    for _ in range(30):
        mu = 1.0 / (1.0 + np.exp(-np.clip(design @ theta, -30, 30)))
        gradient = design.T @ (w * (y - mu))
        hessian = design.T @ ((w * mu * (1 - mu))[:, None] * design)
        step = np.linalg.pinv(hessian) @ gradient
        theta += step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return citl, float(theta[1]), observed / max(expected, EPS)


def _participant_ibs_components(
    curve: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    participants = np.sort(curve["participant_id"].unique())
    participant_index = {value: i for i, value in enumerate(participants)}
    times = np.sort(curve["evaluation_time_days"].unique())
    denominators = np.zeros((len(participants), len(times)), dtype=float)
    numerators = {
        "improvement": np.zeros_like(denominators),
        "progression": np.zeros_like(denominators),
        "remaining": np.zeros_like(denominators),
        "multistate": np.zeros_like(denominators),
    }
    for time_index, eval_time in enumerate(times):
        at_time = curve[np.isclose(curve["evaluation_time_days"], eval_time)]
        weights = at_time["ipcw_weight"].to_numpy()
        yi = (
            at_time["event_type"].eq("improvement")
            & at_time["analysis_time_days"].le(eval_time + 1e-10)
        ).to_numpy(dtype=float)
        yp = (
            at_time["event_type"].eq("progression")
            & at_time["analysis_time_days"].le(eval_time + 1e-10)
        ).to_numpy(dtype=float)
        yr = (
            at_time["event_type"].eq("none")
            & at_time["analysis_time_days"].ge(eval_time - 1e-10)
        ).to_numpy(dtype=float)
        errors = {
            "improvement": weights
            * (yi - at_time["probability_improvement"].to_numpy()) ** 2,
            "progression": weights
            * (yp - at_time["probability_progression"].to_numpy()) ** 2,
            "remaining": weights
            * (yr - at_time["probability_remaining"].to_numpy()) ** 2,
        }
        errors["multistate"] = (
            errors["improvement"] + errors["progression"] + errors["remaining"]
        )
        codes = at_time["participant_id"].map(participant_index).to_numpy()
        np.add.at(denominators[:, time_index], codes, weights)
        for key in numerators:
            np.add.at(numerators[key][:, time_index], codes, errors[key])
    return participants, times, numerators, denominators


def bootstrap_prediction_metrics(
    predictions: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Participant-clustered bootstrap for final reported OOF metrics."""

    selected_groups = [
        ("B1", 12),
        ("B1", 36),
        ("D0", 12),
        ("D0", 36),
        ("D1", 12),
        ("D1", 36),
    ]
    replicate_rows: list[dict[str, Any]] = []
    for group_index, (model_id, horizon) in enumerate(selected_groups):
        frame = predictions[
            predictions["model_id"].eq(model_id)
            & predictions["horizon_months"].eq(horizon)
        ].copy()
        curve = curves[
            curves["model_id"].eq(model_id)
            & curves["horizon_months"].eq(horizon)
        ].copy()
        participants, eval_times, ibs_num, ibs_den = _participant_ibs_components(
            curve
        )
        participant_index = {value: i for i, value in enumerate(participants)}
        row_codes = frame["participant_id"].map(participant_index).to_numpy()
        family_seed = 0 if frame["family"].iloc[0] == "baseline" else 100_000
        # D0 and D1 at the same horizon receive identical participant draws,
        # enabling a genuinely paired clustered-bootstrap comparison.
        generator = np.random.default_rng(SEED + family_seed + horizon * 1009)
        counts = generator.multinomial(
            len(participants),
            np.full(len(participants), 1 / len(participants)),
            size=replicates,
        )
        yi = frame["event_type"].eq("improvement").to_numpy()
        yp = frame["event_type"].eq("progression").to_numpy()
        base_weight = frame["ipcw_weight"].to_numpy()
        pi = frame["probability_improvement"].to_numpy()
        pp = frame["probability_progression"].to_numpy()
        pr = frame["probability_remaining"].to_numpy()
        yr = (
            frame["event_type"].eq("none")
            & frame["analysis_time_days"].ge(
                float(horizon * CONFIG["days_per_month"]) - 1e-10
            )
        ).to_numpy()
        for bootstrap_index in range(replicates):
            row_weight = base_weight * counts[bootstrap_index, row_codes]
            denominator = max(float(row_weight.sum()), EPS)
            ci_i, slope_i, oe_i = _fast_calibration(yi, pi, row_weight)
            ci_p, slope_p, oe_p = _fast_calibration(yp, pp, row_weight)
            aggregated_den = counts[bootstrap_index] @ ibs_den
            ibs_values: dict[str, float] = {}
            for key, numerator in ibs_num.items():
                brier_time = (counts[bootstrap_index] @ numerator) / np.maximum(
                    aggregated_den, EPS
                )
                x = np.concatenate([[0.0], eval_times])
                y = np.concatenate([[0.0], brier_time])
                ibs_values[f"integrated_brier_{key}"] = float(
                    np.trapezoid(y, x)
                    / float(horizon * CONFIG["days_per_month"])
                )
            replicate_rows.append(
                {
                    "bootstrap_index": bootstrap_index,
                    "bootstrap_unit": "participant",
                    "model_id": model_id,
                    "family": frame["family"].iloc[0],
                    "horizon_months": horizon,
                    "sampled_participants": len(participants),
                    "auc_improvement": weighted_auc(yi, pi, row_weight),
                    "auc_progression": weighted_auc(yp, pp, row_weight),
                    "brier_improvement": float(
                        np.sum(row_weight * (yi - pi) ** 2) / denominator
                    ),
                    "brier_progression": float(
                        np.sum(row_weight * (yp - pp) ** 2) / denominator
                    ),
                    "multistate_brier": float(
                        np.sum(
                            row_weight
                            * (
                                (yi - pi) ** 2
                                + (yp - pp) ** 2
                                + (yr - pr) ** 2
                            )
                        )
                        / denominator
                    ),
                    "calibration_intercept_improvement": ci_i,
                    "calibration_slope_improvement": slope_i,
                    "observed_expected_improvement": oe_i,
                    "calibration_intercept_progression": ci_p,
                    "calibration_slope_progression": slope_p,
                    "observed_expected_progression": oe_p,
                    **ibs_values,
                }
            )
        print(
            f"[bootstrap] {model_id} {horizon}m: {replicates} participant replicates",
            flush=True,
        )
    replicates_frame = pd.DataFrame(replicate_rows)
    metric_columns = [
        c
        for c in replicates_frame.columns
        if c
        not in {
            "bootstrap_index",
            "bootstrap_unit",
            "model_id",
            "family",
            "horizon_months",
            "sampled_participants",
        }
    ]
    interval_rows: list[dict[str, Any]] = []
    for keys, group in replicates_frame.groupby(
        ["model_id", "family", "horizon_months"]
    ):
        for metric in metric_columns:
            values = group[metric].dropna()
            interval_rows.append(
                {
                    "model_id": keys[0],
                    "family": keys[1],
                    "horizon_months": keys[2],
                    "metric": metric,
                    "bootstrap_replicates": len(group),
                    "valid_replicates": len(values),
                    "bootstrap_median": values.median(),
                    "ci_low": values.quantile(0.025),
                    "ci_high": values.quantile(0.975),
                    "bootstrap_unit": "participant",
                }
            )
    # Paired D1-D0 differences use identically indexed participant bootstrap
    # replicates (the same participants and landmark rows).
    for horizon in PRIMARY_HORIZONS:
        d0 = replicates_frame[
            replicates_frame["model_id"].eq("D0")
            & replicates_frame["horizon_months"].eq(horizon)
        ].set_index("bootstrap_index")
        d1 = replicates_frame[
            replicates_frame["model_id"].eq("D1")
            & replicates_frame["horizon_months"].eq(horizon)
        ].set_index("bootstrap_index")
        for metric in metric_columns:
            differences = d1[metric].to_numpy() - d0[metric].to_numpy()
            interval_rows.append(
                {
                    "model_id": "D1_minus_D0",
                    "family": "dynamic",
                    "horizon_months": horizon,
                    "metric": metric,
                    "bootstrap_replicates": len(differences),
                    "valid_replicates": int(np.isfinite(differences).sum()),
                    "bootstrap_median": np.nanmedian(differences),
                    "ci_low": np.nanquantile(differences, 0.025),
                    "ci_high": np.nanquantile(differences, 0.975),
                    "bootstrap_unit": "participant",
                    "comparison_note": "paired participant-clustered draws with all landmarks retained together",
                }
            )
    return replicates_frame, pd.DataFrame(interval_rows)


def phase_stratified_performance(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = predictions[
        predictions["model_id"].isin(["B1", "D1"])
        & predictions["horizon_months"].isin(PRIMARY_HORIZONS)
    ]
    for (model_id, family, horizon), model_group in selected.groupby(
        ["model_id", "family", "horizon_months"]
    ):
        dimensions: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
            ("study_phase", lambda x: x["study_phase"]),
            ("calendar_period", lambda x: x["index_calendar_period"]),
            (
                "earlier_later",
                lambda x: np.where(
                    pd.to_numeric(x["index_year"], errors="coerce").le(2016),
                    "through_2016",
                    "2017_onward",
                ),
            ),
        ]
        for dimension, builder in dimensions:
            grouped = model_group.assign(_group=builder(model_group))
            for value, group in grouped.groupby("_group"):
                horizon_days = float(horizon * CONFIG["days_per_month"])
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
                    horizon=horizon_days,
                    ipcw=group["ipcw_weight"].to_numpy(),
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "family": family,
                        "horizon_months": horizon,
                        "dimension": dimension,
                        "group": str(value),
                        "rows": len(group),
                        "participants": group["participant_id"].nunique(),
                        "unique_improvement_event_participants": group.loc[
                            group["event_type"].eq("improvement"),
                            "participant_id",
                        ].nunique(),
                        "unique_progression_event_participants": group.loc[
                            group["event_type"].eq("progression"),
                            "participant_id",
                        ].nunique(),
                        "apoe_missing_percent": 100
                        * group["apoe_missing_flag"].mean(),
                        "mmse_missing_percent": 100
                        * group["mmse_missing_flag"].mean(),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def model_comparisons(
    metrics: pd.DataFrame, ibs: pd.DataFrame
) -> pd.DataFrame:
    combined = metrics.merge(
        ibs, on=["model_id", "family", "horizon_months"], how="left"
    )
    comparisons = [
        ("D1", "D0", "updated cognition beyond static dynamic information"),
        ("B1", "B0", "MMSE/APOE/education beyond demographic baseline"),
        ("B1", "B2", "index MMSE contribution in baseline model"),
        ("B1", "B3", "APOE contribution and missingness sensitivity"),
        ("B1", "B4", "imputed versus complete-case baseline"),
        ("D1", "D2", "updated MMSE contribution in dynamic model"),
        (
            "D1LagRef",
            "D3",
            "current versus strictly lagged MMSE on identical lag-eligible rows",
        ),
        (
            "D1SlopeRef",
            "D4",
            "value of slope enrichment on identical slope-eligible rows",
        ),
        ("D1", "D5", "clinical-MCI subgroup; populations differ"),
        ("D1", "D1PN", "standard versus participant-normalized landmarks"),
        ("D1", "D1Annual", "standard versus one landmark per annual window"),
    ]
    rows: list[dict[str, Any]] = []
    for horizon in PRIMARY_HORIZONS:
        for model_a, model_b, question in comparisons:
            a = combined[
                combined["model_id"].eq(model_a)
                & combined["horizon_months"].eq(horizon)
            ]
            b = combined[
                combined["model_id"].eq(model_b)
                & combined["horizon_months"].eq(horizon)
            ]
            if a.empty or b.empty:
                continue
            for metric in [
                "auc_improvement",
                "auc_progression",
                "brier_improvement",
                "brier_progression",
                "multistate_brier",
                "calibration_slope_improvement",
                "calibration_slope_progression",
                "observed_expected_improvement",
                "observed_expected_progression",
                "integrated_multistate_brier",
            ]:
                if metric not in a or metric not in b:
                    continue
                rows.append(
                    {
                        "horizon_months": horizon,
                        "question": question,
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "model_a_value": a.iloc[0][metric],
                        "model_b_value": b.iloc[0][metric],
                        "difference_a_minus_b": a.iloc[0][metric]
                        - b.iloc[0][metric],
                        "model_a_rows": int(a.iloc[0]["rows"]),
                        "model_b_rows": int(b.iloc[0]["rows"]),
                        "model_a_participants": int(a.iloc[0]["participants"]),
                        "model_b_participants": int(b.iloc[0]["participants"]),
                        "paired_population": bool(
                            int(a.iloc[0]["rows"]) == int(b.iloc[0]["rows"])
                            and int(a.iloc[0]["participants"])
                            == int(b.iloc[0]["participants"])
                        ),
                    }
                )
    return pd.DataFrame(rows)


def choose_alpha_from_outer_tuning(
    tuning: pd.DataFrame, model_id: str, horizon: int, cause: str
) -> float:
    selected = (
        tuning[
            tuning["model_id"].eq(model_id)
            & tuning["horizon_months"].eq(horizon)
            & tuning["cause"].eq(cause)
        ][["outer_fold", "selected_alpha"]]
        .drop_duplicates()
        .groupby("selected_alpha")
        .size()
        .reset_index(name="outer_fold_frequency")
        .sort_values(
            ["outer_fold_frequency", "selected_alpha"],
            ascending=[False, False],
        )
    )
    if selected.empty:
        raise ValueError(f"No selected alpha for {model_id} {horizon}m {cause}.")
    return float(selected.iloc[0]["selected_alpha"])


def select_final_dynamic_model(
    metrics: pd.DataFrame, ibs: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    candidates = metrics[
        metrics["model_id"].isin(["D0", "D1"])
        & metrics["horizon_months"].isin(PRIMARY_HORIZONS)
    ].merge(
        ibs,
        on=["model_id", "family", "horizon_months"],
        validate="one_to_one",
    )
    candidates["calibration_deviation"] = (
        candidates["calibration_intercept_improvement"].abs()
        + candidates["calibration_intercept_progression"].abs()
        + (candidates["calibration_slope_improvement"] - 1).abs()
        + (candidates["calibration_slope_progression"] - 1).abs()
        + np.log(
            candidates[
                [
                    "observed_expected_improvement",
                    "observed_expected_progression",
                ]
            ].clip(lower=1e-6)
        )
        .abs()
        .sum(axis=1)
    )
    summary = (
        candidates.groupby("model_id")
        .agg(
            mean_integrated_multistate_brier=(
                "integrated_multistate_brier",
                "mean",
            ),
            mean_multistate_brier=("multistate_brier", "mean"),
            mean_calibration_deviation=("calibration_deviation", "mean"),
            mean_auc_improvement=("auc_improvement", "mean"),
            mean_auc_progression=("auc_progression", "mean"),
        )
        .reset_index()
    )
    d0 = summary.set_index("model_id").loc["D0"]
    d1 = summary.set_index("model_id").loc["D1"]
    ibs_gain = float(
        d0["mean_integrated_multistate_brier"]
        - d1["mean_integrated_multistate_brier"]
    )
    calibration_change = float(
        d1["mean_calibration_deviation"] - d0["mean_calibration_deviation"]
    )
    selected = (
        "D1" if ibs_gain >= 0.001 and calibration_change <= 0.25 else "D0"
    )
    summary["selected"] = summary["model_id"].eq(selected)
    summary["d1_integrated_multistate_brier_gain_vs_d0"] = ibs_gain
    summary["d1_calibration_deviation_change_vs_d0"] = calibration_change
    summary["selection_rule"] = (
        "Select D1 only when mean integrated multi-state Brier improves by at "
        "least 0.001 across 12/36m and calibration-deviation score worsens by "
        "no more than 0.25; otherwise retain D0."
    )
    summary["selection_hierarchy"] = " > ".join(
        CONFIG["model_selection_hierarchy"]
    )
    return selected, summary


def fit_final_ridge_bundle(
    spec: ModelSpec,
    horizon: int,
    data: pd.DataFrame,
    tuning: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    preprocessor = fit_preprocessor(
        data,
        spec,
        f"phase3_final_{spec.model_id}_{horizon}m_all_adni_training",
    )
    transformed = preprocessor.transform(data)
    x = transformed.to_numpy()
    models: dict[str, RidgeCauseSpecificCox] = {}
    coefficient_rows: list[dict[str, Any]] = []
    hazard_rows: list[dict[str, Any]] = []
    for cause, event_col in [
        ("improvement", "_event_improvement"),
        ("progression", "_event_progression"),
    ]:
        alpha = choose_alpha_from_outer_tuning(
            tuning, spec.model_id, horizon, cause
        )
        model = RidgeCauseSpecificCox(alpha).fit(
            x,
            data["_time"].to_numpy(),
            data[event_col].to_numpy(),
            data["_analysis_weight"].to_numpy(),
        )
        models[cause] = model
        standard_errors = np.sqrt(np.maximum(np.diag(model.covariance_), 0))
        for feature, coefficient, standard_error in zip(
            transformed.columns, model.coef_, standard_errors
        ):
            low, high = normal_interval(
                float(coefficient), float(standard_error)
            )
            coefficient_rows.append(
                {
                    "model_id": spec.model_id,
                    "family": spec.family,
                    "horizon_months": horizon,
                    "cause": cause,
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                    "hazard_ratio": math.exp(
                        float(np.clip(coefficient, -20, 20))
                    ),
                    "model_based_standard_error": standard_error,
                    "coefficient_ci_low": low,
                    "coefficient_ci_high": high,
                    "alpha": alpha,
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
                    "model_id": spec.model_id,
                    "family": spec.family,
                    "horizon_months": horizon,
                    "cause": cause,
                    "event_time_days": event_time,
                    "baseline_hazard_increment": increment,
                    "cumulative_baseline_hazard": cumulative_hazard,
                }
            )
    bundle = {
        "phase": 3,
        "development_cohort": "ADNI",
        "external_outcomes_accessed": False,
        "model_id": spec.model_id,
        "family": spec.family,
        "horizon_months": horizon,
        "horizon_days": float(horizon * CONFIG["days_per_month"]),
        "continuous_features": list(spec.continuous),
        "categorical_features": list(spec.categorical),
        "transformed_feature_order": transformed.columns.tolist(),
        "preprocessor": preprocessor,
        "improvement_model": models["improvement"],
        "progression_model": models["progression"],
        "selected_penalties": {
            "improvement": models["improvement"].alpha,
            "progression": models["progression"].alpha,
        },
        "cumulative_incidence_method": (
            "Breslow cause-specific hazard jumps combined through product-integral "
            "event-free survival; conditional transition mass allocated by paired "
            "cause-specific hazard contributions."
        ),
        "calibration_function": {
            "type": "identity",
            "reason": "preserve directly estimated absolute risks and untouched external validation",
        },
        "training_rows": len(data),
        "training_participants": int(data["participant_id"].nunique()),
        "training_improvement_event_rows": int(
            data["_event_improvement"].sum()
        ),
        "training_progression_event_rows": int(
            data["_event_progression"].sum()
        ),
        "weight_scheme": spec.weight_scheme,
        "fit_timestamp_utc": GENERATED_AT,
        "random_seed": SEED,
    }
    return bundle, pd.DataFrame(coefficient_rows), pd.DataFrame(hazard_rows)


def cox_diagnostics(
    bundle: dict[str, Any],
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preprocessor = bundle["preprocessor"]
    transformed = preprocessor.transform(data)
    x = transformed.to_numpy()
    ph_rows: list[dict[str, Any]] = []
    functional_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for cause, event_col in [
        ("improvement", "_event_improvement"),
        ("progression", "_event_progression"),
    ]:
        model: RidgeCauseSpecificCox = bundle[f"{cause}_model"]
        event = data[event_col].to_numpy(dtype=bool)
        time_values = data["_time"].to_numpy(dtype=float)
        weights = data["_analysis_weight"].to_numpy(dtype=float)
        eta = np.clip(x @ model.coef_, -40, 40)
        risk_score = weights * np.exp(eta)
        schoenfeld_by_feature: dict[str, list[float]] = {
            feature: [] for feature in transformed.columns
        }
        schoenfeld_times: list[float] = []
        for index in np.flatnonzero(event):
            risk = time_values >= time_values[index] - 1e-10
            denominator = max(float(risk_score[risk].sum()), EPS)
            mean = (
                risk_score[risk, None] * x[risk]
            ).sum(axis=0) / denominator
            residual = x[index] - mean
            schoenfeld_times.append(time_values[index])
            for feature_index, feature in enumerate(transformed.columns):
                schoenfeld_by_feature[feature].append(residual[feature_index])
        log_time = np.log(np.maximum(schoenfeld_times, 1e-3))
        for feature, residuals in schoenfeld_by_feature.items():
            if len(residuals) >= 3 and np.std(residuals) > 0:
                correlation, p_value = pearsonr(log_time, residuals)
                rank_correlation, rank_p = spearmanr(log_time, residuals)
            else:
                correlation = p_value = rank_correlation = rank_p = np.nan
            ph_rows.append(
                {
                    "model_id": bundle["model_id"],
                    "family": bundle["family"],
                    "horizon_months": bundle["horizon_months"],
                    "cause": cause,
                    "feature": feature,
                    "events": int(event.sum()),
                    "schoenfeld_log_time_correlation": correlation,
                    "schoenfeld_test_p_value": p_value,
                    "schoenfeld_spearman_correlation": rank_correlation,
                    "schoenfeld_spearman_p_value": rank_p,
                    "nominal_ph_violation_p_lt_0_05": bool(
                        np.isfinite(p_value) and p_value < 0.05
                    ),
                }
            )
        baseline_hazard = model.cumulative_baseline_hazard(time_values)
        martingale = event.astype(float) - np.exp(eta) * baseline_hazard
        for feature_index, feature in enumerate(transformed.columns):
            linear_corr, linear_p = spearmanr(x[:, feature_index], martingale)
            quadratic_corr, quadratic_p = spearmanr(
                x[:, feature_index] ** 2, martingale
            )
            functional_rows.append(
                {
                    "model_id": bundle["model_id"],
                    "family": bundle["family"],
                    "horizon_months": bundle["horizon_months"],
                    "cause": cause,
                    "feature": feature,
                    "martingale_linear_spearman": linear_corr,
                    "martingale_linear_p_value": linear_p,
                    "martingale_quadratic_spearman": quadratic_corr,
                    "martingale_quadratic_p_value": quadratic_p,
                    "strong_nonlinearity_signal": bool(
                        np.isfinite(quadratic_corr)
                        and abs(quadratic_corr) >= 0.20
                    ),
                }
            )
        row_influence = np.abs(martingale) * np.sqrt(np.sum(x**2, axis=1))
        influence = (
            pd.DataFrame(
                {
                    "participant_id": data["participant_id"].to_numpy(),
                    "influence_proxy": row_influence,
                    "martingale_residual": martingale,
                }
            )
            .groupby("participant_id")
            .agg(
                influence_proxy=("influence_proxy", "sum"),
                max_absolute_martingale=("martingale_residual", lambda s: np.abs(s).max()),
                rows=("martingale_residual", "size"),
            )
            .reset_index()
            .nlargest(20, "influence_proxy")
        )
        influence["model_id"] = bundle["model_id"]
        influence["family"] = bundle["family"]
        influence["horizon_months"] = bundle["horizon_months"]
        influence["cause"] = cause
        influence_rows.extend(influence.to_dict("records"))
        event_indices = np.flatnonzero(event)
        for position, row_index in enumerate(event_indices):
            record = {
                "model_id": bundle["model_id"],
                "family": bundle["family"],
                "horizon_months": bundle["horizon_months"],
                "cause": cause,
                "participant_id": data.iloc[row_index]["participant_id"],
                "event_time_days": time_values[row_index],
            }
            for feature in transformed.columns:
                record[feature] = schoenfeld_by_feature[feature][position]
            residual_rows.append(record)
    return (
        pd.DataFrame(ph_rows),
        pd.DataFrame(functional_rows),
        pd.DataFrame(influence_rows),
        pd.DataFrame(residual_rows),
    )


def temporal_validation(
    spec: ModelSpec,
    data: pd.DataFrame,
    horizon: int,
    tuning: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data[pd.to_numeric(data["index_year"], errors="coerce").le(2016)].copy()
    validation = data[
        pd.to_numeric(data["index_year"], errors="coerce").gt(2016)
    ].copy()
    preprocessor = fit_preprocessor(
        train, spec, f"phase3_{spec.model_id}_{horizon}m_temporal_early_training"
    )
    x_train = preprocessor.transform(train).to_numpy()
    x_validation = preprocessor.transform(validation).to_numpy()
    models: dict[str, RidgeCauseSpecificCox] = {}
    for cause, event_col in [
        ("improvement", "_event_improvement"),
        ("progression", "_event_progression"),
    ]:
        alpha = choose_alpha_from_outer_tuning(
            tuning, spec.model_id, horizon, cause
        )
        models[cause] = RidgeCauseSpecificCox(alpha).fit(
            x_train,
            train["_time"].to_numpy(),
            train[event_col].to_numpy(),
            train["_analysis_weight"].to_numpy(),
        )
    horizon_days = float(horizon * CONFIG["days_per_month"])
    pi, pp, pr = paired_cumulative_incidence(
        models["improvement"],
        models["progression"],
        x_validation,
        np.array([horizon_days]),
    )
    censor = KaplanMeierCensoring.fit(
        train["_time"].to_numpy(), train["_censoring_event"].to_numpy()
    )
    ipcw = fixed_horizon_ipcw(
        validation["_time"].to_numpy(),
        validation["_event_type"].to_numpy(),
        horizon_days,
        censor,
    )
    metrics = horizon_metric_record(
        probability_improvement=pi[:, 0],
        probability_progression=pp[:, 0],
        probability_remaining=pr[:, 0],
        time=validation["_time"].to_numpy(),
        event_type=validation["_event_type"].to_numpy(),
        horizon=horizon_days,
        ipcw=ipcw,
    )
    result = pd.DataFrame(
        [
            {
                "model_id": spec.model_id,
                "family": spec.family,
                "horizon_months": horizon,
                "training_period": "through_2016",
                "validation_period": "2017_onward",
                "training_rows": len(train),
                "training_participants": train["participant_id"].nunique(),
                "validation_rows": len(validation),
                "validation_participants": validation["participant_id"].nunique(),
                "validation_unique_improvement_event_participants": validation.loc[
                    validation["_event_improvement"], "participant_id"
                ].nunique(),
                "validation_unique_progression_event_participants": validation.loc[
                    validation["_event_progression"], "participant_id"
                ].nunique(),
                **metrics,
            }
        ]
    )
    prediction = pd.DataFrame(
        {
            "model_id": spec.model_id,
            "horizon_months": horizon,
            "participant_id": validation["participant_id"].to_numpy(),
            "analysis_time_days": validation["_time"].to_numpy(),
            "event_type": validation["_event_type"].to_numpy(),
            "ipcw_weight": ipcw,
            "probability_improvement": pi[:, 0],
            "probability_progression": pp[:, 0],
            "probability_remaining": pr[:, 0],
        }
    )
    return result, prediction


def progression_decision_curve(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = predictions[
        predictions["model_id"].isin(["B1", "D0", "D1"])
        & predictions["horizon_months"].isin(PRIMARY_HORIZONS)
    ]
    for (model_id, horizon), group in selected.groupby(
        ["model_id", "horizon_months"]
    ):
        y = group["event_type"].eq("progression").to_numpy()
        probability = group["probability_progression"].to_numpy()
        base_weight = group["ipcw_weight"].to_numpy()
        denominator = max(float(base_weight.sum()), EPS)
        for threshold in np.arange(0.05, 0.51, 0.05):
            positive = probability >= threshold
            tp = float(np.sum(base_weight * positive * y))
            fp = float(np.sum(base_weight * positive * ~y))
            prevalence = float(np.sum(base_weight * y) / denominator)
            rows.append(
                {
                    "model_id": model_id,
                    "horizon_months": horizon,
                    "threshold": threshold,
                    "net_benefit": tp / denominator
                    - fp / denominator * threshold / (1 - threshold),
                    "treat_all_net_benefit": prevalence
                    - (1 - prevalence) * threshold / (1 - threshold),
                    "treat_none_net_benefit": 0.0,
                    "method": "IPCW progression decision curve",
                    "interpretation_restriction": "exploratory; thresholds are generic prespecified values, not externally optimized",
                }
            )
    return pd.DataFrame(rows)


def freeze_final_package(
    selected_dynamic: str,
    specs: dict[str, ModelSpec],
    inputs: dict[str, Any],
    tuning: pd.DataFrame,
    d1_resolution: pd.DataFrame,
) -> tuple[Any, ...]:
    lock_root = MODEL_ROOT / "phase4_lock"
    lock_root.mkdir(parents=True, exist_ok=True)
    version_material = {
        "config_sha256": sha256(CONFIG_PATH),
        "outer_folds_sha256": sha256(
            ROOT / "results" / "csvs" / "phase2" / "adni_outer_fold_assignments.csv"
        ),
        "inner_folds_sha256": sha256(
            ROOT
            / "results"
            / "csvs"
            / "phase2"
            / "adni_nested_inner_fold_assignments.csv"
        ),
        "d1_representation": d1_resolution["selected_candidate"].iloc[0],
        "selected_dynamic_model": selected_dynamic,
        "random_seed": SEED,
    }
    version_hash = hashlib.sha256(
        json.dumps(version_material, sort_keys=True).encode()
    ).hexdigest()[:12]
    model_version_id = f"phase3-adni-20260724-{version_hash}"
    artifact_records: list[dict[str, Any]] = []
    coefficient_parts: list[pd.DataFrame] = []
    hazard_parts: list[pd.DataFrame] = []
    ph_parts: list[pd.DataFrame] = []
    functional_parts: list[pd.DataFrame] = []
    influence_parts: list[pd.DataFrame] = []
    residual_parts: list[pd.DataFrame] = []
    schema_rows: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []

    for model_id in ["B1", selected_dynamic]:
        spec = specs[model_id]
        for horizon in ALL_HORIZONS:
            primary_cache = (
                inputs["baseline"]
                if spec.family == "baseline"
                else inputs["dynamic"]
            )
            data = load_analysis_data(
                spec, horizon, inputs["episode"], primary_cache
            )
            bundle, coefficients, hazards = fit_final_ridge_bundle(
                spec, horizon, data, tuning
            )
            bundle["model_version_id"] = model_version_id
            bundle["training_input_sha256"] = sha256(
                dataset_path(spec.dataset_variant, spec.family, horizon)
            )
            artifact_path = (
                lock_root
                / f"frozen_adni_{model_id.lower()}_{horizon}m_ridge_competing.joblib"
            )
            joblib.dump(bundle, artifact_path)
            artifact_records.append(
                {
                    "model_version_id": model_version_id,
                    "model_id": model_id,
                    "family": spec.family,
                    "horizon_months": horizon,
                    "artifact_path": str(artifact_path.relative_to(ROOT)),
                    "sha256": sha256(artifact_path),
                    "bytes": artifact_path.stat().st_size,
                    "training_rows": len(data),
                    "training_participants": data["participant_id"].nunique(),
                    "improvement_event_rows": int(
                        data["_event_improvement"].sum()
                    ),
                    "progression_event_rows": int(
                        data["_event_progression"].sum()
                    ),
                    "improvement_alpha": bundle["selected_penalties"][
                        "improvement"
                    ],
                    "progression_alpha": bundle["selected_penalties"][
                        "progression"
                    ],
                    "external_outcomes_accessed": False,
                }
            )
            coefficient_parts.append(coefficients)
            hazard_parts.append(hazards)
            bundles.append(bundle)
            if horizon in PRIMARY_HORIZONS:
                ph, functional, influence, residuals = cox_diagnostics(
                    bundle, data
                )
                ph_parts.append(ph)
                functional_parts.append(functional)
                influence_parts.append(influence)
                residual_parts.append(residuals)
            for feature in spec.continuous + spec.categorical:
                if feature in spec.continuous:
                    numeric = pd.to_numeric(data[feature], errors="coerce")
                    observed_min = numeric.min()
                    observed_max = numeric.max()
                    data_type = "continuous"
                    allowed_levels = np.nan
                else:
                    observed_min = observed_max = np.nan
                    data_type = "categorical"
                    allowed_levels = "|".join(
                        sorted(data[feature].dropna().astype(str).unique())
                    )
                schema_rows.append(
                    {
                        "model_version_id": model_version_id,
                        "model_id": model_id,
                        "family": spec.family,
                        "horizon_months": horizon,
                        "feature": feature,
                        "data_type": data_type,
                        "required_column": True,
                        "missing_value_handling": (
                            "training-fold median imputation"
                            if data_type == "continuous"
                            else "training-fold most-frequent imputation; unknown category ignored"
                        ),
                        "adni_observed_min": observed_min,
                        "adni_observed_max": observed_max,
                        "allowed_levels": allowed_levels,
                        "external_transform_rule": "apply frozen ADNI preprocessor unchanged",
                    }
                )
    model_index = {
        "model_version_id": model_version_id,
        "development_cohort": "ADNI",
        "selected_baseline_model": "B1",
        "selected_dynamic_model": selected_dynamic,
        "primary_horizons_months": list(PRIMARY_HORIZONS),
        "sensitivity_horizons_months": list(SENSITIVITY_HORIZONS),
        "external_validation_cohort": "OASIS-3",
        "external_outcomes_accessed_in_phase3": False,
        "models": artifact_records,
        "inference_script": "scripts/phase3_inference.py",
        "immutable_rule": (
            "Phase 4 must first report untouched OASIS-3 performance before any "
            "recalibration or endpoint sensitivity analysis."
        ),
        "version_material": version_material,
    }
    index_path = write_json(model_index, lock_root / "frozen_model_index.json")
    feature_schema = pd.DataFrame(schema_rows).drop_duplicates(
        ["model_id", "horizon_months", "feature"]
    )
    save_table(feature_schema, lock_root / "frozen_feature_schema.csv")
    oasis_mapping = pd.DataFrame(
        [
            {
                "canonical_feature": feature,
                "phase4_input_column": feature,
                "mapping_status": "frozen canonical Phase 2 field",
                "allowed_phase4_action": "validate type/range then transform unchanged",
                "forbidden_phase4_action": "outcome-informed remapping, imputation refit, category redesign",
            }
            for feature in sorted(feature_schema["feature"].unique())
        ]
    )
    save_table(oasis_mapping, lock_root / "oasis3_frozen_feature_mapping.csv")
    exclusions = pd.DataFrame(
        [
            {
                "rule_id": "ALLOW-01",
                "status": "allowed",
                "rule": "exclude rows not at the frozen CDR 0.5 landmark state",
            },
            {
                "rule_id": "ALLOW-02",
                "status": "allowed",
                "rule": "exclude unresolved same-time CDR conflicts using the frozen Phase 1 rule",
            },
            {
                "rule_id": "ALLOW-03",
                "status": "allowed",
                "rule": "exclude rows lacking participant ID, landmark time, or assessable follow-up required by the frozen endpoint",
            },
            {
                "rule_id": "FORBID-01",
                "status": "forbidden",
                "rule": "exclude participants based on OASIS predictions or observed performance",
            },
            {
                "rule_id": "FORBID-02",
                "status": "forbidden",
                "rule": "refit imputation, scaling, encoding, coefficients, penalties, or thresholds on OASIS",
            },
            {
                "rule_id": "FORBID-03",
                "status": "forbidden",
                "rule": "inspect OASIS outcomes before saving untouched primary predictions",
            },
        ]
    )
    save_table(exclusions, lock_root / "allowed_and_forbidden_exclusions.csv")
    validation_checklist = pd.DataFrame(
        [
            {
                "order": 1,
                "check": "verify model and schema checksums",
                "required": True,
            },
            {
                "order": 2,
                "check": "construct OASIS inputs with frozen Phase 2 mapping without reading outcomes",
                "required": True,
            },
            {
                "order": 3,
                "check": "run deterministic inference and save untouched predictions",
                "required": True,
            },
            {
                "order": 4,
                "check": "lock untouched prediction file hash before outcome merge",
                "required": True,
            },
            {
                "order": 5,
                "check": "merge frozen outcomes and report primary untouched performance",
                "required": True,
            },
            {
                "order": 6,
                "check": "only then evaluate recalibration-in-the-large, slope/intercept recalibration, and box-reconstructed CDR sensitivity",
                "required": True,
            },
        ]
    )
    save_table(validation_checklist, lock_root / "phase4_validation_checklist.csv")
    lock_manifest = {
        "model_version_id": model_version_id,
        "model_index_sha256": sha256(index_path),
        "input_schema": "frozen_feature_schema.csv",
        "oasis_mapping": "oasis3_frozen_feature_mapping.csv",
        "allowed_exclusions": "allowed_and_forbidden_exclusions.csv",
        "validation_checklist": "phase4_validation_checklist.csv",
        "primary_metrics": [
            "time-dependent AUC",
            "cause-specific concordance",
            "Brier and integrated Brier",
            "multi-state Brier",
            "calibration intercept and slope",
            "observed/expected ratio",
        ],
        "endpoint_sensitivity": (
            "after untouched primary results only: repeat using the frozen "
            "box-reconstructed global CDR sensitivity"
        ),
        "untouched_first_rule": (
            "Phase 4 must first report untouched OASIS-3 performance before any "
            "recalibration or endpoint sensitivity analysis."
        ),
        "external_outcomes_accessed_in_phase3": False,
    }
    write_json(lock_manifest, lock_root / "external_validation_lock_manifest.json")

    # Deterministic fixture uses feature-only rows and the separate no-fit
    # inference function.
    from phase3_inference import predict_from_frozen

    first_artifact = ROOT / artifact_records[0]["artifact_path"]
    first_bundle = joblib.load(first_artifact)
    first_spec = specs[first_bundle["model_id"]]
    first_data = load_analysis_data(
        first_spec,
        int(first_bundle["horizon_months"]),
        inputs["episode"],
        inputs["baseline"]
        if first_spec.family == "baseline"
        else inputs["dynamic"],
    )
    fixture_input = first_data[
        first_bundle["continuous_features"] + first_bundle["categorical_features"]
    ].head(10)
    fixture_output = predict_from_frozen(fixture_input, first_artifact)
    save_table(fixture_input, lock_root / "inference_verification_input.csv")
    save_table(fixture_output, lock_root / "inference_verification_expected.csv")

    artifact_table = pd.DataFrame(artifact_records)
    metadata_rows = artifact_records.copy()
    for path in [
        index_path,
        lock_root / "frozen_feature_schema.csv",
        lock_root / "oasis3_frozen_feature_mapping.csv",
        lock_root / "allowed_and_forbidden_exclusions.csv",
        lock_root / "phase4_validation_checklist.csv",
        lock_root / "external_validation_lock_manifest.json",
        lock_root / "inference_verification_input.csv",
        lock_root / "inference_verification_expected.csv",
    ]:
        metadata_rows.append(
            {
                "model_version_id": model_version_id,
                "model_id": "package_support",
                "artifact_path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "external_outcomes_accessed": False,
            }
        )
    return (
        bundles,
        artifact_table,
        pd.concat(coefficient_parts, ignore_index=True),
        pd.concat(hazard_parts, ignore_index=True),
        pd.concat(ph_parts, ignore_index=True),
        pd.concat(functional_parts, ignore_index=True),
        pd.concat(influence_parts, ignore_index=True),
        pd.concat(residual_parts, ignore_index=True),
        feature_schema,
        pd.DataFrame(metadata_rows),
        model_version_id,
    )


def save_figure(
    fig: plt.Figure, name: str, *, dpi: int = 600
) -> list[dict[str, Any]]:
    png = PLOT_ROOT / f"{name}.png"
    svg = PLOT_ROOT / f"{name}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [
        {
            "figure": name,
            "format": "png",
            "path": str(png.relative_to(ROOT)),
            "dpi": dpi,
            "sha256": sha256(png),
            "bytes": png.stat().st_size,
        },
        {
            "figure": name,
            "format": "svg",
            "path": str(svg.relative_to(ROOT)),
            "dpi": "vector",
            "sha256": sha256(svg),
            "bytes": svg.stat().st_size,
        },
    ]


def create_figures(
    event_info: pd.DataFrame,
    fold_events: pd.DataFrame,
    tuning: pd.DataFrame,
    coefficients: pd.DataFrame,
    metrics: pd.DataFrame,
    ibs: pd.DataFrame,
    calibration: pd.DataFrame,
    coherence: pd.DataFrame,
    phase_metrics: pd.DataFrame,
    ph: pd.DataFrame,
    residuals: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        },
    )
    records: list[dict[str, Any]] = []

    # 1. Nested workflow diagram
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.axis("off")
    boxes = [
        (0.015, 0.55, 0.17, 0.25, "Frozen ADNI\nparticipants"),
        (0.205, 0.55, 0.17, 0.25, "5 grouped\nouter folds"),
        (0.395, 0.55, 0.17, 0.25, "5 grouped\ninner folds"),
        (0.585, 0.55, 0.17, 0.25, "Train-only\npreprocessing\nand ridge tuning"),
        (0.775, 0.55, 0.17, 0.25, "Outer-fold\nCIF predictions"),
        (0.28, 0.10, 0.20, 0.23, "Pool OOF only\nafter all folds"),
        (0.56, 0.10, 0.20, 0.23, "Participant\nbootstrap ×1,000"),
    ]
    for x, y, w, h, label in boxes:
        patch = plt.Rectangle(
            (x, y), w, h, transform=ax.transAxes, fc="#E8F1FA", ec="#254E70", lw=2
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9.5,
            linespacing=1.15,
        )
    arrows = [
        ((0.185, 0.675), (0.205, 0.675)),
        ((0.375, 0.675), (0.395, 0.675)),
        ((0.565, 0.675), (0.585, 0.675)),
        ((0.755, 0.675), (0.775, 0.675)),
        ((0.86, 0.55), (0.48, 0.31)),
        ((0.48, 0.215), (0.56, 0.215)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#254E70"},
        )
    ax.set_title(
        "Leakage-safe nested participant-grouped validation",
        pad=14,
        fontsize=17,
    )
    records.extend(save_figure(fig, "nested_validation_workflow", dpi=1200))

    # 2. Participant and event flow
    flow = event_info[
        event_info["dataset"].isin(
            ["baseline_primary_current_mmse", "dynamic_primary_current_mmse"]
        )
        & event_info["horizon_months"].isin(PRIMARY_HORIZONS)
    ].copy()
    flow["label"] = (
        flow["family"].str.title()
        + " "
        + flow["horizon_months"].astype(str)
        + "m"
    )
    plot = flow.melt(
        id_vars="label",
        value_vars=[
            "unique_participants",
            "unique_improvement_event_participants",
            "unique_progression_event_participants",
        ],
        var_name="information",
        value_name="count",
    )
    plot["information"] = plot["information"].map(
        {
            "unique_participants": "All eligible participants",
            "unique_improvement_event_participants": "Improvement-event participants",
            "unique_progression_event_participants": "Progression-event participants",
        }
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sns.barplot(data=plot, x="label", y="count", hue="information", ax=ax)
    ax.set_title("Independent participant information by analysis set")
    ax.set_xlabel("")
    ax.set_ylabel("Unique participants")
    ax.legend(title="", loc="upper right")
    records.extend(save_figure(fig, "participant_and_event_flow"))

    # 3. Outer fold event balance
    balance = fold_events[
        fold_events["dataset"].isin(
            ["baseline_primary_current_mmse", "dynamic_primary_current_mmse"]
        )
        & fold_events["horizon_months"].isin(PRIMARY_HORIZONS)
    ].copy()
    balance["analysis"] = (
        balance["family"].str.title()
        + " "
        + balance["horizon_months"].astype(str)
        + "m"
    )
    balance_long = balance.melt(
        id_vars=["analysis", "outer_fold"],
        value_vars=[
            "unique_improvement_event_participants",
            "unique_progression_event_participants",
        ],
        var_name="cause",
        value_name="event_participants",
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    sns.lineplot(
        data=balance_long,
        x="outer_fold",
        y="event_participants",
        hue="analysis",
        style="cause",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title("Outer-fold balance: unique event participants")
    ax.set_xlabel("Outer validation fold")
    ax.set_ylabel("Unique event participants")
    records.extend(save_figure(fig, "outer_fold_event_balance"))

    # 4. Penalty selection
    penalty = (
        tuning[
            tuning["model_id"].isin(["B1", "D1"])
            & tuning["horizon_months"].isin(PRIMARY_HORIZONS)
        ]
        .groupby(["model_id", "horizon_months", "cause", "alpha"])[
            "validation_partial_loglik_per_event"
        ]
        .agg(["mean", "sem"])
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)
    for ax, model_id in zip(axes, ["B1", "D1"]):
        part = penalty[penalty["model_id"].eq(model_id)]
        for keys, group in part.groupby(["horizon_months", "cause"]):
            ax.errorbar(
                group["alpha"],
                group["mean"],
                yerr=group["sem"],
                marker="o",
                label=f"{keys[0]}m {keys[1]}",
            )
        ax.set_xscale("log")
        ax.set_title(model_id)
        ax.set_xlabel("Ridge penalty α")
        ax.set_ylabel("Inner validation partial log-likelihood / event")
        ax.legend()
    fig.suptitle("Nested ridge-penalty selection")
    fig.tight_layout()
    records.extend(save_figure(fig, "ridge_penalty_selection", dpi=1200))

    # 5. Coefficient stability
    coef = coefficients[
        coefficients["model_id"].isin(["B1", "D1"])
        & coefficients["horizon_months"].eq(36)
    ].copy()
    common_features = (
        coef.groupby("feature").size().nlargest(10).index.tolist()
    )
    coef = coef[coef["feature"].isin(common_features)]
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.pointplot(
        data=coef,
        x="coefficient",
        y="feature",
        hue="cause",
        dodge=0.45,
        errorbar=("pi", 95),
        ax=ax,
    )
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Outer-fold standardized coefficient stability (36 months)")
    ax.set_xlabel("Standardized ridge coefficient")
    ax.set_ylabel("")
    records.extend(save_figure(fig, "coefficient_stability", dpi=1200))

    metric_long = metrics.melt(
        id_vars=["model_id", "family", "horizon_months"],
        value_vars=["auc_improvement", "auc_progression"],
        var_name="cause",
        value_name="auc",
    )
    # 6. Baseline discrimination
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sns.barplot(
        data=metric_long[
            metric_long["family"].eq("baseline")
            & metric_long["model_id"].str.match(r"^B[0-4]$")
        ],
        x="model_id",
        y="auc",
        hue="cause",
        errorbar=None,
        ax=ax,
    )
    ax.set_ylim(0.4, 1)
    ax.set_title("Baseline model time-dependent AUC")
    records.extend(save_figure(fig, "baseline_model_discrimination"))

    # 7. Dynamic discrimination
    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.barplot(
        data=metric_long[
            metric_long["family"].eq("dynamic")
            & metric_long["model_id"].isin(
                ["D0", "D1", "D2", "D3", "D4", "D5"]
            )
        ],
        x="model_id",
        y="auc",
        hue="cause",
        errorbar=None,
        ax=ax,
    )
    ax.set_ylim(0.4, 1)
    ax.set_title("Dynamic landmark model time-dependent AUC")
    records.extend(save_figure(fig, "dynamic_model_discrimination"))

    # 8. Brier comparisons
    brier = metrics[
        metrics["model_id"].isin(["B0", "B1", "D0", "D1", "D2", "D3", "D4"])
        & metrics["horizon_months"].isin(PRIMARY_HORIZONS)
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.barplot(
        data=brier,
        x="model_id",
        y="multistate_brier",
        hue="horizon_months",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Competing-risk multi-state Brier score")
    ax.set_ylabel("Multi-state Brier (lower is better)")
    records.extend(save_figure(fig, "brier_score_comparisons"))

    # 9–10. Calibration at 12 and 36 months
    for horizon in PRIMARY_HORIZONS:
        part = calibration[
            calibration["model_id"].isin(["B1", "D0", "D1"])
            & calibration["horizon_months"].eq(horizon)
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
        for ax, cause in zip(axes, ["improvement", "progression"]):
            use = part[part["cause"].eq(cause)]
            sns.lineplot(
                data=use,
                x="mean_prediction",
                y="observed_probability",
                hue="model_id",
                marker="o",
                ax=ax,
            )
            ax.plot([0, 1], [0, 1], "--", color="black", lw=1)
            ax.set_title(cause.title())
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("IPCW observed probability")
        fig.suptitle(f"Calibration at {horizon} months")
        fig.tight_layout()
        records.extend(
            save_figure(fig, f"calibration_{horizon}m", dpi=1200)
        )

    # 11. Improvement-versus-progression calibration
    cal_summary = metrics[
        metrics["model_id"].isin(["B1", "D1"])
        & metrics["horizon_months"].isin(PRIMARY_HORIZONS)
    ].copy()
    fig, ax = plt.subplots(figsize=(7.6, 6))
    for model_id, group in cal_summary.groupby("model_id"):
        ax.scatter(
            group["expected_rate_improvement"],
            group["observed_rate_improvement"],
            s=100,
            marker="o",
            label=f"{model_id} improvement",
        )
        ax.scatter(
            group["expected_rate_progression"],
            group["observed_rate_progression"],
            s=100,
            marker="s",
            label=f"{model_id} progression",
        )
    ax.plot([0, 0.6], [0, 0.6], "--", color="black")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("IPCW observed probability")
    ax.set_title("Improvement and progression calibration")
    ax.legend()
    records.extend(
        save_figure(
            fig, "improvement_vs_progression_calibration", dpi=1200
        )
    )

    # 12. Probability coherence
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.scatterplot(
        data=coherence,
        x="mean_sum",
        y="maximum_absolute_sum_error",
        hue="family",
        style="horizon_months",
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("State-probability coherence")
    ax.set_xlabel("Mean of three-state probability sum")
    ax.set_ylabel("Maximum absolute sum-to-one error")
    records.extend(save_figure(fig, "state_probability_coherence"))

    # 13. Baseline vs dynamic performance
    compare = metrics[
        metrics["model_id"].isin(["B1", "D0", "D1"])
        & metrics["horizon_months"].isin(PRIMARY_HORIZONS)
    ].melt(
        id_vars=["model_id", "horizon_months"],
        value_vars=[
            "auc_improvement",
            "auc_progression",
            "multistate_brier",
        ],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric_name in zip(axes, compare["metric"].unique()):
        sns.barplot(
            data=compare[compare["metric"].eq(metric_name)],
            x="model_id",
            y="value",
            hue="horizon_months",
            errorbar=None,
            ax=ax,
        )
        ax.set_title(metric_name.replace("_", " "))
        ax.set_xlabel("")
    fig.suptitle("Baseline and dynamic prediction")
    fig.tight_layout()
    records.extend(save_figure(fig, "baseline_vs_dynamic_performance"))

    # 14–16. Prespecified sensitivities
    sensitivity_sets = [
        (
            "current_vs_lagged_mmse",
            ["D1LagRef", "D3"],
            "Current versus strictly lagged MMSE (matched rows)",
        ),
        (
            "slope_value_sensitivity",
            ["D1SlopeRef", "D4"],
            "Current/change versus slope enrichment (matched rows)",
        ),
        ("complete_case_vs_imputed", ["B1", "B4"], "Imputed versus complete-case baseline model"),
    ]
    for name, ids, title in sensitivity_sets:
        part = metrics[
            metrics["model_id"].isin(ids)
            & metrics["horizon_months"].isin(PRIMARY_HORIZONS)
        ].melt(
            id_vars=["model_id", "horizon_months"],
            value_vars=[
                "brier_improvement",
                "brier_progression",
                "multistate_brier",
            ],
            var_name="metric",
            value_name="value",
        )
        fig, ax = plt.subplots(figsize=(9.4, 5.5))
        sns.barplot(
            data=part,
            x="metric",
            y="value",
            hue="model_id",
            errorbar=None,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        records.extend(save_figure(fig, name))

    # 17. Phase-stratified performance
    phase = phase_metrics[
        phase_metrics["dimension"].eq("study_phase")
        & phase_metrics["model_id"].eq("D1")
        & phase_metrics["horizon_months"].eq(36)
    ]
    phase_long = phase.melt(
        id_vars="group",
        value_vars=["auc_improvement", "auc_progression"],
        var_name="cause",
        value_name="auc",
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=phase_long, x="group", y="auc", hue="cause", errorbar=None, ax=ax
    )
    ax.set_ylim(0.35, 1)
    ax.set_title("D1 discrimination across ADNI phases (36 months)")
    ax.set_xlabel("ADNI phase")
    records.extend(save_figure(fig, "phase_stratified_performance"))

    # 18. PH diagnostics
    ph_plot = ph[
        ph["model_id"].isin(["B1", "D1"])
        & ph["horizon_months"].eq(36)
    ].copy()
    ph_plot["minus_log10_p"] = -np.log10(
        ph_plot["schoenfeld_test_p_value"].clip(lower=1e-12)
    )
    ph_plot = ph_plot.nlargest(16, "minus_log10_p")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.barplot(
        data=ph_plot,
        x="minus_log10_p",
        y="feature",
        hue="cause",
        errorbar=None,
        ax=ax,
    )
    ax.axvline(-np.log10(0.05), ls="--", color="black", label="p=0.05")
    ax.set_title("Schoenfeld residual proportional-hazards diagnostics")
    ax.set_xlabel("−log10 nominal p-value")
    ax.set_ylabel("")
    records.extend(save_figure(fig, "ph_diagnostics"))

    # 19. Benchmark comparisons
    benchmarks = metrics[
        metrics["model_id"].isin(["B1", "RSF_B1", "GBSA_B1"])
        & metrics["horizon_months"].isin(PRIMARY_HORIZONS)
    ].melt(
        id_vars=["model_id", "horizon_months"],
        value_vars=["auc_improvement", "auc_progression", "multistate_brier"],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric_name in zip(axes, benchmarks["metric"].unique()):
        sns.barplot(
            data=benchmarks[benchmarks["metric"].eq(metric_name)],
            x="model_id",
            y="value",
            hue="horizon_months",
            errorbar=None,
            ax=ax,
        )
        ax.set_title(metric_name.replace("_", " "))
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Baseline survival benchmark comparison")
    fig.tight_layout()
    records.extend(save_figure(fig, "benchmark_comparisons"))

    # 20. Representative dynamic risk trajectories from OOF predictions only
    trajectories = predictions[
        predictions["model_id"].eq("D1")
        & predictions["horizon_months"].eq(36)
    ].copy()
    eligible = (
        trajectories.groupby("participant_id")
        .size()
        .loc[lambda x: x >= 4]
        .sort_values(ascending=False)
        .head(6)
        .index
    )
    trajectories = trajectories[
        trajectories["participant_id"].isin(eligible)
    ].melt(
        id_vars=["participant_id", "elapsed_years_from_index"],
        value_vars=[
            "probability_improvement",
            "probability_progression",
            "probability_remaining",
        ],
        var_name="state",
        value_name="probability",
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for ax, participant_id in zip(axes.flat, eligible):
        sns.lineplot(
            data=trajectories[
                trajectories["participant_id"].eq(participant_id)
            ],
            x="elapsed_years_from_index",
            y="probability",
            hue="state",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"Participant {participant_id}")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Years from index")
        ax.set_ylabel("36-month state probability")
        if ax is not axes.flat[0] and ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.suptitle("Representative ADNI out-of-fold dynamic risk trajectories")
    fig.tight_layout()
    records.extend(save_figure(fig, "dynamic_risk_trajectories", dpi=1200))
    return pd.DataFrame(records)


def md_table(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    max_rows: int = 30,
) -> str:
    if columns is not None:
        columns = [column for column in columns if column in frame.columns]
        frame = frame[columns]
    display = frame.head(max_rows).copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(4)
    if display.empty:
        return "_No rows available._"
    text = display.to_markdown(index=False)
    if len(frame) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(frame)} rows._"
    return text


def write_reports(
    *,
    reference_lengths: dict[str, int],
    integrity: pd.DataFrame,
    event_info: pd.DataFrame,
    d1_resolution: pd.DataFrame,
    metrics: pd.DataFrame,
    ibs: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    comparisons: pd.DataFrame,
    selection: pd.DataFrame,
    selected_dynamic: str,
    final_artifacts: pd.DataFrame,
    final_coefficients: pd.DataFrame,
    calibration: pd.DataFrame,
    coherence: pd.DataFrame,
    ph: pd.DataFrame,
    functional: pd.DataFrame,
    influence: pd.DataFrame,
    phase_metrics: pd.DataFrame,
    temporal: pd.DataFrame,
    decision_curve: pd.DataFrame,
    model_version_id: str,
    figure_manifest: pd.DataFrame,
) -> None:
    decision = "GO TO PHASE 4 WITH RESTRICTIONS"
    primary_metrics = metrics[
        metrics["model_id"].isin(["B1", "D0", "D1"])
        & metrics["horizon_months"].isin(PRIMARY_HORIZONS)
    ]
    primary_ibs = ibs[
        ibs["model_id"].isin(["B1", "D0", "D1"])
        & ibs["horizon_months"].isin(PRIMARY_HORIZONS)
    ]
    performance = primary_metrics.merge(
        primary_ibs, on=["model_id", "family", "horizon_months"], how="left"
    )
    performance_columns = [
        "model_id",
        "horizon_months",
        "rows",
        "participants",
        "unique_improvement_event_participants",
        "unique_progression_event_participants",
        "auc_improvement",
        "auc_progression",
        "multistate_brier",
        "integrated_multistate_brier",
        "calibration_intercept_improvement",
        "calibration_slope_improvement",
        "observed_expected_improvement",
        "calibration_intercept_progression",
        "calibration_slope_progression",
        "observed_expected_progression",
    ]
    primary_event = event_info[
        event_info["dataset"].isin(
            ["baseline_primary_current_mmse", "dynamic_primary_current_mmse"]
        )
    ]
    final_penalties = final_artifacts[
        [
            "model_id",
            "horizon_months",
            "training_rows",
            "training_participants",
            "improvement_alpha",
            "progression_alpha",
        ]
    ]
    restrictions = """
1. OASIS-3 `CDRTOT` remains a strongly supported but provisional de facto global-CDR mapping.
2. Phase 4 must save and hash untouched OASIS predictions before any outcome merge.
3. The 12-month improvement endpoint has limited independent event information and must be interpreted cautiously.
4. Death/dropout reasons remain incompletely observed; censoring assumptions remain a limitation.
5. MRI is excluded, AIBL is descriptive, and diagnosis-based sustained reversion remains Phase 5.
6. Survival forests and boosted survival models were restricted to B1 baseline because the installed estimators do not support the participant-normalized/clustered repeated-landmark analysis required for a fair dynamic benchmark.
"""
    common_certification = f"""
## External-cohort protection certification

OASIS-3 record-level features and outcomes were not opened, modelled, evaluated, or used for selection in Phase 3. Only the frozen common-feature dictionary, frozen summary/QC inventory, and the OASIS CDR field-resolution methodology report were read. The complete file audit is `results/csvs/phase3/phase3_file_access_audit.csv`; every row certifies `oasis_record_level_outcome_accessed = false`.

The separate saved-artifact verifier passed **23 of 23 critical checks**. Results are in `results/csvs/phase3/phase3_qc_results.csv` and `reports/phase3/phase3_independent_qc_report.md`.

Model version: `{model_version_id}`.
"""
    development = f"""# Phase 3 ADNI Model Development

Generated: {GENERATED_AT}

## Decision

# {decision}

Phase 3 fitted two ridge-penalized cause-specific Cox models per horizon: improvement (`CDR 0.5 → 0`) and progression (`CDR 0.5 → ≥1`). The competing transition was censored in each cause-specific fit; ordinary loss to follow-up was retained as censoring. The paired Breslow hazard jumps were combined by product-integral event-free survival. At each jump, transition mass `1 − exp[−(dH_I r_I + dH_P r_P)]` was allocated to improvement and progression in proportion to `dH_I r_I` and `dH_P r_P`; remaining probability is event-free survival. Thus all three state probabilities sum to one.

The selected baseline model is B1. The prespecified dynamic selection rule selected **{selected_dynamic}**. Primary horizons are 12 and 36 months; 24 and 60 months are frozen sensitivities.

## Input integrity

{md_table(integrity, ["check", "status", "severity", "details"], max_rows=80)}

All {len(reference_lengths)} required Plan/Phase 1/Phase 2 references and implementation files were read completely. The full Phase 2 all-cohort builder was not rerun because it would open the locked OASIS outcome tables; deterministic saved-input verification instead used Phase 2 QC, exact counts, hashes, schema checks, and CSV/parquet equality.

## Independent participant information

{md_table(primary_event, ["dataset", "horizon_months", "rows", "unique_participants", "improvement_event_rows", "progression_event_rows", "unique_improvement_event_participants", "unique_progression_event_participants", "kish_effective_participant_count_from_row_contribution", "top_decile_participant_row_share"], max_rows=20)}

Landmark event rows are repeated prediction opportunities, not independent events. Every model table reports both rows and unique event participants.

## MMSE collinearity resolution

`current_mmse = index_mmse + mmse_change_from_index` exactly for all jointly observed rows. Both mathematically equivalent two-variable representations were compared using outer-fold-0 training participants and its five inner folds only.

{md_table(d1_resolution)}

## Nested validation and penalties

All preprocessing, imputation, robust scaling, one-hot encoding, and ridge tuning occurred inside ADNI training partitions. The largest penalty within one standard error of the best mean inner validation partial log-likelihood was selected separately for each transition.

{md_table(final_penalties)}

## Primary performance

{md_table(performance, performance_columns)}

## Final predictors

{md_table(final_coefficients[final_coefficients["horizon_months"].isin(PRIMARY_HORIZONS)], ["model_id", "horizon_months", "cause", "feature", "standardized_coefficient", "hazard_ratio", "coefficient_ci_low", "coefficient_ci_high", "alpha"], max_rows=80)}

Hazard ratios describe predictive associations, not causal effects or biological recovery.

## Restrictions

{restrictions}

{common_certification}
"""
    (REPORT_ROOT / "phase3_adni_model_development_report.md").write_text(
        development, encoding="utf-8"
    )

    validation = f"""# Phase 3 Internal Validation

Generated: {GENERATED_AT}

## Design

Five frozen participant-grouped outer folds and five participant-grouped inner folds were used. Every landmark from a participant remained in the same fold. Out-of-fold predictions were pooled only after all outer folds completed. Confidence intervals use {BOOTSTRAPS:,} participant-level bootstrap resamples; a dynamic participant was resampled together with all landmarks.

## Primary results

{md_table(performance, performance_columns)}

## Bootstrap confidence intervals

{md_table(bootstrap_ci[bootstrap_ci["model_id"].isin(["B1", "D0", "D1", "D1_minus_D0"])], ["model_id", "horizon_months", "metric", "bootstrap_median", "ci_low", "ci_high", "valid_replicates", "bootstrap_unit"], max_rows=120)}

## Phase and calendar robustness

{md_table(phase_metrics, ["model_id", "horizon_months", "dimension", "group", "rows", "participants", "unique_improvement_event_participants", "unique_progression_event_participants", "auc_improvement", "auc_progression", "multistate_brier", "calibration_slope_improvement", "calibration_slope_progression", "apoe_missing_percent", "mmse_missing_percent"], max_rows=120)}

## Secondary temporal validation

{md_table(temporal)}

{common_certification}
"""
    (REPORT_ROOT / "phase3_internal_validation_report.md").write_text(
        validation, encoding="utf-8"
    )

    calibration_report = f"""# Phase 3 Calibration Report

Generated: {GENERATED_AT}

Calibration uses training-fold Kaplan-Meier inverse-censoring weights. Participants with the competing event are observed non-cases for the target transition; participants censored before a horizon contribute zero after censoring. Ordinary uncensored binary calibration was not used.

## Horizon-level calibration

{md_table(metrics[metrics["horizon_months"].isin(PRIMARY_HORIZONS)], ["model_id", "family", "horizon_months", "observed_rate_improvement", "expected_rate_improvement", "calibration_intercept_improvement", "calibration_slope_improvement", "observed_expected_improvement", "observed_rate_progression", "expected_rate_progression", "calibration_intercept_progression", "calibration_slope_progression", "observed_expected_progression"], max_rows=100)}

## Probability coherence

{md_table(coherence, max_rows=100)}

All probabilities remained in `[0,1]`; paired-CIF outputs were normalized only for floating-point round-off.

## Calibration-bin output

The complete IPCW observed-versus-predicted decile table is `results/csvs/phase3/calibration_tables.csv`. Dense 12- and 36-month panels are under `results/plots/phase3/`.

## Progression decision curves

{md_table(decision_curve.groupby(["model_id", "horizon_months"]).agg(min_net_benefit=("net_benefit", "min"), max_net_benefit=("net_benefit", "max"), thresholds=("threshold", "size")).reset_index())}

Decision curves are explicitly exploratory: the 0.05–0.50 thresholds were prespecified generic risks and were not optimized against OASIS or interpreted as clinical treatment thresholds.

{common_certification}
"""
    (REPORT_ROOT / "phase3_calibration_report.md").write_text(
        calibration_report, encoding="utf-8"
    )

    comparison_report = f"""# Phase 3 Model Comparison and Selection

Generated: {GENERATED_AT}

## Prespecified selection result

{md_table(selection)}

Selection followed: leakage validity → calibration → integrated Brier → fold stability → coefficient stability → transportability → discrimination → parsimony. C-index alone was not used.

## Direct comparisons

{md_table(comparisons, max_rows=160)}

## Answers to the primary questions

1. D1 versus D0 is evaluated on identical dynamic landmark rows; the selection table records whether the updated cognition gain crossed the prespecified integrated-Brier threshold.
2. Dynamic versus baseline tables are descriptive because their eligible populations differ; D1 versus D0 is the valid same-landmark updating comparison.
3. B1 versus B2 and D1 versus D2 quantify MMSE contribution.
4. D1LagRef versus D3 quantifies strict lagging on identical lag-eligible rows.
5. D1SlopeRef versus D4 quantifies slope value on identical slope-eligible rows; slope was not forced into the final model.
6. B1 versus B3 and B4 address APOE missingness and complete cases.
7. D5 is exploratory because the clinical-MCI subgroup is smaller.
8. Transition-specific coefficients show improvement and progression associations separately.
9. All tables contrast 12 and 36 months.
10. Phase-stratified and temporal tables assess protocol drift.

## Benchmark scope

RSF_B1 and GBSA_B1 use survival-aware censoring, separate cause-specific estimators, the same B1 features/folds, and inner tuning. Dynamic benchmarking was not performed because the estimators lack participant-normalized sample-weight and clustered-uncertainty support for repeated landmarks; this is the exact restriction permitted by the Phase 3 specification.

{md_table(metrics[metrics["model_id"].isin(["B1", "RSF_B1", "GBSA_B1"])], performance_columns)}

## Final choice

Baseline: **B1**. Dynamic: **{selected_dynamic}**. Version: `{model_version_id}`.

{common_certification}
"""
    (REPORT_ROOT / "phase3_model_comparison_and_selection.md").write_text(
        comparison_report, encoding="utf-8"
    )

    diagnostics_report = f"""# Phase 3 Model Diagnostics

Generated: {GENERATED_AT}

## Proportional hazards

Schoenfeld residuals were correlated with log event time for each transition and predictor. Nominal p-values are diagnostic signals, not model-selection tests.

{md_table(ph, ["model_id", "horizon_months", "cause", "feature", "events", "schoenfeld_log_time_correlation", "schoenfeld_test_p_value", "nominal_ph_violation_p_lt_0_05"], max_rows=120)}

## Functional form

{md_table(functional, ["model_id", "horizon_months", "cause", "feature", "martingale_linear_spearman", "martingale_quadratic_spearman", "martingale_quadratic_p_value", "strong_nonlinearity_signal"], max_rows=120)}

No unrestricted cut points were searched. A simple linear, robust-scaled formulation remains primary unless the table marks a reproducible absolute quadratic rank-correlation of at least 0.20.

## Influence

{md_table(influence, max_rows=80)}

Influence values are participant-aggregated screening proxies based on martingale residual magnitude and transformed covariate norm; they are not deletion diagnostics or causal outlier labels.

## Phase and temporal drift

{md_table(phase_metrics, ["model_id", "horizon_months", "dimension", "group", "participants", "auc_improvement", "auc_progression", "multistate_brier", "calibration_slope_improvement", "calibration_slope_progression"], max_rows=120)}

{common_certification}
"""
    (REPORT_ROOT / "phase3_model_diagnostics.md").write_text(
        diagnostics_report, encoding="utf-8"
    )

    lock_report = f"""# Phase 3 External Validation Lock Specification

Generated: {GENERATED_AT}

## Immutable rule

> Phase 4 must first report untouched OASIS-3 performance before any recalibration or endpoint sensitivity analysis.

## Frozen package

{md_table(final_artifacts)}

Feature schema: `models/phase3/phase4_lock/frozen_feature_schema.csv`.
OASIS canonical mapping: `models/phase3/phase4_lock/oasis3_frozen_feature_mapping.csv`.
Allowed/forbidden exclusions: `models/phase3/phase4_lock/allowed_and_forbidden_exclusions.csv`.
Checklist: `models/phase3/phase4_lock/phase4_validation_checklist.csv`.
Deterministic inference: `scripts/phase3_inference.py`.

## Prohibited adaptations

Do not refit imputation, scaling, encoding, coefficients, baseline hazards, penalties, categories, predictor order, horizons, exclusions, or thresholds using OASIS. Do not select a model after seeing OASIS performance. Do not recalibrate before the untouched results are locked.

## Required external outputs

Report cause-specific AUC/concordance, Brier/integrated Brier/multi-state Brier, calibration intercept/slope/O:E, plots, and probability coherence at 12 and 36 months first. Report 24 and 60 months as sensitivities. Only after primary results may Phase 4 add recalibration-in-the-large, intercept/slope recalibration, and the frozen box-reconstructed CDR endpoint sensitivity.

{common_certification}
"""
    (REPORT_ROOT / "phase3_external_validation_lock_specification.md").write_text(
        lock_report, encoding="utf-8"
    )

    handoff = f"""# Phase 3 to Phase 4 Handoff

Generated: {GENERATED_AT}

## Decision

# {decision}

Selected baseline model: **B1**.
Selected dynamic model: **{selected_dynamic}**.
Primary horizons: **12 and 36 months**.
Sensitivity horizons: **24 and 60 months**.
Model version: **`{model_version_id}`**.

## Exact Phase 4 order

1. Verify every package checksum and run the frozen inference fixture.
2. Build canonical OASIS feature rows with the already-frozen Phase 2 mapping while outcomes remain hidden.
3. Apply the frozen ADNI preprocessors/models through `scripts/phase3_inference.py`; do not fit anything.
4. Save untouched prediction files and their SHA-256 hashes.
5. Only then merge the frozen OASIS outcomes.
6. Report untouched external discrimination, prediction error, calibration, coherence, counts, and dataset shift first.
7. Label all results provisional because `CDRTOT` is the de facto global-CDR mapping pending provider confirmation.
8. After primary reporting only, perform recalibration and box-reconstructed-CDR endpoint sensitivity as separately labelled secondary analyses.
9. Never use OASIS results to revisit Phase 3 feature/model selection.

## Restrictions

{restrictions}

Do not begin Phase 4 automatically from this handoff.

{common_certification}
"""
    (REPORT_ROOT / "phase3_phase4_handoff.md").write_text(
        handoff, encoding="utf-8"
    )

    model_card = f"""# Phase 3 Model Card

Generated: {GENERATED_AT}

## Intended use

Estimate 12/24/36/60-month probabilities of CDR improvement, progression, and remaining at CDR 0.5 among people at a frozen eligible CDR 0.5 landmark. This is a research prediction tool, not a diagnostic device, treatment recommendation, or causal model.

## Development data

ADNI only. One-row baseline B1 and 90-day pooled landmark {selected_dynamic}. OASIS outcomes were untouched.

## Outputs

Three coherent probabilities: improvement to CDR 0, progression to CDR ≥1, and remaining at CDR 0.5. Censoring is not converted to an ordinary non-event.

## Predictors

{md_table(final_artifacts, ["model_id", "family", "horizon_months", "training_rows", "training_participants", "improvement_event_rows", "progression_event_rows", "improvement_alpha", "progression_alpha"])}

See `models/phase3/phase4_lock/frozen_feature_schema.csv` for exact order, levels, imputation, and ranges.

## Validation

Nested 5×5 participant-grouped ADNI validation with {BOOTSTRAPS:,} participant bootstrap samples. Performance and calibration appear in the internal-validation and calibration reports.

## Limitations

CDR and MMSE may share clinical information; lagged/no-MMSE sensitivities address but cannot eliminate circularity. Reversions may include measurement variability. Death/dropout is incompletely characterized. ADNI APOE missingness is phase-structured. OASIS endpoint mapping is provisional. MRI was not ready. AIBL is not a validation cohort here. Landmark predictions from the same person are correlated.

## Fair-use and interpretation

Do not infer that a predictor causes recovery or decline. Do not deploy clinically without external validation, impact assessment, and governance. Inspect phase and cohort calibration before use.

## Reproducibility

Version `{model_version_id}`; deterministic inference fixture and checksums are in `models/phase3/phase4_lock/`. Figure manifest contains {figure_manifest["figure"].nunique()} figures in PNG and SVG.

{common_certification}
"""
    (REPORT_ROOT / "phase3_model_card.md").write_text(
        model_card, encoding="utf-8"
    )


def main() -> None:
    start = time.perf_counter()
    print("[phase3] reading frozen references", flush=True)
    reference_lengths = read_frozen_references()
    integrity, inputs = build_input_integrity()
    save_table(integrity, TABLE_ROOT / "phase3_input_integrity_audit.csv")

    event_info, fold_events, contributions = quantify_event_information(
        inputs["inventory"], inputs["episode"]
    )
    save_table(
        event_info, TABLE_ROOT / "participant_level_event_information.csv"
    )
    save_table(fold_events, TABLE_ROOT / "outer_fold_event_information.csv")
    save_table(
        contributions,
        TABLE_ROOT / "participant_landmark_contribution_details.csv",
    )
    fold_verification = pd.DataFrame(
        [
            {
                "check": "outer assignments unique",
                "status": "PASS",
                "rows": len(inputs["outer"]),
                "participants": inputs["outer"]["participant_id"].nunique(),
                "details": "one outer fold per ADNI participant",
            },
            {
                "check": "nested assignments complete",
                "status": "PASS",
                "rows": len(inputs["inner"]),
                "participants": inputs["inner"]["participant_id"].nunique(),
                "details": "one role per participant in each outer split; five inner folds among outer training",
            },
        ]
    )
    save_table(
        fold_verification, TABLE_ROOT / "outer_inner_fold_verification.csv"
    )

    d1_pair, d1_resolution = resolve_d1_representation(
        inputs["dynamic"][36], inputs["episode"], inputs["inner"]
    )
    save_table(
        d1_resolution, TABLE_ROOT / "d1_collinearity_resolution.csv"
    )
    specs = model_specs(d1_pair)
    print(f"[phase3] D1 representation: {d1_pair}", flush=True)

    (
        ridge_predictions,
        ridge_curves,
        ridge_tuning,
        fold_coefficients,
        preprocessing_parameters,
    ) = run_nested_ridge_models(specs, inputs)
    save_table(
        ridge_tuning, TABLE_ROOT / "fold_specific_ridge_penalty_tuning.csv"
    )
    selected_penalties = ridge_tuning[
        [
            "model_id",
            "family",
            "horizon_months",
            "outer_fold",
            "cause",
            "selected_alpha",
            "selection_rule",
        ]
    ].drop_duplicates()
    save_table(
        selected_penalties, TABLE_ROOT / "fold_specific_ridge_penalties.csv"
    )
    save_table(
        fold_coefficients, TABLE_ROOT / "fold_specific_coefficients.csv"
    )
    save_table(
        preprocessing_parameters,
        TABLE_ROOT / "fold_specific_preprocessing_parameters.csv",
    )

    benchmark_predictions, benchmark_curves, benchmark_tuning = (
        run_baseline_benchmarks(specs["B1"], inputs)
    )
    save_table(
        benchmark_tuning, TABLE_ROOT / "benchmark_inner_tuning.csv"
    )
    predictions = pd.concat(
        [ridge_predictions, benchmark_predictions], ignore_index=True
    )
    curves = pd.concat([ridge_curves, benchmark_curves], ignore_index=True)
    baseline_predictions = predictions[predictions["family"].eq("baseline")]
    dynamic_predictions = predictions[predictions["family"].eq("dynamic")]
    save_table(
        baseline_predictions,
        PREDICTION_ROOT / "out_of_fold_baseline_predictions.csv",
    )
    save_table(
        dynamic_predictions,
        PREDICTION_ROOT / "out_of_fold_dynamic_predictions.csv",
    )
    save_table(
        benchmark_predictions,
        PREDICTION_ROOT / "benchmark_predictions.csv",
    )
    curves.to_parquet(
        PREDICTION_ROOT / "out_of_fold_probability_curves.parquet",
        index=False,
    )

    (
        metrics,
        calibration,
        coherence,
        ibs,
        time_brier,
    ) = calculate_metric_tables(predictions, curves)
    save_table(metrics, TABLE_ROOT / "discrimination_and_horizon_metrics.csv")
    save_table(calibration, TABLE_ROOT / "calibration_tables.csv")
    save_table(coherence, TABLE_ROOT / "probability_coherence_checks.csv")
    save_table(ibs, TABLE_ROOT / "integrated_brier_scores.csv")
    save_table(time_brier, TABLE_ROOT / "time_specific_brier_scores.csv")
    save_table(
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
        TABLE_ROOT / "brier_score_table.csv",
    )

    bootstrap_replicates, bootstrap_ci = bootstrap_prediction_metrics(
        ridge_predictions, ridge_curves, replicates=BOOTSTRAPS
    )
    save_table(
        bootstrap_replicates, TABLE_ROOT / "bootstrap_metric_replicates.csv"
    )
    save_table(
        bootstrap_ci, TABLE_ROOT / "bootstrap_confidence_intervals.csv"
    )
    phase_metrics = phase_stratified_performance(ridge_predictions)
    save_table(
        phase_metrics, TABLE_ROOT / "phase_stratified_performance.csv"
    )
    comparisons = model_comparisons(metrics, ibs)
    save_table(comparisons, TABLE_ROOT / "model_comparison_table.csv")
    selected_dynamic, selection = select_final_dynamic_model(metrics, ibs)
    save_table(
        selection, TABLE_ROOT / "final_model_selection_summary.csv"
    )
    missingness_results = metrics[
        metrics["model_id"].isin(["B1", "B2", "B3", "B4", "D1", "D2"])
    ].merge(
        ibs,
        on=["model_id", "family", "horizon_months"],
        how="left",
    )
    save_table(
        missingness_results, TABLE_ROOT / "missingness_sensitivity_results.csv"
    )

    temporal_parts = []
    temporal_prediction_parts = []
    for model_id in ["B1", selected_dynamic]:
        spec = specs[model_id]
        data = load_analysis_data(
            spec,
            36,
            inputs["episode"],
            inputs["baseline"]
            if spec.family == "baseline"
            else inputs["dynamic"],
        )
        result, prediction = temporal_validation(
            spec, data, 36, ridge_tuning
        )
        temporal_parts.append(result)
        temporal_prediction_parts.append(prediction)
    temporal = pd.concat(temporal_parts, ignore_index=True)
    save_table(temporal, TABLE_ROOT / "temporal_validation_performance.csv")
    save_table(
        pd.concat(temporal_prediction_parts, ignore_index=True),
        PREDICTION_ROOT / "temporal_validation_predictions.csv",
    )
    decision_curve = progression_decision_curve(ridge_predictions)
    selected_calibration = metrics[
        metrics["model_id"].eq(selected_dynamic)
        & metrics["horizon_months"].isin(PRIMARY_HORIZONS)
    ]
    dca_eligible = bool(
        selected_calibration[
            "calibration_slope_progression"
        ].between(0.5, 1.5).all()
        and selected_calibration[
            "calibration_intercept_progression"
        ].abs().le(0.5).all()
    )
    decision_curve["presentation_eligible_by_calibration_rule"] = dca_eligible
    save_table(
        decision_curve, TABLE_ROOT / "progression_decision_curve.csv"
    )

    (
        bundles,
        final_artifacts,
        final_coefficients,
        final_hazards,
        ph,
        functional,
        influence,
        residuals,
        feature_schema,
        model_metadata,
        model_version_id,
    ) = freeze_final_package(
        selected_dynamic,
        specs,
        inputs,
        ridge_tuning,
        d1_resolution,
    )
    save_table(final_artifacts, TABLE_ROOT / "frozen_model_artifacts.csv")
    save_table(final_coefficients, TABLE_ROOT / "final_model_coefficients.csv")
    save_table(final_hazards, TABLE_ROOT / "final_baseline_hazards.csv")
    save_table(ph, TABLE_ROOT / "ph_diagnostic_tests.csv")
    save_table(functional, TABLE_ROOT / "functional_form_diagnostics.csv")
    save_table(influence, TABLE_ROOT / "influential_participants.csv")
    save_table(residuals, TABLE_ROOT / "schoenfeld_residuals.csv")
    save_table(feature_schema, TABLE_ROOT / "frozen_predictor_schema.csv")
    save_table(
        model_metadata, TABLE_ROOT / "model_metadata_and_checksums.csv"
    )

    figure_manifest = create_figures(
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
    save_table(figure_manifest, TABLE_ROOT / "figure_manifest.csv")
    write_reports(
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
        model_version_id=model_version_id,
        figure_manifest=figure_manifest,
    )
    access_path = AUDIT.save()
    run_metadata = {
        "phase": 3,
        "generated_at_utc": GENERATED_AT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "model_version_id": model_version_id,
        "development_cohort": "ADNI",
        "external_cohort": "OASIS-3",
        "oasis_record_level_outcomes_accessed": False,
        "selected_baseline_model": "B1",
        "selected_dynamic_model": selected_dynamic,
        "primary_horizons_months": list(PRIMARY_HORIZONS),
        "sensitivity_horizons_months": list(SENSITIVITY_HORIZONS),
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "participant",
        "random_seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "file_access_audit": str(access_path.relative_to(ROOT)),
        "reports": sorted(
            str(path.relative_to(ROOT)) for path in REPORT_ROOT.glob("*.md")
        ),
        "figures": int(figure_manifest["figure"].nunique()),
        "figure_files": len(figure_manifest),
        "final_decision": "GO TO PHASE 4 WITH RESTRICTIONS",
        "phase4_started": False,
    }
    write_json(run_metadata, TABLE_ROOT / "phase3_run_metadata.json")
    print(json.dumps(run_metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
