#!/usr/bin/env python3
"""Phase 5 secondary analyses and manuscript-ready synthesis.

The script treats completed Phase 1--4 tables as authoritative for the primary
CDR model study. It reconstructs the previously audited ADNI baseline-MCI
diagnosis trajectories for participant-level secondary analyses, fits only
prespecified low-dimensional ridge association models, and creates the final
tables, figures, and reports. Frozen Phase 3/4 model artifacts are read-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = ROOT / "results" / "csvs" / "phase5"
MANUSCRIPT_TABLE_ROOT = TABLE_ROOT / "manuscript_tables"
PLOT_ROOT = ROOT / "results" / "plots" / "phase5"
REPORT_ROOT = ROOT / "reports" / "phase5"
PROCESSED_ROOT = ROOT / "ProcessedData" / "adni" / "phase5"

PHASE1_TABLES = ROOT / "results" / "csvs" / "phase1"
PHASE3_TABLES = ROOT / "results" / "csvs" / "phase3"
PHASE4_TABLES = ROOT / "results" / "csvs" / "phase4"
PHASE4_REPORTS = ROOT / "reports" / "phase4"

DXSUM_PATH = ROOT / "datasets" / "adni" / "All_Subjects_DXSUM_26Aug2025.csv"
MMSE_PATH = ROOT / "datasets" / "adni" / "All_Subjects_MMSE_26Aug2025.csv"
DEMOGRAPHICS_PATH = (
    ROOT / "datasets" / "adni" / "All_Subjects_PTDEMOG_26Aug2025.csv"
)
APOE_PATH = ROOT / "datasets" / "adni" / "All_Subjects_APOERES_26Aug2025.csv"
CDR_PATH = ROOT / "datasets" / "adni" / "All_Subjects_CDR_26Aug2025.csv"
CDR_MANIFEST_PATH = (
    ROOT / "ProcessedData" / "adni" / "phase1_classifiable_cdr_manifest.csv"
)

GENERATED_AT = datetime.now(timezone.utc).isoformat()
RANDOM_SEED = 20260730
BOOTSTRAP_REPLICATES = 1000
DAY0 = pd.Timestamp("1970-01-01")

for directory in (
    TABLE_ROOT,
    MANUSCRIPT_TABLE_ROOT,
    PLOT_ROOT,
    REPORT_ROOT,
    PROCESSED_ROOT,
):
    directory.mkdir(parents=True, exist_ok=True)

sys.dont_write_bytecode = True
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def md(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    shown = frame if columns is None else frame.loc[:, columns]
    return shown.to_markdown(index=False)


def q1(series: pd.Series) -> float:
    return float(series.quantile(0.25))


def q3(series: pd.Series) -> float:
    return float(series.quantile(0.75))


def clean_numeric(
    series: pd.Series, minimum: float | None = None, maximum: float | None = None
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if minimum is not None:
        values = values.where(values >= minimum)
    if maximum is not None:
        values = values.where(values <= maximum)
    return values


def date_to_day(value: pd.Timestamp) -> float:
    return float((value - DAY0).days)


def ordered_diagnoses() -> pd.DataFrame:
    """Apply the exact audited three-state diagnosis construction."""
    diagnosis = pd.read_csv(DXSUM_PATH, low_memory=False, na_values=[-4])
    diagnosis["state"] = clean_numeric(diagnosis["DIAGNOSIS"])
    diagnosis["assessment_date"] = pd.to_datetime(
        diagnosis["EXAMDATE"], errors="coerce"
    )
    diagnosis = diagnosis[
        diagnosis["state"].isin([1, 2, 3])
        & diagnosis["RID"].notna()
        & diagnosis["assessment_date"].notna()
    ].copy()
    diagnosis["participant_id"] = diagnosis["RID"].astype(int)
    diagnosis["assessment_time_days"] = (
        diagnosis["assessment_date"] - DAY0
    ).dt.days.astype(float)
    diagnosis["source_row"] = np.arange(len(diagnosis))
    diagnosis = diagnosis.sort_values(
        [
            "participant_id",
            "assessment_date",
            "state",
            "update_stamp",
            "source_row",
        ],
        na_position="first",
    )
    # Same-time conflicts are resolved to the more severe state, matching the
    # feasibility audit.
    diagnosis = diagnosis.drop_duplicates(
        ["participant_id", "assessment_date"], keep="last"
    ).sort_values(["participant_id", "assessment_date"])
    diagnosis["state"] = diagnosis["state"].astype(int)
    return diagnosis


def first_consecutive_cn_confirmation(states: np.ndarray) -> int | None:
    for position in range(1, len(states)):
        if states[position - 1] == 1 and states[position] == 1:
            if np.any(states[: position - 1] == 2):
                return position
    return None


def trajectory_records(
    diagnosis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants: list[dict[str, Any]] = []
    records: list[pd.DataFrame] = []
    for participant_id, group in diagnosis.groupby("participant_id", sort=True):
        group = group.sort_values("assessment_date").reset_index(drop=True)
        if int(group.loc[0, "state"]) != 2:
            continue
        group = group.copy()
        group["diagnosis_assessment_number"] = np.arange(1, len(group) + 1)
        group["months_from_baseline_diagnosis"] = (
            group["assessment_date"] - group.loc[0, "assessment_date"]
        ).dt.days / 30.4375
        records.append(group)

        states = group["state"].to_numpy(dtype=int)
        later = group.iloc[1:]
        later_states = states[1:]
        any_cn = bool(np.any(later_states == 1))
        any_ad = bool(np.any(later_states == 3))
        last_mci_position = int(np.where(states == 2)[0].max())
        terminal_tail = group.iloc[last_mci_position + 1 :]
        sustained = (
            len(terminal_tail) >= 2
            and terminal_tail["state"].eq(1).all()
        )
        preceding_mci_run = 0
        for state in states[: last_mci_position + 1][::-1]:
            if state == 2:
                preceding_mci_run += 1
            else:
                break
        strongest = sustained and preceding_mci_run >= 2

        if len(later) == 0:
            trajectory_group = "no_later_assessment"
        elif any_ad:
            trajectory_group = "progression"
        elif not any_cn:
            trajectory_group = "persistent_mci"
        elif sustained:
            trajectory_group = "sustained_reversion"
        else:
            trajectory_group = "unstable_or_single_cn"

        apparent_positions = np.where(later_states == 1)[0] + 1
        first_apparent_position = (
            int(apparent_positions[0]) if len(apparent_positions) else None
        )
        confirmed_position = first_consecutive_cn_confirmation(states)
        first_ad_positions = np.where(later_states == 3)[0] + 1
        first_ad_position = (
            int(first_ad_positions[0]) if len(first_ad_positions) else None
        )
        sustained_position = (
            last_mci_position + 2 if sustained else None
        )

        apparent_relapse = False
        if first_apparent_position is not None:
            apparent_relapse = bool(
                np.any(states[first_apparent_position + 1 :] > 1)
            )
        confirmed_relapse = False
        if confirmed_position is not None:
            confirmed_relapse = bool(
                np.any(states[confirmed_position + 1 :] > 1)
            )
        progression_lower_state = False
        if first_ad_position is not None:
            progression_lower_state = bool(
                np.any(states[first_ad_position + 1 :] < 3)
            )

        baseline_date = group.loc[0, "assessment_date"]
        participants.append(
            {
                "participant_id": int(participant_id),
                "baseline_diagnosis_date": baseline_date,
                "baseline_diagnosis_time_days": date_to_day(baseline_date),
                "baseline_phase": group.loc[0, "PHASE"],
                "baseline_site_id": group.loc[0, "SITEID"],
                "baseline_ptid": group.loc[0, "PTID"],
                "first_later_diagnosis_date": (
                    group.loc[1, "assessment_date"]
                    if len(group) > 1
                    else pd.NaT
                ),
                "last_diagnosis_date": group.iloc[-1]["assessment_date"],
                "diagnosis_assessments": len(group),
                "later_diagnosis_assessments": len(later),
                "follow_up_months": float(
                    (
                        group.iloc[-1]["assessment_date"] - baseline_date
                    ).days
                    / 30.4375
                ),
                "trajectory_group": trajectory_group,
                "apparent_reversion_flag": first_apparent_position is not None,
                "confirmed_reversion_flag": confirmed_position is not None,
                "sustained_reversion_flag": sustained,
                "strongest_reversion_flag": strongest,
                "progression_flag": any_ad,
                "persistent_mci_flag": trajectory_group == "persistent_mci",
                "unstable_or_single_cn_flag": (
                    trajectory_group == "unstable_or_single_cn"
                ),
                "apparent_reversion_time_months": (
                    float(
                        group.loc[
                            first_apparent_position,
                            "months_from_baseline_diagnosis",
                        ]
                    )
                    if first_apparent_position is not None
                    else np.nan
                ),
                "confirmed_reversion_time_months": (
                    float(
                        group.loc[
                            confirmed_position,
                            "months_from_baseline_diagnosis",
                        ]
                    )
                    if confirmed_position is not None
                    else np.nan
                ),
                "sustained_reversion_time_months": (
                    float(
                        group.loc[
                            sustained_position,
                            "months_from_baseline_diagnosis",
                        ]
                    )
                    if sustained_position is not None
                    else np.nan
                ),
                "progression_time_months": (
                    float(
                        group.loc[
                            first_ad_position,
                            "months_from_baseline_diagnosis",
                        ]
                    )
                    if first_ad_position is not None
                    else np.nan
                ),
                "apparent_reversion_assessment_number": (
                    first_apparent_position + 1
                    if first_apparent_position is not None
                    else np.nan
                ),
                "confirmed_reversion_assessment_number": (
                    confirmed_position + 1
                    if confirmed_position is not None
                    else np.nan
                ),
                "sustained_reversion_assessment_number": (
                    sustained_position + 1
                    if sustained_position is not None
                    else np.nan
                ),
                "progression_assessment_number": (
                    first_ad_position + 1
                    if first_ad_position is not None
                    else np.nan
                ),
                "apparent_reversion_later_relapse_flag": apparent_relapse,
                "confirmed_reversion_later_relapse_flag": confirmed_relapse,
                "progression_later_lower_state_flag": progression_lower_state,
                "terminal_cn_assessments": (
                    int(len(terminal_tail)) if sustained else 0
                ),
                "mci_run_before_terminal_cn": preceding_mci_run,
            }
        )
    participant_frame = pd.DataFrame(participants)
    record_frame = pd.concat(records, ignore_index=True)
    expected = {
        "no_later_assessment": 350,
        "persistent_mci": 746,
        "progression": 399,
        "sustained_reversion": 60,
        "unstable_or_single_cn": 45,
    }
    observed = participant_frame["trajectory_group"].value_counts().to_dict()
    if len(participant_frame) != 1600 or observed != expected:
        raise RuntimeError(
            f"Audited diagnosis labels were not reproduced: n={len(participant_frame)}, "
            f"observed={observed}"
        )
    if int(participant_frame["strongest_reversion_flag"].sum()) != 50:
        raise RuntimeError("Strongest reversion definition did not reproduce n=50.")
    return participant_frame, record_frame


def first_valid_by_participant(
    frame: pd.DataFrame, column: str
) -> pd.Series:
    subset = frame.dropna(subset=[column]).drop_duplicates("RID")
    return subset.set_index("RID")[column]


def apoe_dose(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    alleles = str(value).replace("|", "/").split("/")
    try:
        return float(sum(int(allele) == 4 for allele in alleles))
    except ValueError:
        return np.nan


def nearest_measurement(
    frame: pd.DataFrame,
    participant_id: int,
    baseline_date: pd.Timestamp,
    first_later_date: pd.Timestamp | pd.NaT,
    value_column: str,
) -> tuple[float, float]:
    latest_allowed = baseline_date + pd.Timedelta(days=180)
    if pd.notna(first_later_date):
        latest_allowed = min(
            latest_allowed, first_later_date - pd.Timedelta(microseconds=1)
        )
    subset = frame[
        frame["RID"].eq(participant_id)
        & frame[value_column].notna()
        & frame["_date"].between(
            baseline_date - pd.Timedelta(days=180), latest_allowed
        )
    ].copy()
    if subset.empty:
        return np.nan, np.nan
    subset["_absolute_offset"] = (
        subset["_date"] - baseline_date
    ).abs().dt.days
    chosen = subset.sort_values(
        ["_absolute_offset", "_date", "_source_row"]
    ).iloc[0]
    return float(chosen[value_column]), float(
        (chosen["_date"] - baseline_date).days
    )


def add_clinical_predictors(participants: pd.DataFrame) -> pd.DataFrame:
    demographics = pd.read_csv(
        DEMOGRAPHICS_PATH, low_memory=False, na_values=[-4]
    )
    demographics["_date"] = pd.to_datetime(
        demographics["VISDATE"], errors="coerce"
    )
    sex_map = first_valid_by_participant(demographics, "PTGENDER")
    education_map = first_valid_by_participant(demographics, "PTEDUCAT")
    dob_map = first_valid_by_participant(demographics, "PTDOB")

    apoe = pd.read_csv(APOE_PATH, low_memory=False, na_values=[-4])
    genotype_map = first_valid_by_participant(apoe, "GENOTYPE")

    mmse = pd.read_csv(MMSE_PATH, low_memory=False, na_values=[-4])
    mmse["_date"] = pd.to_datetime(mmse["VISDATE"], errors="coerce")
    mmse["_source_row"] = np.arange(len(mmse))
    mmse["MMSCORE"] = clean_numeric(mmse["MMSCORE"], 0, 30)

    cdr = pd.read_csv(CDR_PATH, low_memory=False, na_values=[-4])
    cdr["_date"] = pd.to_datetime(cdr["VISDATE"], errors="coerce")
    cdr["_source_row"] = np.arange(len(cdr))
    cdr["CDGLOBAL"] = clean_numeric(cdr["CDGLOBAL"]).where(
        clean_numeric(cdr["CDGLOBAL"]).isin([0, 0.5, 1, 2, 3])
    )

    rows: list[dict[str, Any]] = []
    for record in participants.itertuples(index=False):
        participant_id = int(record.participant_id)
        dob = pd.to_datetime(
            dob_map.get(participant_id, np.nan),
            format="%m/%Y",
            errors="coerce",
        )
        if pd.notna(dob):
            dob = dob + pd.Timedelta(days=14)
            age = (
                record.baseline_diagnosis_date - dob
            ).days / 365.25
        else:
            age = np.nan
        mmse_value, mmse_offset = nearest_measurement(
            mmse,
            participant_id,
            record.baseline_diagnosis_date,
            record.first_later_diagnosis_date,
            "MMSCORE",
        )
        cdr_value, cdr_offset = nearest_measurement(
            cdr,
            participant_id,
            record.baseline_diagnosis_date,
            record.first_later_diagnosis_date,
            "CDGLOBAL",
        )
        sex_code = clean_numeric(
            pd.Series([sex_map.get(participant_id, np.nan)])
        ).iloc[0]
        rows.append(
            {
                "participant_id": participant_id,
                "baseline_age": age,
                "sex": (
                    "Male"
                    if sex_code == 1
                    else "Female"
                    if sex_code == 2
                    else np.nan
                ),
                "sex_male": (
                    1.0 if sex_code == 1 else 0.0 if sex_code == 2 else np.nan
                ),
                "education_years": clean_numeric(
                    pd.Series(
                        [education_map.get(participant_id, np.nan)]
                    ),
                    0,
                    30,
                ).iloc[0],
                "apoe_genotype": genotype_map.get(
                    participant_id, np.nan
                ),
                "apoe_e4_dose": apoe_dose(
                    genotype_map.get(participant_id, np.nan)
                ),
                "baseline_mmse": mmse_value,
                "baseline_mmse_day_offset": mmse_offset,
                "baseline_cdr": cdr_value,
                "baseline_cdr_day_offset": cdr_offset,
            }
        )
    predictors = pd.DataFrame(rows)
    return participants.merge(
        predictors, on="participant_id", how="left", validate="one_to_one"
    )


def add_cdr_comparison(participants: pd.DataFrame) -> pd.DataFrame:
    cdr = pd.read_csv(CDR_MANIFEST_PATH, low_memory=False)
    cdr = cdr[
        cdr["participant_id"].notna()
        & cdr["time_sort_days"].notna()
        & cdr["cdr_state"].isin([0, 1, 2])
    ].copy()
    cdr["participant_id"] = cdr["participant_id"].astype(int)
    rows: list[dict[str, Any]] = []
    for record in participants.itertuples(index=False):
        group = cdr[
            cdr["participant_id"].eq(record.participant_id)
            & cdr["time_sort_days"].ge(
                record.baseline_diagnosis_time_days
            )
        ].sort_values("time_sort_days")
        states = group["cdr_state"].to_numpy(dtype=int)
        transition_positions = (
            np.where((states[:-1] == 1) & (states[1:] == 0))[0] + 1
            if len(states) > 1
            else np.array([], dtype=int)
        )
        improvement = len(transition_positions) > 0
        first_position = (
            int(transition_positions[0]) if improvement else None
        )
        first_time = (
            float(
                (
                    group.iloc[first_position]["time_sort_days"]
                    - record.baseline_diagnosis_time_days
                )
                / 30.4375
            )
            if first_position is not None
            else np.nan
        )
        relapse = (
            bool(
                group.iloc[first_position + 1 :]["cdr_state"].astype(int).gt(0).any()
            )
            if first_position is not None
            else False
        )
        rows.append(
            {
                "participant_id": int(record.participant_id),
                "cdr_assessments_on_or_after_diagnosis_index": len(group),
                "cdr_followup_available_flag": len(group) >= 2,
                "first_post_index_cdr_state": (
                    int(states[0]) if len(states) else np.nan
                ),
                "cdr_defined_improvement_flag": improvement,
                "cdr_defined_first_improvement_months": first_time,
                "cdr_defined_later_relapse_flag": relapse,
                "cdr_defined_improvement_events": int(
                    len(transition_positions)
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    result = participants.merge(
        comparison, on="participant_id", how="left", validate="one_to_one"
    )
    result["definition_agreement_category"] = np.select(
        [
            result["sustained_reversion_flag"]
            & result["cdr_defined_improvement_flag"],
            result["sustained_reversion_flag"]
            & ~result["cdr_defined_improvement_flag"],
            ~result["sustained_reversion_flag"]
            & result["cdr_defined_improvement_flag"],
        ],
        [
            "both_sustained_diagnosis_and_cdr_improvement",
            "diagnosis_sustained_only",
            "cdr_improvement_only",
        ],
        default="neither_definition",
    )
    return result


def group_mask(frame: pd.DataFrame, group: str) -> pd.Series:
    if group == "strongest_reversion":
        return frame["strongest_reversion_flag"]
    return frame["trajectory_group"].eq(group)


def group_characteristics(frame: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "sustained_reversion",
        "strongest_reversion",
        "persistent_mci",
        "progression",
        "unstable_or_single_cn",
    ]
    numeric_variables = {
        "baseline_age": "years",
        "education_years": "years",
        "apoe_e4_dose": "alleles",
        "baseline_mmse": "points",
        "baseline_cdr": "global stage",
        "follow_up_months": "months",
        "later_diagnosis_assessments": "assessments",
        "diagnosis_assessments": "assessments",
    }
    rows: list[dict[str, Any]] = []
    for group in groups:
        subset = frame[group_mask(frame, group)]
        for variable, unit in numeric_variables.items():
            values = clean_numeric(subset[variable]).dropna()
            rows.append(
                {
                    "group": group,
                    "group_participants": len(subset),
                    "variable": variable,
                    "unit": unit,
                    "observed_n": len(values),
                    "missing_n": int(subset[variable].isna().sum()),
                    "missing_percent": float(
                        subset[variable].isna().mean() * 100
                    ),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "sd": float(values.std()) if len(values) > 1 else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "q1": q1(values) if len(values) else np.nan,
                    "q3": q3(values) if len(values) else np.nan,
                    "category_count": np.nan,
                    "category_percent": np.nan,
                }
            )
        observed_sex = subset["sex"].dropna()
        female_n = int(observed_sex.eq("Female").sum())
        rows.append(
            {
                "group": group,
                "group_participants": len(subset),
                "variable": "female_sex",
                "unit": "n_percent",
                "observed_n": len(observed_sex),
                "missing_n": int(subset["sex"].isna().sum()),
                "missing_percent": float(subset["sex"].isna().mean() * 100),
                "mean": np.nan,
                "sd": np.nan,
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "category_count": female_n,
                "category_percent": (
                    float(female_n / len(observed_sex) * 100)
                    if len(observed_sex)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def group_comparison_tests(frame: pd.DataFrame) -> pd.DataFrame:
    exclusive_groups = [
        "sustained_reversion",
        "persistent_mci",
        "progression",
        "unstable_or_single_cn",
    ]
    rows = []
    for variable in (
        "baseline_age",
        "education_years",
        "apoe_e4_dose",
        "baseline_mmse",
        "follow_up_months",
        "later_diagnosis_assessments",
    ):
        samples = [
            clean_numeric(
                frame.loc[
                    frame["trajectory_group"].eq(group), variable
                ]
            ).dropna()
            for group in exclusive_groups
        ]
        statistic, p_value = stats.kruskal(*samples)
        rows.append(
            {
                "variable": variable,
                "test": "Kruskal-Wallis",
                "groups": "|".join(exclusive_groups),
                "statistic": statistic,
                "p_value": p_value,
                "interpretation": "unadjusted association; not causal",
            }
        )
    contingency = pd.crosstab(
        frame.loc[
            frame["trajectory_group"].isin(exclusive_groups),
            "trajectory_group",
        ],
        frame.loc[
            frame["trajectory_group"].isin(exclusive_groups), "sex"
        ],
    )
    statistic, p_value, _, _ = stats.chi2_contingency(contingency)
    rows.append(
        {
            "variable": "sex",
            "test": "Pearson chi-square",
            "groups": "|".join(exclusive_groups),
            "statistic": statistic,
            "p_value": p_value,
            "interpretation": "unadjusted association; not causal",
        }
    )
    return pd.DataFrame(rows)


MODEL_FEATURES = [
    "baseline_age",
    "sex_male",
    "education_years",
    "apoe_e4_dose",
    "baseline_mmse",
]
MODEL_LABELS = {
    "baseline_age": "Age (per SD)",
    "sex_male": "Male versus female",
    "education_years": "Education (per SD)",
    "apoe_e4_dose": "APOE e4 dose (per SD)",
    "baseline_mmse": "MMSE (per SD)",
}


def ridge_design(
    frame: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    work = frame[feature_names].copy()
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in feature_names:
        values = clean_numeric(work[feature])
        median = float(values.median())
        values = values.fillna(median)
        medians[feature] = median
        if feature == "sex_male":
            scale = 1.0
            center = 0.0
        else:
            center = float(values.mean())
            scale = float(values.std(ddof=0))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
        scales[feature] = scale
        work[feature] = (values - center) / scale
    return work.to_numpy(dtype=float), medians, scales


def ridge_coefficients(
    frame: pd.DataFrame,
    outcome: pd.Series,
    feature_names: list[str],
) -> np.ndarray:
    x, _, _ = ridge_design(frame, feature_names)
    y = outcome.to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("Bootstrap sample has one outcome class.")
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    model.fit(x, y)
    return model.coef_[0]


def bootstrap_ridge_association(
    data: pd.DataFrame,
    outcome_column: str,
    comparison: str,
    feature_names: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    coefficients = ridge_coefficients(
        data, data[outcome_column], feature_names
    )
    bootstrap = np.full(
        (BOOTSTRAP_REPLICATES, len(feature_names)), np.nan
    )
    n = len(data)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, n, size=n)
        sample = data.iloc[indices].reset_index(drop=True)
        try:
            bootstrap[replicate] = ridge_coefficients(
                sample, sample[outcome_column], feature_names
            )
        except (ValueError, RuntimeError):
            continue
    rows = []
    for index, feature in enumerate(feature_names):
        valid = bootstrap[:, index][np.isfinite(bootstrap[:, index])]
        rows.append(
            {
                "comparison": comparison,
                "outcome": outcome_column,
                "feature": feature,
                "feature_label": MODEL_LABELS.get(feature, feature),
                "ridge_C": 1.0,
                "coefficient": coefficients[index],
                "adjusted_odds_ratio": math.exp(coefficients[index]),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "valid_bootstrap_replicates": len(valid),
                "bootstrap_unit": "participant",
                "confidence_interval_method": (
                    "participant-resampled percentile bootstrap"
                ),
                "odds_ratio_ci_low": (
                    math.exp(float(np.quantile(valid, 0.025)))
                    if len(valid)
                    else np.nan
                ),
                "odds_ratio_ci_high": (
                    math.exp(float(np.quantile(valid, 0.975)))
                    if len(valid)
                    else np.nan
                ),
                "analysis_type": (
                    "shrinkage-adjusted association; not prediction or causal"
                ),
            }
        )
    return pd.DataFrame(rows)


def adjusted_associations(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    comparisons = [
        (
            "sustained_reversion_vs_persistent_mci",
            ["sustained_reversion", "persistent_mci"],
            "sustained_reversion",
            "sustained_case",
        ),
        (
            "sustained_reversion_vs_progression",
            ["sustained_reversion", "progression"],
            "sustained_reversion",
            "sustained_case",
        ),
    ]
    outputs = []
    for name, groups, case_group, outcome in comparisons:
        data = frame[
            frame["trajectory_group"].isin(groups)
        ].copy()
        data[outcome] = data["trajectory_group"].eq(case_group).astype(int)
        outputs.append(
            bootstrap_ridge_association(
                data, outcome, name, MODEL_FEATURES, rng
            )
        )

    strongest = frame[
        frame["strongest_reversion_flag"]
        | frame["trajectory_group"].eq("persistent_mci")
    ].copy()
    strongest["strongest_case"] = strongest[
        "strongest_reversion_flag"
    ].astype(int)
    outputs.append(
        bootstrap_ridge_association(
            strongest,
            "strongest_case",
            "strongest_reversion_vs_persistent_mci",
            MODEL_FEATURES,
            rng,
        )
    )
    return pd.concat(outputs, ignore_index=True)


def observation_process_association(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[
        ~frame["trajectory_group"].eq("no_later_assessment")
    ].copy()
    data["sustained_case"] = data["sustained_reversion_flag"].astype(int)
    features = ["follow_up_months", "later_diagnosis_assessments"]
    labels = {
        "follow_up_months": "Follow-up duration (per SD)",
        "later_diagnosis_assessments": "Later assessments (per SD)",
    }
    rng = np.random.default_rng(RANDOM_SEED + 11)
    result = bootstrap_ridge_association(
        data,
        "sustained_case",
        "sustained_vs_other_followed_observation_process",
        features,
        rng,
    )
    result["feature_label"] = result["feature"].map(labels)
    result["analysis_type"] = (
        "post-baseline ascertainment association; not prediction or causal"
    )
    return result


def followup_opportunity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in (24, 36, 48, 60):
        eligible = frame["follow_up_months"].ge(threshold)
        subset = frame[eligible]
        rows.append(
            {
                "restriction_type": "minimum_follow_up_months",
                "restriction_value": threshold,
                "eligible_participants": len(subset),
                "sustained_reverters": int(
                    subset["sustained_reversion_flag"].sum()
                ),
                "strongest_reverters": int(
                    subset["strongest_reversion_flag"].sum()
                ),
                "progressors": int(subset["progression_flag"].sum()),
                "persistent_mci": int(subset["persistent_mci_flag"].sum()),
                "sustained_percent": float(
                    subset["sustained_reversion_flag"].mean() * 100
                ),
            }
        )
    for threshold in (2, 4, 6):
        subset = frame[
            frame["later_diagnosis_assessments"].ge(threshold)
        ]
        rows.append(
            {
                "restriction_type": "minimum_later_assessments",
                "restriction_value": threshold,
                "eligible_participants": len(subset),
                "sustained_reverters": int(
                    subset["sustained_reversion_flag"].sum()
                ),
                "strongest_reverters": int(
                    subset["strongest_reversion_flag"].sum()
                ),
                "progressors": int(subset["progression_flag"].sum()),
                "persistent_mci": int(subset["persistent_mci_flag"].sum()),
                "sustained_percent": float(
                    subset["sustained_reversion_flag"].mean() * 100
                ),
            }
        )
    return pd.DataFrame(rows)


def fixed_horizon_diagnosis(
    participant_records: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for horizon in (24, 36, 48):
        for participant_id, group in participant_records.groupby(
            "participant_id"
        ):
            group = group.sort_values("assessment_date")
            eligible = (
                group.iloc[-1]["months_from_baseline_diagnosis"] >= horizon
            )
            if not eligible:
                status = "excluded_before_horizon"
            else:
                observed = group[
                    group["months_from_baseline_diagnosis"].le(horizon)
                ]
                states = observed["state"].to_numpy(dtype=int)
                if np.any(states[1:] == 3):
                    status = "progression"
                else:
                    last_mci = int(np.where(states == 2)[0].max())
                    tail = states[last_mci + 1 :]
                    if len(tail) >= 2 and np.all(tail == 1):
                        status = "sustained_reversion"
                    elif np.all(states == 2):
                        status = "persistent_mci"
                    else:
                        status = "unstable_unresolved"
            rows.append(
                {
                    "participant_id": participant_id,
                    "horizon_months": horizon,
                    "status": status,
                    "eligible_at_horizon": eligible,
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["horizon_months", "status"])
        .size()
        .rename("participants")
        .reset_index()
    )
    return summary


def timing_relapse_summary(frame: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        (
            "apparent_reversion",
            "apparent_reversion_flag",
            "apparent_reversion_time_months",
            "apparent_reversion_assessment_number",
            "apparent_reversion_later_relapse_flag",
        ),
        (
            "confirmed_reversion",
            "confirmed_reversion_flag",
            "confirmed_reversion_time_months",
            "confirmed_reversion_assessment_number",
            "confirmed_reversion_later_relapse_flag",
        ),
        (
            "sustained_reversion",
            "sustained_reversion_flag",
            "sustained_reversion_time_months",
            "sustained_reversion_assessment_number",
            None,
        ),
        (
            "strongest_reversion",
            "strongest_reversion_flag",
            "sustained_reversion_time_months",
            "sustained_reversion_assessment_number",
            None,
        ),
        (
            "progression",
            "progression_flag",
            "progression_time_months",
            "progression_assessment_number",
            "progression_later_lower_state_flag",
        ),
        (
            "persistent_mci",
            "persistent_mci_flag",
            None,
            None,
            None,
        ),
    ]
    rows = []
    for definition, flag, time_col, assessment_col, relapse_col in definitions:
        subset = frame[frame[flag]]
        time = (
            clean_numeric(subset[time_col]).dropna()
            if time_col
            else pd.Series(dtype=float)
        )
        assessment = (
            clean_numeric(subset[assessment_col]).dropna()
            if assessment_col
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "definition": definition,
                "participants": len(subset),
                "time_median_months": (
                    float(time.median()) if len(time) else np.nan
                ),
                "time_q1_months": q1(time) if len(time) else np.nan,
                "time_q3_months": q3(time) if len(time) else np.nan,
                "assessment_number_median": (
                    float(assessment.median())
                    if len(assessment)
                    else np.nan
                ),
                "assessment_number_q1": (
                    q1(assessment) if len(assessment) else np.nan
                ),
                "assessment_number_q3": (
                    q3(assessment) if len(assessment) else np.nan
                ),
                "later_relapse_or_lower_state": (
                    int(subset[relapse_col].sum())
                    if relapse_col
                    else 0
                ),
                "interpretation": (
                    "observed diagnostic trajectory; not biological recovery"
                ),
            }
        )
    result = pd.DataFrame(rows)
    expected_counts = {
        "apparent_reversion": 110,
        "confirmed_reversion": 74,
        "sustained_reversion": 60,
        "strongest_reversion": 50,
        "progression": 399,
        "persistent_mci": 746,
    }
    observed = dict(zip(result["definition"], result["participants"]))
    if observed != expected_counts:
        raise RuntimeError(
            f"Reversion definition counts changed: {observed}"
        )
    return result


def phase_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in (
        "sustained_reversion",
        "strongest_reversion",
        "persistent_mci",
        "progression",
        "unstable_or_single_cn",
        "no_later_assessment",
    ):
        subset = frame[group_mask(frame, group)]
        counts = subset["baseline_phase"].fillna("missing").value_counts()
        for phase, count in counts.items():
            rows.append(
                {
                    "group": group,
                    "baseline_adni_phase": phase,
                    "participants": int(count),
                    "group_percent": float(count / len(subset) * 100),
                }
            )
    return pd.DataFrame(rows)


def missingness_by_group(frame: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "baseline_age",
        "sex",
        "education_years",
        "apoe_e4_dose",
        "baseline_mmse",
        "baseline_cdr",
    ]
    rows = []
    for group in (
        "sustained_reversion",
        "strongest_reversion",
        "persistent_mci",
        "progression",
        "unstable_or_single_cn",
        "no_later_assessment",
    ):
        subset = frame[group_mask(frame, group)]
        for variable in variables:
            rows.append(
                {
                    "group": group,
                    "variable": variable,
                    "participants": len(subset),
                    "missing_n": int(subset[variable].isna().sum()),
                    "missing_percent": float(
                        subset[variable].isna().mean() * 100
                    ),
                }
            )
    return pd.DataFrame(rows)


def agreement_tables(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnosis_set = set(
        frame.loc[
            frame["sustained_reversion_flag"], "participant_id"
        ]
    )
    cdr_set = set(
        frame.loc[
            frame["cdr_defined_improvement_flag"], "participant_id"
        ]
    )
    overlap = diagnosis_set & cdr_set
    union = diagnosis_set | cdr_set
    summary = pd.DataFrame(
        [
            {
                "baseline_clinical_mci_participants": len(frame),
                "participants_with_at_least_two_post_index_cdr_assessments": int(
                    frame["cdr_followup_available_flag"].sum()
                ),
                "sustained_diagnosis_reverters": len(diagnosis_set),
                "cdr_defined_improvers": len(cdr_set),
                "overlap_participants": len(overlap),
                "diagnosis_only": len(diagnosis_set - cdr_set),
                "cdr_only": len(cdr_set - diagnosis_set),
                "union_participants": len(union),
                "jaccard_similarity": len(overlap) / len(union),
                "diagnosis_reverters_with_cdr_improvement_percent": (
                    len(overlap) / len(diagnosis_set) * 100
                ),
                "cdr_improvers_with_sustained_diagnosis_reversion_percent": (
                    len(overlap) / len(cdr_set) * 100
                ),
                "cdr_improvers_with_later_cdr_relapse": int(
                    frame.loc[
                        frame["cdr_defined_improvement_flag"],
                        "cdr_defined_later_relapse_flag",
                    ].sum()
                ),
                "interpretation": (
                    "definition agreement only; neither endpoint proves biological recovery"
                ),
            }
        ]
    )
    disagreement = (
        frame["definition_agreement_category"]
        .value_counts()
        .rename_axis("agreement_category")
        .rename("participants")
        .reset_index()
    )
    disagreement["percent_of_baseline_mci"] = (
        disagreement["participants"] / len(frame) * 100
    )

    timing = frame[
        frame["sustained_reversion_flag"]
        & frame["cdr_defined_improvement_flag"]
    ][
        [
            "participant_id",
            "sustained_reversion_time_months",
            "cdr_defined_first_improvement_months",
            "apparent_reversion_later_relapse_flag",
            "cdr_defined_later_relapse_flag",
        ]
    ].copy()
    timing["diagnosis_minus_cdr_timing_months"] = (
        timing["sustained_reversion_time_months"]
        - timing["cdr_defined_first_improvement_months"]
    )

    characteristics = []
    for category, subset in frame.groupby(
        "definition_agreement_category"
    ):
        for variable in (
            "baseline_age",
            "education_years",
            "apoe_e4_dose",
            "baseline_mmse",
            "follow_up_months",
            "later_diagnosis_assessments",
        ):
            values = clean_numeric(subset[variable]).dropna()
            characteristics.append(
                {
                    "agreement_category": category,
                    "participants": len(subset),
                    "variable": variable,
                    "observed_n": len(values),
                    "median": float(values.median())
                    if len(values)
                    else np.nan,
                    "q1": q1(values) if len(values) else np.nan,
                    "q3": q3(values) if len(values) else np.nan,
                    "missing_percent": float(
                        subset[variable].isna().mean() * 100
                    ),
                }
            )
    return (
        summary,
        disagreement,
        timing,
        pd.DataFrame(characteristics),
    )


def cohort_characteristics() -> pd.DataFrame:
    rows = []
    label_map = {"adni": "ADNI", "oasis3": "OASIS-3", "aibl": "AIBL"}
    for cohort in ("adni", "oasis3", "aibl"):
        baseline = pd.read_csv(
            ROOT
            / "ProcessedData"
            / cohort
            / "phase2"
            / "baseline"
            / "baseline_12m_clinical_core.csv",
            low_memory=False,
        )
        # One row per participant in every baseline file.
        age = clean_numeric(baseline["age_at_landmark"])
        education = clean_numeric(baseline["education_years"])
        mmse = clean_numeric(baseline["index_mmse"])
        apoe = clean_numeric(baseline["apoe_e4_dose"])
        rows.append(
            {
                "cohort": label_map[cohort],
                "role": (
                    "development/internal validation"
                    if cohort == "adni"
                    else "locked external validation"
                    if cohort == "oasis3"
                    else "descriptive only"
                ),
                "index_participants": baseline["participant_id"].nunique(),
                "age_median": float(age.median()),
                "age_q1": q1(age),
                "age_q3": q3(age),
                "age_verified": cohort != "aibl",
                "female_n": int(baseline["sex"].eq("Female").sum()),
                "female_percent": float(
                    baseline["sex"].eq("Female").mean() * 100
                ),
                "education_median": (
                    float(education.median())
                    if education.notna().any()
                    else np.nan
                ),
                "education_missing_percent": float(
                    education.isna().mean() * 100
                ),
                "apoe_missing_percent": float(apoe.isna().mean() * 100),
                "mmse_median": float(mmse.median()),
                "mmse_q1": q1(mmse),
                "mmse_q3": q3(mmse),
                "endpoint": "CDR 0.5 to 0 / >=1",
            }
        )
    return pd.DataFrame(rows)


def transition_counts_table() -> pd.DataFrame:
    frames = []
    for cohort in ("adni", "oasis3", "aibl"):
        frame = pd.read_csv(
            PHASE1_TABLES / cohort / "horizon_feasibility_summary.csv"
        )
        frames.append(frame[frame["horizon_months"].isin([12, 36, 60])])
    return pd.concat(frames, ignore_index=True)


def internal_performance_table() -> pd.DataFrame:
    metrics = pd.read_csv(
        PHASE3_TABLES / "discrimination_and_horizon_metrics.csv"
    )
    integrated = pd.read_csv(
        PHASE3_TABLES / "integrated_brier_scores.csv"
    )
    selected = metrics[
        metrics["model_id"].isin(["B1", "D0", "D1"])
        & metrics["horizon_months"].isin([12, 36])
    ].merge(
        integrated,
        on=["model_id", "family", "horizon_months"],
        how="left",
        validate="one_to_one",
    )
    return selected


def external_performance_table() -> pd.DataFrame:
    metrics = pd.read_csv(PHASE4_TABLES / "untouched_external_metrics.csv")
    integrated = pd.read_csv(
        PHASE4_TABLES / "untouched_integrated_brier_scores.csv"
    )
    selected = metrics[
        metrics["model_id"].isin(["B1", "D0", "D1"])
        & metrics["horizon_months"].isin([12, 36])
    ].merge(
        integrated,
        on=["model_id", "family", "horizon_months"],
        how="left",
        validate="one_to_one",
    )
    return selected


def transportability_table() -> pd.DataFrame:
    comparison = pd.read_csv(
        PHASE4_TABLES / "adni_oasis_performance_comparison.csv"
    )
    return comparison[
        comparison["model_id"].isin(["B1", "D0", "D1"])
        & comparison["horizon_months"].isin([12, 36])
    ].copy()


def calibration_recalibration_table() -> pd.DataFrame:
    effect = pd.read_csv(
        PHASE4_TABLES / "recalibration_effect_comparison.csv"
    )
    return effect[
        effect["model_id"].isin(["B1", "D0", "D1"])
        & effect["horizon_months"].isin([12, 36])
    ].copy()


def recalibration_parameter_summary() -> pd.DataFrame:
    parameters = pd.read_csv(
        PHASE4_TABLES / "cross_fitted_recalibration_parameters.csv"
    )
    parameters = parameters[
        parameters["model_id"].isin(["B1", "D0", "D1"])
        & parameters["horizon_months"].isin([12, 36])
    ].copy()
    summary = (
        parameters.groupby(
            ["model_id", "horizon_months", "method", "cause"],
            as_index=False,
        )
        .agg(
            cross_validation_folds=("fold", "nunique"),
            median_training_rows=("training_rows", "median"),
            median_training_participants=(
                "training_participants",
                "median",
            ),
            median_training_events=("training_events", "median"),
            median_intercept=("intercept", "median"),
            intercept_q1=("intercept", lambda x: x.quantile(0.25)),
            intercept_q3=("intercept", lambda x: x.quantile(0.75)),
            median_slope=("slope", "median"),
            slope_q1=("slope", lambda x: x.quantile(0.25)),
            slope_q3=("slope", lambda x: x.quantile(0.75)),
            predictor_coefficients_refit=(
                "predictor_coefficients_refit",
                "max",
            ),
        )
    )
    summary["validation_design"] = (
        "5-fold participant-grouped cross-fitting in OASIS-3"
    )
    summary["interpretation"] = (
        "local model updating; not untouched external validation"
    )
    return summary


def transportability_classification_framework() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "classification": "transportable_without_updating",
                "operational_definition": (
                    "useful discrimination and acceptable untouched "
                    "calibration without local parameter updating"
                ),
                "phase5_status": "not_met",
                "reason": (
                    "untouched OASIS improvement risk was materially "
                    "underpredicted"
                ),
            },
            {
                "classification": "transportable_with_recalibration",
                "operational_definition": (
                    "useful untouched discrimination with correctable "
                    "local calibration shift"
                ),
                "phase5_status": "selected",
                "reason": (
                    "D1 retained useful discrimination and D1-vs-D0 value; "
                    "cross-fitted local updating improved prediction error"
                ),
            },
            {
                "classification": "limited_transportability",
                "operational_definition": (
                    "transport restricted to a horizon, transition, or "
                    "narrowly matched population"
                ),
                "phase5_status": "not_selected",
                "reason": (
                    "some low-event analyses remain imprecise, but the "
                    "primary D1 pattern persisted across 12 and 36 months"
                ),
            },
            {
                "classification": "not_transportable",
                "operational_definition": (
                    "external discrimination and/or prediction error do not "
                    "support use even with reasonable local calibration"
                ),
                "phase5_status": "not_met",
                "reason": (
                    "external rank discrimination remained useful and "
                    "endpoint sensitivity did not reverse conclusions"
                ),
            },
        ]
    )


def shift_summary_table() -> pd.DataFrame:
    continuous = pd.read_csv(
        PHASE4_TABLES / "predictor_distribution_shift.csv"
    )
    continuous = continuous[
        continuous["model_id"].isin(["B1", "D0", "D1", "D3"])
    ].copy()
    feature_summary = (
        continuous.sort_values(
            "absolute_standardized_mean_difference", ascending=False
        )
        .drop_duplicates("feature")
        .head(12)
    )
    feature_rows = pd.DataFrame(
        {
            "shift_type": "continuous_predictor",
            "item": feature_summary["feature"],
            "model_id": feature_summary["model_id"],
            "horizon_months": feature_summary["horizon_months"],
            "adni_value": feature_summary["adni_mean"],
            "oasis_value": feature_summary["oasis_mean"],
            "difference_metric": feature_summary[
                "standardized_mean_difference"
            ],
            "difference_metric_name": "standardized_mean_difference",
            "interpretation": "case-mix or measurement-process shift",
        }
    )
    event = pd.read_csv(PHASE4_TABLES / "event_rate_shift.csv")
    event = event[
        event["model_id"].isin(["B1", "D0", "D1"])
        & event["horizon_months"].isin([12, 36])
    ]
    event_rows = pd.DataFrame(
        {
            "shift_type": "event_rate",
            "item": event["cause"],
            "model_id": event["model_id"],
            "horizon_months": event["horizon_months"],
            "adni_value": event["adni_ipcw_observed_rate"],
            "oasis_value": event["oasis_ipcw_observed_rate"],
            "difference_metric": event["oasis_to_adni_ratio"],
            "difference_metric_name": "oasis_to_adni_ratio",
            "interpretation": "outcome-incidence shift",
        }
    )
    return pd.concat([feature_rows, event_rows], ignore_index=True)


def endpoint_sensitivity_table() -> pd.DataFrame:
    comparison = pd.read_csv(
        PHASE4_TABLES / "reconstructed_cdr_metric_comparison.csv"
    )
    return comparison[
        comparison["model_id"].isin(["B1", "D0", "D1"])
        & comparison["horizon_months"].isin([12, 36])
        & comparison["metric"].isin(
            [
                "auc_improvement",
                "auc_progression",
                "c_index_improvement",
                "c_index_progression",
                "multistate_brier",
                "calibration_slope_improvement",
                "calibration_slope_progression",
                "observed_expected_improvement",
                "observed_expected_progression",
            ]
        )
    ].copy()


def sensitivity_summary_table() -> pd.DataFrame:
    metrics = pd.read_csv(PHASE4_TABLES / "untouched_external_metrics.csv")
    rows = []
    for model_id, label in (
        ("B3", "No-APOE baseline sensitivity"),
        ("D3", "Strictly lagged-MMSE dynamic sensitivity"),
    ):
        subset = metrics[
            metrics["model_id"].eq(model_id)
            & metrics["horizon_months"].isin([12, 36])
        ]
        for record in subset.itertuples(index=False):
            rows.append(
                {
                    "analysis": label,
                    "model_id": model_id,
                    "horizon_months": record.horizon_months,
                    "rows": record.rows,
                    "participants": record.participants,
                    "auc_improvement": record.auc_improvement,
                    "auc_progression": record.auc_progression,
                    "multistate_brier": record.multistate_brier,
                    "conclusion": (
                        "directionally consistent but population/support differs"
                        if model_id == "D3"
                        else "similar to B1; APOE removal did not improve transport"
                    ),
                }
            )
    for model_id in ("B1", "D1"):
        subset = metrics[
            metrics["model_id"].eq(model_id)
            & metrics["horizon_months"].isin([24, 60])
        ]
        for record in subset.itertuples(index=False):
            rows.append(
                {
                    "analysis": "Extended horizon sensitivity",
                    "model_id": model_id,
                    "horizon_months": record.horizon_months,
                    "rows": record.rows,
                    "participants": record.participants,
                    "auc_improvement": record.auc_improvement,
                    "auc_progression": record.auc_progression,
                    "multistate_brier": record.multistate_brier,
                    "conclusion": "secondary horizon; not used for model selection",
                }
            )
    clinical = pd.read_csv(
        PHASE4_TABLES / "clinical_mci_subgroup_results.csv"
    )
    for record in clinical.itertuples(index=False):
        rows.append(
            {
                "analysis": "Clinical-MCI subgroup",
                "model_id": record.model_id,
                "horizon_months": record.horizon_months,
                "rows": record.rows,
                "participants": record.participants,
                "auc_improvement": record.auc_improvement,
                "auc_progression": record.auc_progression,
                "multistate_brier": record.multistate_brier,
                "conclusion": "small event counts; descriptive sensitivity",
            }
        )
    return pd.DataFrame(rows)


def aibl_descriptive_table() -> pd.DataFrame:
    horizon = pd.read_csv(
        PHASE1_TABLES / "aibl" / "horizon_feasibility_summary.csv"
    )
    index = pd.read_csv(
        ROOT / "ProcessedData" / "aibl" / "phase1_primary_index_cohort.csv"
    )
    rows = []
    for record in horizon.itertuples(index=False):
        rows.append(
            {
                "horizon_months": record.horizon_months,
                "eligible_index_participants": record.total_index_participants,
                "known_outcome_participants": (
                    record.participants_with_known_outcome_by_horizon
                ),
                "cdr_improvement": record.transition_to_cdr_0,
                "cdr_progression": record.transition_to_cdr_ge1,
                "remain_cdr_0_5": record.remain_cdr_0_5,
                "censored_before_horizon": record.censored_before_horizon,
                "verified_age_available": int(
                    (
                        index["index_age"].notna()
                        & index["age_derivation_verified_flag"].astype(bool)
                    ).sum()
                ),
                "unverified_age_proxy_available": int(
                    index["index_age"].notna().sum()
                ),
                "education_available": int(
                    index["education_years"].notna().sum()
                ),
                "participants_with_two_mmse": int(
                    pd.read_csv(
                        PHASE1_TABLES
                        / "aibl"
                        / "longitudinal_depth_summary.csv"
                    )
                    .query("outcome_group == 'ALL'")
                    ["mmse_at_least_2"]
                    .iloc[0]
                ),
                "confirmatory_validation_supported": False,
                "reason": (
                    "no verified age, education 100% missing, sparse repeated MMSE, "
                    "and few known events after censoring"
                ),
            }
        )
    return pd.DataFrame(rows)


def limitations_bias_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "domain": "External endpoint",
                "limitation_or_bias": (
                    "OASIS CDRTOT is a provisional de facto global CDR"
                ),
                "control_or_analysis": (
                    "reconstructed six-box global CDR sensitivity"
                ),
                "residual_risk": "endpoint misclassification remains",
            },
            {
                "domain": "Observation process",
                "limitation_or_bias": (
                    "death and dropout reasons are incomplete"
                ),
                "control_or_analysis": (
                    "administrative last-observation censoring and follow-up audit"
                ),
                "residual_risk": "informative censoring cannot be excluded",
            },
            {
                "domain": "State definition",
                "limitation_or_bias": (
                    "CDR 0.5 is not identical to clinical MCI"
                ),
                "control_or_analysis": (
                    "participant-level CDR-versus-diagnosis agreement analysis"
                ),
                "residual_risk": "definitions answer different clinical questions",
            },
            {
                "domain": "Measurement overlap",
                "limitation_or_bias": (
                    "MMSE and CDR measure overlapping cognitive domains"
                ),
                "control_or_analysis": (
                    "strictly lagged-MMSE D3 and no-MMSE Phase 3 sensitivities"
                ),
                "residual_risk": "construct overlap remains",
            },
            {
                "domain": "Transportability",
                "limitation_or_bias": (
                    "OASIS improvement incidence is substantially higher"
                ),
                "control_or_analysis": (
                    "untouched validation followed by separate cross-fitted recalibration"
                ),
                "residual_risk": "local recalibration is required",
            },
            {
                "domain": "Repeated landmarks",
                "limitation_or_bias": (
                    "within-participant correlation and leakage"
                ),
                "control_or_analysis": (
                    "participant-grouped folds and participant-cluster bootstrap"
                ),
                "residual_risk": "landmark weighting choices affect estimand",
            },
            {
                "domain": "Diagnosis reversion",
                "limitation_or_bias": (
                    "sustained status requires long future observation"
                ),
                "control_or_analysis": (
                    "follow-up/visit audit, fixed-horizon summaries, strongest definition"
                ),
                "residual_risk": "guarantee-time and ascertainment bias remain",
            },
            {
                "domain": "MRI",
                "limitation_or_bias": (
                    "harmonized sample support is inadequate"
                ),
                "control_or_analysis": "MRI excluded from primary models",
                "residual_risk": "imaging incremental value is unresolved",
            },
            {
                "domain": "AIBL",
                "limitation_or_bias": (
                    "verified age/education and longitudinal depth are inadequate"
                ),
                "control_or_analysis": "descriptive counts only",
                "residual_risk": "third-cohort validation is unavailable",
            },
            {
                "domain": "Causal interpretation",
                "limitation_or_bias": (
                    "observational predictors and state labels are not interventions"
                ),
                "control_or_analysis": (
                    "association, prediction, and model updating labelled separately"
                ),
                "residual_risk": "no causal claims are supported",
            },
        ]
    )


def representative_trajectories() -> pd.DataFrame:
    predictions = pd.read_csv(
        PHASE3_TABLES / "predictions" / "out_of_fold_dynamic_predictions.csv"
    )
    predictions = predictions[
        predictions["model_id"].eq("D1")
        & predictions["horizon_months"].eq(36)
    ].sort_values(["participant_id", "elapsed_years_from_index"])
    eligible = predictions.groupby("participant_id").filter(
        lambda group: len(group) >= 3
    )
    change = eligible.groupby("participant_id").agg(
        first_progression=("probability_progression", "first"),
        last_progression=("probability_progression", "last"),
        first_improvement=("probability_improvement", "first"),
        last_improvement=("probability_improvement", "last"),
        rows=("row_key", "size"),
    )
    change["progression_change"] = (
        change["last_progression"] - change["first_progression"]
    )
    change["improvement_change"] = (
        change["last_improvement"] - change["first_improvement"]
    )
    progression_id = int(change["progression_change"].idxmax())
    improvement_id = int(change["improvement_change"].idxmax())
    remaining = change.drop(index=[progression_id, improvement_id])
    stable_id = int(
        (
            remaining["progression_change"].abs()
            + remaining["improvement_change"].abs()
        ).idxmin()
    )
    labels = {
        progression_id: "progression-risk-increase",
        improvement_id: "improvement-risk-increase",
        stable_id: "stable-risk-example",
    }
    result = eligible[
        eligible["participant_id"].isin(labels)
    ].copy()
    result["trajectory_label"] = result["participant_id"].map(labels)
    result["selection_rule"] = (
        "deterministic representative OOF D1 36m pattern; descriptive only"
    )
    return result[
        [
            "participant_id",
            "row_key",
            "elapsed_years_from_index",
            "probability_improvement",
            "probability_progression",
            "probability_remaining",
            "trajectory_label",
            "selection_rule",
        ]
    ]


def save_figure(
    fig: plt.Figure,
    stem: str,
    dpi: int,
) -> None:
    fig.tight_layout()
    fig.savefig(
        PLOT_ROOT / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        PLOT_ROOT / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def figure_study_flow(cohorts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axis("off")
    colors = ["#d7e9f7", "#e1f2e8", "#f7ead6"]
    y_positions = [0.72, 0.43, 0.14]
    for color, y, row in zip(colors, y_positions, cohorts.itertuples()):
        box = patches.FancyBboxPatch(
            (0.07, y),
            0.86,
            0.19,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="#34495e",
            linewidth=2,
        )
        ax.add_patch(box)
        verification = (
            "verified age"
            if bool(row.age_verified)
            else "age proxy unverified"
        )
        ax.text(
            0.5,
            y + 0.125,
            f"{row.cohort}: {row.index_participants:,} CDR 0.5 index participants",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        ax.text(
            0.5,
            y + 0.055,
            f"{row.role}; {verification}",
            ha="center",
            va="center",
            fontsize=11,
        )
    ax.text(
        0.5,
        0.97,
        "Study cohorts and analysis roles",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    save_figure(fig, "figure_01_study_and_cohort_flow", 1200)


def figure_transition_diagram() -> None:
    transition = pd.read_csv(
        PHASE1_TABLES / "cdr_transition_summary_all_cohorts.csv"
    )
    transition = transition[
        transition["transition_label"].isin(
            ["CDR 0.5 -> 0", "CDR 0.5 -> >=1"]
        )
    ]
    pivot = transition.pivot(
        index="cohort", columns="transition_label", values="events"
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.8))
    positions = {"CDR 0": (0.1, 0.5), "CDR 0.5": (0.5, 0.5), "CDR ≥1": (0.9, 0.5)}
    for ax, cohort in zip(axes, ["adni", "oasis3", "aibl"]):
        ax.axis("off")
        for label, (x, y) in positions.items():
            circle = patches.Circle(
                (x, y), 0.12, facecolor="#eef3f7", edgecolor="#2c3e50", lw=2
            )
            ax.add_patch(circle)
            ax.text(x, y, label, ha="center", va="center", fontsize=11, fontweight="bold")
        improvement = int(pivot.loc[cohort, "CDR 0.5 -> 0"])
        progression = int(pivot.loc[cohort, "CDR 0.5 -> >=1"])
        ax.annotate(
            "",
            xy=(0.23, 0.54),
            xytext=(0.38, 0.54),
            arrowprops=dict(arrowstyle="->", lw=2.5, color="#2a9d8f"),
        )
        ax.annotate(
            "",
            xy=(0.77, 0.46),
            xytext=(0.62, 0.46),
            arrowprops=dict(arrowstyle="->", lw=2.5, color="#d95f02"),
        )
        ax.text(0.3, 0.65, f"n={improvement}", ha="center", color="#167c6b", fontsize=11)
        ax.text(0.7, 0.34, f"n={progression}", ha="center", color="#a94600", fontsize=11)
        ax.set_title(cohort.upper() if cohort != "oasis3" else "OASIS-3", fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.1, 0.9)
    fig.suptitle("Observed CDR transitions from state 0.5", fontweight="bold")
    save_figure(fig, "figure_02_cdr_state_transition_diagram", 1200)


def figure_workflow() -> None:
    stages = [
        "ADNI\nlandmarks",
        "Participant-grouped\nnested validation",
        "Frozen D1\nmodel",
        "Outcome-hidden\nOASIS prediction",
        "Untouched\nexternal validation",
        "Separate\nrecalibration",
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.axis("off")
    x = np.linspace(0.08, 0.92, len(stages))
    for idx, (position, label) in enumerate(zip(x, stages)):
        color = "#d9eef2" if idx < 3 else "#f4e5cf" if idx < 5 else "#eadcf2"
        box = patches.FancyBboxPatch(
            (position - 0.07, 0.4),
            0.14,
            0.23,
            boxstyle="round,pad=0.015",
            facecolor=color,
            edgecolor="#34495e",
            lw=2,
        )
        ax.add_patch(box)
        ax.text(position, 0.515, label, ha="center", va="center", fontsize=10.5)
        if idx < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x[idx + 1] - 0.075, 0.515),
                xytext=(position + 0.075, 0.515),
                arrowprops=dict(arrowstyle="->", lw=2, color="#34495e"),
            )
    ax.text(
        0.5,
        0.82,
        "Dynamic model development and locked external-validation workflow",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.22,
        "Untouched validation remains primary; OASIS-based recalibration is model updating",
        ha="center",
        fontsize=11.5,
        color="#6c3483",
    )
    save_figure(fig, "figure_03_dynamic_external_validation_workflow", 1200)


def figure_internal_external(transport: pd.DataFrame) -> None:
    subset = transport[
        transport["metric"].isin(["auc_improvement", "auc_progression"])
    ].copy()
    subset["label"] = (
        subset["model_id"]
        + " "
        + subset["horizon_months"].astype(str)
        + "m "
        + subset["metric"].str.replace("auc_", "")
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    y = np.arange(len(subset))
    ax.scatter(
        subset["adni_internal"],
        y,
        label="ADNI internal OOF",
        color="#1f77b4",
        s=65,
    )
    ax.scatter(
        subset["oasis_external"],
        y,
        label="OASIS untouched",
        color="#d95f02",
        s=65,
        marker="s",
    )
    for pos, record in enumerate(subset.itertuples()):
        ax.plot(
            [record.adni_internal, record.oasis_external],
            [pos, pos],
            color="#9aa3a8",
            lw=1.5,
            zorder=0,
        )
    ax.set_yticks(y, subset["label"])
    ax.set_xlabel("Time-dependent AUC")
    ax.set_xlim(0.60, 0.84)
    ax.axvline(0.70, color="#bbbbbb", lw=1, ls="--")
    ax.legend(loc="lower right")
    ax.set_title("Internal versus untouched external discrimination", fontweight="bold")
    save_figure(fig, "figure_04_internal_vs_external_discrimination", 600)


def figure_untouched_calibration() -> None:
    curves = pd.read_csv(PHASE4_TABLES / "untouched_calibration_curves.csv")
    curves = curves[
        curves["model_id"].eq("D1")
        & curves["horizon_months"].isin([12, 36])
        & curves["cause"].isin(["improvement", "progression"])
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), sharex=True, sharey=True)
    for ax, ((horizon, cause), subset) in zip(
        axes.flat, curves.groupby(["horizon_months", "cause"], sort=True)
    ):
        ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1.5)
        ax.plot(
            subset["mean_prediction"],
            subset["observed_probability"],
            "o-",
            color="#3b7ea1" if cause == "improvement" else "#d95f02",
            lw=2,
            ms=5,
        )
        ax.set_title(f"D1 {horizon}m {cause}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed probability")
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)
        ax.grid(alpha=0.2)
    fig.suptitle("Untouched OASIS-3 calibration", fontweight="bold")
    save_figure(fig, "figure_05_untouched_calibration", 1200)


def figure_recalibration(effect: pd.DataFrame) -> None:
    subset = effect[
        effect["model_id"].eq("D1")
        & effect["horizon_months"].isin([12, 36])
    ].copy()
    order = ["untouched", "calibration_in_the_large", "intercept_and_slope"]
    colors = ["#7f8c8d", "#8e44ad", "#5b2c6f"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.5))
    width = 0.24
    x = np.arange(2)
    for offset, (method, color) in enumerate(zip(order, colors)):
        method_data = subset[subset["method"].eq(method)].sort_values(
            "horizon_months"
        )
        axes[0].bar(
            x + (offset - 1) * width,
            method_data["multistate_brier"],
            width,
            label=method.replace("_", " "),
            color=color,
        )
        axes[1].bar(
            x + (offset - 1) * width,
            method_data["observed_expected_improvement"],
            width,
            label=method.replace("_", " "),
            color=color,
        )
    for ax in axes:
        ax.set_xticks(x, ["12 months", "36 months"])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Multistate Brier score")
    axes[0].set_title("Prediction error")
    axes[1].set_ylabel("Improvement O/E")
    axes[1].axhline(1, color="#222222", ls="--", lw=1.5)
    axes[1].set_title("Improvement calibration")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Untouched validation versus cross-fitted model updating",
        fontweight="bold",
    )
    save_figure(fig, "figure_06_untouched_vs_recalibrated", 1200)


def figure_d1_d0() -> None:
    ci = pd.read_csv(PHASE4_TABLES / "d1_vs_d0_paired_bootstrap_ci.csv")
    ci = ci[
        ci["metric"].isin(
            ["auc_improvement", "auc_progression", "multistate_brier"]
        )
    ].copy()
    ci["label"] = (
        ci["horizon_months"].astype(str)
        + "m "
        + ci["metric"].str.replace("_", " ")
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    y = np.arange(len(ci))
    ax.errorbar(
        ci["bootstrap_median_difference"],
        y,
        xerr=[
            ci["bootstrap_median_difference"] - ci["ci_low"],
            ci["ci_high"] - ci["bootstrap_median_difference"],
        ],
        fmt="o",
        color="#2c7fb8",
        ecolor="#7fcdbb",
        capsize=4,
        markersize=7,
    )
    ax.axvline(0, color="#555555", ls="--")
    ax.set_yticks(y, ci["label"])
    ax.set_xlabel("D1 minus D0 (paired participant bootstrap)")
    ax.set_title("External value of longitudinal MMSE updating", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, "figure_07_d1_vs_d0_external_comparison", 600)


def figure_shift(shift: pd.DataFrame) -> None:
    continuous = shift[shift["shift_type"].eq("continuous_predictor")].head(8)
    event = shift[
        shift["shift_type"].eq("event_rate")
        & shift["model_id"].eq("D1")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.8))
    axes[0].barh(
        continuous["item"],
        continuous["difference_metric"],
        color=np.where(
            continuous["difference_metric"].ge(0), "#3182bd", "#e6550d"
        ),
    )
    axes[0].axvline(0, color="#555555", lw=1)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Standardized mean difference")
    axes[0].set_title("Predictor shift")
    event = event.copy()
    event["label"] = (
        event["horizon_months"].astype(str)
        + "m "
        + event["item"]
    )
    axes[1].barh(
        event["label"], event["difference_metric"], color="#756bb1"
    )
    axes[1].axvline(1, color="#555555", ls="--")
    axes[1].set_xlabel("OASIS / ADNI observed-rate ratio")
    axes[1].set_title("Event-rate shift")
    axes[1].invert_yaxis()
    fig.suptitle("Development-to-external dataset shift", fontweight="bold")
    save_figure(fig, "figure_08_predictor_and_event_rate_shift", 1200)


def figure_overlap(summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.axis("off")
    left = patches.Circle(
        (0.42, 0.5), 0.28, color="#5dade2", alpha=0.55, ec="#21618c", lw=2
    )
    right = patches.Circle(
        (0.62, 0.5), 0.28, color="#f5b041", alpha=0.55, ec="#9c640c", lw=2
    )
    ax.add_patch(left)
    ax.add_patch(right)
    ax.text(0.29, 0.5, str(int(row.diagnosis_only)), fontsize=20, fontweight="bold")
    ax.text(0.52, 0.5, str(int(row.overlap_participants)), fontsize=20, fontweight="bold")
    ax.text(0.74, 0.5, str(int(row.cdr_only)), fontsize=20, fontweight="bold")
    ax.text(0.28, 0.82, "Sustained diagnosis\nreversion", ha="center", fontsize=13)
    ax.text(0.76, 0.82, "CDR-defined\nimprovement", ha="center", fontsize=13)
    ax.text(
        0.52,
        0.12,
        f"Jaccard = {row.jaccard_similarity:.3f}\nNeither definition implies biological recovery",
        ha="center",
        fontsize=12,
    )
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 0.95)
    ax.set_title("Clinical-state definition changes observed reversion", fontweight="bold")
    save_figure(fig, "figure_09_cdr_vs_diagnosis_reversion_overlap", 600)


def figure_dynamic_trajectories(trajectories: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.8), sharey=True)
    for ax, (label, subset) in zip(
        axes, trajectories.groupby("trajectory_label", sort=True)
    ):
        subset = subset.sort_values("elapsed_years_from_index")
        ax.plot(
            subset["elapsed_years_from_index"],
            subset["probability_improvement"],
            "o-",
            label="Improvement",
            color="#2a9d8f",
            lw=2,
        )
        ax.plot(
            subset["elapsed_years_from_index"],
            subset["probability_progression"],
            "s-",
            label="Progression",
            color="#d95f02",
            lw=2,
        )
        ax.set_title(label.replace("-", " "))
        ax.set_xlabel("Years from index")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Frozen D1 36m probability")
    axes[0].legend(fontsize=9)
    fig.suptitle("Representative out-of-fold dynamic risk trajectories", fontweight="bold")
    save_figure(fig, "figure_10_representative_dynamic_risk_trajectories", 600)


def figure_reversion_trajectory(timing: pd.DataFrame) -> None:
    subset = timing[
        timing["definition"].isin(
            [
                "apparent_reversion",
                "confirmed_reversion",
                "sustained_reversion",
                "strongest_reversion",
            ]
        )
    ].copy()
    labels = ["Apparent", "Confirmed", "Sustained", "Strongest"]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(subset))
    bars = ax1.bar(
        x,
        subset["participants"],
        color=["#b3cde3", "#8c96c6", "#8856a7", "#810f7c"],
    )
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Participants")
    ax1.set_ylim(0, 125)
    for bar, count in zip(bars, subset["participants"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(int(count)),
            ha="center",
            fontweight="bold",
        )
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        subset["time_median_months"],
        "o-",
        color="#e6550d",
        lw=2,
        label="Median qualifying time",
    )
    ax2.set_ylabel("Median qualifying time (months)")
    for position, record in enumerate(subset.itertuples()):
        ax1.text(
            position,
            7,
            f"relapse {int(record.later_relapse_or_lower_state)}",
            ha="center",
            fontsize=9,
            rotation=90,
            color="#333333",
        )
    ax1.set_title("Diagnosis-reversion confirmation and relapse", fontweight="bold")
    save_figure(fig, "figure_11_reversion_trajectory_and_relapse", 600)


def figure_graphical_summary(
    external: pd.DataFrame,
    d1d0: pd.DataFrame,
    overlap: pd.DataFrame,
    recalibration: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0))
    d1 = external[external["model_id"].eq("D1")].sort_values(
        "horizon_months"
    )
    axes[0, 0].plot(
        d1["horizon_months"], d1["auc_improvement"], "o-", label="Improvement"
    )
    axes[0, 0].plot(
        d1["horizon_months"], d1["auc_progression"], "s-", label="Progression"
    )
    axes[0, 0].set_ylim(0.6, 0.8)
    axes[0, 0].set_title("Untouched D1 AUC")
    axes[0, 0].legend(fontsize=8)

    paired = d1d0[
        d1d0["metric"].isin(["auc_improvement", "auc_progression"])
    ]
    for cause, marker in (
        ("auc_improvement", "o"),
        ("auc_progression", "s"),
    ):
        subset = paired[paired["metric"].eq(cause)]
        axes[0, 1].plot(
            subset["horizon_months"],
            subset["bootstrap_median_difference"],
            marker=marker,
            label=cause.replace("auc_", ""),
        )
    axes[0, 1].axhline(0, color="#555555", ls="--")
    axes[0, 1].set_title("D1 minus D0 AUC")
    axes[0, 1].legend(fontsize=8)

    row = overlap.iloc[0]
    axes[1, 0].bar(
        ["Diagnosis\nonly", "Both", "CDR\nonly"],
        [row.diagnosis_only, row.overlap_participants, row.cdr_only],
        color=["#5dade2", "#8e6e53", "#f5b041"],
    )
    axes[1, 0].set_title(f"Reversion overlap (J={row.jaccard_similarity:.2f})")

    recal = recalibration[
        recalibration["model_id"].eq("D1")
        & recalibration["horizon_months"].eq(36)
    ]
    axes[1, 1].bar(
        recal["method"].str.replace("_", "\n"),
        recal["multistate_brier"],
        color=["#7f8c8d", "#8e44ad", "#5b2c6f"],
    )
    axes[1, 1].set_title("D1 36m prediction error")
    axes[1, 1].tick_params(axis="x", labelsize=7)
    fig.suptitle(
        "Dynamic prediction transports, but calibration and endpoint definition matter",
        fontweight="bold",
    )
    save_figure(fig, "figure_12_final_graphical_summary", 1200)


def create_figures(
    cohorts: pd.DataFrame,
    transport: pd.DataFrame,
    recalibration: pd.DataFrame,
    shift: pd.DataFrame,
    agreement: pd.DataFrame,
    timing: pd.DataFrame,
    trajectories: pd.DataFrame,
    external: pd.DataFrame,
) -> None:
    figure_study_flow(cohorts)
    figure_transition_diagram()
    figure_workflow()
    figure_internal_external(transport)
    figure_untouched_calibration()
    figure_recalibration(recalibration)
    figure_d1_d0()
    figure_shift(shift)
    figure_overlap(agreement)
    figure_dynamic_trajectories(trajectories)
    figure_reversion_trajectory(timing)
    figure_graphical_summary(
        external,
        pd.read_csv(PHASE4_TABLES / "d1_vs_d0_paired_bootstrap_ci.csv"),
        agreement,
        recalibration,
    )


def manuscript_tables(
    cohort: pd.DataFrame,
    transitions: pd.DataFrame,
    internal: pd.DataFrame,
    external: pd.DataFrame,
    recalibration: pd.DataFrame,
    shift: pd.DataFrame,
    endpoint: pd.DataFrame,
    reversion_characteristics: pd.DataFrame,
    agreement: pd.DataFrame,
    sensitivities: pd.DataFrame,
    limitations: pd.DataFrame,
) -> dict[str, Path]:
    table_frames = {
        "table_01_cohort_characteristics.csv": cohort,
        "table_02_adni_oasis_transition_counts.csv": transitions,
        "table_03_adni_internal_model_performance.csv": internal,
        "table_04_untouched_oasis_external_performance.csv": external,
        "table_05_d1_vs_d0_external_comparison.csv": pd.read_csv(
            PHASE4_TABLES / "d1_vs_d0_paired_bootstrap_ci.csv"
        ),
        "table_06_calibration_and_recalibration.csv": recalibration,
        "table_07_distribution_shift_summary.csv": shift,
        "table_08_endpoint_sensitivity.csv": endpoint,
        "table_09_diagnosis_reversion_group_characteristics.csv": (
            reversion_characteristics
        ),
        "table_10_cdr_vs_diagnosis_reversion_agreement.csv": agreement,
        "table_11_sensitivity_analysis_summary.csv": sensitivities,
        "table_12_limitations_and_bias_controls.csv": limitations,
    }
    return {
        name: save_csv(frame, MANUSCRIPT_TABLE_ROOT / name)
        for name, frame in table_frames.items()
    }


def phase5_reports(
    participant: pd.DataFrame,
    characteristics: pd.DataFrame,
    timing: pd.DataFrame,
    associations: pd.DataFrame,
    observation: pd.DataFrame,
    phase: pd.DataFrame,
    missingness: pd.DataFrame,
    agreement: pd.DataFrame,
    disagreement: pd.DataFrame,
    agreement_timing: pd.DataFrame,
    internal: pd.DataFrame,
    external: pd.DataFrame,
    transport: pd.DataFrame,
    recalibration: pd.DataFrame,
    recalibration_parameters: pd.DataFrame,
    transportability_framework: pd.DataFrame,
    shift: pd.DataFrame,
    endpoint: pd.DataFrame,
    aibl: pd.DataFrame,
    limitations: pd.DataFrame,
) -> None:
    primary_groups = characteristics[
        characteristics["variable"].isin(
            [
                "baseline_age",
                "education_years",
                "apoe_e4_dose",
                "baseline_mmse",
                "follow_up_months",
                "later_diagnosis_assessments",
                "female_sex",
            ]
        )
    ]
    reversion_report = f"""# Phase 5 ADNI Diagnosis-Based Reversion Analysis

Generated: {GENERATED_AT}

## Audited population

The first classifiable clinician diagnosis was MCI for **1,600** ADNI
participants. Exactly **1,250** had a later classifiable diagnosis. The
mutually exclusive trajectory groups reproduce the prior feasibility audit:
60 sustained reverters, 399 progressors, 746 persistent-MCI participants,
45 unstable/single-CN trajectories, and 350 without a later assessment.
The strongest confirmation definition is met by 50 of the 60 sustained
reverters.

## Group characteristics

{md(primary_groups)}

## Timing and relapse

{md(timing)}

Apparent and confirmed reversions can relapse; sustained and strongest
definitions exclude later relapse by construction. This is diagnostic-state
reclassification, not evidence of biological recovery.

## Shrinkage-adjusted associations

{md(associations)}

These prespecified five-variable ridge models are association analyses. They
are neither internally validated prediction models nor causal models.

## Observation-process association

{md(observation)}

Follow-up duration and visit count occur after baseline and cannot be treated
as baseline predictors. Their association with sustained classification
quantifies ascertainment/guarantee-time bias.

## ADNI phase distribution

{md(phase)}

## Clinical-core missingness

{md(missingness)}

## Interpretation

Sustained reverters differ from persistent-MCI and progressive trajectories,
but their substantially longer and deeper follow-up makes unrestricted
end-of-follow-up labels observation-dependent. The 50-person strongest
definition does not remove this dependence; it increases confirmation at the
cost of greater guarantee-time selection.
"""
    (REPORT_ROOT / "phase5_adni_diagnosis_reversion_report.md").write_text(
        reversion_report, encoding="utf-8"
    )

    transport_summary = transport[
        transport["metric"].isin(
            [
                "auc_improvement",
                "auc_progression",
                "multistate_brier",
                "calibration_slope_improvement",
                "calibration_slope_progression",
                "observed_expected_improvement",
                "observed_expected_progression",
            ]
        )
    ]
    recal_selected = recalibration[
        recalibration["model_id"].isin(["B1", "D0", "D1"])
    ][
        [
            "model_id",
            "horizon_months",
            "method",
            "auc_improvement",
            "auc_progression",
            "multistate_brier",
            "calibration_slope_improvement",
            "observed_expected_improvement",
            "calibration_slope_progression",
            "observed_expected_progression",
            "validation_design",
        ]
    ]
    transport_report = f"""# Phase 5 Final Transportability Synthesis

Generated: {GENERATED_AT}

## Classification

**TRANSPORTABLE WITH RECALIBRATION**

The untouched OASIS-3 results are the primary external-validation evidence.
OASIS-derived intercept and slope updates are reported separately as model
updating.

{md(transportability_framework)}

## Internal-to-external comparison

{md(transport_summary)}

## Untouched-versus-recalibrated performance

{md(recal_selected)}

## Cross-fitted recalibration parameters

{md(recalibration_parameters)}

Discrimination is shown for transparency. It is not expected to change under
a single monotonic transform, but may vary slightly here because each
participant's updated probability came from a different cross-fitted
calibration model. No original predictor coefficient was refit.

## Interpretation

D1 retained useful external discrimination and improved over D0, most clearly
for 12-month progression and at both transitions at 36 months. Improvement
risk was materially underpredicted in OASIS-3. Completed shift analyses point
to higher OASIS improvement incidence, age/education and follow-up
differences, MMSE timing differences, and endpoint/assessment-process
differences. No primary model was changed in response.
"""
    (REPORT_ROOT / "phase5_transportability_synthesis.md").write_text(
        transport_report, encoding="utf-8"
    )

    overlap_report = f"""# Phase 5 CDR-versus-Diagnosis Improvement Agreement

Generated: {GENERATED_AT}

## Agreement

{md(agreement)}

## Disagreement categories

{md(disagreement)}

## Timing among participants meeting both definitions

{md(agreement_timing.describe(include="all").reset_index())}

The modest Jaccard similarity demonstrates that observed "reversion" depends
strongly on the clinical-state definition. CDR 0.5 is not identical to
clinical MCI, and neither CDR improvement nor diagnosis reversion establishes
biological recovery.
"""
    (REPORT_ROOT / "phase5_cdr_diagnosis_agreement_report.md").write_text(
        overlap_report, encoding="utf-8"
    )

    aibl_report = f"""# Phase 5 AIBL Descriptive Analysis

Generated: {GENERATED_AT}

{md(aibl)}

AIBL has 241 eligible CDR 0.5 index participants, but outcome observation
falls rapidly with the horizon. All supplied age values are unverified
year-only proxies, education is missing for every index participant, and only
66 participants have at least two MMSE measurements. Performance estimates
would be unstable and potentially misleading. AIBL is therefore descriptive
only; no prediction or calibration estimate is reported.
"""
    (REPORT_ROOT / "phase5_aibl_descriptive_analysis.md").write_text(
        aibl_report, encoding="utf-8"
    )

    final_report = f"""# Phase 5 Final Report

Generated: {GENERATED_AT}

## Final scientific conclusion

Longitudinal MMSE updating in D1 added externally reproducible value over the
static D0 comparator while retaining useful discrimination in OASIS-3.
Untouched improvement probabilities were substantially miscalibrated because
improvement was more common in OASIS-3; cross-fitted local updating improved
prediction error but is not untouched validation. The main result remained
stable under reconstructed global CDR.

## Primary study synthesis

{md(external[[
    "model_id", "horizon_months", "rows", "participants",
    "improvement_events", "progression_events", "auc_improvement",
    "auc_progression", "c_index_improvement", "c_index_progression",
    "multistate_brier", "calibration_slope_improvement",
    "observed_expected_improvement", "calibration_slope_progression",
    "observed_expected_progression"
]])}

## Diagnosis-based reversion

The secondary ADNI study reproduced 60 sustained and 50 strongest reverters,
399 progressors, and 746 persistent-MCI participants with follow-up. Follow-up
was much longer for sustained reverters, so the unrestricted label remains
vulnerable to guarantee-time and ascertainment bias.

## Endpoint-definition comparison

{md(agreement)}

## Distribution and endpoint sensitivity

{md(shift)}

{md(endpoint)}

## Limitations and bias control

{md(limitations)}

## Scope boundaries

- OASIS `CDRTOT` is a provisional de facto global CDR in the delivered export.
- Death and dropout reasons are incomplete.
- CDR 0.5 is not identical to clinical MCI.
- MMSE and CDR measure overlapping cognitive domains.
- OASIS recalibration is model updating.
- AIBL is descriptive only.
- MRI was excluded because harmonized sample support was inadequate.

## Submission readiness and exact remaining manual tasks

All computational Phase 5 analyses, aggregate tables, figures, model
documentation, and the manuscript draft are complete subject to independent
QC. Before submission, authors must:

1. add the final author list, affiliations, ORCID identifiers, and
   corresponding-author details;
2. insert cohort-specific ethics/IRB, informed-consent, and data-use language
   from the governing agreements;
3. add funding, consortium acknowledgements, competing interests, and author
   contribution statements;
4. complete and verify literature citations and reference formatting;
5. confirm the selected journal's current word, abstract, figure/table,
   reporting-checklist, and data-availability requirements;
6. perform a final author review of clinical wording and all figure legends;
7. export journal-formatted tables/figures and complete any required
   repository or data-access statements without redistributing raw cohort
   data.

## Phase 5 decision

**READY FOR MANUSCRIPT SUBMISSION AFTER MANUAL AUTHOR/ETHICS/DATA-SHARING
COMPLETION.**
"""
    (REPORT_ROOT / "phase5_final_report.md").write_text(
        final_report, encoding="utf-8"
    )

    executive = f"""# Project Executive Summary

Generated: {GENERATED_AT}

This project developed participant-grouped, ridge competing-risk models in
ADNI for bidirectional transitions from CDR 0.5 to improvement (CDR 0) or
progression (CDR >=1), and externally evaluated the unchanged models in
OASIS-3. D1, the dynamic model using updated MMSE, outperformed D0 externally
and retained AUCs of {external.query("model_id == 'D1' and horizon_months == 12").iloc[0].auc_improvement:.3f}/
{external.query("model_id == 'D1' and horizon_months == 12").iloc[0].auc_progression:.3f}
at 12 months and
{external.query("model_id == 'D1' and horizon_months == 36").iloc[0].auc_improvement:.3f}/
{external.query("model_id == 'D1' and horizon_months == 36").iloc[0].auc_progression:.3f}
at 36 months. Improvement was about twice as common in OASIS, producing
substantial underprediction. Cross-fitted recalibration improved error but is
model updating.

Diagnosis-based sustained MCI reversion in ADNI overlapped only partially with
CDR improvement (Jaccard {agreement.iloc[0].jaccard_similarity:.3f}), showing
that "reversion" is definition-dependent. AIBL lacks sufficient verified
demographics and longitudinal depth for confirmatory validation.

Strongest contribution: calibration-first, bidirectional dynamic prediction
with genuine locked external validation and explicit endpoint-definition
sensitivity.

Main limitation: cross-cohort outcome and observation-process differences,
especially the provisional OASIS endpoint and incomplete death/dropout data.
"""
    (REPORT_ROOT / "phase5_project_executive_summary.md").write_text(
        executive, encoding="utf-8"
    )


def manuscript_draft(
    participant: pd.DataFrame,
    external: pd.DataFrame,
    internal: pd.DataFrame,
    agreement: pd.DataFrame,
    timing: pd.DataFrame,
    aibl: pd.DataFrame,
) -> None:
    d1_12 = external.query(
        "model_id == 'D1' and horizon_months == 12"
    ).iloc[0]
    d1_36 = external.query(
        "model_id == 'D1' and horizon_months == 36"
    ).iloc[0]
    bootstrap_ci = pd.read_csv(
        PHASE4_TABLES / "participant_bootstrap_confidence_intervals.csv"
    )

    def auc_ci(horizon: int, metric: str) -> tuple[float, float]:
        row = bootstrap_ci[
            bootstrap_ci["model_id"].eq("D1")
            & bootstrap_ci["horizon_months"].eq(horizon)
            & bootstrap_ci["metric"].eq(metric)
        ].iloc[0]
        return float(row["ci_low"]), float(row["ci_high"])

    d1_12_i_ci = auc_ci(12, "auc_improvement")
    d1_12_p_ci = auc_ci(12, "auc_progression")
    d1_36_i_ci = auc_ci(36, "auc_improvement")
    d1_36_p_ci = auc_ci(36, "auc_progression")
    draft = f"""# Manuscript Draft

Generated from frozen machine-readable results: {GENERATED_AT}

## Title options

1. Dynamic and Externally Validated Prediction of Cognitive-Stage
   Improvement and Progression in Alzheimer Cohorts
2. Calibration and Transportability of Longitudinal Multi-State Cognitive
   Prediction from ADNI to OASIS-3
3. Updating Bidirectional Cognitive-Stage Risk with Longitudinal MMSE:
   Internal and External Validation

## Abstract

### Background

Prediction studies in Alzheimer disease frequently treat cognitive decline as
irreversible, use baseline-only classifiers, and emphasize discrimination
without external calibration. We evaluated whether a parsimonious dynamic
model could update bidirectional risks of improvement and progression from
Clinical Dementia Rating (CDR) 0.5 and transport from ADNI to OASIS-3.

### Methods

Participant-visit and landmark datasets were constructed for ADNI and
OASIS-3. Outcomes were transition from CDR 0.5 to CDR 0, persistence at 0.5,
or progression to CDR >=1 over 12- and 36-month primary horizons. Ridge
cause-specific models were tuned and validated with nested participant-grouped
folds in ADNI. B1 was baseline-only, D0 was the static dynamic comparator, and
D1 updated MMSE and change from index. Models, preprocessing, penalties,
coefficients, and baseline hazards were frozen before any OASIS outcome was
opened. External discrimination, calibration, prediction error, and
participant-bootstrap confidence intervals were assessed unchanged.
Reconstructed six-box CDR and OASIS cross-fitted recalibration were secondary.
Diagnosis-based sustained MCI reversion was examined separately in ADNI.

### Results

D1 included 845 OASIS landmarks from 437 participants. At 12 months its
untouched improvement/progression AUCs were {d1_12.auc_improvement:.3f}
(95% participant-bootstrap CI {d1_12_i_ci[0]:.3f}-{d1_12_i_ci[1]:.3f}) and
{d1_12.auc_progression:.3f} (95% CI
{d1_12_p_ci[0]:.3f}-{d1_12_p_ci[1]:.3f}); at 36 months they were
{d1_36.auc_improvement:.3f} (95% CI
{d1_36_i_ci[0]:.3f}-{d1_36_i_ci[1]:.3f}) and
{d1_36.auc_progression:.3f} (95% CI
{d1_36_p_ci[0]:.3f}-{d1_36_p_ci[1]:.3f}). On identical
landmarks, D1 exceeded D0 for 12-month progression and both transitions at 36
months. Improvement O/E was {d1_12.observed_expected_improvement:.2f} at 12
months and {d1_36.observed_expected_improvement:.2f} at 36 months, reflecting
substantial underprediction as improvement occurred more often in OASIS-3.
Cross-fitted recalibration improved prediction error but was model updating,
not untouched validation. Reconstructed CDR did not materially change the
main conclusions. Among 1,600 ADNI participants with baseline clinical MCI,
60 showed sustained diagnosis reversion, 50 met the strongest definition, 399
progressed, and 746 remained MCI with follow-up. Diagnosis reversion and CDR
improvement overlapped in {int(agreement.iloc[0].overlap_participants)}
participants (Jaccard {agreement.iloc[0].jaccard_similarity:.3f}).

### Conclusions

Longitudinal cognitive updating added transportable discrimination for
bidirectional cognitive-stage transitions, but external calibration depended
strongly on cohort event incidence and assessment process. The model is
research-grade and requires local recalibration before new-cohort use.
Observed reversion is sensitive to clinical-state definition and should not be
interpreted as biological recovery.

## Introduction

Many Alzheimer disease prediction studies frame mild impairment as a
one-directional conversion problem. This framing ignores observed improvement,
oscillation between clinical states, and the practical need to update risk as
new assessments become available. It also risks overstating usefulness when
models are evaluated only in the development cohort.

External validation and calibration address different questions.
Discrimination asks whether participants are ranked appropriately, whereas
calibration asks whether probabilities correspond to observed risks. A model
may preserve rank ordering yet systematically under- or over-predict when
transported to a cohort with different recruitment, follow-up, assessment
timing, or outcome incidence.

We therefore conducted a calibration-first dynamic multi-state prediction
study. Models were developed and internally validated in ADNI and evaluated
unchanged in OASIS-3. We modelled improvement and progression from CDR 0.5,
compared longitudinal D1 with the static dynamic comparator D0, examined
endpoint reconstruction and recalibration, and completed a separate
diagnosis-based ADNI reversion analysis.

## Methods

### Cohorts and roles

ADNI was the sole development cohort. OASIS-3 was locked external validation.
AIBL was descriptive because verified age and education and repeated dynamic
MMSE support were inadequate. No OASIS participant-level information informed
ADNI fitting, tuning, preprocessing, or model selection.

### CDR endpoint and landmarks

The index state was CDR 0.5. Improvement was the first transition to CDR 0 and
progression the first transition to CDR >=1; otherwise the participant
remained at CDR 0.5 or was censored at the last classifiable assessment.
Primary horizons were 12 and 36 months, with 24 and 60 months secondary where
frozen artifacts existed. The delivered OASIS `CDRTOT` field was treated as a
provisional de facto global CDR. Reconstructed global CDR from six boxes was
secondary.

### Predictors and models

The portable clinical core contained age, sex, education, APOE epsilon-4 dose,
baseline MMSE, and for dynamic models elapsed time, current MMSE, and MMSE
change. MMSE and CDR overlap cognitively; D3 therefore required strictly
pre-landmark MMSE. MRI was excluded because harmonized sample support was
inadequate. Ridge cause-specific models estimated improvement and progression
hazards, which were combined into coherent three-state probabilities.

### Validation and leakage control

All transformations and tuning occurred inside ADNI participant-grouped
folds. No participant crossed validation partitions. OASIS features were built
without outcomes, predictions were frozen and hashed, an independent script
regenerated them, and only then were outcomes opened. Participant-clustered
bootstrap sampling retained all landmarks for a sampled participant.

### External evaluation and recalibration

Untouched OASIS evaluation included time-dependent AUC, concordance, Brier and
integrated Brier scores, multistate error, calibration intercept/slope, O/E,
remaining-state calibration, and probability coherence. Cross-fitted
calibration-in-the-large and intercept/slope recalibration used OASIS data only
after untouched results were frozen and are labelled model updating.

### Diagnosis-based reversion

ADNI participants whose first classifiable diagnosis was MCI were grouped as
sustained reversion, strongest confirmed reversion, persistent MCI,
progression, or unstable/single-CN trajectory. Sustained reversion required at
least two terminal CN assessments after the last MCI and no later MCI/AD.
The strongest definition additionally required at least two consecutive MCI
assessments. Five prespecified clinical-core covariates were evaluated in
low-dimensional ridge association models. Follow-up and visit-count analyses
quantified ascertainment rather than baseline prediction.

### Statistical interpretation

Prediction, association, and causal inference were separated. No causal effect
was estimated. Reversion labels describe observed clinical states, not
biological recovery. Death and dropout reasons were incomplete, so censoring
at the last assessment may be informative.

## Results

### Internal validation

{md(internal[[
    "model_id", "horizon_months", "rows", "participants",
    "auc_improvement", "auc_progression", "c_index_improvement",
    "c_index_progression", "multistate_brier",
    "calibration_slope_improvement", "observed_expected_improvement",
    "calibration_slope_progression", "observed_expected_progression"
]])}

### Untouched external validation

{md(external[[
    "model_id", "horizon_months", "rows", "participants",
    "improvement_events", "progression_events",
    "auc_improvement", "auc_progression", "c_index_improvement",
    "c_index_progression", "multistate_brier",
    "calibration_slope_improvement", "observed_expected_improvement",
    "calibration_slope_progression", "observed_expected_progression"
]])}

The 12-month B1 progression analysis contained only 18 events and was
low-precision. D1 preserved useful discrimination but substantially
underpredicted improvement. OASIS improvement incidence was approximately
twice ADNI incidence.

### Dynamic value, endpoint sensitivity, and recalibration

D1 improved progression AUC over D0 by approximately 0.071 at 12 months. At
36 months it improved improvement/progression AUC by approximately 0.024/0.029
and reduced multistate Brier error by approximately 0.014. Reconstructed
six-box CDR changed D1 AUC by no more than about 0.012 in absolute value and
did not reverse conclusions. Recalibration reduced D1 36-month multistate
Brier from 0.707 to 0.629 with intercept-and-slope updating, but this estimate
used OASIS outcomes and is not untouched validation.

### Diagnosis reversion and endpoint agreement

{md(timing)}

Sustained reverters had substantially longer follow-up and more assessments
than persistent-MCI participants. Among baseline clinical-MCI participants,
CDR improvement was observed in {int(agreement.iloc[0].cdr_defined_improvers)}
and sustained diagnosis reversion in
{int(agreement.iloc[0].sustained_diagnosis_reverters)}; only
{int(agreement.iloc[0].overlap_participants)} met both. This partial overlap
supports outcome-definition dependence.

### AIBL

{md(aibl)}

No AIBL performance estimate was calculated.

## Discussion

### Principal findings

This study demonstrates that a low-dimensional dynamic model can retain useful
external discrimination for bidirectional CDR transitions while exposing
important calibration limits. D1's improvement over D0 shows that updated
cognitive information adds value beyond merely re-issuing a baseline model at
successive landmarks. The gain was most consistent for progression, whereas
improvement probabilities were sensitive to event-incidence shift.

### Calibration and transportability

The model was classified as transportable with recalibration, not
transportable unchanged. This distinction is central. Untouched OASIS results
measure genuine transport; recalibrated results estimate performance after
local model updating. Higher improvement incidence, case-mix differences,
longer and differently scheduled follow-up, and endpoint/assessment-process
differences plausibly explain miscalibration.

### Reversion is definition-dependent

Diagnosis-based sustained reversion and CDR-defined improvement overlapped
only partially. Diagnosis requires clinician classification and the sustained
definition uses future confirmation, whereas CDR improvement reflects a
staging threshold and can be transient. Neither should be labelled biological
recovery. Long follow-up and repeated assessment strongly influence who can
meet a sustained definition.

### Strengths

Strengths include participant-grouped validation, a genuinely locked external
test, outcome-hidden prediction freezing, calibration-first reporting,
participant-clustered uncertainty, explicit modelling of improvement and
progression, reconstructed-endpoint sensitivity, independent verification,
and transparent negative calibration findings.

### Limitations

OASIS `CDRTOT` was a provisional de facto global CDR in the delivered export.
Death and dropout reasons were incomplete and may induce informative
censoring. CDR 0.5 is not identical to clinical MCI. MMSE and CDR assess
overlapping domains. OASIS recruitment, follow-up, and improvement incidence
differed from ADNI. Recalibration used OASIS data. AIBL was descriptive only.
MRI was excluded because harmonized support was insufficient. The diagnosis
reversion analysis is vulnerable to guarantee-time and ascertainment bias.

### Clinical implications

The results support continued investigation of longitudinal risk updating and
local calibration, but not immediate clinical deployment. A new site must
audit endpoint construction, predictor mapping, and event incidence and then
perform local recalibration before any research use. Decision thresholds
should not be adopted from this study.

## Conclusion

Dynamic updating improved externally observed transition prediction over a
static dynamic comparator, but calibration transported less well than
discrimination. Explicit local recalibration and endpoint auditing are
necessary. The strongest contribution is a reproducible, calibration-first
demonstration of bidirectional dynamic prediction under genuine external
validation. The principal limitation is cross-cohort outcome and observation
process heterogeneity.

## Data availability

ADNI, OASIS-3, and AIBL data remain subject to their respective access and
data-use agreements. Derived manifests, aggregate tables, scripts, checksums,
and executed notebooks are organized in the project workspace. Raw
participant-level data cannot be redistributed through the manuscript
package.

## Manual completion placeholders

- Authors, affiliations, ORCID identifiers, and corresponding author.
- Ethics/IRB and cohort consent language copied from each data-use agreement.
- Funding and consortium acknowledgements.
- Competing interests.
- Final literature citations and reference formatting.
- Journal-specific word/display-item limits and reporting checklist.
"""
    (REPORT_ROOT / "phase5_manuscript_draft.md").write_text(
        draft, encoding="utf-8"
    )


def final_model_card(
    external: pd.DataFrame,
    recalibration: pd.DataFrame,
) -> None:
    card = f"""# Final Research Model Card

Generated: {GENERATED_AT}

## Model identity

- Selected model: D1 dynamic ridge competing-transition model.
- Comparators/sensitivities: B1, D0, B3, D3.
- Frozen source: `models/phase3/phase4_lock_v2`.
- Horizons: 12 and 36 months primary; 24 and 60 months secondary where frozen.
- States: CDR 0 improvement, CDR 0.5 persistence, CDR >=1 progression.

## Intended use

Research evaluation of bidirectional cognitive-stage transition probabilities
among cohort participants observed at CDR 0.5, using harmonized longitudinal
clinical data and local validation.

## Non-intended use

The model is **research-grade and is not approved for clinical
decision-making**. It must not diagnose Alzheimer disease, select treatment,
determine eligibility, replace clinician judgment, infer biological recovery,
or estimate causal treatment effects.

## Populations

Developed in ADNI and externally evaluated unchanged in OASIS-3. AIBL was not
used for performance validation. Generalizability outside research cohorts,
under-represented demographic groups, non-AD dementias, and clinical service
populations is unknown.

## Predictors

Age, sex, education, APOE epsilon-4 dose, baseline MMSE, elapsed time, current
MMSE, and MMSE change from index. Frozen preprocessing includes ADNI-derived
imputation, robust scaling, and categorical levels. D3 uses only strictly
pre-landmark MMSE.

## Internal validation

Nested participant-grouped folds, participant-clustered bootstrap intervals,
and leakage-safe preprocessing were used. Every landmark from one participant
remained in one fold.

## External validation

{md(external[external["model_id"].eq("D1")][[
    "horizon_months", "rows", "participants", "improvement_events",
    "progression_events", "auc_improvement", "auc_progression",
    "c_index_improvement", "c_index_progression", "multistate_brier",
    "calibration_slope_improvement", "observed_expected_improvement",
    "calibration_slope_progression", "observed_expected_progression"
]])}

## Calibration and model updating

Untouched improvement risk was underpredicted. OASIS cross-fitted
calibration-in-the-large and intercept/slope updating improved calibration and
error:

{md(recalibration[recalibration["model_id"].eq("D1")][[
    "horizon_months", "method", "multistate_brier",
    "calibration_slope_improvement", "observed_expected_improvement",
    "calibration_slope_progression", "observed_expected_progression"
]])}

These updated results are not untouched validation.

## Endpoint and missingness limitations

OASIS `CDRTOT` is provisional; reconstructed six-box CDR was consistent but
did not eliminate endpoint uncertainty. CDR 0.5 is not clinical MCI. MMSE and
CDR overlap. APOE and longitudinal MMSE availability vary by cohort and phase.
Death/dropout reasons are incomplete.

## Subgroup limitations

Clinical-MCI subgroup event counts were small. Performance by race/ethnicity,
site, socioeconomic status, and treatment exposure is not adequately
established. AIBL cannot support confirmation under the current extract.

## Deployment restrictions

1. Reproduce the input schema and endpoint locally.
2. Verify probabilities and deterministic inference.
3. Evaluate discrimination and calibration with participant-grouped methods.
4. Perform required local recalibration before use in a new cohort.
5. Do not reuse OASIS-updated parameters as if they were untouched validation.
6. Do not use the model clinically without prospective validation, governance,
   regulatory review, and decision-impact assessment.
"""
    (REPORT_ROOT / "phase5_final_model_card.md").write_text(
        card, encoding="utf-8"
    )


def alternate_directions() -> None:
    content = """# Alternate Study Directions

These projects are preserved as future work. They were not rejected; each is
blocked by current data support or harmonization.

| Direction | Scientific gap | Current blocker | Required new data or harmonization | Recommended future scope |
|---|---|---|---|---|
| Real missing-modality and cohort-level modality-shift modelling | Most models treat protocol-driven missingness as ordinary missingness | Modalities and assay availability differ structurally by cohort/phase | Cohort-aligned modality indicators, assay metadata, and external cohorts with overlapping modalities | Pre-register availability-aware models and test whole-modality absence without leakage |
| Vascular-neurodegenerative interaction analysis | Vascular burden may modify progression and improvement pathways | Vascular definitions and timing are not harmonized across cohorts | Validated hypertension, diabetes, stroke, WMH, medication, and timing maps | ADNI discovery with an independently harmonized replication cohort |
| Biomarker-discordance and resilience phenotyping | Cognitive trajectories may diverge from amyloid/tau status | Quantitative assays and thresholds are not cross-cohort equivalent | Harmonized amyloid/tau measures, assay batches, thresholds, and sufficient longitudinal outcomes | Descriptive phenotypes followed by externally validated trajectory analysis |
| MRI-enriched transportability | Imaging may add value beyond the clinical core | Scanner/processing heterogeneity and low harmonized sample retention | Common MRI pipeline, intracranial-volume validation, scanner/site metadata, larger overlap | Locked clinical-core versus prespecified MRI increment with external validation |
| Diagnosis-based multi-state reversion after adjudication | Clinical diagnosis reversion is under-modelled | OASIS terminology/form changes and incomplete endpoint adjudication | Common diagnosis ontology, consensus/adjudication fields, death/dropout data | Low-dimensional competing-transition analysis with fixed horizons |
| Richer AIBL validation | A third cohort could test wider transport | Verified age unavailable, education missing, sparse repeated MMSE/imaging | Validated demographics, richer longitudinal CDR/MMSE, quantitative MRI, outcome follow-up | Descriptive QC first, then locked reduced-core validation if event thresholds pass |
"""
    path = ROOT / "LiteratureAndPlan" / "Alternate_Study_Directions.md"
    path.write_text(content, encoding="utf-8")


def reproducibility_report(
    source_manifest: pd.DataFrame,
    table_paths: dict[str, Path],
) -> None:
    report = f"""# Phase 5 Reproducibility Report

Generated: {GENERATED_AT}

## Environment

- Python: {platform.python_version()}
- numpy: {np.__version__}
- pandas: {pd.__version__}
- scipy: {scipy.__version__}
- scikit-learn: {sklearn.__version__}
- random seed: {RANDOM_SEED}
- association bootstrap replicates: {BOOTSTRAP_REPLICATES}

## Frozen boundaries

No B1, D0, D1, B3, or D3 model was refit or modified. Phase 3/4 performance,
calibration, bootstrap, and endpoint-sensitivity values were read from their
machine-readable frozen outputs. The only fitted Phase 5 models are explicitly
labelled low-dimensional ridge **association** models for diagnosis reversion.

## Reversion reconstruction

Raw ADNI diagnosis records were mapped 1=CN, 2=MCI, 3=AD/dementia-coded,
unmapped records were excluded, and same-time conflicts were collapsed to the
more severe state. Exact prior counts were hard assertions: 1,600 baseline
MCI, 60 sustained reversion, 50 strongest reversion, 399 progression, and 746
persistent MCI with follow-up.

## Source checksums

{md(source_manifest)}

## Manuscript tables

{md(pd.DataFrame([
    {"table": name, "path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    for name, path in table_paths.items()
]))}

## Figure regeneration

All 12 Phase 5 figures are generated by
`scripts/phase5_secondary_synthesis.py` from saved CSV tables. SVG companions
are saved for every figure. Final DPI and saved-artifact checks are performed
by the separate `scripts/phase5_qc.py`.
"""
    (REPORT_ROOT / "phase5_reproducibility_report.md").write_text(
        report, encoding="utf-8"
    )


def source_manifest() -> pd.DataFrame:
    paths = [
        DXSUM_PATH,
        MMSE_PATH,
        DEMOGRAPHICS_PATH,
        APOE_PATH,
        CDR_PATH,
        CDR_MANIFEST_PATH,
        ROOT / "LiteratureAndPlan" / "Plan.md",
        ROOT
        / "reports"
        / "feasibility"
        / "sustained_mci_reversion_feasibility_audit.md",
        PHASE3_TABLES / "discrimination_and_horizon_metrics.csv",
        PHASE3_TABLES / "bootstrap_confidence_intervals.csv",
        PHASE4_TABLES / "untouched_external_metrics.csv",
        PHASE4_TABLES / "participant_bootstrap_confidence_intervals.csv",
        PHASE4_TABLES / "recalibration_effect_comparison.csv",
        PHASE4_TABLES / "reconstructed_cdr_metric_comparison.csv",
        PHASE4_TABLES / "predictor_distribution_shift.csv",
        PHASE4_TABLES / "event_rate_shift.csv",
        PHASE4_TABLES / "phase4_run_metadata.json",
    ]
    return pd.DataFrame(
        [
            {
                "relative_path": rel(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "access_role": (
                    "participant-level Phase 5 secondary source"
                    if path
                    in {
                        DXSUM_PATH,
                        MMSE_PATH,
                        DEMOGRAPHICS_PATH,
                        APOE_PATH,
                        CDR_PATH,
                        CDR_MANIFEST_PATH,
                    }
                    else "completed Phase 1-4 source of truth"
                ),
            }
            for path in paths
        ]
    )


def main() -> None:
    sources = source_manifest()
    save_csv(sources, TABLE_ROOT / "phase5_source_provenance.csv")

    diagnosis = ordered_diagnoses()
    participants, diagnosis_records = trajectory_records(diagnosis)
    participants = add_clinical_predictors(participants)
    participants = add_cdr_comparison(participants)

    save_csv(
        participants,
        PROCESSED_ROOT / "diagnosis_reversion_participant_analysis.csv",
    )
    save_csv(
        diagnosis_records,
        PROCESSED_ROOT / "diagnosis_reversion_ordered_records.csv",
    )

    counts = (
        participants["trajectory_group"]
        .value_counts()
        .rename_axis("trajectory_group")
        .rename("participants")
        .reset_index()
    )
    counts = pd.concat(
        [
            counts,
            pd.DataFrame(
                [
                    {
                        "trajectory_group": "strongest_reversion_nested",
                        "participants": int(
                            participants["strongest_reversion_flag"].sum()
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    save_csv(counts, TABLE_ROOT / "reversion_group_counts.csv")

    characteristics = group_characteristics(participants)
    tests = group_comparison_tests(participants)
    timing = timing_relapse_summary(participants)
    associations = adjusted_associations(participants)
    observation = observation_process_association(participants)
    opportunity = followup_opportunity(participants)
    fixed_horizon = fixed_horizon_diagnosis(diagnosis_records)
    phase = phase_distribution(participants)
    missingness = missingness_by_group(participants)
    agreement, disagreement, agreement_timing, agreement_chars = (
        agreement_tables(participants)
    )

    save_csv(
        characteristics, TABLE_ROOT / "reversion_group_characteristics.csv"
    )
    save_csv(tests, TABLE_ROOT / "reversion_group_comparison_tests.csv")
    save_csv(timing, TABLE_ROOT / "reversion_timing_and_relapse.csv")
    save_csv(
        associations, TABLE_ROOT / "reversion_adjusted_associations.csv"
    )
    save_csv(
        observation,
        TABLE_ROOT / "reversion_observation_process_associations.csv",
    )
    save_csv(
        opportunity, TABLE_ROOT / "reversion_followup_opportunity.csv"
    )
    save_csv(
        fixed_horizon, TABLE_ROOT / "diagnosis_fixed_horizon_summary.csv"
    )
    save_csv(phase, TABLE_ROOT / "reversion_adni_phase_distribution.csv")
    save_csv(
        missingness,
        TABLE_ROOT / "reversion_clinical_core_missingness.csv",
    )
    save_csv(
        agreement, TABLE_ROOT / "cdr_diagnosis_agreement_summary.csv"
    )
    save_csv(
        disagreement,
        TABLE_ROOT / "cdr_diagnosis_disagreement_categories.csv",
    )
    save_csv(
        agreement_timing,
        TABLE_ROOT / "cdr_diagnosis_timing_comparison.csv",
    )
    save_csv(
        agreement_chars,
        TABLE_ROOT / "cdr_diagnosis_agreement_characteristics.csv",
    )

    cohorts = cohort_characteristics()
    transitions = transition_counts_table()
    internal = internal_performance_table()
    external = external_performance_table()
    transport = transportability_table()
    recalibration = calibration_recalibration_table()
    recalibration_parameters = recalibration_parameter_summary()
    transportability_framework = (
        transportability_classification_framework()
    )
    shift = shift_summary_table()
    endpoint = endpoint_sensitivity_table()
    sensitivities = sensitivity_summary_table()
    aibl = aibl_descriptive_table()
    limitations = limitations_bias_table()
    trajectories = representative_trajectories()

    save_csv(cohorts, TABLE_ROOT / "cohort_characteristics.csv")
    save_csv(transitions, TABLE_ROOT / "transition_counts_by_horizon.csv")
    save_csv(
        internal, TABLE_ROOT / "adni_internal_performance_synthesis.csv"
    )
    save_csv(
        external, TABLE_ROOT / "oasis_untouched_performance_synthesis.csv"
    )
    save_csv(
        transport, TABLE_ROOT / "internal_external_transportability.csv"
    )
    save_csv(
        recalibration, TABLE_ROOT / "calibration_recalibration_synthesis.csv"
    )
    save_csv(
        recalibration_parameters,
        TABLE_ROOT / "recalibration_parameter_summary.csv",
    )
    save_csv(
        transportability_framework,
        TABLE_ROOT / "transportability_classification_framework.csv",
    )
    save_csv(shift, TABLE_ROOT / "distribution_shift_synthesis.csv")
    save_csv(endpoint, TABLE_ROOT / "endpoint_sensitivity_synthesis.csv")
    save_csv(sensitivities, TABLE_ROOT / "sensitivity_analysis_summary.csv")
    save_csv(aibl, TABLE_ROOT / "aibl_descriptive_summary.csv")
    save_csv(limitations, TABLE_ROOT / "limitations_bias_controls.csv")
    save_csv(
        trajectories,
        TABLE_ROOT / "representative_dynamic_risk_trajectories.csv",
    )

    table_paths = manuscript_tables(
        cohorts,
        transitions,
        internal,
        external,
        recalibration,
        shift,
        endpoint,
        characteristics,
        agreement,
        sensitivities,
        limitations,
    )
    create_figures(
        cohorts,
        transport,
        recalibration,
        shift,
        agreement,
        timing,
        trajectories,
        external,
    )
    phase5_reports(
        participants,
        characteristics,
        timing,
        associations,
        observation,
        phase,
        missingness,
        agreement,
        disagreement,
        agreement_timing,
        internal,
        external,
        transport,
        recalibration,
        recalibration_parameters,
        transportability_framework,
        shift,
        endpoint,
        aibl,
        limitations,
    )
    manuscript_draft(
        participants, external, internal, agreement, timing, aibl
    )
    final_model_card(external, recalibration)
    alternate_directions()
    reproducibility_report(sources, table_paths)

    metadata = {
        "phase": 5,
        "generated_at_utc": GENERATED_AT,
        "random_seed": RANDOM_SEED,
        "association_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "frozen_models_refit_or_modified": False,
        "phase3_phase4_outputs_used_as_source_of_truth": True,
        "diagnosis_reversion_counts": {
            "baseline_mci": 1600,
            "with_followup": 1250,
            "sustained": 60,
            "strongest": 50,
            "progression": 399,
            "persistent_with_followup": 746,
            "unstable_or_single_cn": 45,
            "no_later_assessment": 350,
        },
        "transportability_classification": (
            "TRANSPORTABLE WITH RECALIBRATION"
        ),
        "aibl_role": "descriptive only",
        "mri_role": "excluded from primary models",
        "final_qc_pending": True,
        "phase5_complete": False,
        "decision": "HOLD FOR FINAL PROJECT QC",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    save_json(metadata, TABLE_ROOT / "phase5_run_metadata.json")
    print(
        json.dumps(
            {
                "baseline_mci": len(participants),
                "sustained": int(
                    participants["sustained_reversion_flag"].sum()
                ),
                "strongest": int(
                    participants["strongest_reversion_flag"].sum()
                ),
                "cdr_improvers": int(
                    participants["cdr_defined_improvement_flag"].sum()
                ),
                "overlap": int(agreement.iloc[0]["overlap_participants"]),
                "manuscript_tables": len(table_paths),
                "figures": 12,
                "final_qc_pending": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
