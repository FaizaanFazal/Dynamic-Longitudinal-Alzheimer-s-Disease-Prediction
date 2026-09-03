#!/usr/bin/env python3
"""Build and audit the Phase 1 longitudinal CDR data foundation.

This module intentionally stops before feature engineering, imputation, or model
fitting. Raw inputs are read-only. Every derived artifact is written beneath
ProcessedData, results, or reports using the project's established conventions.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "phase1_analysis_parameters.json"
PROCESSED_ROOT = ROOT / "ProcessedData"
TABLE_ROOT = ROOT / "results" / "csvs" / "phase1"
PLOT_ROOT = ROOT / "results" / "plots" / "phase1"
REPORT_ROOT = ROOT / "reports" / "phase1"

with CONFIG_PATH.open(encoding="utf-8") as handle:
    PARAMS = json.load(handle)

DAYS_PER_MONTH = float(PARAMS["days_per_month"])
MMSE_WINDOW_DAYS = int(PARAMS["mmse_baseline_window_days"])
MATCH_TOLERANCE_DAYS = int(PARAMS["manifest_same_visit_match_tolerance_days"])
NEAR_DUPLICATE_DAYS = int(PARAMS["near_duplicate_days"])
HORIZONS = tuple(int(x) for x in PARAMS["prediction_horizons_months"])
VALID_CDR = {float(x) for x in PARAMS["valid_cdr_values"]}


@dataclass(frozen=True)
class CohortSources:
    cohort: str
    cdr: Path
    mmse: Path
    diagnosis: Path
    demographics: Path
    apoe: Path
    gds: Path | None = None
    codebook: Path | None = None


def _first(pattern: str, root: Path) -> Path:
    matches = sorted(p for p in root.rglob(pattern) if "__MACOSX" not in p.parts)
    if not matches:
        raise FileNotFoundError(f"No source file matched {pattern!r} below {root}")
    return matches[0]


ADNI_ROOT = ROOT / "datasets" / "adni"
OASIS_ROOT = ROOT / "datasets" / "oasis3"
AIBL_ROOT = ROOT / "datasets" / "aibl"

SOURCES = {
    "adni": CohortSources(
        cohort="adni",
        cdr=ADNI_ROOT / "All_Subjects_CDR_26Aug2025.csv",
        mmse=ADNI_ROOT / "All_Subjects_MMSE_26Aug2025.csv",
        diagnosis=ADNI_ROOT / "All_Subjects_DXSUM_26Aug2025.csv",
        demographics=ADNI_ROOT / "All_Subjects_PTDEMOG_26Aug2025.csv",
        apoe=ADNI_ROOT / "All_Subjects_APOERES_26Aug2025.csv",
        gds=ADNI_ROOT / "All_Subjects_GDSCALE_29Sep2025.csv",
    ),
    "oasis3": CohortSources(
        cohort="oasis3",
        cdr=_first("OASIS3_UDSb4_cdr.csv", OASIS_ROOT),
        mmse=_first("OASIS3_UDSb4_cdr.csv", OASIS_ROOT),
        diagnosis=_first("OASIS3_UDSd1_diagnoses.csv", OASIS_ROOT),
        demographics=_first("OASIS3_demographics.csv", OASIS_ROOT),
        apoe=_first("OASIS3_demographics.csv", OASIS_ROOT),
        gds=_first("OASIS3_UDSb6_gds.csv", OASIS_ROOT),
        codebook=_first("OASIS3_Clinical and Psychometrics_dictionary.xlsx", OASIS_ROOT),
    ),
    "aibl": CohortSources(
        cohort="aibl",
        cdr=_first("aibl_cdr_*.csv", AIBL_ROOT),
        mmse=_first("aibl_mmse_*.csv", AIBL_ROOT),
        diagnosis=_first("aibl_pdxconv_*.csv", AIBL_ROOT),
        demographics=_first("aibl_ptdemog_*.csv", AIBL_ROOT),
        apoe=_first("aibl_apoeres_*.csv", AIBL_ROOT),
    ),
}


def ensure_output_dirs() -> None:
    for path in [TABLE_ROOT, PLOT_ROOT, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    for cohort in SOURCES:
        (PROCESSED_ROOT / cohort).mkdir(parents=True, exist_ok=True)
        (TABLE_ROOT / cohort).mkdir(parents=True, exist_ok=True)
        (PLOT_ROOT / cohort).mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def iso_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d").where(dt.notna(), pd.NA)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_table(frame: pd.DataFrame, path: Path, parquet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if parquet:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)


def _format_values(values: Iterable[Any]) -> str:
    cleaned = []
    for value in values:
        if pd.isna(value):
            cleaned.append("<missing>")
        elif isinstance(value, float) and value.is_integer():
            cleaned.append(str(int(value)))
        else:
            cleaned.append(str(value))
    return " | ".join(sorted(set(cleaned)))


def _state_from_cdr(value: Any) -> float:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value) or float(value) not in VALID_CDR:
        return np.nan
    if float(value) == 0:
        return 0.0
    if float(value) == 0.5:
        return 1.0
    return 2.0


def state_label(value: Any) -> str:
    labels = {0: "CDR 0", 1: "CDR 0.5", 2: "CDR >=1"}
    return labels.get(int(value), "Unclassifiable") if pd.notna(value) else "Unclassifiable"


def _sex(value: Any) -> str | pd._libs.missing.NAType:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if value == 1:
        return "Male"
    if value == 2:
        return "Female"
    return pd.NA


def _apoe_string_and_dose(value: Any) -> tuple[Any, float]:
    if pd.isna(value):
        return pd.NA, np.nan
    text = str(value).strip().replace(".0", "")
    digits = re.findall(r"[234]", text)
    if len(digits) != 2:
        return pd.NA, np.nan
    return f"{digits[0]}/{digits[1]}", float(sum(x == "4" for x in digits))


def _aibl_apoe(row: pd.Series) -> tuple[Any, float]:
    a = pd.to_numeric(pd.Series([row.get("APGEN1")]), errors="coerce").iloc[0]
    b = pd.to_numeric(pd.Series([row.get("APGEN2")]), errors="coerce").iloc[0]
    if pd.isna(a) or pd.isna(b) or a not in {2, 3, 4} or b not in {2, 3, 4}:
        return pd.NA, np.nan
    return f"{int(a)}/{int(b)}", float((a == 4) + (b == 4))


def _stable_subject_record(
    frame: pd.DataFrame,
    participant_col: str,
    preferred_cols: list[str],
) -> pd.DataFrame:
    """Select one deterministic, most-complete record per participant."""
    work = frame.copy()
    work[participant_col] = work[participant_col].astype("string")
    work["_complete"] = work[preferred_cols].notna().sum(axis=1)
    work["_row"] = np.arange(len(work))
    work = work.sort_values(
        [participant_col, "_complete", "_row"], ascending=[True, False, True], kind="mergesort"
    )
    return work.drop_duplicates(participant_col, keep="first").drop(columns=["_complete", "_row"])


def _resolve_anchor_records(
    anchor: pd.DataFrame,
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve duplicate time points and retain unresolved conflicts explicitly."""
    required = {"participant_id", "time_sort_days", "cdr_global_source", "source_row_number"}
    missing = required - set(anchor.columns)
    if missing:
        raise ValueError(f"{cohort} anchor missing columns: {sorted(missing)}")

    work = anchor.copy()
    work["cdr_global_source"] = numeric(work["cdr_global_source"])
    work["cdr_is_valid"] = work["cdr_global_source"].isin(VALID_CDR)
    work["source_quality_error"] = numeric(work.get("source_quality_error", pd.Series(0, index=work.index))).fillna(0)
    clinical_cols = [
        c for c in ["cdr_global_source", "cdr_sum_boxes", "study_phase", "site_id", "original_visit_id"] if c in work
    ]
    work["_completeness"] = work[clinical_cols].notna().sum(axis=1)
    work["_update"] = pd.to_datetime(work.get("source_update_timestamp"), errors="coerce")

    invalid_audit = work.loc[~work["cdr_is_valid"]].copy()
    invalid_audit["invalid_reason"] = np.where(
        invalid_audit["cdr_global_source"].isna(), "missing_cdr", "cdr_outside_accepted_values"
    )

    selected_rows: list[pd.Series] = []
    conflict_rows: list[dict[str, Any]] = []
    group_cols = ["participant_id", "time_sort_days"]
    for (participant, time_value), group in work.groupby(group_cols, dropna=False, sort=False):
        ordered = group.sort_values(
            ["cdr_is_valid", "source_quality_error", "_completeness", "_update", "source_row_number"],
            ascending=[False, True, False, False, True],
            kind="mergesort",
        )
        chosen = ordered.iloc[0].copy()
        valid_values = sorted(set(ordered.loc[ordered["cdr_is_valid"], "cdr_global_source"].astype(float)))
        same_time_conflict = len(valid_values) > 1
        chosen["source_row_count_at_time"] = len(group)
        chosen["duplicate_participant_time_flag"] = len(group) > 1
        chosen["same_time_cdr_conflict_flag"] = same_time_conflict
        chosen["same_time_cdr_values"] = _format_values(group["cdr_global_source"])
        chosen["duplicate_resolution_rule"] = (
            "unresolved_conflicting_valid_cdr_values"
            if same_time_conflict
            else "valid_cdr_then_no_qc_error_then_completeness_then_latest_update_then_source_row"
        )
        chosen["duplicate_resolution_unresolved_flag"] = same_time_conflict
        chosen["cdr_global"] = np.nan if same_time_conflict or not valid_values else valid_values[0]
        chosen["cdr_state"] = _state_from_cdr(chosen["cdr_global"])
        chosen["cdr_state_label"] = state_label(chosen["cdr_state"])
        selected_rows.append(chosen)

        if len(group) > 1 or same_time_conflict:
            conflict_rows.append(
                {
                    "cohort": cohort,
                    "conflict_type": "cdr_duplicate_or_conflict",
                    "participant_id": participant,
                    "time_sort_days": time_value,
                    "visit_date": chosen.get("visit_date"),
                    "original_visit_ids": _format_values(group["original_visit_id"]),
                    "source_row_count": len(group),
                    "cdr_values": _format_values(group["cdr_global_source"]),
                    "diagnosis_values": "",
                    "source_tables": _format_values(group["source_cdr_table"]),
                    "resolution_applied": chosen["duplicate_resolution_rule"],
                    "unresolved_flag": same_time_conflict,
                    "selected_source_row": chosen["source_row_number"],
                }
            )

    manifest = pd.DataFrame(selected_rows).drop(columns=["_completeness", "_update"], errors="ignore")
    manifest = manifest.sort_values(["participant_id", "time_sort_days", "source_row_number"], kind="mergesort")
    if cohort == "oasis3":
        manifest["relative_time_days"] = manifest["time_sort_days"]
    else:
        first = manifest.groupby("participant_id")["time_sort_days"].transform("min")
        manifest["relative_time_days"] = manifest["time_sort_days"] - first
    manifest["relative_time_months"] = manifest["relative_time_days"] / DAYS_PER_MONTH
    previous = manifest.groupby("participant_id")["time_sort_days"].shift(1)
    following = manifest.groupby("participant_id")["time_sort_days"].shift(-1)
    manifest["days_from_previous_cdr"] = manifest["time_sort_days"] - previous
    manifest["near_duplicate_visit_flag"] = (
        manifest["days_from_previous_cdr"].between(0, NEAR_DUPLICATE_DAYS, inclusive="both")
        | (following - manifest["time_sort_days"]).between(0, NEAR_DUPLICATE_DAYS, inclusive="both")
    )
    manifest["manifest_key"] = (
        cohort + ":" + manifest["participant_id"].astype(str) + ":" + manifest["time_sort_days"].astype(str)
    )
    conflicts = pd.DataFrame(conflict_rows)
    return manifest.reset_index(drop=True), conflicts, invalid_audit.reset_index(drop=True)


def _measurement_frame(
    frame: pd.DataFrame,
    participant_col: str,
    visit_col: str,
    time_values: pd.Series,
    value_col: str,
    source_table: str,
    valid_min: float | None = None,
    valid_max: float | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "participant_id": frame[participant_col].astype("string").str.replace(r"\.0$", "", regex=True),
            "measurement_visit_id": frame[visit_col].astype("string"),
            "measurement_time_days": numeric(time_values),
            "measurement_value_source": frame[value_col],
            "measurement_value": numeric(frame[value_col]),
            "measurement_source_table": source_table,
            "measurement_source_row": np.arange(len(frame)),
        }
    )
    if valid_min is not None:
        out.loc[out["measurement_value"] < valid_min, "measurement_value"] = np.nan
    if valid_max is not None:
        out.loc[out["measurement_value"] > valid_max, "measurement_value"] = np.nan
    out = out[out["measurement_time_days"].notna()].copy()
    return out.sort_values(["participant_id", "measurement_time_days", "measurement_source_row"])


def _best_nearby_measurement(
    participant: str,
    time_days: float,
    visit_id: Any,
    measurements_by_subject: dict[str, pd.DataFrame],
    tolerance_days: int,
    on_or_before: bool = False,
) -> dict[str, Any]:
    candidates = measurements_by_subject.get(str(participant))
    empty = {
        "value": np.nan,
        "source_value": np.nan,
        "measurement_time_days": np.nan,
        "day_offset": np.nan,
        "match_rule": "no_measurement_in_window",
        "source_table": pd.NA,
        "source_row": np.nan,
    }
    if candidates is None or candidates.empty or pd.isna(time_days):
        return empty
    cand = candidates[candidates["measurement_value"].notna()].copy()
    if on_or_before:
        cand = cand[cand["measurement_time_days"] <= time_days]
    if cand.empty:
        return empty
    cand["day_offset"] = cand["measurement_time_days"] - float(time_days)
    cand["abs_offset"] = cand["day_offset"].abs()
    cand = cand[cand["abs_offset"] <= tolerance_days]
    if cand.empty:
        return empty
    exact_visit = cand["measurement_visit_id"].astype(str).eq(str(visit_id))
    cand["visit_priority"] = (~exact_visit).astype(int)
    if on_or_before:
        cand = cand.sort_values(
            ["visit_priority", "measurement_time_days", "measurement_source_row"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    else:
        cand = cand.sort_values(
            ["visit_priority", "abs_offset", "measurement_time_days", "measurement_source_row"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )
    row = cand.iloc[0]
    if row["visit_priority"] == 0 and row["day_offset"] == 0:
        rule = "same_visit_same_time"
    elif row["visit_priority"] == 0:
        rule = "same_visit_within_window"
    elif row["day_offset"] == 0:
        rule = "same_time"
    else:
        rule = "nearest_on_or_before" if on_or_before else "nearest_absolute_time"
    return {
        "value": row["measurement_value"],
        "source_value": row["measurement_value_source"],
        "measurement_time_days": row["measurement_time_days"],
        "day_offset": row["day_offset"],
        "match_rule": rule,
        "source_table": row["measurement_source_table"],
        "source_row": row["measurement_source_row"],
    }


def _attach_measurements(
    manifest: pd.DataFrame,
    measurements: pd.DataFrame,
    prefix: str,
    tolerance_days: int = MATCH_TOLERANCE_DAYS,
    on_or_before: bool = False,
) -> pd.DataFrame:
    grouped = {str(k): v.copy() for k, v in measurements.groupby("participant_id", sort=False)}
    matches = [
        _best_nearby_measurement(
            row.participant_id,
            row.time_sort_days,
            row.original_visit_id,
            grouped,
            tolerance_days=tolerance_days,
            on_or_before=on_or_before,
        )
        for row in manifest.itertuples(index=False)
    ]
    matched = pd.DataFrame(matches, index=manifest.index).add_prefix(f"{prefix}_")
    return pd.concat([manifest, matched], axis=1)


def _diagnosis_measurements(
    frame: pd.DataFrame,
    participant_col: str,
    visit_col: str,
    time_values: pd.Series,
    diagnosis_values: pd.Series,
    original_values: pd.Series,
    source_table: str,
    method: pd.Series | None = None,
    form_family: pd.Series | None = None,
) -> pd.DataFrame:
    n = len(frame)
    return pd.DataFrame(
        {
            "participant_id": frame[participant_col].astype("string").str.replace(r"\.0$", "", regex=True),
            "measurement_visit_id": frame[visit_col].astype("string"),
            "measurement_time_days": numeric(time_values),
            "measurement_value": diagnosis_values.astype("string"),
            "measurement_value_source": original_values.astype("string"),
            "measurement_source_table": source_table,
            "measurement_source_row": np.arange(n),
            "diagnosis_method_value": method.astype("string") if method is not None else pd.Series(pd.NA, index=frame.index, dtype="string"),
            "diagnosis_form_family": form_family.astype("string") if form_family is not None else pd.Series(pd.NA, index=frame.index, dtype="string"),
        }
    )


def _attach_diagnosis(
    manifest: pd.DataFrame,
    diagnoses: pd.DataFrame,
    tolerance_days: int = MATCH_TOLERANCE_DAYS,
) -> pd.DataFrame:
    grouped = {str(k): v.copy() for k, v in diagnoses.groupby("participant_id", sort=False)}
    records = []
    for row in manifest.itertuples(index=False):
        match = _best_nearby_measurement(
            row.participant_id,
            row.time_sort_days,
            row.original_visit_id,
            grouped,
            tolerance_days=tolerance_days,
            on_or_before=False,
        )
        candidates = grouped.get(str(row.participant_id))
        method = pd.NA
        family = pd.NA
        if candidates is not None and pd.notna(match["source_row"]):
            chosen = candidates[candidates["measurement_source_row"].eq(match["source_row"])]
            if not chosen.empty:
                method = chosen.iloc[0].get("diagnosis_method_value", pd.NA)
                family = chosen.iloc[0].get("diagnosis_form_family", pd.NA)
        records.append(
            {
                "clinician_diagnosis": match["value"],
                "clinician_diagnosis_source_value": match["source_value"],
                "diagnosis_time_days": match["measurement_time_days"],
                "diagnosis_day_offset": match["day_offset"],
                "diagnosis_match_rule": match["match_rule"],
                "diagnosis_source_table": match["source_table"],
                "diagnosis_source_row": match["source_row"],
                "diagnosis_method": method,
                "diagnosis_form_family": family,
            }
        )
    return pd.concat([manifest, pd.DataFrame(records, index=manifest.index)], axis=1)


def _diagnosis_conflict_records(diagnoses: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = []
    valid = diagnoses[diagnoses["measurement_time_days"].notna()].copy()
    for (participant, time_value), group in valid.groupby(["participant_id", "measurement_time_days"], sort=False):
        values = sorted(set(group["measurement_value"].dropna().astype(str)) - {"Unclassified"})
        if len(group) <= 1 and len(values) <= 1:
            continue
        unresolved = len(values) > 1
        rows.append(
            {
                "cohort": cohort,
                "conflict_type": "diagnosis_duplicate_or_conflict",
                "participant_id": participant,
                "time_sort_days": time_value,
                "visit_date": pd.NA,
                "original_visit_ids": _format_values(group["measurement_visit_id"]),
                "source_row_count": len(group),
                "cdr_values": "",
                "diagnosis_values": _format_values(group["measurement_value"]),
                "source_tables": _format_values(group["measurement_source_table"]),
                "resolution_applied": "diagnosis retained for sensitivity only; earliest deterministic source row selected" if unresolved else "exact/same-state duplicate collapsed",
                "unresolved_flag": unresolved,
                "selected_source_row": group["measurement_source_row"].min(),
            }
        )
    return pd.DataFrame(rows)


def _mark_diagnosis_conflicts(manifest: pd.DataFrame, diagnosis_conflicts: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if diagnosis_conflicts.empty:
        out["same_time_diagnosis_conflict_flag"] = False
        return out
    keys = set(
        zip(
            diagnosis_conflicts.loc[diagnosis_conflicts["unresolved_flag"].fillna(False), "participant_id"].astype(str),
            diagnosis_conflicts.loc[diagnosis_conflicts["unresolved_flag"].fillna(False), "time_sort_days"],
        )
    )
    out["same_time_diagnosis_conflict_flag"] = [
        (str(p), t) in keys for p, t in zip(out["participant_id"], out["time_sort_days"])
    ]
    return out


def _adni_age(dob: Any, dob_year: Any, visit_date: Any) -> tuple[float, str]:
    visit = pd.to_datetime(visit_date, errors="coerce")
    if pd.isna(visit):
        return np.nan, "unavailable"
    match = re.fullmatch(r"\s*(\d{1,2})/(\d{4})\s*", str(dob)) if pd.notna(dob) else None
    if match:
        birth = pd.Timestamp(int(match.group(2)), int(match.group(1)), 15)
        return (visit - birth).days / 365.25, "PTDOB_month_year_midmonth_approximation"
    year = pd.to_numeric(pd.Series([dob_year]), errors="coerce").iloc[0]
    if pd.notna(year):
        birth = pd.Timestamp(int(year), 7, 1)
        return (visit - birth).days / 365.25, "PTDOBYY_midyear_approximation"
    return np.nan, "unavailable"


def _aibl_age(dob: Any, visit_date: Any) -> tuple[float, str]:
    visit = pd.to_datetime(visit_date, errors="coerce")
    match = re.search(r"(19\d{2}|20\d{2})", str(dob)) if pd.notna(dob) else None
    if pd.isna(visit) or not match:
        return np.nan, "unavailable"
    birth = pd.Timestamp(int(match.group(1)), 7, 1)
    return (visit - birth).days / 365.25, "PTDOB_year_only_midyear_unverified"


def _oasis_diagnosis_mapping(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    mci_cols = ["MCIAMEM", "MCIAPLUS", "MCINON1", "MCINON2"]
    norm = numeric(frame["NORMCOG"]).eq(1)
    dementia = numeric(frame["DEMENTED"]).eq(1)
    mci = pd.concat([numeric(frame[c]).eq(1) for c in mci_cols], axis=1).any(axis=1)
    impaired = numeric(frame["IMPNOMCI"]).eq(1)
    conflict = pd.concat([norm, dementia, mci], axis=1).sum(axis=1).gt(1)
    alz = numeric(frame.get("alzdis", pd.Series(np.nan, index=frame.index))).eq(1)
    old_ad = (
        numeric(frame.get("PROBAD", pd.Series(np.nan, index=frame.index))).eq(1)
        | numeric(frame.get("POSSAD", pd.Series(np.nan, index=frame.index))).eq(1)
    )
    mapped = pd.Series("Unclassified", index=frame.index, dtype="string")
    mapped.loc[impaired] = "Impaired-not-MCI"
    mapped.loc[mci] = "MCI"
    mapped.loc[dementia] = "Dementia-non-AD-or-unspecified"
    mapped.loc[dementia & (alz | old_ad)] = "Dementia-AD-coded"
    mapped.loc[norm] = "CN"
    mapped.loc[conflict] = "Diagnosis-conflict"
    uds3 = frame.get("dxmethod", pd.Series(np.nan, index=frame.index)).notna()
    uds2 = frame.get("WHODIDDX", pd.Series(np.nan, index=frame.index)).notna()
    family = pd.Series("unidentified", index=frame.index, dtype="string")
    family.loc[uds2 & ~uds3] = "UDS2-style"
    family.loc[uds3 & ~uds2] = "UDS3-style"
    family.loc[uds2 & uds3] = "mixed-fields"
    method = pd.Series("unknown", index=frame.index, dtype="string")
    whodid = numeric(frame.get("WHODIDDX", pd.Series(np.nan, index=frame.index)))
    dxmethod = numeric(frame.get("dxmethod", pd.Series(np.nan, index=frame.index)))
    method.loc[whodid.eq(1)] = "single-clinician"
    method.loc[whodid.eq(2)] = "formal-consensus"
    method.loc[dxmethod.eq(1)] = "single-clinician"
    method.loc[dxmethod.eq(2)] = "formal-consensus"
    method.loc[dxmethod.eq(3)] = "other-informal-group"
    return mapped, method, family


def build_adni_manifest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    src = SOURCES["adni"]
    cdr = read_csv(src.cdr)
    cdr_date = parse_date(cdr["VISDATE"])
    anchor = pd.DataFrame(
        {
            "cohort": "adni",
            "participant_id": cdr["RID"].astype("Int64").astype("string"),
            "participant_id_source": cdr["RID"],
            "original_visit_id": cdr["VISCODE2"].astype("string"),
            "original_visit_id_secondary": cdr["VISCODE"].astype("string"),
            "visit_date": iso_date(cdr_date),
            "time_sort_days": (cdr_date - pd.Timestamp("1970-01-01")).dt.days,
            "cdr_global_source": cdr["CDGLOBAL"],
            "cdr_sum_boxes": numeric(cdr["CDRSB"]),
            "cdr_domain_values_source": cdr[["CDMEMORY", "CDORIENT", "CDJUDGE", "CDCOMMUN", "CDHOME", "CDCARE"]].astype("string").fillna("<missing>").agg("|".join, axis=1),
            "study_phase": cdr["PHASE"].astype("string"),
            "site_id": cdr["SITEID"].astype("string"),
            "source_form_version": cdr["DD_CRF_VERSION_LABEL"].astype("string"),
            "cdr_source_method": cdr["CDSOURCE"].astype("string"),
            "source_cdr_table": src.cdr.name,
            "source_row_number": np.arange(len(cdr)),
            "source_quality_error": cdr["HAS_QC_ERROR"],
            "source_update_timestamp": cdr["update_stamp"],
            "scheduled_visit_flag": cdr["VISCODE2"].astype(str).str.fullmatch(r"bl|m\d+", case=False),
            "scheduled_visit_definition": "VISCODE2 is bl or m<number>",
        }
    )
    manifest, conflicts, invalid = _resolve_anchor_records(anchor, "adni")

    demo = read_csv(src.demographics)
    demo_one = _stable_subject_record(demo, "RID", ["PTGENDER", "PTDOB", "PTDOBYY", "PTEDUCAT"])
    demo_one["participant_id"] = demo_one["RID"].astype("Int64").astype("string")
    demo_map = demo_one.set_index("participant_id")
    sex = []
    education = []
    ages = []
    age_methods = []
    for row in manifest.itertuples(index=False):
        if row.participant_id in demo_map.index:
            d = demo_map.loc[row.participant_id]
            sex.append(_sex(d.get("PTGENDER")))
            edu = pd.to_numeric(pd.Series([d.get("PTEDUCAT")]), errors="coerce").iloc[0]
            education.append(edu if pd.notna(edu) and 0 <= edu <= 40 else np.nan)
            age, method = _adni_age(d.get("PTDOB"), d.get("PTDOBYY"), row.visit_date)
            ages.append(age)
            age_methods.append(method)
        else:
            sex.append(pd.NA); education.append(np.nan); ages.append(np.nan); age_methods.append("unavailable")
    manifest["sex"] = pd.Series(sex, dtype="string")
    manifest["education_years"] = education
    manifest["age_at_visit"] = ages
    manifest["age_derivation_method"] = age_methods
    manifest["age_derivation_verified_flag"] = manifest["age_at_visit"].notna()

    apoe = read_csv(src.apoe)
    apoe_one = _stable_subject_record(apoe, "RID", ["GENOTYPE"])
    apoe_one["participant_id"] = apoe_one["RID"].astype("Int64").astype("string")
    geno_map = apoe_one.set_index("participant_id")["GENOTYPE"].to_dict()
    parsed = [_apoe_string_and_dose(geno_map.get(str(p))) for p in manifest["participant_id"]]
    manifest["apoe_genotype"] = pd.Series([x[0] for x in parsed], dtype="string")
    manifest["apoe_e4_dose"] = [x[1] for x in parsed]

    mmse_raw = read_csv(src.mmse)
    mmse_time = (parse_date(mmse_raw["VISDATE"]) - pd.Timestamp("1970-01-01")).dt.days
    mmse = _measurement_frame(mmse_raw, "RID", "VISCODE2", mmse_time, "MMSCORE", src.mmse.name, 0, 30)
    manifest = _attach_measurements(manifest, mmse, "mmse")
    manifest = manifest.rename(columns={"mmse_value": "mmse", "mmse_source_value": "mmse_source_value"})

    dx_raw = read_csv(src.diagnosis)
    dx_time = (parse_date(dx_raw["EXAMDATE"]) - pd.Timestamp("1970-01-01")).dt.days
    dx_code = numeric(dx_raw["DIAGNOSIS"])
    dx_map = dx_code.map({1: "CN", 2: "MCI", 3: "Dementia-AD-coded"}).fillna("Unclassified")
    diagnoses = _diagnosis_measurements(
        dx_raw, "RID", "VISCODE2", dx_time, dx_map, dx_raw["DIAGNOSIS"], src.diagnosis.name,
        method=pd.Series("clinician-diagnosis-method-not-supplied", index=dx_raw.index),
        form_family=dx_raw["PHASE"].astype("string"),
    )
    manifest = _attach_diagnosis(manifest, diagnoses)
    diagnosis_conflicts = _diagnosis_conflict_records(diagnoses, "adni")
    manifest = _mark_diagnosis_conflicts(manifest, diagnosis_conflicts)
    conflicts = pd.concat([conflicts, diagnosis_conflicts], ignore_index=True, sort=False)

    gds_raw = read_csv(src.gds)
    gds_time = (parse_date(gds_raw["VISDATE"]) - pd.Timestamp("1970-01-01")).dt.days
    gds = _measurement_frame(gds_raw, "RID", "VISCODE2", gds_time, "GDTOTAL", src.gds.name, 0, 15)
    manifest = _attach_measurements(manifest, gds, "gds", tolerance_days=MMSE_WINDOW_DAYS, on_or_before=True)
    manifest = manifest.rename(columns={"gds_value": "gds_total"})
    manifest["form_family"] = manifest["study_phase"]
    manifest["cdr_field_validation_status"] = "validated_by_range_and_existing_project_notes; ADNI codebook not supplied"
    return finalize_manifest(manifest), conflicts, invalid, {"mmse": mmse, "diagnosis": diagnoses, "gds": gds}


def build_oasis_manifest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    src = SOURCES["oasis3"]
    cdr = read_csv(src.cdr)
    time = numeric(cdr["days_to_visit"])
    anchor = pd.DataFrame(
        {
            "cohort": "oasis3",
            "participant_id": cdr["OASISID"].astype("string"),
            "participant_id_source": cdr["OASISID"],
            "original_visit_id": cdr["OASIS_session_label"].astype("string"),
            "original_visit_id_secondary": pd.Series(pd.NA, index=cdr.index, dtype="string"),
            "visit_date": pd.Series(pd.NA, index=cdr.index, dtype="string"),
            "time_sort_days": time,
            "cdr_global_source": cdr["CDRTOT"],
            "cdr_sum_boxes": numeric(cdr["CDRSUM"]),
            "cdr_domain_values_source": cdr[["memory", "orient", "judgment", "commun", "homehobb", "perscare"]].astype("string").fillna("<missing>").agg("|".join, axis=1),
            "study_phase": pd.Series("OASIS-3", index=cdr.index, dtype="string"),
            "site_id": pd.Series("Knight ADRC/WUSTL", index=cdr.index, dtype="string"),
            "source_form_version": pd.Series(pd.NA, index=cdr.index, dtype="string"),
            "cdr_source_method": pd.Series("UDS B4", index=cdr.index, dtype="string"),
            "source_cdr_table": src.cdr.name,
            "source_row_number": np.arange(len(cdr)),
            "source_quality_error": np.zeros(len(cdr)),
            "source_update_timestamp": pd.NaT,
            "scheduled_visit_flag": pd.Series(pd.NA, index=cdr.index, dtype="boolean"),
            "scheduled_visit_definition": "not identifiable in supplied OASIS-3 clinical extract",
            "age_at_visit_source": cdr["age at visit"],
        }
    )
    manifest, conflicts, invalid = _resolve_anchor_records(anchor, "oasis3")

    demo = read_csv(src.demographics)
    demo_one = _stable_subject_record(demo, "OASISID", ["GENDER", "EDUC", "APOE", "AgeatEntry"])
    demo_one["participant_id"] = demo_one["OASISID"].astype("string")
    demo_map = demo_one.set_index("participant_id")
    manifest["age_at_visit"] = numeric(manifest["age_at_visit_source"])
    manifest.loc[~manifest["age_at_visit"].between(18, 110), "age_at_visit"] = np.nan
    manifest["age_derivation_method"] = "source age at visit"
    manifest["age_derivation_verified_flag"] = manifest["age_at_visit"].between(18, 110)
    manifest["sex"] = manifest["participant_id"].map(lambda p: _sex(demo_map.loc[p, "GENDER"]) if p in demo_map.index else pd.NA).astype("string")
    manifest["education_years"] = manifest["participant_id"].map(
        lambda p: pd.to_numeric(pd.Series([demo_map.loc[p, "EDUC"]]), errors="coerce").iloc[0] if p in demo_map.index else np.nan
    )
    parsed = [_apoe_string_and_dose(demo_map.loc[p, "APOE"] if p in demo_map.index else np.nan) for p in manifest["participant_id"]]
    manifest["apoe_genotype"] = pd.Series([x[0] for x in parsed], dtype="string")
    manifest["apoe_e4_dose"] = [x[1] for x in parsed]

    mmse = _measurement_frame(cdr, "OASISID", "OASIS_session_label", time, "MMSE", src.cdr.name, 0, 30)
    manifest = _attach_measurements(manifest, mmse, "mmse", tolerance_days=0)
    manifest = manifest.rename(columns={"mmse_value": "mmse", "mmse_source_value": "mmse_source_value"})

    dx_raw = read_csv(src.diagnosis)
    dx_map, dx_method, form_family = _oasis_diagnosis_mapping(dx_raw)
    diagnoses = _diagnosis_measurements(
        dx_raw, "OASISID", "OASIS_session_label", dx_raw["days_to_visit"], dx_map,
        dx_raw[["NORMCOG", "DEMENTED", "MCIAMEM", "MCIAPLUS", "MCINON1", "MCINON2", "IMPNOMCI", "alzdis"]].astype("string").fillna("<missing>").agg("|".join, axis=1),
        src.diagnosis.name, method=dx_method, form_family=form_family,
    )
    manifest = _attach_diagnosis(manifest, diagnoses, tolerance_days=0)
    diagnosis_conflicts = _diagnosis_conflict_records(diagnoses, "oasis3")
    manifest = _mark_diagnosis_conflicts(manifest, diagnosis_conflicts)
    conflicts = pd.concat([conflicts, diagnosis_conflicts], ignore_index=True, sort=False)
    manifest["form_family"] = manifest["diagnosis_form_family"].fillna("unidentified")
    manifest["study_phase"] = manifest["form_family"]

    gds_raw = read_csv(src.gds)
    gds = _measurement_frame(gds_raw, "OASISID", "OASIS_session_label", gds_raw["days_to_visit"], "GDS", src.gds.name, 0, 15)
    manifest = _attach_measurements(manifest, gds, "gds", tolerance_days=MMSE_WINDOW_DAYS, on_or_before=True)
    manifest = manifest.rename(columns={"gds_value": "gds_total"})
    manifest["cdr_field_validation_status"] = (
        "provisional: delivered CDRTOT has global-stage range and CDRSUM is sum-of-boxes, "
        "but supplied workbook labels CDRGLOB as global and CDRTOT as UDS2 total"
    )
    return finalize_manifest(manifest), conflicts, invalid, {"mmse": mmse, "diagnosis": diagnoses, "gds": gds}


def build_aibl_manifest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    src = SOURCES["aibl"]
    cdr = read_csv(src.cdr)
    cdr_date = pd.to_datetime(cdr["EXAMDATE"], format="%m/%d/%Y", errors="coerce")
    anchor = pd.DataFrame(
        {
            "cohort": "aibl",
            "participant_id": cdr["RID"].astype("Int64").astype("string"),
            "participant_id_source": cdr["RID"],
            "original_visit_id": cdr["VISCODE"].astype("string"),
            "original_visit_id_secondary": pd.Series(pd.NA, index=cdr.index, dtype="string"),
            "visit_date": iso_date(cdr_date),
            "time_sort_days": (cdr_date - pd.Timestamp("1970-01-01")).dt.days,
            "cdr_global_source": cdr["CDGLOBAL"],
            "cdr_sum_boxes": np.nan,
            "cdr_domain_values_source": pd.Series(pd.NA, index=cdr.index, dtype="string"),
            "study_phase": pd.Series("AIBL extract 3.3.0", index=cdr.index, dtype="string"),
            "site_id": cdr["SITEID"].astype("string"),
            "source_form_version": pd.Series("01-Jun-2018", index=cdr.index, dtype="string"),
            "cdr_source_method": pd.Series(pd.NA, index=cdr.index, dtype="string"),
            "source_cdr_table": src.cdr.name,
            "source_row_number": np.arange(len(cdr)),
            "source_quality_error": np.zeros(len(cdr)),
            "source_update_timestamp": pd.NaT,
            "scheduled_visit_flag": cdr["VISCODE"].astype(str).str.fullmatch(r"bl|m\d+", case=False),
            "scheduled_visit_definition": "VISCODE is bl or m<number>",
        }
    )
    manifest, conflicts, invalid = _resolve_anchor_records(anchor, "aibl")
    demo = read_csv(src.demographics)
    demo_one = _stable_subject_record(demo, "RID", ["PTGENDER", "PTDOB"])
    demo_one["participant_id"] = demo_one["RID"].astype("Int64").astype("string")
    demo_map = demo_one.set_index("participant_id")
    ages, methods, sexes = [], [], []
    for row in manifest.itertuples(index=False):
        if row.participant_id in demo_map.index:
            d = demo_map.loc[row.participant_id]
            age, method = _aibl_age(d.get("PTDOB"), row.visit_date)
            ages.append(age); methods.append(method); sexes.append(_sex(d.get("PTGENDER")))
        else:
            ages.append(np.nan); methods.append("unavailable"); sexes.append(pd.NA)
    manifest["age_at_visit"] = ages
    manifest["age_derivation_method"] = methods
    manifest["age_derivation_verified_flag"] = False
    manifest["sex"] = pd.Series(sexes, dtype="string")
    manifest["education_years"] = np.nan

    apoe_raw = read_csv(src.apoe)
    apoe_one = _stable_subject_record(apoe_raw, "RID", ["APGEN1", "APGEN2"])
    apoe_one["participant_id"] = apoe_one["RID"].astype("Int64").astype("string")
    apoe_map = apoe_one.set_index("participant_id")
    parsed = [_aibl_apoe(apoe_map.loc[p]) if p in apoe_map.index else (pd.NA, np.nan) for p in manifest["participant_id"]]
    manifest["apoe_genotype"] = pd.Series([x[0] for x in parsed], dtype="string")
    manifest["apoe_e4_dose"] = [x[1] for x in parsed]

    mmse_raw = read_csv(src.mmse)
    mmse_date = pd.to_datetime(mmse_raw["EXAMDATE"], format="%m/%d/%Y", errors="coerce")
    mmse_time = (mmse_date - pd.Timestamp("1970-01-01")).dt.days
    mmse = _measurement_frame(mmse_raw, "RID", "VISCODE", mmse_time, "MMSCORE", src.mmse.name, 0, 30)
    manifest = _attach_measurements(manifest, mmse, "mmse", tolerance_days=0)
    manifest = manifest.rename(columns={"mmse_value": "mmse", "mmse_source_value": "mmse_source_value"})

    dx_raw = read_csv(src.diagnosis)
    visit_month = dx_raw["VISCODE"].astype(str).str.extract(r"m(\d+)", expand=False).astype(float)
    dx_time_lookup = manifest.set_index(["participant_id", "original_visit_id"])["time_sort_days"].to_dict()
    dx_participant = dx_raw["RID"].astype("Int64").astype("string")
    dx_time = pd.Series([dx_time_lookup.get((p, str(v)), np.nan) for p, v in zip(dx_participant, dx_raw["VISCODE"])], index=dx_raw.index)
    dx_code = numeric(dx_raw["DXCURREN"])
    dx_map = dx_code.map({1: "CN", 2: "MCI", 3: "Dementia-AD-coded", 7: "Dementia-other"}).fillna("Unclassified")
    diagnoses = _diagnosis_measurements(
        dx_raw, "RID", "VISCODE", dx_time, dx_map, dx_raw["DXCURREN"], src.diagnosis.name,
        method=pd.Series("not supplied", index=dx_raw.index),
        form_family=pd.Series("AIBL extract 3.3.0", index=dx_raw.index),
    )
    manifest = _attach_diagnosis(manifest, diagnoses, tolerance_days=0)
    diagnosis_conflicts = _diagnosis_conflict_records(diagnoses, "aibl")
    manifest = _mark_diagnosis_conflicts(manifest, diagnosis_conflicts)
    conflicts = pd.concat([conflicts, diagnosis_conflicts], ignore_index=True, sort=False)
    manifest["gds_total"] = np.nan
    manifest["gds_match_rule"] = "not supplied"
    manifest["form_family"] = "AIBL extract 3.3.0"
    manifest["cdr_field_validation_status"] = "range validated; AIBL codebook not supplied; age derivation remains unverified"
    return finalize_manifest(manifest), conflicts, invalid, {"mmse": mmse, "diagnosis": diagnoses, "gds": pd.DataFrame()}


def finalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    out["mmse_same_time_as_cdr_flag"] = out.get("mmse_day_offset", pd.Series(np.nan, index=out.index)).eq(0)
    out["diagnosis_same_time_as_cdr_flag"] = out.get("diagnosis_day_offset", pd.Series(np.nan, index=out.index)).eq(0)
    out["valid_time_flag"] = out["time_sort_days"].notna()
    out["valid_age_flag"] = numeric(out["age_at_visit"]).between(18, 110)
    out["valid_mmse_flag"] = numeric(out.get("mmse", pd.Series(np.nan, index=out.index))).between(0, 30)
    out["classifiable_cdr_flag"] = out["cdr_state"].notna() & ~out["duplicate_resolution_unresolved_flag"].fillna(False)
    year = pd.to_datetime(out["visit_date"], errors="coerce").dt.year
    out["calendar_period"] = pd.cut(
        year, [-np.inf, 2009, 2014, 2019, np.inf],
        labels=["through 2009", "2010-2014", "2015-2019", "2020 onward"],
    ).astype("string")
    out["qc_flags"] = ""
    flag_rules = {
        "unresolved_same_time_cdr_conflict": out["duplicate_resolution_unresolved_flag"].fillna(False),
        "near_duplicate_visit": out["near_duplicate_visit_flag"].fillna(False),
        "invalid_or_missing_age": ~out["valid_age_flag"],
        "invalid_or_missing_mmse": ~out["valid_mmse_flag"],
        "missing_sex": out["sex"].isna(),
        "same_time_diagnosis_conflict": out.get("same_time_diagnosis_conflict_flag", pd.Series(False, index=out.index)).fillna(False),
        "provisional_cdr_field_validation": out["cdr_field_validation_status"].astype(str).str.startswith("provisional"),
    }
    for label, mask in flag_rules.items():
        out.loc[mask, "qc_flags"] = out.loc[mask, "qc_flags"].map(lambda x: f"{x};{label}".strip(";"))
    preferred = [
        "manifest_key", "cohort", "participant_id", "participant_id_source", "original_visit_id",
        "original_visit_id_secondary", "visit_date", "relative_time_days", "relative_time_months", "time_sort_days",
        "age_at_visit", "age_derivation_method", "age_derivation_verified_flag", "sex", "education_years",
        "apoe_genotype", "apoe_e4_dose", "cdr_global", "cdr_global_source", "cdr_sum_boxes",
        "cdr_domain_values_source", "cdr_state", "cdr_state_label", "mmse", "mmse_source_value",
        "mmse_measurement_time_days", "mmse_day_offset", "mmse_match_rule", "clinician_diagnosis",
        "clinician_diagnosis_source_value", "diagnosis_time_days", "diagnosis_day_offset", "diagnosis_match_rule",
        "diagnosis_source_table", "diagnosis_method", "study_phase", "form_family", "site_id",
        "calendar_period", "same_time_diagnosis_conflict_flag",
        "source_form_version", "cdr_source_method", "gds_total", "scheduled_visit_flag",
        "scheduled_visit_definition", "duplicate_participant_time_flag", "same_time_cdr_conflict_flag",
        "duplicate_resolution_unresolved_flag", "duplicate_resolution_rule", "source_row_count_at_time",
        "near_duplicate_visit_flag", "days_from_previous_cdr", "classifiable_cdr_flag", "valid_time_flag",
        "valid_age_flag", "valid_mmse_flag", "mmse_same_time_as_cdr_flag", "diagnosis_same_time_as_cdr_flag",
        "cdr_field_validation_status", "source_cdr_table", "source_row_number", "qc_flags",
    ]
    return out[[c for c in preferred if c in out] + [c for c in out if c not in preferred]].reset_index(drop=True)


TRANSITION_LABELS = {
    (0, 1): "CDR 0 -> 0.5",
    (0, 2): "CDR 0 -> >=1",
    (1, 0): "CDR 0.5 -> 0",
    (1, 2): "CDR 0.5 -> >=1",
    (2, 1): "CDR >=1 -> 0.5",
    (2, 0): "CDR >=1 -> 0",
}


def build_transition_history(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    classifiable = manifest[
        manifest["classifiable_cdr_flag"].fillna(False) & manifest["time_sort_days"].notna()
    ].copy()
    classifiable = classifiable.sort_values(["participant_id", "time_sort_days"], kind="mergesort")
    pairs: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    cohort = str(manifest["cohort"].iloc[0])
    for participant, history in classifiable.groupby("participant_id", sort=False):
        history = history.sort_values("time_sort_days", kind="mergesort").reset_index(drop=True)
        participant_changes: list[dict[str, Any]] = []
        for i in range(1, len(history)):
            before, after = history.iloc[i - 1], history.iloc[i]
            from_state, to_state = int(before["cdr_state"]), int(after["cdr_state"])
            interval = float(after["time_sort_days"] - before["time_sort_days"])
            later_state = history.iloc[i + 1]["cdr_state"] if i + 1 < len(history) else np.nan
            record = {
                "cohort": cohort,
                "participant_id": participant,
                "from_state": from_state,
                "to_state": to_state,
                "from_state_label": state_label(from_state),
                "to_state_label": state_label(to_state),
                "transition_label": TRANSITION_LABELS.get((from_state, to_state), f"State {from_state} -> {to_state}"),
                "from_manifest_key": before["manifest_key"],
                "to_manifest_key": after["manifest_key"],
                "from_visit_id": before["original_visit_id"],
                "to_visit_id": after["original_visit_id"],
                "from_visit_date": before["visit_date"],
                "to_visit_date": after["visit_date"],
                "from_time_days": before["time_sort_days"],
                "to_time_days": after["time_sort_days"],
                "transition_interval_days": interval,
                "transition_interval_months": interval / DAYS_PER_MONTH,
                "destination_phase_or_form": after.get("form_family", after.get("study_phase", pd.NA)),
                "destination_study_phase": after.get("study_phase", pd.NA),
                "destination_form_family": after.get("form_family", pd.NA),
                "destination_diagnosis_method": after.get("diagnosis_method", pd.NA),
                "destination_calendar_period": after.get("calendar_period", pd.NA),
                "destination_relative_time_band": (
                    "before/at entry" if after.get("relative_time_months", np.nan) <= 0
                    else "0-5 years" if after.get("relative_time_months", np.nan) <= 60
                    else "5-10 years" if after.get("relative_time_months", np.nan) <= 120
                    else ">10 years"
                ),
                "under_30_days_flag": interval < 30,
                "under_90_days_flag": interval < 90,
                "meets_minimum_30_days_flag": interval >= 30,
                "meets_minimum_90_days_flag": interval >= 90,
                "scheduled_pair_flag": (
                    bool(before["scheduled_visit_flag"]) and bool(after["scheduled_visit_flag"])
                    if pd.notna(before["scheduled_visit_flag"]) and pd.notna(after["scheduled_visit_flag"])
                    else pd.NA
                ),
                "near_duplicate_pair_flag": interval <= NEAR_DUPLICATE_DAYS,
                "later_state": later_state,
                "later_state_label": state_label(later_state),
                "implausible_direct_transition_flag": (from_state, to_state) in {(0, 2), (2, 0)},
                "state_changed_flag": from_state != to_state,
            }
            pairs.append(record)
            if from_state != to_state:
                participant_changes.append(record.copy())
                changes.append(record.copy())
        label_counts = Counter(x["transition_label"] for x in participant_changes)
        worsening = sum(
            x["from_state"] < x["to_state"] for x in participant_changes
        )
        improvement = sum(
            x["from_state"] > x["to_state"] for x in participant_changes
        )
        summaries.append(
            {
                "cohort": cohort,
                "participant_id": participant,
                "classifiable_visit_count": len(history),
                "first_cdr_state": int(history.iloc[0]["cdr_state"]),
                "last_cdr_state": int(history.iloc[-1]["cdr_state"]),
                "follow_up_days": float(history.iloc[-1]["time_sort_days"] - history.iloc[0]["time_sort_days"]),
                "follow_up_months": float(history.iloc[-1]["time_sort_days"] - history.iloc[0]["time_sort_days"]) / DAYS_PER_MONTH,
                "state_change_event_count": len(participant_changes),
                "worsening_event_count": worsening,
                "improvement_event_count": improvement,
                "repeated_worsening_flag": worsening >= 2,
                "repeated_improvement_flag": improvement >= 2,
                "cdr_0_5_to_0_event_count": label_counts["CDR 0.5 -> 0"],
                "repeated_cdr_0_5_to_0_flag": label_counts["CDR 0.5 -> 0"] >= 2,
                "cdr_0_5_to_ge1_event_count": label_counts["CDR 0.5 -> >=1"],
                "repeated_cdr_0_5_to_ge1_flag": label_counts["CDR 0.5 -> >=1"] >= 2,
            }
        )
    changes_df = pd.DataFrame(changes)
    if not changes_df.empty:
        changes_df["participant_transition_sequence"] = changes_df.groupby(
            ["participant_id", "transition_label"]
        ).cumcount() + 1
        changes_df["repeated_transition_event_flag"] = changes_df["participant_transition_sequence"].gt(1)
    return changes_df, pd.DataFrame(summaries), pd.DataFrame(pairs)


def transition_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if events.empty:
        return pd.DataFrame()
    for (cohort, label), group in events.groupby(["cohort", "transition_label"], sort=False):
        times = numeric(group["transition_interval_months"]).dropna()
        rows.append(
            {
                "cohort": cohort,
                "transition_label": label,
                "events": len(group),
                "participants": group["participant_id"].nunique(),
                "median_interval_months": times.median(),
                "q1_interval_months": times.quantile(0.25),
                "q3_interval_months": times.quantile(0.75),
                "minimum_interval_months": times.min(),
                "maximum_interval_months": times.max(),
                "under_30_day_events": int(group["under_30_days_flag"].sum()),
                "under_90_day_events": int(group["under_90_days_flag"].sum()),
                "direct_implausible_events": int(group["implausible_direct_transition_flag"].sum()),
                "participants_with_repeated_event": int(
                    group.groupby("participant_id").size().ge(2).sum()
                ),
                "destination_phase_or_form_counts": json.dumps(
                    group["destination_phase_or_form"].fillna("missing").value_counts().to_dict(), sort_keys=True
                ),
                "later_state_counts": json.dumps(
                    group["later_state_label"].fillna("No later state").value_counts().to_dict(), sort_keys=True
                ),
            }
        )
    return pd.DataFrame(rows)


def _index_mmse_match(
    participant: str,
    index_time: float,
    visit_id: str,
    mmse_by_subject: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    return _best_nearby_measurement(
        participant,
        index_time,
        visit_id,
        mmse_by_subject,
        tolerance_days=MMSE_WINDOW_DAYS,
        on_or_before=True,
    )


def _index_flow_candidates(
    manifest: pd.DataFrame,
    mmse: pd.DataFrame,
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the first eligible CDR 0.5 record and participant flow."""
    classifiable = manifest[
        manifest["classifiable_cdr_flag"].fillna(False) & manifest["valid_time_flag"].fillna(False)
    ].copy()
    classifiable = classifiable.sort_values(["participant_id", "time_sort_days"], kind="mergesort")
    mmse_by = {str(k): v.copy() for k, v in mmse.groupby("participant_id", sort=False)}
    candidate_records = []
    ever_half = set(classifiable.loc[classifiable["cdr_state"].eq(1), "participant_id"].astype(str))
    no_prior_dementia = set()
    age_available = set()
    sex_available = set()
    mmse_available = set()

    for participant, history in classifiable.groupby("participant_id", sort=False):
        history = history.sort_values("time_sort_days", kind="mergesort")
        candidate_rows = history[history["cdr_state"].eq(1)]
        for _, row in candidate_rows.iterrows():
            prior_dementia = history[
                (history["time_sort_days"] < row["time_sort_days"]) & history["cdr_state"].eq(2)
            ].shape[0] > 0
            if prior_dementia:
                continue
            no_prior_dementia.add(str(participant))
            if not bool(row["valid_age_flag"]):
                continue
            age_available.add(str(participant))
            if pd.isna(row["sex"]):
                continue
            sex_available.add(str(participant))
            match = _index_mmse_match(
                str(participant), float(row["time_sort_days"]), str(row["original_visit_id"]), mmse_by
            )
            if pd.isna(match["value"]):
                continue
            mmse_available.add(str(participant))
            selected = row.to_dict()
            selected.update(
                {
                    "index_mmse": match["value"],
                    "index_mmse_source_value": match["source_value"],
                    "index_mmse_time_days": match["measurement_time_days"],
                    "index_mmse_day_offset": match["day_offset"],
                    "index_mmse_match_rule": match["match_rule"],
                    "index_mmse_source_table": match["source_table"],
                    "index_mmse_source_row": match["source_row"],
                }
            )
            candidate_records.append(selected)
            break

    index = pd.DataFrame(candidate_records)
    all_participants = classifiable["participant_id"].nunique()
    counts = [
        ("Classifiable CDR participants", all_participants),
        ("Ever observed at CDR 0.5", len(ever_half)),
        ("At least one CDR 0.5 candidate with no earlier CDR >=1", len(no_prior_dementia)),
        ("Age available/derivable at candidate", len(age_available)),
        ("Sex available at candidate", len(sex_available)),
        (f"Valid MMSE on/before candidate within {MMSE_WINDOW_DAYS} days", len(mmse_available)),
        ("Final first eligible index participants", len(index)),
    ]
    flow = []
    previous = None
    for order, (step, count) in enumerate(counts, start=1):
        flow.append(
            {
                "cohort": cohort,
                "step_order": order,
                "flow_step": step,
                "participants_remaining": count,
                "excluded_at_step": (previous - count) if previous is not None else np.nan,
            }
        )
        previous = count
    return index, pd.DataFrame(flow)


def build_index_cohort(
    manifest: pd.DataFrame,
    measurements: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = str(manifest["cohort"].iloc[0])
    index, flow = _index_flow_candidates(manifest, measurements["mmse"], cohort)
    if index.empty:
        return index, flow
    classifiable = manifest[manifest["classifiable_cdr_flag"].fillna(False)].copy()
    later_counts = []
    for row in index.itertuples(index=False):
        later = classifiable[
            classifiable["participant_id"].astype(str).eq(str(row.participant_id))
            & classifiable["time_sort_days"].gt(row.time_sort_days)
        ]
        later_counts.append(len(later))
    index["later_classifiable_cdr_count"] = later_counts
    index["at_least_one_later_cdr_flag"] = index["later_classifiable_cdr_count"].ge(1)
    index["at_least_two_later_cdr_flag"] = index["later_classifiable_cdr_count"].ge(2)
    index["index_mmse_same_time_flag"] = numeric(index["index_mmse_day_offset"]).eq(0)
    index["index_mmse_strictly_before_flag"] = numeric(index["index_mmse_day_offset"]).lt(0)
    index["index_diagnosis_observed_on_or_before_flag"] = numeric(index["diagnosis_day_offset"]).le(0)
    valid_index_dx = index["index_diagnosis_observed_on_or_before_flag"].fillna(False)
    index["clinician_diagnosed_mci_at_index_flag"] = (
        index["clinician_diagnosis"].eq("MCI") & valid_index_dx
    )
    index["clinician_diagnosis_other_than_mci_at_index_flag"] = (
        index["clinician_diagnosis"].notna()
        & ~index["clinician_diagnosis"].eq("Unclassified")
        & ~index["clinician_diagnosis"].eq("MCI")
        & valid_index_dx
    )
    index["lagged_mmse_available_flag"] = index["index_mmse_strictly_before_flag"]
    index["exclude_same_day_mmse_sensitivity_flag"] = index["index_mmse_same_time_flag"]
    index["model_without_mmse_sensitivity_flag"] = True
    index["clinical_mci_subgroup_flag"] = index["clinician_diagnosed_mci_at_index_flag"]
    index["cdr_only_no_domain_predictors_flag"] = True
    index["index_time_days"] = index["time_sort_days"]
    index["index_visit_date"] = index["visit_date"]
    index["index_visit_id"] = index["original_visit_id"]
    index["index_age"] = index["age_at_visit"]
    index["index_year"] = pd.to_datetime(index["index_visit_date"], errors="coerce").dt.year
    index["index_calendar_period"] = pd.cut(
        index["index_year"], [-np.inf, 2009, 2014, 2019, np.inf],
        labels=["through 2009", "2010-2014", "2015-2019", "2020 onward"],
    ).astype("string")
    index["index_relative_time_band"] = pd.cut(
        numeric(index["relative_time_months"]), [-np.inf, 0, 60, 120, np.inf],
        labels=["before/at entry", "0-5 years", "5-10 years", ">10 years"], include_lowest=True,
    ).astype("string")
    index["primary_analysis_role"] = np.where(
        cohort == "aibl", "descriptive_only_age_unverified", "primary_index_candidate"
    )
    index["primary_eligibility_age_verified_flag"] = index["age_derivation_verified_flag"].fillna(False)
    keep = [
        "cohort", "participant_id", "manifest_key", "index_visit_id", "index_visit_date", "index_time_days",
        "relative_time_days", "relative_time_months", "index_age", "age_derivation_method",
        "age_derivation_verified_flag", "sex", "education_years", "apoe_genotype", "apoe_e4_dose",
        "index_mmse", "index_mmse_source_value", "index_mmse_time_days", "index_mmse_day_offset",
        "index_mmse_match_rule", "clinician_diagnosis", "clinician_diagnosis_source_value",
        "diagnosis_day_offset", "diagnosis_method", "diagnosis_source_table", "study_phase", "form_family", "site_id", "gds_total",
        "later_classifiable_cdr_count", "at_least_one_later_cdr_flag", "at_least_two_later_cdr_flag",
        "index_mmse_same_time_flag", "index_mmse_strictly_before_flag", "clinician_diagnosed_mci_at_index_flag",
        "clinician_diagnosis_other_than_mci_at_index_flag", "lagged_mmse_available_flag",
        "exclude_same_day_mmse_sensitivity_flag", "model_without_mmse_sensitivity_flag",
        "clinical_mci_subgroup_flag", "cdr_only_no_domain_predictors_flag", "index_year",
        "index_calendar_period", "index_relative_time_band", "primary_analysis_role", "primary_eligibility_age_verified_flag",
        "cdr_field_validation_status", "qc_flags",
    ]
    return index[[c for c in keep if c in index]].reset_index(drop=True), flow


def build_primary_episode_outcomes(
    index: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if index.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows = []
    horizon_rows = []
    classifiable = manifest[manifest["classifiable_cdr_flag"].fillna(False)].copy()
    unresolved = manifest[manifest["duplicate_resolution_unresolved_flag"].fillna(False)].copy()
    for idx in index.itertuples(index=False):
        subject = classifiable[
            classifiable["participant_id"].astype(str).eq(str(idx.participant_id))
            & classifiable["time_sort_days"].ge(idx.index_time_days)
        ].sort_values("time_sort_days", kind="mergesort")
        post = subject[subject["time_sort_days"].gt(idx.index_time_days)]
        event_visits = post[~post["cdr_state"].eq(1)]
        event = event_visits.iloc[0] if not event_visits.empty else None
        last = subject.iloc[-1]
        if event is not None:
            event_type = "improvement" if int(event["cdr_state"]) == 0 else "progression"
            event_time = float(event["time_sort_days"] - idx.index_time_days)
            endpoint_time = event_time
        else:
            event_type = "no_transition_observed"
            event_time = np.nan
            endpoint_time = float(last["time_sort_days"] - idx.index_time_days)
        later_conflicts = unresolved[
            unresolved["participant_id"].astype(str).eq(str(idx.participant_id))
            & unresolved["time_sort_days"].gt(idx.index_time_days)
        ]
        rows.append(
            {
                "cohort": idx.cohort,
                "participant_id": idx.participant_id,
                "index_manifest_key": idx.manifest_key,
                "index_time_days": idx.index_time_days,
                "first_event_type": event_type,
                "first_event_state": event["cdr_state"] if event is not None else np.nan,
                "first_event_manifest_key": event["manifest_key"] if event is not None else pd.NA,
                "first_event_time_days": event_time,
                "first_event_time_months": event_time / DAYS_PER_MONTH if pd.notna(event_time) else np.nan,
                "censoring_time_days": float(last["time_sort_days"] - idx.index_time_days),
                "censoring_time_months": float(last["time_sort_days"] - idx.index_time_days) / DAYS_PER_MONTH,
                "analysis_endpoint_time_days": endpoint_time,
                "post_index_classifiable_visits": len(post),
                "post_index_unresolved_conflict_count": len(later_conflicts),
                "event_destination_phase": event["study_phase"] if event is not None else pd.NA,
                "event_destination_form_family": event["form_family"] if event is not None else pd.NA,
                "event_mmse_same_time_flag": bool(event["mmse_same_time_as_cdr_flag"]) if event is not None else pd.NA,
                "last_classifiable_manifest_key": last["manifest_key"],
            }
        )
        for horizon_months in HORIZONS:
            horizon_days = horizon_months * DAYS_PER_MONTH
            conflict_before_horizon = later_conflicts[later_conflicts["time_sort_days"].le(idx.index_time_days + horizon_days)]
            event_by_horizon = event is not None and event_time <= horizon_days
            observed_through_horizon = float(last["time_sort_days"] - idx.index_time_days) >= horizon_days
            unresolved_before_outcome = (
                not conflict_before_horizon.empty
                and (not event_by_horizon or float(conflict_before_horizon["time_sort_days"].min() - idx.index_time_days) < event_time)
            )
            if unresolved_before_outcome:
                status = "unresolved_conflict"
            elif event_by_horizon:
                status = "transition_to_cdr_0" if event_type == "improvement" else "transition_to_cdr_ge1"
            elif observed_through_horizon:
                status = "remain_cdr_0_5_through_horizon"
            else:
                status = "censored_before_horizon"
            horizon_rows.append(
                {
                    "cohort": idx.cohort,
                    "participant_id": idx.participant_id,
                    "horizon_months": horizon_months,
                    "horizon_days": horizon_days,
                    "horizon_status": status,
                    "event_by_horizon_flag": event_by_horizon and not unresolved_before_outcome,
                    "transition_to_cdr_0_flag": status == "transition_to_cdr_0",
                    "transition_to_cdr_ge1_flag": status == "transition_to_cdr_ge1",
                    "remain_cdr_0_5_flag": status == "remain_cdr_0_5_through_horizon",
                    "censored_before_horizon_flag": status == "censored_before_horizon",
                    "unresolved_conflict_flag": status == "unresolved_conflict",
                    "observed_through_horizon_flag": observed_through_horizon,
                    "outcome_known_by_horizon_flag": event_by_horizon or observed_through_horizon,
                    "sufficient_observation_no_transition_flag": status == "remain_cdr_0_5_through_horizon",
                }
            )
    outcomes = pd.DataFrame(rows)
    horizon_person = pd.DataFrame(horizon_rows)
    summary = (
        horizon_person.groupby(["cohort", "horizon_months"], as_index=False)
        .agg(
            total_index_participants=("participant_id", "nunique"),
            participants_observed_through_horizon=("observed_through_horizon_flag", "sum"),
            participants_with_known_outcome_by_horizon=("outcome_known_by_horizon_flag", "sum"),
            transition_to_cdr_0=("transition_to_cdr_0_flag", "sum"),
            transition_to_cdr_ge1=("transition_to_cdr_ge1_flag", "sum"),
            remain_cdr_0_5=("remain_cdr_0_5_flag", "sum"),
            censored_before_horizon=("censored_before_horizon_flag", "sum"),
            unresolved_conflicts=("unresolved_conflict_flag", "sum"),
            transition_before_horizon=("event_by_horizon_flag", "sum"),
            sufficient_observation_no_transition=("sufficient_observation_no_transition_flag", "sum"),
        )
    )
    for destination in ["transition_to_cdr_0", "transition_to_cdr_ge1", "remain_cdr_0_5", "censored_before_horizon"]:
        summary[f"{destination}_percent_total"] = np.where(
            summary["total_index_participants"].gt(0),
            summary[destination] / summary["total_index_participants"] * 100,
            np.nan,
        )
    summary["known_outcome_percent"] = np.where(
        summary["total_index_participants"].gt(0),
        summary["participants_with_known_outcome_by_horizon"] / summary["total_index_participants"] * 100,
        np.nan,
    )
    return outcomes, horizon_person, summary


def build_longitudinal_depth(
    index: pd.DataFrame,
    outcomes: pd.DataFrame,
    manifest: pd.DataFrame,
    mmse: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if index.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    outcome_map = outcomes.set_index("participant_id")
    rows = []
    interval_rows = []
    for idx in index.itertuples(index=False):
        outcome = outcome_map.loc[idx.participant_id]
        endpoint = idx.index_time_days + outcome["analysis_endpoint_time_days"]
        event = outcome["first_event_type"] in {"improvement", "progression"}
        subject_mmse = mmse[
            mmse["participant_id"].astype(str).eq(str(idx.participant_id))
            & mmse["measurement_time_days"].ge(idx.index_time_days - MMSE_WINDOW_DAYS)
            & (
                mmse["measurement_time_days"].lt(endpoint)
                if event
                else mmse["measurement_time_days"].le(endpoint)
            )
            & mmse["measurement_value"].notna()
        ].copy()
        subject_mmse = subject_mmse.sort_values("measurement_time_days").drop_duplicates(
            "measurement_time_days", keep="first"
        )
        subject_cdr = manifest[
            manifest["participant_id"].astype(str).eq(str(idx.participant_id))
            & manifest["classifiable_cdr_flag"].fillna(False)
            & manifest["time_sort_days"].ge(idx.index_time_days)
            & manifest["time_sort_days"].le(endpoint)
        ].sort_values("time_sort_days")
        mmse_intervals = subject_mmse["measurement_time_days"].diff().dropna()
        cdr_intervals = subject_cdr["time_sort_days"].diff().dropna()
        for interval in cdr_intervals:
            interval_rows.append(
                {
                    "cohort": idx.cohort,
                    "participant_id": idx.participant_id,
                    "instrument": "CDR",
                    "interval_days": interval,
                    "interval_months": interval / DAYS_PER_MONTH,
                }
            )
        for interval in mmse_intervals:
            interval_rows.append(
                {
                    "cohort": idx.cohort,
                    "participant_id": idx.participant_id,
                    "instrument": "MMSE",
                    "interval_days": interval,
                    "interval_months": interval / DAYS_PER_MONTH,
                }
            )
        cdr_state1_landmarks = subject_cdr[
            subject_cdr["cdr_state"].eq(1)
            & (subject_cdr["time_sort_days"].lt(endpoint) if event else subject_cdr["time_sort_days"].le(endpoint))
        ]
        cv = (
            float(cdr_intervals.std(ddof=1) / cdr_intervals.mean())
            if len(cdr_intervals) >= 2 and cdr_intervals.mean() > 0
            else np.nan
        )
        followup = float(outcome["analysis_endpoint_time_days"])
        if outcome["first_event_type"] in {"improvement", "progression"}:
            group = outcome["first_event_type"]
        elif followup < 12 * DAYS_PER_MONTH:
            group = "censored_early"
        else:
            group = "no_transition_persistence"
        rows.append(
            {
                "cohort": idx.cohort,
                "participant_id": idx.participant_id,
                "outcome_group": group,
                "mmse_assessments_before_event_or_censor": len(subject_mmse),
                "cdr_assessments_index_to_event_or_censor": len(subject_cdr),
                "follow_up_days": followup,
                "follow_up_months": followup / DAYS_PER_MONTH,
                "median_mmse_interval_days": mmse_intervals.median(),
                "median_cdr_interval_days": cdr_intervals.median(),
                "cdr_interval_coefficient_of_variation": cv,
                "irregular_intervals_flag": bool(pd.notna(cv) and cv > 0.25),
                "potential_dynamic_landmarks": len(cdr_state1_landmarks),
                "multiple_potential_landmarks_flag": len(cdr_state1_landmarks) >= 2,
                "current_mmse_feasible_flag": len(subject_mmse) >= 1,
                "mmse_change_feasible_flag": len(subject_mmse) >= 2,
                "two_point_slope_feasible_flag": len(subject_mmse) >= 2,
                "stable_slope_feasible_flag": len(subject_mmse) >= 3,
                "mixed_trajectory_contribution_flag": len(subject_mmse) >= 2,
            }
        )
    participant_depth = pd.DataFrame(rows)
    summaries = []
    for (cohort, group), data in participant_depth.groupby(["cohort", "outcome_group"], dropna=False):
        summaries.append(_depth_summary_row(cohort, group, data))
    for cohort, data in participant_depth.groupby("cohort"):
        summaries.append(_depth_summary_row(cohort, "ALL", data))
    return participant_depth, pd.DataFrame(summaries), pd.DataFrame(interval_rows)


def _depth_summary_row(cohort: str, group: str, data: pd.DataFrame) -> dict[str, Any]:
    mmse_count = data["mmse_assessments_before_event_or_censor"]
    return {
        "cohort": cohort,
        "outcome_group": group,
        "participants": data["participant_id"].nunique(),
        "mmse_at_least_1": int(mmse_count.ge(1).sum()),
        "mmse_at_least_2": int(mmse_count.ge(2).sum()),
        "mmse_at_least_3": int(mmse_count.ge(3).sum()),
        "mmse_at_least_4": int(mmse_count.ge(4).sum()),
        "median_mmse_assessments": mmse_count.median(),
        "median_cdr_assessments": data["cdr_assessments_index_to_event_or_censor"].median(),
        "median_follow_up_months": data["follow_up_months"].median(),
        "median_mmse_interval_days": data["median_mmse_interval_days"].median(),
        "median_cdr_interval_days": data["median_cdr_interval_days"].median(),
        "participants_with_irregular_intervals": int(data["irregular_intervals_flag"].sum()),
        "participants_with_multiple_landmarks": int(data["multiple_potential_landmarks_flag"].sum()),
        "current_mmse_feasible": int(data["current_mmse_feasible_flag"].sum()),
        "mmse_change_feasible": int(data["mmse_change_feasible_flag"].sum()),
        "two_point_slope_feasible": int(data["two_point_slope_feasible_flag"].sum()),
        "stable_slope_feasible": int(data["stable_slope_feasible_flag"].sum()),
        "mixed_trajectory_contribution": int(data["mixed_trajectory_contribution_flag"].sum()),
    }


def attach_episode_group(index: pd.DataFrame, outcomes: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    out = index.merge(
        outcomes[["participant_id", "first_event_type", "first_event_time_months", "censoring_time_months"]],
        on="participant_id", how="left", validate="one_to_one",
    )
    out = out.merge(
        depth[["participant_id", "outcome_group", "mmse_assessments_before_event_or_censor", "cdr_assessments_index_to_event_or_censor", "potential_dynamic_landmarks"]],
        on="participant_id", how="left", validate="one_to_one",
    )
    out["follow_up_duration_band"] = pd.cut(
        numeric(out["censoring_time_months"]), [-np.inf, 12, 36, 60, np.inf],
        labels=["<12 months", "12-<36 months", "36-<60 months", ">=60 months"], right=False,
    ).astype("string")
    out["visit_count_band"] = pd.cut(
        numeric(out["cdr_assessments_index_to_event_or_censor"]), [0, 1, 2, 3, np.inf],
        labels=["1", "2", "3", ">=4"], include_lowest=True,
    ).astype("string")
    return out


def clinical_core_missingness(index_enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = {
        "age": "index_age",
        "sex": "sex",
        "education": "education_years",
        "apoe_e4_dose": "apoe_e4_dose",
        "mmse": "index_mmse",
        "visit_time": "index_time_days",
        "clinician_diagnosis": "clinician_diagnosis",
        "gds": "gds_total",
        "study_phase": "study_phase",
        "diagnosis_source": "diagnosis_source_table",
    }
    strata = {
        "overall": None,
        "cohort": "cohort",
        "outcome_group": "outcome_group",
        "study_phase": "study_phase",
        "follow_up_duration": "follow_up_duration_band",
        "index_time_band": "index_relative_time_band",
        "visit_count": "visit_count_band",
    }
    rows = []
    for stratum_type, column in strata.items():
        groups = [("ALL", index_enriched)] if column is None else index_enriched.groupby(column, dropna=False)
        for value, group in groups:
            for variable, source_col in variables.items():
                missing = group[source_col].isna() | group[source_col].astype("string").isin(["Unclassified", "unknown", "<NA>"])
                rows.append(
                    {
                        "stratum_type": stratum_type,
                        "stratum_value": str(value),
                        "variable": variable,
                        "source_column": source_col,
                        "participants": len(group),
                        "missing_n": int(missing.sum()),
                        "missing_percent": float(missing.mean() * 100) if len(group) else np.nan,
                    }
                )
    missingness = pd.DataFrame(rows)
    structure_rows = []
    for variable in variables:
        for stratum_type in ["cohort", "outcome_group", "study_phase", "follow_up_duration", "visit_count"]:
            subset = missingness[
                missingness["variable"].eq(variable) & missingness["stratum_type"].eq(stratum_type)
                & missingness["participants"].ge(10)
            ]
            spread = subset["missing_percent"].max() - subset["missing_percent"].min() if not subset.empty else np.nan
            structure_rows.append(
                {
                    "variable": variable,
                    "stratum_type": stratum_type,
                    "groups_with_at_least_10": len(subset),
                    "missingness_range_percentage_points": spread,
                    "structural_missingness_signal_flag": bool(pd.notna(spread) and spread >= 10),
                    "interpretation": "descriptive signal only; not evidence of a missing-data mechanism",
                }
            )
    return missingness, pd.DataFrame(structure_rows)


def circularity_audit(
    manifests: dict[str, pd.DataFrame],
    indices: dict[str, pd.DataFrame],
    outcomes: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    crosstabs = []
    for cohort, index in indices.items():
        if index.empty:
            continue
        outcome = outcomes[cohort]
        rows.extend(
            [
                {
                    "cohort": cohort,
                    "measure": "index CDR with same-time baseline MMSE",
                    "numerator": int(index["index_mmse_same_time_flag"].sum()),
                    "denominator": len(index),
                },
                {
                    "cohort": cohort,
                    "measure": "index CDR with strictly earlier MMSE",
                    "numerator": int(index["index_mmse_strictly_before_flag"].sum()),
                    "denominator": len(index),
                },
                {
                    "cohort": cohort,
                    "measure": "outcome CDR with same-time MMSE",
                    "numerator": int(outcome["event_mmse_same_time_flag"].fillna(False).sum()),
                    "denominator": int(outcome["first_event_type"].isin(["improvement", "progression"]).sum()),
                },
                {
                    "cohort": cohort,
                    "measure": "index clinician diagnosis is MCI",
                    "numerator": int(index["clinician_diagnosed_mci_at_index_flag"].sum()),
                    "denominator": len(index),
                },
            ]
        )
        manifest = manifests[cohort]
        valid = manifest[
            manifest["classifiable_cdr_flag"].fillna(False) & manifest["clinician_diagnosis"].notna()
        ].copy()
        for (state, diagnosis), group in valid.groupby(["cdr_state_label", "clinician_diagnosis"], dropna=False):
            crosstabs.append(
                {
                    "cohort": cohort,
                    "cdr_state_label": state,
                    "clinician_diagnosis": diagnosis,
                    "visits": len(group),
                    "participants": group["participant_id"].nunique(),
                }
            )
    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit["percent"] = np.where(audit["denominator"].gt(0), audit["numerator"] / audit["denominator"] * 100, np.nan)
    return audit, pd.DataFrame(crosstabs)


def phase_form_audits(
    transitions: pd.DataFrame,
    index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_main = (
        transitions.groupby(["cohort", "destination_study_phase", "transition_label"], dropna=False)
        .agg(events=("participant_id", "size"), participants=("participant_id", "nunique"))
        .reset_index()
    ) if not transitions.empty else pd.DataFrame()
    if not phase_main.empty:
        phase_main["audit_dimension"] = "study_phase"
        phase_main["destination_group"] = phase_main["destination_study_phase"]
    calendar = (
        transitions.groupby(["cohort", "destination_calendar_period", "transition_label"], dropna=False)
        .agg(events=("participant_id", "size"), participants=("participant_id", "nunique"))
        .reset_index()
    ) if not transitions.empty else pd.DataFrame()
    if not calendar.empty:
        calendar["audit_dimension"] = "calendar_period"
        calendar["destination_group"] = calendar["destination_calendar_period"]
    index_phase = (
        index.groupby(["cohort", "study_phase"], dropna=False)
        .agg(events=("participant_id", "size"), participants=("participant_id", "nunique"))
        .reset_index().rename(columns={"study_phase": "destination_study_phase"})
    ) if not index.empty else pd.DataFrame()
    if not index_phase.empty:
        index_phase["transition_label"] = "INDEX"
        index_phase["audit_dimension"] = "study_phase"
        index_phase["destination_group"] = index_phase["destination_study_phase"]
    index_calendar = (
        index.groupby(["cohort", "index_calendar_period"], dropna=False)
        .agg(events=("participant_id", "size"), participants=("participant_id", "nunique"))
        .reset_index().rename(columns={"index_calendar_period": "destination_calendar_period"})
    ) if not index.empty and "index_calendar_period" in index else pd.DataFrame()
    if not index_calendar.empty:
        index_calendar["transition_label"] = "INDEX"
        index_calendar["audit_dimension"] = "calendar_period"
        index_calendar["destination_group"] = index_calendar["destination_calendar_period"]
    phase = pd.concat([phase_main, calendar, index_phase, index_calendar], ignore_index=True, sort=False)
    form = (
        transitions.groupby(["cohort", "destination_form_family", "destination_diagnosis_method", "transition_label"], dropna=False)
        .agg(events=("participant_id", "size"), participants=("participant_id", "nunique"))
        .reset_index()
    ) if not transitions.empty else pd.DataFrame()
    index_phase = (
        index.groupby(["cohort", "study_phase", "form_family", "diagnosis_method"], dropna=False)
        .agg(index_participants=("participant_id", "nunique"))
        .reset_index()
    ) if not index.empty else pd.DataFrame()
    form = pd.concat([form, index_phase.assign(transition_label="INDEX")], ignore_index=True, sort=False)
    relative = (
        transitions.groupby(["cohort", "destination_relative_time_band", "transition_label"], dropna=False)
        .agg(events=("participant_id", "size"), participants=("participant_id", "nunique"))
        .reset_index()
    ) if not transitions.empty else pd.DataFrame()
    if not relative.empty:
        relative["audit_dimension"] = "relative_time_band"
        relative["destination_form_family"] = relative["destination_relative_time_band"]
    form["audit_dimension"] = form.get("audit_dimension", "form_family")
    form = pd.concat([form, relative], ignore_index=True, sort=False)
    return phase, form


def _strict_diagnosis_history(cohort: str) -> pd.DataFrame:
    src = SOURCES[cohort]
    raw = read_csv(src.diagnosis)
    if cohort == "adni":
        participant = raw["RID"].astype("Int64").astype("string")
        time = (parse_date(raw["EXAMDATE"]) - pd.Timestamp("1970-01-01")).dt.days
        state = numeric(raw["DIAGNOSIS"]).map({1: 0, 2: 1, 3: 2})
    elif cohort == "oasis3":
        participant = raw["OASISID"].astype("string")
        time = numeric(raw["days_to_visit"])
        normal = numeric(raw["NORMCOG"]).eq(1)
        mci = pd.concat(
            [numeric(raw[c]).eq(1) for c in ["MCIAMEM", "MCIAPLUS", "MCINON1", "MCINON2"]], axis=1
        ).any(axis=1)
        ad_dementia = numeric(raw["DEMENTED"]).eq(1) & (
            numeric(raw["PROBAD"]).eq(1) | numeric(raw["POSSAD"]).eq(1) | numeric(raw["alzdis"]).eq(1)
        )
        state = pd.Series(np.nan, index=raw.index)
        state.loc[normal] = 0
        state.loc[mci] = 1
        state.loc[ad_dementia] = 2
    else:
        participant = raw["RID"].astype("Int64").astype("string")
        visit = raw["VISCODE"].astype(str)
        time = np.where(visit.eq("bl"), 0, numeric(visit.str.extract(r"m(\d+)", expand=False)) * DAYS_PER_MONTH)
        state = numeric(raw["DXCURREN"]).map({1: 0, 2: 1, 3: 2})
    history = pd.DataFrame({"participant_id": participant, "time": time, "state": state}).dropna()
    history["state"] = history["state"].astype(int)
    history = (
        history.sort_values(["participant_id", "time", "state"], ascending=[True, True, False])
        .drop_duplicates(["participant_id", "time"], keep="first")
    )
    return history


def diagnosis_cdr_transition_comparison(
    all_events: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cohort in SOURCES:
        cdr_events = all_events[all_events["cohort"].eq(cohort)]
        for (a, b), label in TRANSITION_LABELS.items():
            subset = cdr_events[cdr_events["from_state"].eq(a) & cdr_events["to_state"].eq(b)]
            rows.append({"cohort": cohort, "definition": "CDR state", "transition_label": label, "events": len(subset), "participants": subset["participant_id"].nunique()})
        history = _strict_diagnosis_history(cohort)
        diag_records = []
        for participant, group in history.groupby("participant_id"):
            group = group.sort_values("time")
            prior = group.shift(1)
            for i in range(1, len(group)):
                a, b = int(group.iloc[i - 1]["state"]), int(group.iloc[i]["state"])
                if a != b:
                    diag_records.append((participant, a, b))
        diag = pd.DataFrame(diag_records, columns=["participant_id", "from_state", "to_state"])
        for (a, b), label in TRANSITION_LABELS.items():
            subset = diag[diag["from_state"].eq(a) & diag["to_state"].eq(b)] if not diag.empty else diag
            rows.append({"cohort": cohort, "definition": "strict clinician diagnosis", "transition_label": label, "events": len(subset), "participants": subset["participant_id"].nunique() if not subset.empty else 0})
    return pd.DataFrame(rows)


def previous_count_comparison(current: pd.DataFrame) -> pd.DataFrame:
    prior = [
        ("oasis3", "CDR state", "CDR 0.5 -> 0", 291, 201, "demographics report CDR sensitivity"),
        ("oasis3", "CDR state", "CDR 0.5 -> >=1", 231, 223, "demographics report CDR sensitivity"),
        ("oasis3", "CDR state", "CDR 0 -> >=1", 19, 19, "demographics report CDR sensitivity"),
        ("adni", "strict clinician diagnosis", "CDR 0.5 -> 0", 150, 134, "demographics/feasibility diagnosis MCI->CN"),
        ("adni", "strict clinician diagnosis", "CDR 0.5 -> >=1", 453, np.nan, "feasibility strict diagnosis matrix"),
        ("oasis3", "strict clinician diagnosis", "CDR 0.5 -> 0", 88, 79, "demographics/feasibility strict diagnosis"),
        ("oasis3", "strict clinician diagnosis", "CDR 0.5 -> >=1", 61, np.nan, "feasibility strict diagnosis matrix"),
        ("aibl", "strict clinician diagnosis", "CDR 0.5 -> 0", 11, 11, "demographics/feasibility diagnosis"),
        ("aibl", "strict clinician diagnosis", "CDR 0.5 -> >=1", 19, np.nan, "feasibility strict diagnosis matrix"),
    ]
    prior_df = pd.DataFrame(prior, columns=["cohort", "definition", "transition_label", "previous_events", "previous_participants", "previous_source"])
    merged = prior_df.merge(current, on=["cohort", "definition", "transition_label"], how="left")
    merged = merged.rename(columns={"events": "reconstructed_events", "participants": "reconstructed_participants"})
    merged["event_difference"] = merged["reconstructed_events"] - merged["previous_events"]
    merged["participant_difference"] = merged["reconstructed_participants"] - merged["previous_participants"]
    merged["reconciliation"] = np.where(
        merged["event_difference"].eq(0) & (merged["participant_difference"].eq(0) | merged["previous_participants"].isna()),
        "exact match",
        "difference investigated in Phase 1 report",
    )
    return merged


def source_inventory() -> pd.DataFrame:
    purposes = {
        "cdr": "CDR anchor and cognitive-state outcome",
        "mmse": "MMSE longitudinal predictor availability",
        "diagnosis": "Clinician diagnosis and agreement/sensitivity",
        "demographics": "Age, sex, and education",
        "apoe": "APOE genotype and epsilon-4 dose",
        "gds": "Geriatric Depression Scale availability",
        "codebook": "OASIS-3 clinical variable definitions",
    }
    identifiers = {"adni": "RID", "oasis3": "OASISID", "aibl": "RID"}
    time_fields = {
        "adni": {"cdr": "VISDATE/VISCODE2", "mmse": "VISDATE/VISCODE2", "diagnosis": "EXAMDATE/VISCODE2", "demographics": "VISDATE", "apoe": "VISCODE/APTESTDT", "gds": "VISDATE/VISCODE2"},
        "oasis3": {"cdr": "days_to_visit/OASIS_session_label", "mmse": "days_to_visit/OASIS_session_label", "diagnosis": "days_to_visit/OASIS_session_label", "demographics": "AgeatEntry", "apoe": "time-invariant", "gds": "days_to_visit/OASIS_session_label", "codebook": "not applicable"},
        "aibl": {"cdr": "EXAMDATE/VISCODE", "mmse": "EXAMDATE/VISCODE", "diagnosis": "VISCODE", "demographics": "VISCODE", "apoe": "VISCODE"},
    }
    relevant = {
        "cdr": "participant, visit/time, CDR global, CDR-SB/domains, phase/form/site/QC",
        "mmse": "participant, visit/time, MMSE total",
        "diagnosis": "participant, visit/time, clinician diagnosis, source/method/form",
        "demographics": "participant, age/DOB, sex, education",
        "apoe": "participant, genotype or alleles",
        "gds": "participant, visit/time, GDS total",
        "codebook": "variable name, question, values, UDS version",
    }
    rows = []
    seen: set[tuple[str, Path]] = set()
    for cohort, src in SOURCES.items():
        for role in ["cdr", "mmse", "diagnosis", "demographics", "apoe", "gds", "codebook"]:
            path = getattr(src, role)
            if path is None or (cohort, path) in seen:
                continue
            seen.add((cohort, path))
            if path.suffix.lower() == ".csv":
                data = read_csv(path)
                participant = identifiers[cohort]
                visit_columns = [x for x in re.split(r"/", time_fields[cohort].get(role, "")) if x in data]
                dup_cols = [participant] + visit_columns if participant in data else visit_columns
                duplicate_count = int(data.duplicated(dup_cols, keep=False).sum()) if dup_cols else np.nan
                rows.append(
                    {
                        "cohort": cohort,
                        "file_name": path.name,
                        "relative_path": str(path.relative_to(ROOT)),
                        "table_purpose": purposes[role],
                        "participant_identifier": participant,
                        "visit_or_time_field": time_fields[cohort].get(role, ""),
                        "row_count": len(data),
                        "column_count": len(data.columns),
                        "relevant_clinical_variables": relevant[role],
                        "duplicate_participant_visit_rows": duplicate_count,
                        "exact_duplicate_rows": int(data.duplicated(keep=False).sum()),
                        "file_version_or_date": " ".join(re.findall(r"\d{2}[A-Za-z]{3}\d{4}|\d{2}-[A-Za-z]{3}-\d{4}|\d+\.\d+\.\d+", str(path))),
                        "sha256": file_sha256(path),
                    }
                )
            else:
                workbook = pd.ExcelFile(path)
                rows.append(
                    {
                        "cohort": cohort,
                        "file_name": path.name,
                        "relative_path": str(path.relative_to(ROOT)),
                        "table_purpose": purposes[role],
                        "participant_identifier": "not applicable",
                        "visit_or_time_field": "not applicable",
                        "row_count": np.nan,
                        "column_count": len(workbook.sheet_names),
                        "relevant_clinical_variables": relevant[role],
                        "duplicate_participant_visit_rows": np.nan,
                        "exact_duplicate_rows": np.nan,
                        "file_version_or_date": "supplied OASIS-3 workbook; UDS2/UDS3 sheets",
                        "sha256": file_sha256(path),
                    }
                )
    return pd.DataFrame(rows)


def sentinel_map() -> pd.DataFrame:
    records = [
        ("adni", "All_Subjects_CDR_26Aug2025.csv", "CDGLOBAL", "-1", "invalid/outside accepted global CDR range", "recode to missing for harmonized CDR only", "range validation; supplied ADNI codebook absent", "provisional"),
        ("adni", "All_Subjects_MMSE_26Aug2025.csv", "MMSCORE", "-1", "invalid/outside 0-30 MMSE range", "recode to missing for harmonized MMSE only", "instrument range; supplied ADNI codebook absent", "provisional"),
        ("adni", "All_Subjects_PTDEMOG_26Aug2025.csv", "PTGENDER", "-4", "not a valid 1/2 project sex code", "recode to missing for harmonized sex only", "existing project convention; supplied ADNI codebook absent", "provisional"),
        ("adni", "All_Subjects_PTDEMOG_26Aug2025.csv", "PTEDUCAT", "-4|-1", "negative years invalid", "recode to missing for education only", "logical range; supplied ADNI codebook absent", "provisional"),
        ("oasis3", "OASIS3_demographics.csv", "APOE", "0", "not a two-allele genotype", "recode to missing for harmonized APOE only", "supplied workbook does not define APOE encoding", "unresolved"),
        ("aibl", "aibl_cdr_01-Jun-2018.csv", "CDGLOBAL", "-4", "invalid/outside accepted global CDR range", "recode to missing for harmonized CDR only", "range validation; AIBL codebook absent", "provisional"),
        ("aibl", "aibl_mmse_01-Jun-2018.csv", "MMSCORE", "-4", "invalid/outside 0-30 MMSE range", "recode to missing for harmonized MMSE only", "instrument range; AIBL codebook absent", "provisional"),
        ("aibl", "aibl_pdxconv_01-Jun-2018.csv", "DXCURREN", "-4|7", "-4 unclassified; 7 other dementia", "-4 missing; retain 7 as other dementia, not AD", "existing report mapping; AIBL codebook absent", "provisional"),
    ]
    return pd.DataFrame(records, columns=["cohort", "source_table", "source_variable", "source_code", "meaning_or_validity", "harmonization_action", "evidence", "validation_status"])


def variable_provenance() -> pd.DataFrame:
    mappings = {
        "adni": {
            "participant_id": ("CDR", "RID", "string conversion"), "original_visit_id": ("CDR", "VISCODE2", "preserved"),
            "visit_date": ("CDR", "VISDATE", "ISO date"), "relative_time_days": ("CDR", "VISDATE", "days since participant's first CDR record"),
            "age_at_visit": ("PTDEMOG+CDR", "PTDOB/PTDOBYY+VISDATE", "month/year or midyear approximate age"),
            "sex": ("PTDEMOG", "PTGENDER", "1=Male, 2=Female project convention"), "education_years": ("PTDEMOG", "PTEDUCAT", "valid nonnegative years"),
            "apoe_genotype": ("APOERES", "GENOTYPE", "normalized allele string"), "apoe_e4_dose": ("APOERES", "GENOTYPE", "count allele 4"),
            "cdr_global": ("CDR", "CDGLOBAL", "valid values 0,0.5,1,2,3; conflicts unresolved"), "mmse": ("MMSE", "MMSCORE", "same visit/nearest within 30 days"),
            "clinician_diagnosis": ("DXSUM", "DIAGNOSIS", "1=CN,2=MCI,3=AD/dementia"), "diagnosis_method": ("DXSUM", "not supplied", "explicitly unavailable"),
            "study_phase": ("CDR", "PHASE", "preserved"), "form_family": ("CDR", "PHASE", "phase used as source family"), "gds_total": ("GDSCALE", "GDTOTAL", "nearest on/before within 180 days"),
        },
        "oasis3": {
            "participant_id": ("UDSb4", "OASISID", "preserved"), "original_visit_id": ("UDSb4", "OASIS_session_label", "preserved"),
            "visit_date": ("UDSb4", "not supplied", "unavailable"), "relative_time_days": ("UDSb4", "days_to_visit", "preserved days from study entry"),
            "age_at_visit": ("UDSb4", "age at visit", "preserved"), "sex": ("demographics", "GENDER", "dictionary: 1=Male,2=Female"),
            "education_years": ("demographics", "EDUC", "preserved; years inferred from observed scale/name"), "apoe_genotype": ("demographics", "APOE", "two digits normalized to allele string"),
            "apoe_e4_dose": ("demographics", "APOE", "count allele 4"), "cdr_global": ("UDSb4", "CDRTOT", "provisional de facto global field; dictionary mismatch flagged"),
            "mmse": ("UDSb4", "MMSE", "same clinical row"), "clinician_diagnosis": ("UDSd1", "NORMCOG/DEMENTED/MCI fields/alzdis", "strict mutually exclusive mapping"),
            "diagnosis_method": ("UDSd1", "WHODIDDX/dxmethod", "UDS-version-specific mapping"), "study_phase": ("UDSd1", "field family", "UDS2/UDS3/unidentified"),
            "form_family": ("UDSd1", "field availability", "UDS2/UDS3/unidentified"), "gds_total": ("UDSb6", "GDS", "nearest on/before within 180 days"),
        },
        "aibl": {
            "participant_id": ("cdr", "RID", "string conversion"), "original_visit_id": ("cdr", "VISCODE", "preserved"),
            "visit_date": ("cdr", "EXAMDATE", "ISO date"), "relative_time_days": ("cdr", "EXAMDATE", "days since first CDR"),
            "age_at_visit": ("ptdemog+cdr", "PTDOB+EXAMDATE", "year-only midyear approximation; unverified"), "sex": ("ptdemog", "PTGENDER", "1=Male,2=Female project convention; unverified"),
            "education_years": ("not supplied", "not supplied", "unavailable"), "apoe_genotype": ("apoeres", "APGEN1/APGEN2", "normalized allele string"),
            "apoe_e4_dose": ("apoeres", "APGEN1/APGEN2", "count allele 4"), "cdr_global": ("cdr", "CDGLOBAL", "valid range; -4 invalid"),
            "mmse": ("mmse", "MMSCORE", "same visit/time"), "clinician_diagnosis": ("pdxconv", "DXCURREN", "1=CN,2=MCI,3=AD/dementia,7=other"),
            "diagnosis_method": ("not supplied", "not supplied", "unavailable"), "study_phase": ("file release", "extract 3.3.0", "constant"),
            "form_family": ("file release", "extract 3.3.0", "constant"), "gds_total": ("not supplied", "not supplied", "unavailable"),
        },
    }
    rows = []
    common = {
        "cdr_state": ("canonical", "cdr_global", "0->0; 0.5->1; >=1->2"),
        "duplicate_resolution_unresolved_flag": ("canonical", "participant+time groups", "true when valid CDR values conflict"),
        "near_duplicate_visit_flag": ("canonical", "time", f"adjacent CDR assessment within {NEAR_DUPLICATE_DAYS} days"),
        "classifiable_cdr_flag": ("canonical", "cdr/state/QC", "valid CDR and no unresolved same-time conflict"),
    }
    for cohort, fields in mappings.items():
        for field, (table, column, transformation) in {**fields, **common}.items():
            rows.append(
                {
                    "cohort": cohort, "canonical_field": field, "source_table": table, "source_column": column,
                    "transformation": transformation, "units": "days" if field.endswith("time_days") or field == "relative_time_days" else "field-specific",
                    "timing_rule": "at CDR visit; baseline MMSE separately constrained on/before index" if field in {"mmse", "clinician_diagnosis", "gds_total"} else "visit-level or time-invariant",
                    "codebook_reference": "supplied OASIS workbook/UDS DED" if cohort == "oasis3" else "existing project notes; source codebook not supplied",
                    "missingness_rule": "variable-level range/code validation only; no global sentinel replacement",
                    "qc_rule": "preserve source value; flag invalid, ambiguous, duplicate, or temporal mismatch",
                    "analysis_role": "outcome" if field in {"cdr_global", "cdr_state"} else "candidate predictor/metadata",
                }
            )
    return pd.DataFrame(rows)


def source_field_validation() -> pd.DataFrame:
    rows = [
        ("adni", "participant identifier", "RID", "validated", "Stable numeric subject ID in all core tables"),
        ("adni", "visit identifier", "VISCODE2 (VISCODE preserved)", "validated", "VISCODE2 used with actual dates; labels never determine order"),
        ("adni", "visit date/time", "VISDATE; EXAMDATE in diagnosis", "validated", "Actual date used; days rebased per participant"),
        ("adni", "CDR global", "CDGLOBAL", "provisional", "Range is appropriate; supplied extract has no ADNI codebook"),
        ("adni", "MMSE", "MMSCORE", "provisional", "0-30 instrument range enforced; source codebook absent"),
        ("adni", "diagnosis", "DIAGNOSIS", "provisional", "Existing project mapping 1=CN, 2=MCI, 3=AD/dementia"),
        ("adni", "age", "PTDOB/PTDOBYY + VISDATE", "validated-with-approximation", "Birth month/year or year only; mid-period approximation documented"),
        ("adni", "sex", "PTGENDER", "provisional", "Existing project convention 1=Male, 2=Female; source codebook absent"),
        ("adni", "education", "PTEDUCAT", "provisional", "Plausible years; negative values invalidated variable-specifically"),
        ("adni", "APOE", "GENOTYPE", "validated-by-values", "Alleles 2/3/4; epsilon-4 dose counted"),
        ("adni", "phase/form", "PHASE/DD_CRF_VERSION_LABEL", "validated", "Both preserved; form label is mostly missing"),
        ("oasis3", "participant identifier", "OASISID", "validated", "Documented by supplied workbook"),
        ("oasis3", "clinical session", "OASIS_session_label", "validated", "Documented; exact temporal ordering uses days_to_visit"),
        ("oasis3", "relative time", "days_to_visit", "validated", "Documented as day 0000 = entry; no calendar dates supplied"),
        ("oasis3", "CDR global", "CDRTOT", "unresolved-provisional", "Delivered values behave as global CDR, but supplied workbook says CDRGLOB is global and labels CDRTOT as a UDS2 total; CDRGLOB is absent"),
        ("oasis3", "CDR sum of boxes", "CDRSUM", "validated", "Workbook documents standard CDR sum of boxes and observed range agrees"),
        ("oasis3", "MMSE", "MMSE", "validated-by-range", "On UDSb4 delivered row; workbook B4 sheet contains only an unlabeled mmse placeholder"),
        ("oasis3", "diagnosis", "NORMCOG/DEMENTED/MCI*/IMPNOMCI/alzdis", "validated", "UDS2 and UDS3 definitions verified in supplied workbook/DED"),
        ("oasis3", "diagnosis method", "WHODIDDX/dxmethod", "validated", "UDS2 single/consensus and UDS3 single/formal/other definitions documented"),
        ("oasis3", "age", "age at visit/AgeatEntry", "validated", "Visit age documented; source value retained"),
        ("oasis3", "sex", "GENDER", "validated", "Workbook: 1=Male, 2=Female"),
        ("oasis3", "education", "EDUC", "partially-documented", "Field supplied but workbook lacks units; observed range is consistent with years"),
        ("oasis3", "APOE", "APOE", "partially-documented", "Observed two-digit allele encoding is clear; workbook does not document values; code 0 unresolved"),
        ("oasis3", "UDS form family", "WHODIDDX/dxmethod field availability", "derived-validated", "UDS2/UDS3-specific fields documented"),
        ("aibl", "participant identifier", "RID", "validated-by-consistency", "Shared across supplied core tables; codebook absent"),
        ("aibl", "visit/time", "VISCODE/EXAMDATE", "validated-by-consistency", "Actual date orders CDR/MMSE; VISCODE identifies scheduled cycle"),
        ("aibl", "CDR", "CDGLOBAL", "provisional", "Plausible range; -4 invalid; codebook absent"),
        ("aibl", "MMSE", "MMSCORE", "provisional", "0-30 range enforced; codebook absent"),
        ("aibl", "diagnosis", "DXCURREN", "provisional", "Existing project mapping; codebook absent"),
        ("aibl", "sex", "PTGENDER", "provisional", "Existing 1/2 convention; codebook absent"),
        ("aibl", "APOE", "APGEN1/APGEN2", "validated-by-values", "Alleles limited to 2/3/4"),
        ("aibl", "age", "PTDOB year + EXAMDATE", "unresolved", "Year can be parsed, but no supplied dictionary confirms privacy transformation; descriptive estimate only"),
        ("aibl", "education", "not found", "unavailable", "No education field found in supplied AIBL tables"),
    ]
    return pd.DataFrame(rows, columns=["cohort", "concept", "source_field", "validation_status", "decision_notes"])


def run_qc(
    manifests: dict[str, pd.DataFrame],
    indices: dict[str, pd.DataFrame],
    outcomes: dict[str, pd.DataFrame],
    transitions: dict[str, pd.DataFrame],
    transition_summaries: dict[str, pd.DataFrame],
    horizon_person: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(cohort: str, check: str, passed: bool, severity: str, details: str) -> None:
        checks.append(
            {"cohort": cohort, "check": check, "status": "PASS" if passed else ("WARN" if severity == "warning" else "FAIL"), "severity": severity, "details": details}
        )

    for cohort, manifest in manifests.items():
        add(cohort, "unique manifest keys", manifest["manifest_key"].is_unique, "critical", f"rows={len(manifest):,}; unique={manifest['manifest_key'].nunique():,}")
        ordered = manifest.sort_values(["participant_id", "time_sort_days"])
        negative_order = ordered.groupby("participant_id")["time_sort_days"].diff().lt(0).sum()
        add(cohort, "valid temporal ordering", negative_order == 0, "critical", f"negative adjacent intervals={negative_order}")
        invalid_cdr = manifest.loc[manifest["cdr_global"].notna() & ~manifest["cdr_global"].isin(VALID_CDR)]
        add(cohort, "valid harmonized CDR range", invalid_cdr.empty, "critical", f"invalid rows={len(invalid_cdr)}")
        invalid_mmse = manifest.loc[manifest["mmse"].notna() & ~numeric(manifest["mmse"]).between(0, 30)]
        add(cohort, "valid harmonized MMSE range", invalid_mmse.empty, "critical", f"invalid rows={len(invalid_mmse)}")
        invalid_age = manifest.loc[manifest["age_at_visit"].notna() & ~numeric(manifest["age_at_visit"]).between(18, 110)]
        add(cohort, "no impossible harmonized ages", invalid_age.empty, "critical", f"invalid rows={len(invalid_age)}")
        add(cohort, "deterministic duplicate selection", manifest["source_row_number"].notna().all(), "critical", "Every canonical time point retains a source row")
        index = indices[cohort]
        future_mmse = int((numeric(index["index_mmse_time_days"]) > numeric(index["index_time_days"])).sum()) if not index.empty else 0
        add(cohort, "no future MMSE used at index", future_mmse == 0, "critical", f"future MMSE rows={future_mmse}")
        prior_dementia_violations = 0
        classifiable = manifest[manifest["classifiable_cdr_flag"].fillna(False)]
        for row in index.itertuples(index=False):
            prior_dementia_violations += int(
                not classifiable[
                    classifiable["participant_id"].astype(str).eq(str(row.participant_id))
                    & classifiable["time_sort_days"].lt(row.index_time_days)
                    & classifiable["cdr_state"].eq(2)
                ].empty
            )
        add(cohort, "no index after prior dementia-stage CDR", prior_dementia_violations == 0, "critical", f"violations={prior_dementia_violations}")
        outcome = outcomes[cohort]
        negative_followup = int((numeric(outcome["analysis_endpoint_time_days"]) < 0).sum()) if not outcome.empty else 0
        add(cohort, "no negative index follow-up", negative_followup == 0, "critical", f"violations={negative_followup}")
        event_nonpositive = int((numeric(outcome["first_event_time_days"]).dropna() <= 0).sum()) if not outcome.empty else 0
        add(cohort, "all primary transition times after index", event_nonpositive == 0, "critical", f"violations={event_nonpositive}")
        events = transitions[cohort]
        event_nonpositive_all = int((numeric(events["transition_interval_days"]) <= 0).sum()) if not events.empty else 0
        add(cohort, "all chronological transition intervals positive", event_nonpositive_all == 0, "critical", f"violations={event_nonpositive_all}")
        summary = transition_summaries[cohort]
        summary_events = int(summary["events"].sum()) if not summary.empty else 0
        add(cohort, "transition summary agrees with row-level events", summary_events == len(events), "critical", f"summary={summary_events}; row-level={len(events)}")
        horizon = horizon_person[cohort]
        horizon_ok = all(group["participant_id"].nunique() == len(index) for _, group in horizon.groupby("horizon_months")) if not horizon.empty else index.empty
        add(cohort, "horizon rows agree with index denominator", horizon_ok, "critical", f"index n={len(index)}")

    global_ids = pd.concat(
        [m[["cohort", "participant_id"]].drop_duplicates().assign(global_id=lambda d: d["cohort"] + ":" + d["participant_id"].astype(str)) for m in manifests.values()],
        ignore_index=True,
    )
    add("all", "cohort-qualified participant IDs do not cross labels", global_ids["global_id"].is_unique, "critical", f"qualified IDs={global_ids['global_id'].nunique():,}")
    add("oasis3", "OASIS CDR global source field confirmed by supplied dictionary", False, "warning", "CDRTOT behaves as global but dictionary labels CDRGLOB as global; confirmation required before locked external validation")
    add("aibl", "AIBL age derivation verified by supplied codebook", False, "warning", "Year-only PTDOB is parseable but no AIBL codebook is supplied")
    qc = pd.DataFrame(checks)
    critical_failures = qc[(qc["severity"].eq("critical")) & (qc["status"].eq("FAIL"))]
    if not critical_failures.empty:
        raise RuntimeError("Critical Phase 1 QC failures:\n" + critical_failures.to_string(index=False))
    return qc


def _save_figure(fig: plt.Figure, stem: str, dpi: int = 600) -> None:
    fig.savefig(PLOT_ROOT / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(PLOT_ROOT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(
    flow: pd.DataFrame,
    transitions: pd.DataFrame,
    outcomes: pd.DataFrame,
    horizon: pd.DataFrame,
    depth_summary: pd.DataFrame,
    intervals: pd.DataFrame,
    missingness: pd.DataFrame,
    phase_counts: pd.DataFrame,
    form_counts: pd.DataFrame,
    diagnosis_crosstab: pd.DataFrame,
) -> None:
    colors = {"adni": "#315A7D", "oasis3": "#B85C38", "aibl": "#5B7F48"}
    cohorts = [c for c in ["adni", "oasis3", "aibl"] if c in flow["cohort"].unique()]

    fig, axes = plt.subplots(1, len(cohorts), figsize=(5 * len(cohorts), 5), squeeze=False)
    for ax, cohort in zip(axes[0], cohorts):
        data = flow[flow["cohort"].eq(cohort)].sort_values("step_order")
        y = np.arange(len(data))
        ax.barh(y, data["participants_remaining"], color=colors[cohort])
        ax.set_yticks(y, [str(x).replace(" candidate", "\ncandidate") for x in data["flow_step"]], fontsize=8)
        ax.invert_yaxis(); ax.set_title(cohort.upper(), fontsize=13); ax.set_xlabel("Participants")
        for yi, value in zip(y, data["participants_remaining"]):
            ax.text(value, yi, f" {int(value):,}", va="center", fontsize=9)
    fig.suptitle("Phase 1 index-cohort flow", fontsize=16)
    fig.tight_layout()
    _save_figure(fig, "cohort_flow_diagram", 600)

    fig, axes = plt.subplots(1, len(cohorts), figsize=(5 * len(cohorts), 4.8), squeeze=False)
    positions = {0: (0.1, 0.5), 1: (0.5, 0.5), 2: (0.9, 0.5)}
    for ax, cohort in zip(axes[0], cohorts):
        data = transitions[transitions["cohort"].eq(cohort)]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_title(cohort.upper(), fontsize=14)
        for state, (x, y) in positions.items():
            ax.scatter([x], [y], s=1600, color="#E9EEF2", edgecolor="#263238", zorder=3)
            ax.text(x, y, state_label(state), ha="center", va="center", fontsize=10, zorder=4)
        for (a, b), label in TRANSITION_LABELS.items():
            subset = data[data["from_state"].eq(a) & data["to_state"].eq(b)]
            n, p = len(subset), subset["participant_id"].nunique()
            if n == 0: continue
            x1, y1 = positions[a]; x2, y2 = positions[b]
            curve = 0.22 if a < b else -0.22
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2 + np.log1p(n) / 3, color=colors[cohort], connectionstyle=f"arc3,rad={curve}"))
            midx, midy = (x1 + x2) / 2, 0.68 if a < b else 0.32
            ax.text(midx, midy, f"{n:,} events\n{p:,} people", ha="center", va="center", fontsize=8)
    fig.suptitle("Observed CDR state-change events (all chronological episodes)", fontsize=15)
    fig.tight_layout()
    _save_figure(fig, "cdr_transition_diagrams", 1200)

    fig, ax = plt.subplots(figsize=(9, 5))
    for cohort in cohorts:
        values = outcomes.loc[outcomes["cohort"].eq(cohort), "censoring_time_months"].dropna()
        ax.hist(values, bins=30, alpha=0.45, label=f"{cohort.upper()} (n={len(values):,})", color=colors[cohort])
    ax.set(title="Follow-up from first eligible CDR 0.5 index", xlabel="Months", ylabel="Participants")
    ax.legend(fontsize=10); fig.tight_layout(); _save_figure(fig, "index_follow_up_distribution", 600)

    fig, ax = plt.subplots(figsize=(9, 5))
    for cohort in cohorts:
        values = intervals.loc[(intervals["cohort"].eq(cohort)) & (intervals["instrument"].eq("CDR")), "interval_months"].clip(upper=60)
        ax.hist(values, bins=30, alpha=0.4, label=cohort.upper(), color=colors[cohort])
    ax.set(title="CDR inter-assessment intervals (truncated at 60 months for display)", xlabel="Months", ylabel="Intervals")
    ax.legend(); fig.tight_layout(); _save_figure(fig, "inter_visit_interval_distribution", 600)

    plot_h = horizon.melt(
        id_vars=["cohort", "horizon_months"],
        value_vars=["transition_to_cdr_0", "transition_to_cdr_ge1", "remain_cdr_0_5", "censored_before_horizon"],
        var_name="status", value_name="participants",
    )
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5 * len(cohorts), 5), squeeze=False)
    status_colors = ["#3A923A", "#C44E52", "#4C72B0", "#A0A0A0"]
    for ax, cohort in zip(axes[0], cohorts):
        pivot = plot_h[plot_h["cohort"].eq(cohort)].pivot(index="horizon_months", columns="status", values="participants").fillna(0)
        pivot.plot(kind="bar", ax=ax, color=status_colors, fontsize=9)
        ax.set_title(cohort.upper()); ax.set_xlabel("Horizon (months)"); ax.set_ylabel("Participants")
        ax.legend(fontsize=7, title="")
    fig.suptitle("Fixed-horizon outcome feasibility", fontsize=16); fig.tight_layout(); _save_figure(fig, "horizon_event_counts", 600)

    overall_depth = depth_summary[depth_summary["outcome_group"].eq("ALL")].set_index("cohort")
    fig, ax = plt.subplots(figsize=(9, 5))
    cols = ["mmse_at_least_1", "mmse_at_least_2", "mmse_at_least_3", "mmse_at_least_4"]
    overall_depth[cols].plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set(title="Pre-event/censor MMSE longitudinal depth", xlabel="Cohort", ylabel="Participants")
    ax.set_xticklabels([x.upper() for x in overall_depth.index], rotation=0); ax.legend([">=1", ">=2", ">=3", ">=4"], title="MMSE assessments")
    fig.tight_layout(); _save_figure(fig, "longitudinal_mmse_depth", 600)

    miss = missingness[(missingness["stratum_type"].eq("cohort"))].pivot(index="variable", columns="stratum_value", values="missing_percent")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(miss.values, cmap="magma_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(miss.columns)), [str(x).upper() for x in miss.columns], fontsize=10)
    ax.set_yticks(range(len(miss.index)), miss.index, fontsize=10)
    for i in range(len(miss.index)):
        for j in range(len(miss.columns)):
            val = miss.iloc[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8, color="white" if val > 55 else "black")
    fig.colorbar(image, ax=ax, label="Missing (%)"); ax.set_title("Clinical-core missingness at index")
    fig.tight_layout(); _save_figure(fig, "clinical_core_missingness_heatmap", 1200)

    adni_phase = phase_counts[
        phase_counts["cohort"].eq("adni")
        & phase_counts["audit_dimension"].eq("study_phase")
        & ~phase_counts["transition_label"].eq("INDEX")
    ]
    if not adni_phase.empty:
        pivot = adni_phase.pivot_table(index="destination_study_phase", columns="transition_label", values="events", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(11, 5)); pivot.plot(kind="bar", ax=ax)
        ax.set(title="ADNI CDR state-change events by destination phase", xlabel="ADNI phase", ylabel="Events")
        ax.legend(fontsize=8, ncol=2); fig.tight_layout(); _save_figure(fig, "adni_transition_counts_by_phase", 600)

    oasis_form = form_counts[
        form_counts["cohort"].eq("oasis3")
        & form_counts["audit_dimension"].eq("form_family")
        & ~form_counts["transition_label"].eq("INDEX")
    ]
    if not oasis_form.empty:
        pivot = oasis_form.pivot_table(index="destination_form_family", columns="transition_label", values="events", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(11, 5)); pivot.plot(kind="bar", ax=ax)
        ax.set(title="OASIS-3 CDR state-change events by destination diagnosis-form family", xlabel="Form family", ylabel="Events")
        ax.legend(fontsize=8, ncol=2); fig.tight_layout(); _save_figure(fig, "oasis3_transition_counts_by_form_family", 600)

    diag = diagnosis_crosstab.groupby(["cohort", "cdr_state_label", "clinician_diagnosis"])["visits"].sum().reset_index()
    top_dx = diag.groupby("clinician_diagnosis")["visits"].sum().nlargest(7).index
    diag = diag[diag["clinician_diagnosis"].isin(top_dx)]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5 * len(cohorts), 5), squeeze=False)
    for ax, cohort in zip(axes[0], cohorts):
        pivot = diag[diag["cohort"].eq(cohort)].pivot(index="cdr_state_label", columns="clinician_diagnosis", values="visits").fillna(0)
        if pivot.empty: ax.axis("off"); continue
        image = ax.imshow(np.log1p(pivot.values), cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=9); ax.set_title(cohort.upper())
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{int(pivot.iloc[i,j]):,}", ha="center", va="center", fontsize=7)
    fig.suptitle("Clinician diagnosis versus CDR-derived state", fontsize=15); fig.tight_layout(); _save_figure(fig, "clinician_diagnosis_vs_cdr_state", 1200)


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None, digits: int = 1) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy() if max_rows else frame.copy()
    float_cols = view.select_dtypes(include=["float"]).columns
    view[float_cols] = view[float_cols].round(digits)
    return view.to_markdown(index=False)


def write_data_dictionary(manifests: dict[str, pd.DataFrame], provenance: pd.DataFrame) -> Path:
    descriptions = {
        "manifest_key": "Unique cohort-qualified participant-time key.", "cohort": "Canonical cohort name.",
        "participant_id": "Cohort-specific participant identifier as text.", "original_visit_id": "Original clinical visit/session label.",
        "visit_date": "ISO visit date where supplied; unavailable in OASIS-3.", "relative_time_days": "Cohort-relative visit time in days.",
        "relative_time_months": "Relative days divided by 30.4375.", "time_sort_days": "Internal monotonic time coordinate; Unix-day scale for dated cohorts, source relative day for OASIS-3.",
        "age_at_visit": "Age at CDR visit in years; derivation status is separate.", "sex": "Male/Female harmonized value.",
        "education_years": "Education value interpreted as years; unavailable in AIBL.", "apoe_genotype": "Normalized APOE allele pair.",
        "apoe_e4_dose": "Number of epsilon-4 alleles (0, 1, or 2).", "cdr_global": "Harmonized valid global CDR; missing when invalid/conflicted.",
        "cdr_global_source": "Unmodified numeric source CDR value.", "cdr_state": "0=CDR 0, 1=CDR 0.5, 2=CDR >=1.",
        "cdr_state_label": "Readable CDR state label.", "cdr_sum_boxes": "Source CDR-SB/sum-of-boxes; prohibited as primary predictor.",
        "cdr_domain_values_source": "Pipe-delimited preserved source CDR domains; prohibited as primary predictors.",
        "mmse": "Valid MMSE matched to the CDR visit (not the separately frozen index MMSE).", "mmse_day_offset": "MMSE time minus CDR time in days.",
        "clinician_diagnosis": "Harmonized clinician diagnosis for agreement/sensitivity analyses.", "diagnosis_day_offset": "Diagnosis time minus CDR time in days.",
        "diagnosis_method": "Clinician/consensus method where supplied.", "study_phase": "ADNI phase, OASIS UDS form family, or AIBL release.",
        "form_family": "Source form family/release.", "gds_total": "Nearest valid GDS on/before visit within 180 days where supplied.",
        "scheduled_visit_flag": "Sensitivity-ready scheduled-visit indicator; unavailable in OASIS-3.",
        "duplicate_participant_time_flag": "More than one raw CDR row at participant-time.", "same_time_cdr_conflict_flag": "More than one valid CDR value at participant-time.",
        "duplicate_resolution_unresolved_flag": "Conflict could not be resolved without choosing an outcome value.",
        "near_duplicate_visit_flag": f"Another CDR visit is within {NEAR_DUPLICATE_DAYS} days.", "classifiable_cdr_flag": "Valid state and no unresolved same-time conflict.",
        "qc_flags": "Semicolon-delimited visit-level QC findings.", "cdr_field_validation_status": "Cohort-specific source-field validation status.",
    }
    all_columns = sorted(set().union(*(set(m.columns) for m in manifests.values())))
    rows = []
    for col in all_columns:
        examples = []
        cohorts = []
        dtypes = []
        for cohort, frame in manifests.items():
            if col in frame:
                cohorts.append(cohort); dtypes.append(str(frame[col].dtype))
                examples.extend(frame[col].dropna().astype(str).unique()[:2])
        source = provenance[provenance["canonical_field"].eq(col)]
        rows.append(
            {
                "field": col,
                "description": descriptions.get(col, "Source/provenance or QC support field; see variable_provenance.csv."),
                "dtype": " | ".join(sorted(set(dtypes))),
                "cohorts": ", ".join(cohorts),
                "example_values": " | ".join(examples[:4]),
                "primary_role": "outcome" if col in {"cdr_global", "cdr_state", "cdr_state_label"} else ("do not use as predictor" if col in {"cdr_sum_boxes", "cdr_domain_values_source"} else "predictor/metadata/QC"),
            }
        )
    path = REPORT_ROOT / "canonical_manifest_data_dictionary.md"
    text = "# Canonical Visit Manifest Data Dictionary\n\n"
    text += "Generated by `scripts/phase1_data_foundation.py`. The canonical state mapping is 0→State 0, 0.5→State 1, and ≥1→State 2. State 1 means CDR 0.5/MCI-like staging, not necessarily clinician-diagnosed MCI.\n\n"
    text += markdown_table(pd.DataFrame(rows)) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _time_to_event_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, event), data in outcomes.groupby(["cohort", "first_event_type"]):
        event_times = data["first_event_time_months"].dropna()
        rows.append(
            {
                "cohort": cohort,
                "first_event_type": event,
                "participants": len(data),
                "median_event_time_months": event_times.median(),
                "event_time_q1": event_times.quantile(0.25),
                "event_time_q3": event_times.quantile(0.75),
                "median_available_follow_up_months": data["censoring_time_months"].median(),
                "follow_up_q1": data["censoring_time_months"].quantile(0.25),
                "follow_up_q3": data["censoring_time_months"].quantile(0.75),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    inventory: pd.DataFrame,
    field_validation: pd.DataFrame,
    manifests: dict[str, pd.DataFrame],
    invalid_cdr: pd.DataFrame,
    conflicts: pd.DataFrame,
    flow: pd.DataFrame,
    transitions: pd.DataFrame,
    transition_stats: pd.DataFrame,
    participant_transitions: pd.DataFrame,
    outcomes: pd.DataFrame,
    horizon: pd.DataFrame,
    depth: pd.DataFrame,
    missingness: pd.DataFrame,
    missingness_structure: pd.DataFrame,
    circularity: pd.DataFrame,
    phase_counts: pd.DataFrame,
    form_counts: pd.DataFrame,
    diagnosis_transition_compare: pd.DataFrame,
    previous_compare: pd.DataFrame,
    qc: pd.DataFrame,
) -> Path:
    manifest_summary = pd.DataFrame(
        [
            {
                "cohort": cohort,
                "canonical time points": len(frame),
                "participants": frame["participant_id"].nunique(),
                "classifiable visits": int(frame["classifiable_cdr_flag"].sum()),
                "CDR 0 visits": int(frame["cdr_state"].eq(0).sum()),
                "CDR 0.5 visits": int(frame["cdr_state"].eq(1).sum()),
                "CDR >=1 visits": int(frame["cdr_state"].eq(2).sum()),
                "unresolved same-time conflicts": int(frame["duplicate_resolution_unresolved_flag"].sum()),
                "near-duplicate visits": int(frame["near_duplicate_visit_flag"].sum()),
            }
            for cohort, frame in manifests.items()
        ]
    )
    repeated = (
        participant_transitions.groupby("cohort", as_index=False)
        .agg(
            participants=("participant_id", "nunique"),
            participants_with_state_change=("state_change_event_count", lambda x: int(x.gt(0).sum())),
            repeated_improvement_participants=("repeated_improvement_flag", "sum"),
            repeated_worsening_participants=("repeated_worsening_flag", "sum"),
            repeated_cdr_0_5_to_0_participants=("repeated_cdr_0_5_to_0_flag", "sum"),
            repeated_cdr_0_5_to_ge1_participants=("repeated_cdr_0_5_to_ge1_flag", "sum"),
        )
    )
    tte = _time_to_event_summary(outcomes)
    overall_missing = missingness[
        missingness["stratum_type"].eq("cohort")
    ][["stratum_value", "variable", "participants", "missing_n", "missing_percent"]].rename(columns={"stratum_value": "cohort"})
    structure_flags = missingness_structure[missingness_structure["structural_missingness_signal_flag"]]
    invalid_summary = (
        invalid_cdr.groupby(["cohort", "invalid_reason"], as_index=False).size().rename(columns={"size": "source_rows"})
        if not invalid_cdr.empty else pd.DataFrame()
    )
    conflict_summary = (
        conflicts.groupby(["cohort", "conflict_type"], as_index=False).agg(
            duplicate_time_groups=("participant_id", "size"), unresolved_groups=("unresolved_flag", "sum")
        ) if not conflicts.empty else pd.DataFrame()
    )
    index_counts = flow[flow["flow_step"].eq("Final first eligible index participants")][["cohort", "participants_remaining"]]
    primary_event_counts = (
        outcomes.groupby(["cohort", "first_event_type"], as_index=False).size().rename(columns={"size": "participants"})
    )
    qc_summary = qc.groupby(["severity", "status"], as_index=False).size().rename(columns={"size": "checks"})

    restrictions = [
        "OASIS-3 `CDRTOT` is used provisionally as global CDR because its observed values are exclusively global stages and `CDRSUM` is the delivered sum-of-boxes field. The supplied workbook nevertheless names `CDRGLOB` as global and labels `CDRTOT` as an older UDS2 total. Obtain written OASIS confirmation or a corrected export before the external pipeline is locked.",
        "ADNI and AIBL source codebooks are absent from the supplied project. Their familiar code mappings are retained as provisional, with source values preserved and variable-level validation flags.",
        "AIBL age is a year-only approximation and education is unavailable. AIBL remains descriptive and is not a confirmatory validation cohort.",
        "OASIS-3 has relative days but no clinical calendar dates or reliable scheduled-visit indicator in the supplied tables.",
        "Death/dropout reasons remain incomplete; Phase 1 censors at the last classifiable CDR visit and does not claim full competing-risk control.",
        "Same-day MMSE is common and can be part of the assessment workflow. Phase 2 must preserve lagged-MMSE, no-same-day-MMSE, and no-MMSE sensitivity datasets.",
    ]

    text: list[str] = [
        "# Phase 1: Data Foundation and Endpoint Validation\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n",
        "## Decision\n",
        "# GO TO PHASE 2 WITH RESTRICTIONS\n",
        "Canonical CDR manifests, index cohorts, event clocks, fixed-horizon labels, longitudinal-depth audits, missingness audits, and reproducible QC are now available. No predictive model was fitted. The data support Phase 2 preparation, but the OASIS-3 global-CDR field-label discrepancy must be resolved before locked confirmatory external validation, and AIBL must remain descriptive.\n",
        "### Frozen Phase 2 hand-off\n",
        "- Primary state: first eligible CDR 0.5 (MCI-like staging), no earlier CDR ≥1.\n",
        "- Destination events: first later CDR 0 and first later CDR ≥1; persistence is horizon-specific only.\n",
        "- Recommended primary horizons: **12 and 36 months**. Use **24 months** as a sensitivity/comparability horizon and **60 months** as a prespecified long-term secondary horizon because attrition and observation selection increase with time.\n",
        "- Recommended clinical core for Phase 2: age, sex, education, APOE ε4 dose, MMSE, elapsed time, and pre-landmark MMSE change/slope only where supported. Preserve a reduced model without education/APOE and circularity sensitivities without same-day MMSE or without MMSE.\n",
        "- CDR-SB and CDR domains remain outcome-adjacent and are prohibited from the primary predictor set.\n",
        "## 1. Work completed\n",
        "The reusable script `scripts/phase1_data_foundation.py` reconstructs all Phase 1 artifacts from raw files. `scripts/phase1_qc.py` independently rechecks the saved manifests and endpoint tables. Raw data are never modified. Outputs are written as CSV and, for canonical/index datasets, Parquet.\n",
        "## 2. Source-file inventory\n",
        markdown_table(inventory[["cohort", "file_name", "table_purpose", "participant_identifier", "visit_or_time_field", "row_count", "column_count", "duplicate_participant_visit_rows", "exact_duplicate_rows", "file_version_or_date"]]),
        "\nThe full inventory includes source-relative paths and SHA-256 hashes in `results/csvs/phase1/source_file_inventory.csv`.\n",
        "## 3. Exact field and codebook decisions\n",
        markdown_table(field_validation),
        "\nSentinels are handled variable-by-variable. Values are not globally replaced. The action and evidence for every handled core code are saved in `sentinel_code_map.csv`.\n",
        "## 4. Canonical-manifest logic\n",
        "A CDR assessment anchors each clinical time point. Dates/relative days, not visit labels, determine chronology. Demographics and APOE are treated as participant-level fields. MMSE and diagnosis first match the same visit and time, then a nearby time only when permitted. The index MMSE is independently frozen to the nearest valid observation on or before index within 180 days.\n",
        markdown_table(manifest_summary),
        "\n### Invalid/missing source CDR records\n", markdown_table(invalid_summary),
        "\n### Duplicate/conflict groups\n", markdown_table(conflict_summary),
        "\nDuplicate resolution is deterministic: valid CDR, no QC error, greatest clinical completeness, latest update, then original row. No valid same-time CDR conflict remains in this extract. Two ADNI same-time diagnosis groups contain conflicting mapped diagnoses; they are flagged and retained for diagnosis-sensitivity review but do not alter the CDR endpoint. If a valid CDR conflict occurs in a future extract, the harmonized state will be left unresolved and excluded from primary state construction.\n",
        "## 5. CDR state construction\n",
        "CDR 0 maps to State 0; CDR 0.5 maps to State 1; CDR 1, 2, or 3 maps to State 2. State 1 is described as **CDR 0.5 or MCI-like cognitive staging**, never automatically as clinician-diagnosed MCI. Flags support ≥30-day, ≥90-day, scheduled-visit, near-duplicate, and unresolved-conflict sensitivities.\n",
        "## 6. Primary index-cohort flow\n", markdown_table(flow),
        "\nAIBL rows are generated for descriptive feasibility, but their `primary_analysis_role` records the unverified age derivation.\n",
        "## 7. Complete transition audit\n", markdown_table(transition_stats),
        "\n### Repeated episodes\n", markdown_table(repeated),
        "\nAll state changes are retained, including direct CDR 0→≥1 and ≥1→0 transitions. Short intervals and the immediately later state are available in the row-level event table.\n",
        "## 8. Prediction-horizon feasibility\n", markdown_table(horizon),
        "\n`participants_observed_through_horizon` requires a classifiable visit at or after the horizon. `participants_with_known_outcome_by_horizon` also counts a documented transition before the horizon. Persistence means remaining at CDR 0.5 through that specific horizon; it is not a permanent phenotype.\n",
        "### Time-to-first-event and censoring\n", markdown_table(tte),
        "\n### First-event totals\n", markdown_table(primary_event_counts),
        "## 9. Dynamic longitudinal depth\n", markdown_table(depth),
        "\nCurrent MMSE is feasible with one measurement. Change and a two-point slope require two; a more stable participant-specific slope is flagged only with at least three. Phase 2 should begin with current MMSE plus change, use a pre-landmark two-point slope cautiously, and reserve mixed-model trajectory estimates for a training-only secondary representation.\n",
        "## 10. Clinical-core missingness\n", markdown_table(overall_missing),
        "\n### Descriptive structural-missingness signals (≥10 percentage-point spread; groups n≥10)\n", markdown_table(structure_flags, max_rows=80),
        "\nThese are descriptive signals, not proof of MCAR/MAR/MNAR. No missingness flag is designated as a predictor.\n",
        "## 11. Outcome/predictor circularity\n", markdown_table(circularity),
        "\nMMSE timing flags, diagnosis agreement tables, CDR-SB/domain provenance, and no-MMSE/no-same-day-MMSE sensitivity flags are frozen. CDR-SB and individual domains are not allowed in the primary clinical core.\n",
        "## 12. Phase and form-family audits\n",
        "### ADNI destination phase\n", markdown_table(phase_counts[phase_counts["cohort"].eq("adni")], max_rows=100),
        "\n### OASIS-3 form family and diagnosis method\n", markdown_table(form_counts[form_counts["cohort"].eq("oasis3")], max_rows=120),
        "\nOASIS diagnosis-method metadata remain dominated by single-clinician assessments, consistent with the earlier feasibility audit; formal consensus validation is not supported by the supplied extract.\n",
        "## 13. CDR versus diagnosis definitions\n", markdown_table(diagnosis_transition_compare),
        "\nCDR-defined improvement is not equivalent to diagnosis-defined MCI reversion. The CDR endpoint is used because it is more internally stable across ADNI/OASIS-3; diagnosis mappings remain sensitivity analyses.\n",
        "## 14. Reconciliation with previous reports\n", markdown_table(previous_compare),
        "\nDifferences, when present, reflect the frozen Phase 1 rules: actual-time ordering, explicit same-time conflict handling, exact strict-diagnosis mapping, and change-only transition counting. In OASIS-3, the documented strict reconstruction produces one fewer MCI→AD and one fewer reciprocal AD→MCI episode than the earlier matrix; all other strict cells match. Re-review found no duplicate participant-time diagnosis rows. The prior one-event excess cannot be reproduced without changing the documented AD requirement (`DEMENTED=1` plus UDS2 `PROBAD/POSSAD` or UDS3 `alzdis`), so the new row-level reconstruction is retained and the earlier two cells are treated as an earlier counting discrepancy. Previous diagnosis counts are never forced to equal CDR-state counts.\n",
        "## 15. Automated QC\n", markdown_table(qc),
        "\n### QC summary\n", markdown_table(qc_summary),
        "\nCritical failures stop the pipeline. Warnings represent documented source limitations rather than silent exceptions.\n",
        "## 16. Unresolved blockers and restrictions\n",
    ]
    text.extend(f"- {item}\n" for item in restrictions)
    text.extend(
        [
            "## 17. Phase 2 recommendations\n",
            "1. Freeze separate baseline and dynamic landmark datasets from these index episodes; never merge future outcomes into predictor files.\n",
            "2. Preserve actual elapsed time and participant grouping. Implement 30-day/90-day and scheduled-visit sensitivities.\n",
            "3. Use training-fold-only imputation and transformation. Never use OASIS-3 outcome information or cohort-wide external statistics.\n",
            "4. Start with the low-dimensional clinical core. Audit education/APOE complete-case retention and prepare the reduced portability model.\n",
            "5. Use current MMSE and change as the first dynamic representations; add slopes only after training-fold construction confirms adequate pre-landmark depth.\n",
            "6. Keep ADNI as development, OASIS-3 untouched as external validation after the CDR field is confirmed, and AIBL descriptive.\n",
            "## 18. Output map\n",
            "- Full audit manifests: `ProcessedData/{adni,oasis3,aibl}/phase1_canonical_visit_manifest.{csv,parquet}`\n",
            "- Classifiable locked manifests: `ProcessedData/{adni,oasis3,aibl}/phase1_classifiable_cdr_manifest.{csv,parquet}`\n",
            "- Index cohorts: `ProcessedData/{cohort}/phase1_primary_index_cohort.{csv,parquet}`\n",
            "- Event/horizon/depth/missingness/provenance/QC tables: `results/csvs/phase1/`\n",
            "- Figures (600/1200 DPI PNG plus SVG): `results/plots/phase1/`\n",
            "- Manifest dictionary: `reports/phase1/canonical_manifest_data_dictionary.md`\n",
            "- This report: `reports/phase1/phase1_data_foundation_endpoint_validation_report.md`\n",
        ]
    )
    path = REPORT_ROOT / "phase1_data_foundation_endpoint_validation_report.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path


def write_run_metadata(inventory: pd.DataFrame, qc: pd.DataFrame) -> Path:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {"pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": plt.matplotlib.__version__},
        "config": PARAMS,
        "source_files": inventory[["cohort", "relative_path", "sha256"]].to_dict(orient="records"),
        "critical_qc_failures": int(((qc["severity"] == "critical") & (qc["status"] == "FAIL")).sum()),
        "warnings": qc[qc["status"].eq("WARN")][["cohort", "check", "details"]].to_dict(orient="records"),
        "predictive_models_fitted": False,
    }
    path = TABLE_ROOT / "phase1_run_metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_phase1() -> dict[str, Any]:
    ensure_output_dirs()
    inventory = source_inventory()
    provenance = variable_provenance()
    sentinels = sentinel_map()
    fields = source_field_validation()
    save_table(inventory, TABLE_ROOT / "source_file_inventory.csv")
    save_table(provenance, TABLE_ROOT / "variable_provenance.csv")
    save_table(sentinels, TABLE_ROOT / "sentinel_code_map.csv")
    save_table(fields, TABLE_ROOT / "source_field_validation.csv")

    builders = {"adni": build_adni_manifest, "oasis3": build_oasis_manifest, "aibl": build_aibl_manifest}
    manifests: dict[str, pd.DataFrame] = {}
    conflicts: dict[str, pd.DataFrame] = {}
    invalid: dict[str, pd.DataFrame] = {}
    measurements: dict[str, dict[str, pd.DataFrame]] = {}
    transitions: dict[str, pd.DataFrame] = {}
    transition_participants: dict[str, pd.DataFrame] = {}
    transition_pairs: dict[str, pd.DataFrame] = {}
    transition_stats: dict[str, pd.DataFrame] = {}
    indices: dict[str, pd.DataFrame] = {}
    flows: dict[str, pd.DataFrame] = {}
    outcomes: dict[str, pd.DataFrame] = {}
    horizons_person: dict[str, pd.DataFrame] = {}
    horizons_summary: dict[str, pd.DataFrame] = {}
    depths_person: dict[str, pd.DataFrame] = {}
    depths_summary: dict[str, pd.DataFrame] = {}
    intervals: dict[str, pd.DataFrame] = {}

    for cohort, builder in builders.items():
        print(f"Building {cohort.upper()} canonical manifest...", flush=True)
        manifest, conflict, invalid_cdr, cohort_measurements = builder()
        manifests[cohort] = manifest; conflicts[cohort] = conflict; invalid[cohort] = invalid_cdr; measurements[cohort] = cohort_measurements
        save_table(manifest, PROCESSED_ROOT / cohort / "phase1_canonical_visit_manifest.csv", parquet=True)
        save_table(
            manifest[manifest["classifiable_cdr_flag"].fillna(False)].reset_index(drop=True),
            PROCESSED_ROOT / cohort / "phase1_classifiable_cdr_manifest.csv",
            parquet=True,
        )
        save_table(conflict, TABLE_ROOT / cohort / "duplicate_and_conflict_records.csv")
        save_table(invalid_cdr, TABLE_ROOT / cohort / "invalid_and_missing_cdr_source_records.csv")

        event, participant_summary, pairs = build_transition_history(manifest)
        transitions[cohort] = event; transition_participants[cohort] = participant_summary; transition_pairs[cohort] = pairs
        stats = transition_summary(event); transition_stats[cohort] = stats
        save_table(event, TABLE_ROOT / cohort / "cdr_transition_events.csv")
        save_table(participant_summary, TABLE_ROOT / cohort / "participant_transition_summary.csv")
        save_table(pairs, TABLE_ROOT / cohort / "all_adjacent_cdr_pairs.csv")
        save_table(stats, TABLE_ROOT / cohort / "cdr_transition_summary.csv")

        index, flow = build_index_cohort(manifest, cohort_measurements)
        indices[cohort] = index; flows[cohort] = flow
        save_table(index, PROCESSED_ROOT / cohort / "phase1_primary_index_cohort.csv", parquet=True)
        save_table(flow, TABLE_ROOT / cohort / "index_cohort_flow.csv")
        episode, horizon_person, horizon_summary = build_primary_episode_outcomes(index, manifest)
        outcomes[cohort] = episode; horizons_person[cohort] = horizon_person; horizons_summary[cohort] = horizon_summary
        save_table(episode, TABLE_ROOT / cohort / "primary_episode_time_to_event.csv")
        save_table(horizon_person, TABLE_ROOT / cohort / "participant_horizon_outcomes.csv")
        save_table(horizon_summary, TABLE_ROOT / cohort / "horizon_feasibility_summary.csv")
        depth_person, depth_summary, interval = build_longitudinal_depth(index, episode, manifest, cohort_measurements["mmse"])
        depths_person[cohort] = depth_person; depths_summary[cohort] = depth_summary; intervals[cohort] = interval
        save_table(depth_person, TABLE_ROOT / cohort / "participant_longitudinal_depth.csv")
        save_table(depth_summary, TABLE_ROOT / cohort / "longitudinal_depth_summary.csv")
        save_table(interval, TABLE_ROOT / cohort / "inter_assessment_intervals.csv")

    all_conflicts = pd.concat(conflicts.values(), ignore_index=True, sort=False)
    all_invalid = pd.concat([x.assign(cohort=k) for k, x in invalid.items()], ignore_index=True, sort=False)
    all_transitions = pd.concat(transitions.values(), ignore_index=True, sort=False)
    all_transition_participants = pd.concat(transition_participants.values(), ignore_index=True, sort=False)
    all_transition_stats = pd.concat(transition_stats.values(), ignore_index=True, sort=False)
    all_flow = pd.concat(flows.values(), ignore_index=True, sort=False)
    all_outcomes = pd.concat(outcomes.values(), ignore_index=True, sort=False)
    all_horizon_person = pd.concat(horizons_person.values(), ignore_index=True, sort=False)
    all_horizon_summary = pd.concat(horizons_summary.values(), ignore_index=True, sort=False)
    all_depth_person = pd.concat(depths_person.values(), ignore_index=True, sort=False)
    all_depth_summary = pd.concat(depths_summary.values(), ignore_index=True, sort=False)
    all_intervals = pd.concat(intervals.values(), ignore_index=True, sort=False)
    save_table(all_conflicts, TABLE_ROOT / "duplicate_and_conflict_records_all_cohorts.csv")
    save_table(all_invalid, TABLE_ROOT / "invalid_and_missing_cdr_records_all_cohorts.csv")
    save_table(all_transitions, TABLE_ROOT / "cdr_transition_events_all_cohorts.csv")
    save_table(all_transition_participants, TABLE_ROOT / "participant_transition_summary_all_cohorts.csv")
    save_table(all_transition_stats, TABLE_ROOT / "cdr_transition_summary_all_cohorts.csv")
    save_table(all_flow, TABLE_ROOT / "index_cohort_flow_all_cohorts.csv")
    save_table(all_outcomes, TABLE_ROOT / "primary_episode_time_to_event_all_cohorts.csv")
    save_table(all_horizon_person, TABLE_ROOT / "participant_horizon_outcomes_all_cohorts.csv")
    save_table(all_horizon_summary, TABLE_ROOT / "horizon_feasibility_all_cohorts.csv")
    save_table(all_depth_person, TABLE_ROOT / "participant_longitudinal_depth_all_cohorts.csv")
    save_table(all_depth_summary, TABLE_ROOT / "longitudinal_depth_all_cohorts.csv")
    save_table(all_intervals, TABLE_ROOT / "inter_assessment_intervals_all_cohorts.csv")

    enriched_indices = {
        cohort: attach_episode_group(indices[cohort], outcomes[cohort], depths_person[cohort])
        for cohort in SOURCES
    }
    all_index_enriched = pd.concat(enriched_indices.values(), ignore_index=True, sort=False)
    missingness, missingness_structure = clinical_core_missingness(all_index_enriched)
    circularity, diagnosis_cross = circularity_audit(manifests, indices, outcomes)
    phase_counts, form_counts = phase_form_audits(all_transitions, pd.concat(indices.values(), ignore_index=True, sort=False))
    diagnosis_compare = diagnosis_cdr_transition_comparison(all_transitions)
    prior_compare = previous_count_comparison(diagnosis_compare)
    save_table(missingness, TABLE_ROOT / "clinical_core_missingness.csv")
    save_table(missingness_structure, TABLE_ROOT / "missingness_structure_flags.csv")
    save_table(circularity, TABLE_ROOT / "circularity_timing_audit.csv")
    save_table(diagnosis_cross, TABLE_ROOT / "clinician_diagnosis_vs_cdr_state.csv")
    save_table(phase_counts, TABLE_ROOT / "transition_counts_by_phase.csv")
    save_table(form_counts, TABLE_ROOT / "transition_and_index_counts_by_form_family.csv")
    save_table(diagnosis_compare, TABLE_ROOT / "diagnosis_vs_cdr_transition_comparison.csv")
    save_table(prior_compare, TABLE_ROOT / "comparison_with_previous_reports.csv")

    qc = run_qc(manifests, indices, outcomes, transitions, transition_stats, horizons_person)
    save_table(qc, TABLE_ROOT / "qc_results.csv")
    create_figures(all_flow, all_transitions, all_outcomes, all_horizon_summary, all_depth_summary, all_intervals, missingness, phase_counts, form_counts, diagnosis_cross)
    dictionary_path = write_data_dictionary(manifests, provenance)
    report_path = write_report(
        inventory, fields, manifests, all_invalid, all_conflicts, all_flow, all_transitions,
        all_transition_stats, all_transition_participants, all_outcomes, all_horizon_summary,
        all_depth_summary, missingness, missingness_structure, circularity, phase_counts,
        form_counts, diagnosis_compare, prior_compare, qc,
    )
    metadata_path = write_run_metadata(inventory, qc)
    print(f"Phase 1 report: {report_path}")
    return {
        "report": report_path,
        "dictionary": dictionary_path,
        "metadata": metadata_path,
        "manifests": manifests,
        "indices": indices,
        "outcomes": outcomes,
        "horizons": horizons_summary,
        "depth": depths_summary,
        "qc": qc,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    run_phase1()
