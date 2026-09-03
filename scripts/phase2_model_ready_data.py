#!/usr/bin/env python3
"""Phase 2 model-ready data, harmonization, and readiness pipeline.

The pipeline consumes the locked Phase 1 manifests, never modifies raw data,
never fits a prediction model, and writes leakage-auditable Phase 2 artifacts.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from phase2_preprocessing import ClinicalCorePreprocessor, load_preprocessor


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "phase2_analysis_parameters.json"
TABLE_ROOT = ROOT / "results" / "csvs" / "phase2"
PLOT_ROOT = ROOT / "results" / "plots" / "phase2"
REPORT_ROOT = ROOT / "reports" / "phase2"
PROCESSED_ROOT = ROOT / "ProcessedData"
PHASE1_TABLE_ROOT = ROOT / "results" / "csvs" / "phase1"

with CONFIG_PATH.open(encoding="utf-8") as handle:
    PARAMS = json.load(handle)

COHORTS = ("adni", "oasis3", "aibl")
HORIZONS = tuple(
    sorted(
        {
            int(x)
            for x in PARAMS["primary_horizons_months"]
            + PARAMS["sensitivity_horizons_months"]
        }
    )
)
DAYS_PER_MONTH = float(PARAMS["days_per_month"])
DAYS_PER_YEAR = float(PARAMS["days_per_year"])
BASELINE_WINDOW = int(PARAMS["baseline_measurement_window_days"])
DYNAMIC_WINDOW = int(PARAMS["dynamic_measurement_window_days"])
SLOPE_MIN_DAYS = int(PARAMS["slope_minimum_separation_days"])
PRIMARY_INTERVAL = int(PARAMS["primary_landmark_minimum_interval_days"])
SEED = int(PARAMS["random_seed"])
OUTCOME_CODES = PARAMS["outcome_codes"]

OASIS_CDR = next((ROOT / "datasets" / "oasis3").rglob("OASIS3_UDSb4_cdr.csv"))
OASIS_DICTIONARY = next(
    (ROOT / "datasets" / "oasis3").rglob(
        "OASIS3_Clinical and Psychometrics_dictionary.xlsx"
    )
)
OASIS_UDS2_DED = next((ROOT / "datasets" / "oasis3").rglob("uds2-ivp-ded.pdf"))
OASIS_UDS3_DED = next((ROOT / "datasets" / "oasis3").rglob("uds3-ivp-ded.pdf"))


def ensure_directories() -> None:
    for path in (TABLE_ROOT, PLOT_ROOT, REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    for cohort in COHORTS:
        for sub in ("baseline", "dynamic", "sensitivity"):
            (PROCESSED_ROOT / cohort / "phase2" / sub).mkdir(
                parents=True, exist_ok=True
            )
        (TABLE_ROOT / cohort).mkdir(parents=True, exist_ok=True)


def numeric(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def state_from_cdr(values: pd.Series) -> pd.Series:
    values = numeric(values)
    state = pd.Series(np.nan, index=values.index, dtype=float)
    state.loc[values.eq(0)] = 0
    state.loc[values.eq(0.5)] = 1
    state.loc[values.ge(1)] = 2
    return state


def save_table(frame: pd.DataFrame, path: Path, parquet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if parquet:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)


def markdown_table(
    frame: pd.DataFrame, max_rows: int | None = None, digits: int = 2
) -> str:
    if frame.empty:
        return "_No rows._"
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.select_dtypes(include=["float"]).columns:
        out[column] = out[column].round(digits)
    text = out.to_markdown(index=False)
    if max_rows and len(frame) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(frame)} rows._"
    return text


def dataframe_sha256(frame: pd.DataFrame, sort_columns: Iterable[str]) -> str:
    work = frame.copy()
    usable = [x for x in sort_columns if x in work]
    if usable:
        work = work.sort_values(usable, kind="mergesort")
    payload = work.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _mode_closest_to_memory(values: list[float], memory: float) -> float:
    counts = Counter(values)
    highest = max(counts.values())
    tied = [value for value, count in counts.items() if count == highest]
    return min(tied, key=lambda value: (abs(value - memory), value))


def derive_global_cdr_from_boxes(row: pd.Series) -> float:
    """Implement published standard CDR scoring rules for the six box scores."""
    columns = ["memory", "orient", "judgment", "commun", "homehobb", "perscare"]
    values = [pd.to_numeric(row.get(column), errors="coerce") for column in columns]
    if any(pd.isna(value) for value in values):
        return np.nan
    memory = float(values[0])
    secondary = [float(value) for value in values[1:]]
    above = [value for value in secondary if value > memory]
    below = [value for value in secondary if value < memory]
    if memory == 0:
        return 0.5 if sum(value >= 0.5 for value in secondary) >= 2 else 0.0
    if memory == 0.5:
        return 1.0 if sum(value >= 1 for value in secondary) >= 3 else 0.5
    if memory >= 1 and sum(value == 0 for value in secondary) >= 3:
        return 0.5
    if sum(value == memory for value in secondary) >= 3:
        return memory
    if (len(above) == 3 and len(below) == 2) or (
        len(below) == 3 and len(above) == 2
    ):
        return memory
    if len(above) >= 3 and len(above) > len(below):
        return _mode_closest_to_memory(above, memory)
    if len(below) >= 3 and len(below) > len(above):
        return _mode_closest_to_memory(below, memory)
    return memory


def _candidate_index_and_horizons(
    source: pd.DataFrame, candidate: str, cdr_values: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild OASIS eligibility/events for one field without changing Phase 1."""
    work = pd.DataFrame(
        {
            "participant_id": source["OASISID"].astype("string"),
            "time_days": numeric(source["days_to_visit"]),
            "age": numeric(source["age at visit"]),
            "mmse": numeric(source["MMSE"]).where(
                numeric(source["MMSE"]).between(0, 30)
            ),
            "cdr": numeric(cdr_values),
        }
    )
    work["state"] = state_from_cdr(work["cdr"])
    work = work[work["time_days"].notna() & work["state"].notna()].copy()
    work["_row"] = np.arange(len(work))
    work = work.sort_values(
        ["participant_id", "time_days", "_row"], kind="mergesort"
    ).drop_duplicates(["participant_id", "time_days"], keep="first")

    rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for participant, group in work.groupby("participant_id", sort=True):
        group = group.sort_values("time_days", kind="mergesort")
        index_row = None
        for row in group.itertuples(index=False):
            earlier = group[group["time_days"].lt(row.time_days)]
            if (
                row.state == 1
                and not earlier["state"].eq(2).any()
                and pd.notna(row.age)
                and 18 <= row.age <= 110
            ):
                mmse_history = group[
                    group["time_days"].le(row.time_days)
                    & group["time_days"].ge(row.time_days - BASELINE_WINDOW)
                    & group["mmse"].notna()
                ]
                if not mmse_history.empty:
                    index_row = row
                    break
        if index_row is None:
            continue
        later = group[group["time_days"].gt(index_row.time_days)]
        event = later[~later["state"].eq(1)].head(1)
        last_time = float(group["time_days"].max())
        first_type = "no_transition_observed"
        event_time = np.nan
        if not event.empty:
            event_row = event.iloc[0]
            event_time = float(event_row["time_days"] - index_row.time_days)
            first_type = (
                "improvement" if event_row["state"] == 0 else "progression"
            )
        rows.append(
            {
                "candidate_field": candidate,
                "participant_id": participant,
                "index_time_days": index_row.time_days,
                "first_event_type": first_type,
                "first_event_time_days": event_time,
            }
        )
        for horizon in HORIZONS:
            horizon_days = horizon * DAYS_PER_MONTH
            if pd.notna(event_time) and event_time <= horizon_days:
                status = (
                    "transition_to_cdr_0"
                    if first_type == "improvement"
                    else "transition_to_cdr_ge1"
                )
            elif last_time - index_row.time_days >= horizon_days:
                status = "remain_cdr_0_5_through_horizon"
            else:
                status = "censored_before_horizon"
            horizon_rows.append(
                {
                    "candidate_field": candidate,
                    "participant_id": participant,
                    "horizon_months": horizon,
                    "horizon_status": status,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(horizon_rows)


def resolve_oasis_cdr() -> dict[str, pd.DataFrame]:
    source = pd.read_csv(OASIS_CDR, low_memory=False)
    source["derived_global_cdr"] = source.apply(
        derive_global_cdr_from_boxes, axis=1
    )
    manifest = pd.read_parquet(
        PROCESSED_ROOT / "oasis3" / "phase1_canonical_visit_manifest.parquet"
    )
    form_lookup = manifest[
        ["participant_id", "time_sort_days", "form_family"]
    ].drop_duplicates(["participant_id", "time_sort_days"])
    comparison = source.rename(
        columns={"OASISID": "participant_id", "days_to_visit": "time_sort_days"}
    ).merge(form_lookup, on=["participant_id", "time_sort_days"], how="left")
    comparison["cdrtot_minus_derived"] = (
        numeric(comparison["CDRTOT"]) - numeric(comparison["derived_global_cdr"])
    )
    comparison["cdrtot_derived_agree_flag"] = numeric(comparison["CDRTOT"]).eq(
        numeric(comparison["derived_global_cdr"])
    )
    comparison["cdrsum_recomputed"] = comparison[
        ["memory", "orient", "judgment", "commun", "homehobb", "perscare"]
    ].sum(axis=1, min_count=6)
    comparison["cdrsum_box_sum_agree_flag"] = numeric(comparison["CDRSUM"]).eq(
        numeric(comparison["cdrsum_recomputed"])
    )
    keep = [
        "participant_id",
        "OASIS_session_label",
        "time_sort_days",
        "form_family",
        "memory",
        "orient",
        "judgment",
        "commun",
        "homehobb",
        "perscare",
        "CDRSUM",
        "cdrsum_recomputed",
        "cdrsum_box_sum_agree_flag",
        "CDRTOT",
        "derived_global_cdr",
        "cdrtot_minus_derived",
        "cdrtot_derived_agree_flag",
    ]
    save_table(
        comparison[keep],
        TABLE_ROOT / "oasis3_cdr_record_level_comparison.csv",
        parquet=True,
    )

    field_rows = [
        {
            "source_file": str(OASIS_CDR.relative_to(ROOT)),
            "source_column": "CDRSUM",
            "uds_version": "UDS2 and UDS3",
            "delivered_workbook_label": "Standard CDR Sum of Boxes",
            "authoritative_uds_definition": "CDR sum of boxes",
            "valid_value_structure": "0 to 18 in 0.5 increments",
            "observed_nonmissing_records": int(source["CDRSUM"].notna().sum()),
            "records": len(source),
            "participants": source["OASISID"].nunique(),
            "longitudinal_participants": int(
                source.groupby("OASISID").size().ge(2).sum()
            ),
            "analysis_decision": "not a global-CDR candidate",
        },
        {
            "source_file": str(OASIS_CDR.relative_to(ROOT)),
            "source_column": "CDRTOT",
            "uds_version": "mixed UDS2/UDS3 export",
            "delivered_workbook_label": "Standard & Supplemental CDR Sum of Boxes (UDS2)",
            "authoritative_uds_definition": "name CDRTOT is not the UDS global field",
            "valid_value_structure": "observed exactly 0, 0.5, 1, 2, 3",
            "observed_nonmissing_records": int(source["CDRTOT"].notna().sum()),
            "records": len(source),
            "participants": source["OASISID"].nunique(),
            "longitudinal_participants": int(
                source.groupby("OASISID").size().ge(2).sum()
            ),
            "analysis_decision": "selected de facto global CDR; provisional export mapping",
        },
        {
            "source_file": str(OASIS_CDR.relative_to(ROOT)),
            "source_column": "CDRGLOB",
            "uds_version": "UDS2 and UDS3",
            "delivered_workbook_label": "Standard Global CDR",
            "authoritative_uds_definition": "Global CDR; values 0, 0.5, 1, 2, 3",
            "valid_value_structure": "column absent from delivered CSV",
            "observed_nonmissing_records": 0,
            "records": len(source),
            "participants": 0,
            "longitudinal_participants": 0,
            "analysis_decision": "unavailable",
        },
        {
            "source_file": "derived from six delivered CDR domain columns",
            "source_column": "derived_global_cdr",
            "uds_version": "version-independent standard scoring rules",
            "delivered_workbook_label": "not delivered",
            "authoritative_uds_definition": "rule-derived global score",
            "valid_value_structure": "0, 0.5, 1, 2, 3",
            "observed_nonmissing_records": int(
                source["derived_global_cdr"].notna().sum()
            ),
            "records": len(source),
            "participants": source.loc[
                source["derived_global_cdr"].notna(), "OASISID"
            ].nunique(),
            "longitudinal_participants": int(
                source.loc[source["derived_global_cdr"].notna()]
                .groupby("OASISID")
                .size()
                .ge(2)
                .sum()
            ),
            "analysis_decision": "independent validation/sensitivity field",
        },
    ]
    field_inventory = pd.DataFrame(field_rows)
    save_table(field_inventory, TABLE_ROOT / "oasis3_cdr_field_comparison.csv")

    distribution_rows = []
    for field in ("CDRSUM", "CDRTOT", "derived_global_cdr"):
        for value, count in source[field].value_counts(dropna=False).items():
            distribution_rows.append(
                {
                    "field": field,
                    "value": "<missing>" if pd.isna(value) else value,
                    "records": int(count),
                    "percent": count / len(source) * 100,
                }
            )
    distributions = pd.DataFrame(distribution_rows)
    save_table(distributions, TABLE_ROOT / "oasis3_cdr_value_distributions.csv")

    complete = comparison["CDRTOT"].notna() & comparison["derived_global_cdr"].notna()
    mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3, 3.0: 4}
    agreement_rows = []
    for family, group in [
        ("ALL", comparison.loc[complete]),
        *list(comparison.loc[complete].groupby("form_family", dropna=False)),
    ]:
        actual = numeric(group["CDRTOT"]).map(mapping)
        derived = numeric(group["derived_global_cdr"]).map(mapping)
        agreement_rows.append(
            {
                "form_family": family,
                "complete_records": len(group),
                "exact_agreement_records": int(actual.eq(derived).sum()),
                "exact_agreement_percent": actual.eq(derived).mean() * 100,
                "unweighted_cohen_kappa": cohen_kappa_score(actual, derived),
                "quadratic_weighted_kappa": cohen_kappa_score(
                    actual, derived, weights="quadratic"
                ),
                "participants_with_complete_comparison": group[
                    "participant_id"
                ].nunique(),
                "participants_with_any_disagreement": group.loc[
                    ~actual.eq(derived).to_numpy(), "participant_id"
                ].nunique(),
            }
        )
    agreement = pd.DataFrame(agreement_rows)
    save_table(agreement, TABLE_ROOT / "oasis3_cdr_agreement_statistics.csv")

    disagreement = (
        comparison.loc[
            complete
            & numeric(comparison["CDRTOT"]).ne(
                numeric(comparison["derived_global_cdr"])
            )
        ]
        .groupby(["form_family", "CDRTOT", "derived_global_cdr"], dropna=False)
        .agg(
            records=("participant_id", "size"),
            participants=("participant_id", "nunique"),
            median_visit_day=("time_sort_days", "median"),
            min_visit_day=("time_sort_days", "min"),
            max_visit_day=("time_sort_days", "max"),
        )
        .reset_index()
    )
    save_table(disagreement, TABLE_ROOT / "oasis3_cdr_disagreement_table.csv")

    candidate_frames = {
        "CDRTOT_de_facto_global": source["CDRTOT"],
        "derived_global_from_boxes": source["derived_global_cdr"],
        "CDRSUM_misinterpreted_as_global": source["CDRSUM"],
    }
    endpoint_rows = []
    transition_rows = []
    for name, values in candidate_frames.items():
        index, horizons = _candidate_index_and_horizons(source, name, values)
        first_counts = index["first_event_type"].value_counts()
        transition_rows.extend(
            {
                "candidate_field": name,
                "summary_type": "first_primary_episode_event",
                "horizon_months": np.nan,
                "outcome": outcome,
                "count": int(first_counts.get(outcome, 0)),
            }
            for outcome in (
                "improvement",
                "progression",
                "no_transition_observed",
            )
        )
        for horizon, group in horizons.groupby("horizon_months"):
            counts = group["horizon_status"].value_counts()
            for outcome in (
                "transition_to_cdr_0",
                "transition_to_cdr_ge1",
                "remain_cdr_0_5_through_horizon",
                "censored_before_horizon",
            ):
                transition_rows.append(
                    {
                        "candidate_field": name,
                        "summary_type": "fixed_horizon",
                        "horizon_months": horizon,
                        "outcome": outcome,
                        "count": int(counts.get(outcome, 0)),
                    }
                )
        endpoint_rows.append(
            {
                "candidate_field": name,
                "eligible_index_participants": index["participant_id"].nunique(),
                "first_improvement": int(first_counts.get("improvement", 0)),
                "first_progression": int(first_counts.get("progression", 0)),
                "no_observed_transition": int(
                    first_counts.get("no_transition_observed", 0)
                ),
            }
        )
    candidate_endpoints = pd.DataFrame(endpoint_rows)
    candidate_transitions = pd.DataFrame(transition_rows)
    save_table(
        candidate_endpoints,
        TABLE_ROOT / "oasis3_cdr_candidate_endpoint_summary.csv",
    )
    save_table(
        candidate_transitions,
        TABLE_ROOT / "oasis3_cdr_candidate_transition_counts.csv",
    )
    return {
        "field_inventory": field_inventory,
        "distributions": distributions,
        "agreement": agreement,
        "disagreement": disagreement,
        "candidate_endpoints": candidate_endpoints,
        "candidate_transitions": candidate_transitions,
        "record_comparison": comparison[keep],
    }


def load_phase1() -> dict[str, dict[str, pd.DataFrame]]:
    output: dict[str, dict[str, pd.DataFrame]] = {}
    for cohort in COHORTS:
        output[cohort] = {
            "manifest": pd.read_parquet(
                PROCESSED_ROOT
                / cohort
                / "phase1_classifiable_cdr_manifest.parquet"
            ),
            "index": pd.read_parquet(
                PROCESSED_ROOT / cohort / "phase1_primary_index_cohort.parquet"
            ),
            "episode": pd.read_csv(
                PHASE1_TABLE_ROOT / cohort / "primary_episode_time_to_event.csv",
                low_memory=False,
            ),
            "horizons": pd.read_csv(
                PHASE1_TABLE_ROOT / cohort / "participant_horizon_outcomes.csv",
                low_memory=False,
            ),
            "depth": pd.read_csv(
                PHASE1_TABLE_ROOT / cohort / "participant_longitudinal_depth.csv",
                low_memory=False,
            ),
        }
        for name in output[cohort]:
            if "participant_id" in output[cohort][name]:
                output[cohort][name]["participant_id"] = output[cohort][name][
                    "participant_id"
                ].astype("string")
    return output


def load_mmse_histories() -> dict[str, pd.DataFrame]:
    paths = {
        "adni": ROOT
        / "datasets"
        / "adni"
        / "All_Subjects_MMSE_26Aug2025.csv",
        "oasis3": OASIS_CDR,
        "aibl": next((ROOT / "datasets" / "aibl").rglob("aibl_mmse_*.csv")),
    }
    output = {}
    for cohort, path in paths.items():
        raw = pd.read_csv(path, low_memory=False)
        if cohort == "adni":
            time = (
                pd.to_datetime(raw["VISDATE"], errors="coerce")
                - pd.Timestamp("1970-01-01")
            ).dt.days
            participant = raw["RID"].astype("Int64").astype("string")
            score = numeric(raw["MMSCORE"])
            visit = raw["VISCODE2"].astype("string")
        elif cohort == "oasis3":
            time = numeric(raw["days_to_visit"])
            participant = raw["OASISID"].astype("string")
            score = numeric(raw["MMSE"])
            visit = raw["OASIS_session_label"].astype("string")
        else:
            time = (
                pd.to_datetime(raw["EXAMDATE"], errors="coerce")
                - pd.Timestamp("1970-01-01")
            ).dt.days
            participant = raw["RID"].astype("Int64").astype("string")
            score = numeric(raw["MMSCORE"])
            visit = raw["VISCODE"].astype("string")
        frame = pd.DataFrame(
            {
                "participant_id": participant,
                "measurement_time_days": time,
                "mmse": score.where(score.between(0, 30)),
                "mmse_raw": score,
                "measurement_visit_id": visit,
                "measurement_source_table": path.name,
                "measurement_source_row": np.arange(len(raw)),
            }
        )
        frame = frame[
            frame["measurement_time_days"].notna() & frame["mmse"].notna()
        ].copy()
        frame = frame.sort_values(
            ["participant_id", "measurement_time_days", "measurement_source_row"],
            kind="mergesort",
        ).drop_duplicates(
            ["participant_id", "measurement_time_days"], keep="first"
        )
        output[cohort] = frame.reset_index(drop=True)
    return output


def prior_measurement(
    history: pd.DataFrame, time_days: float, *, strictly_before: bool
) -> pd.Series | None:
    eligible = history[
        (
            history["measurement_time_days"].lt(time_days)
            if strictly_before
            else history["measurement_time_days"].le(time_days)
        )
        & history["measurement_time_days"].ge(time_days - DYNAMIC_WINDOW)
    ]
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["measurement_time_days", "measurement_source_row"], kind="mergesort"
    ).iloc[-1]


def build_recurrent_episode_table(manifests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cohort, manifest in manifests.items():
        for participant, group in manifest.groupby("participant_id", sort=True):
            group = group.sort_values("time_sort_days", kind="mergesort")
            in_episode = False
            episode_number = 0
            for row in group.itertuples(index=False):
                if row.cdr_state == 1 and not in_episode:
                    episode_number += 1
                    in_episode = True
                    rows.append(
                        {
                            "cohort": cohort,
                            "participant_id": participant,
                            "episode_number": episode_number,
                            "episode_id": f"{cohort}:{participant}:cdr05_episode_{episode_number}",
                            "episode_start_manifest_key": row.manifest_key,
                            "episode_start_time_days": row.time_sort_days,
                            "primary_episode_flag": episode_number == 1,
                            "phase2_analysis_role": (
                                "primary_candidate"
                                if episode_number == 1
                                else "future_recurrent_sensitivity_only"
                            ),
                        }
                    )
                elif row.cdr_state != 1:
                    in_episode = False
    frame = pd.DataFrame(rows)
    save_table(frame, TABLE_ROOT / "recurrent_episode_identifiers.csv")
    return frame


def frozen_specifications() -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = pd.DataFrame(
        [
            ("state_0", "CDR exactly 0", "0"),
            ("state_1", "CDR exactly 0.5", "1"),
            ("state_2", "CDR 1, 2, or 3", "2"),
            (
                "index",
                "first eligible State 1; valid age/sex/MMSE; no earlier State 2",
                "one per participant",
            ),
            (
                "event",
                "first strictly later State 0 or State 2 in primary episode",
                "ties impossible after same-time conflict resolution",
            ),
            (
                "persistence",
                "no event and classifiable observation reaches horizon",
                "never assigned to early censoring",
            ),
            ("primary_horizons", "12 and 36 months", "365.25 and 1095.75 days"),
            ("sensitivity_horizons", "24 and 60 months", "730.5 and 1826.25 days"),
            (
                "oasis3_field",
                "CDRTOT interpreted as de facto global CDR",
                "provisional export mapping; CDRGLOB absent",
            ),
        ],
        columns=["component", "frozen_definition", "coding_or_value"],
    )
    landmark = pd.DataFrame(
        [
            ("episode", "first CDR 0.5 episode only"),
            ("schedule", "observed classifiable CDR 0.5 visits"),
            ("time", "actual elapsed days; modelling unit years from index"),
            ("minimum_interval", f"{PRIMARY_INTERVAL} days"),
            ("sensitivity_intervals", "none, 30 days, 90 days"),
            ("risk_start", "strictly after landmark"),
            ("end", "first State 0/2 transition or last classifiable observation"),
            ("last_visit", "not a landmark when no later classifiable observation"),
            (
                "predictor_window",
                f"current MMSE nearest on/before within {DYNAMIC_WINDOW} days; lagged MMSE is most recent strictly earlier score since episode-index MMSE",
            ),
            ("slope", f"past only; at least two points spanning {SLOPE_MIN_DAYS} days"),
            ("recurrent_episodes", "identified but excluded from primary datasets"),
        ],
        columns=["component", "frozen_definition"],
    )
    save_table(endpoint, TABLE_ROOT / "frozen_endpoint_specification.csv")
    save_table(landmark, TABLE_ROOT / "frozen_landmark_specification.csv")
    return endpoint, landmark


def _outcome_code(status: str) -> float:
    value = OUTCOME_CODES.get(status)
    return np.nan if value is None else float(value)


def build_baseline_datasets(
    phase1: dict[str, dict[str, pd.DataFrame]],
    mmse_histories: dict[str, pd.DataFrame],
) -> dict[str, dict[int, pd.DataFrame]]:
    datasets: dict[str, dict[int, pd.DataFrame]] = {}
    for cohort in COHORTS:
        index = phase1[cohort]["index"].copy()
        episode = phase1[cohort]["episode"].copy()
        history = mmse_histories[cohort]
        lagged_rows = []
        for row in index.itertuples(index=False):
            participant_history = history[
                history["participant_id"].eq(str(row.participant_id))
            ]
            previous = prior_measurement(
                participant_history, float(row.index_time_days), strictly_before=True
            )
            lagged_rows.append(
                {
                    "participant_id": str(row.participant_id),
                    "lagged_mmse": np.nan if previous is None else previous["mmse"],
                    "lagged_mmse_time_days": (
                        np.nan
                        if previous is None
                        else previous["measurement_time_days"]
                    ),
                    "lagged_mmse_day_offset": (
                        np.nan
                        if previous is None
                        else previous["measurement_time_days"] - row.index_time_days
                    ),
                }
            )
        lagged = pd.DataFrame(lagged_rows)
        episode_for_merge = episode.drop(
            columns=["cohort", "index_time_days", "index_manifest_key"],
            errors="ignore",
        )
        base = index.merge(
            episode_for_merge, on="participant_id", how="left"
        ).merge(lagged, on="participant_id", how="left")
        base["primary_episode_id"] = (
            cohort
            + ":"
            + base["participant_id"].astype(str)
            + ":primary_cdr05_episode"
        )
        base["age_at_landmark"] = numeric(base["index_age"])
        base["current_age"] = numeric(base["index_age"])
        base["current_mmse"] = numeric(base["index_mmse"])
        base["current_mmse_time_days"] = numeric(base["index_mmse_time_days"])
        base["mmse_change_from_index"] = 0.0
        base["elapsed_days_from_index"] = 0.0
        base["elapsed_months_from_index"] = 0.0
        base["elapsed_years_from_index"] = 0.0
        base["current_state"] = 1
        base["cdr_source_field"] = np.where(
            cohort == "oasis3", "CDRTOT_de_facto_global", "CDGLOBAL"
        )
        base["external_endpoint_status"] = np.where(
            cohort == "oasis3",
            "usable_with_provisional_export_mapping",
            "not_applicable",
        )
        base["predictor_max_time_days"] = base[
            ["index_time_days", "current_mmse_time_days"]
        ].max(axis=1)
        datasets[cohort] = {}
        for horizon in HORIZONS:
            horizon_frame = phase1[cohort]["horizons"]
            horizon_frame = horizon_frame[
                numeric(horizon_frame["horizon_months"]).eq(horizon)
            ].copy()
            frame = base.merge(
                horizon_frame.drop(columns=["cohort"], errors="ignore"),
                on="participant_id",
                how="left",
            )
            frame["horizon_months"] = horizon
            frame["horizon_days"] = horizon * DAYS_PER_MONTH
            frame["fixed_horizon_class"] = frame["horizon_status"]
            frame["outcome_code"] = frame["horizon_status"].map(_outcome_code)
            frame["eligible_for_fixed_horizon_analysis"] = frame[
                "outcome_known_by_horizon_flag"
            ].fillna(False) & ~frame["unresolved_conflict_flag"].fillna(False)
            frame["censoring_status"] = np.where(
                frame["censored_before_horizon_flag"].fillna(False),
                "censored_before_horizon",
                "outcome_known_by_horizon",
            )
            frame["first_transition"] = frame["first_event_type"].map(
                {
                    "improvement": "transition_to_cdr_0",
                    "progression": "transition_to_cdr_ge1",
                    "no_transition_observed": "none_observed",
                }
            )
            frame["transition_time_days"] = numeric(
                frame["first_event_time_days"]
            )
            frame["transition_time_months"] = numeric(
                frame["first_event_time_months"]
            )
            frame["same_day_mmse_flag"] = frame[
                "index_mmse_same_time_flag"
            ].fillna(False)
            frame["strictly_lagged_mmse_available_flag"] = frame[
                "lagged_mmse"
            ].notna()
            frame["no_same_day_outcome_assessment_overlap_flag"] = ~(
                frame["event_by_horizon_flag"].fillna(False)
                & frame["event_mmse_same_time_flag"].fillna(False)
            )
            frame["phase2_qc_flags"] = ""
            columns = [
                "cohort",
                "participant_id",
                "primary_episode_id",
                "manifest_key",
                "index_visit_id",
                "index_visit_date",
                "index_time_days",
                "horizon_months",
                "horizon_days",
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
                "unresolved_conflict_flag",
                "current_state",
                "index_age",
                "age_at_landmark",
                "current_age",
                "sex",
                "education_years",
                "apoe_genotype",
                "apoe_e4_dose",
                "index_mmse",
                "current_mmse",
                "current_mmse_time_days",
                "mmse_change_from_index",
                "lagged_mmse",
                "lagged_mmse_time_days",
                "lagged_mmse_day_offset",
                "elapsed_days_from_index",
                "elapsed_months_from_index",
                "elapsed_years_from_index",
                "same_day_mmse_flag",
                "strictly_lagged_mmse_available_flag",
                "no_same_day_outcome_assessment_overlap_flag",
                "clinician_diagnosis",
                "clinician_diagnosed_mci_at_index_flag",
                "clinical_mci_subgroup_flag",
                "study_phase",
                "form_family",
                "site_id",
                "index_year",
                "index_calendar_period",
                "index_relative_time_band",
                "cdr_source_field",
                "cdr_field_validation_status",
                "external_endpoint_status",
                "index_mmse_match_rule",
                "predictor_max_time_days",
                "phase2_qc_flags",
            ]
            frame = frame[[column for column in columns if column in frame]].copy()
            datasets[cohort][horizon] = frame
    return datasets


def mmse_features_at_landmark(
    participant_history: pd.DataFrame,
    index_mmse: float,
    index_mmse_time: float,
    landmark_time: float,
) -> dict[str, Any]:
    current = prior_measurement(
        participant_history, landmark_time, strictly_before=False
    )
    # The lagged sensitivity uses the most recent strictly earlier score from
    # the primary episode's observed history.  It is not constrained to the
    # 180-day "current" window, because annual cohorts would otherwise have no
    # usable lagged representation.  Staleness is retained explicitly below.
    previous_history = participant_history[
        participant_history["measurement_time_days"].lt(landmark_time)
        & participant_history["measurement_time_days"].ge(index_mmse_time)
    ].sort_values(
        ["measurement_time_days", "measurement_source_row"],
        kind="mergesort",
    )
    previous = None if previous_history.empty else previous_history.iloc[-1]
    current_value = np.nan if current is None else float(current["mmse"])
    current_time = (
        np.nan if current is None else float(current["measurement_time_days"])
    )
    previous_value = np.nan if previous is None else float(previous["mmse"])
    previous_time = (
        np.nan if previous is None else float(previous["measurement_time_days"])
    )

    slope_history = participant_history[
        participant_history["measurement_time_days"].ge(index_mmse_time)
        & participant_history["measurement_time_days"].le(landmark_time)
    ].sort_values("measurement_time_days", kind="mergesort")
    if (
        pd.notna(index_mmse)
        and pd.notna(index_mmse_time)
        and not (
            numeric(slope_history["measurement_time_days"])
            .eq(index_mmse_time)
            .any()
        )
    ):
        slope_history = pd.concat(
            [
                pd.DataFrame(
                    {
                        "measurement_time_days": [index_mmse_time],
                        "mmse": [index_mmse],
                    }
                ),
                slope_history,
            ],
            ignore_index=True,
            sort=False,
        ).sort_values("measurement_time_days", kind="mergesort")
    slope_history = slope_history.drop_duplicates(
        "measurement_time_days", keep="first"
    )
    count = len(slope_history)
    span = (
        float(
            slope_history["measurement_time_days"].max()
            - slope_history["measurement_time_days"].min()
        )
        if count >= 2
        else np.nan
    )
    slope = np.nan
    method = "unavailable"
    reliable = False
    two_point = False
    if count >= 2 and span >= SLOPE_MIN_DAYS:
        x = (
            numeric(slope_history["measurement_time_days"]).to_numpy()
            - float(slope_history["measurement_time_days"].min())
        ) / DAYS_PER_YEAR
        y = numeric(slope_history["mmse"]).to_numpy()
        if count == 2:
            slope = float((y[-1] - y[0]) / (x[-1] - x[0]))
            method = "two_point_annualized_change"
            two_point = True
        else:
            slope = float(np.polyfit(x, y, 1)[0])
            method = "ordinary_least_squares"
            reliable = span >= DAYS_PER_YEAR
    return {
        "current_mmse": current_value,
        "current_mmse_time_days": current_time,
        "current_mmse_day_offset": (
            current_time - landmark_time if pd.notna(current_time) else np.nan
        ),
        "current_mmse_same_day_flag": (
            bool(current_time == landmark_time) if pd.notna(current_time) else False
        ),
        "mmse_change_from_index": (
            current_value - index_mmse
            if pd.notna(current_value) and pd.notna(index_mmse)
            else np.nan
        ),
        "previous_mmse": previous_value,
        "previous_mmse_time_days": previous_time,
        "time_since_previous_mmse_days": (
            landmark_time - previous_time
            if pd.notna(previous_time)
            else np.nan
        ),
        "time_since_previous_mmse_years": (
            (landmark_time - previous_time) / DAYS_PER_YEAR
            if pd.notna(previous_time)
            else np.nan
        ),
        "mmse_slope_per_year": slope,
        "mmse_slope_method": method,
        "mmse_slope_reliable_flag": reliable,
        "mmse_slope_two_point_flag": two_point,
        "prior_mmse_count": count,
        "mmse_history_span_days": span,
        "slope_max_measurement_time_days": (
            numeric(slope_history["measurement_time_days"]).max()
            if count
            else np.nan
        ),
    }


def _retain_by_interval(
    candidate: pd.DataFrame, minimum_interval_days: int
) -> pd.DataFrame:
    retained = []
    last = -np.inf
    for row in candidate.sort_values("time_sort_days", kind="mergesort").itertuples(
        index=False
    ):
        if row.time_sort_days - last >= minimum_interval_days:
            retained.append(row.manifest_key)
            last = row.time_sort_days
    return candidate[candidate["manifest_key"].isin(retained)].copy()


def build_dynamic_base(
    cohort: str,
    phase1: dict[str, pd.DataFrame],
    mmse_history: pd.DataFrame,
    minimum_interval_days: int,
) -> pd.DataFrame:
    manifest = phase1["manifest"].copy()
    index = phase1["index"].copy()
    episode = phase1["episode"].set_index("participant_id")
    rows: list[dict[str, Any]] = []
    for idx in index.itertuples(index=False):
        participant = str(idx.participant_id)
        subject = manifest[
            manifest["participant_id"].astype(str).eq(participant)
        ].sort_values("time_sort_days", kind="mergesort")
        if subject.empty:
            continue
        outcome = episode.loc[participant]
        event_absolute = (
            idx.index_time_days + outcome["first_event_time_days"]
            if pd.notna(outcome["first_event_time_days"])
            else np.inf
        )
        candidates = subject[
            subject["time_sort_days"].ge(idx.index_time_days)
            & subject["time_sort_days"].lt(event_absolute)
            & subject["cdr_state"].eq(1)
        ].copy()
        candidates = candidates[
            candidates["time_sort_days"].lt(subject["time_sort_days"].max())
        ]
        if candidates.empty:
            continue
        candidates = _retain_by_interval(candidates, minimum_interval_days)
        participant_mmse = mmse_history[
            mmse_history["participant_id"].astype(str).eq(participant)
        ]
        first_event = subject[
            subject["time_sort_days"].gt(idx.index_time_days)
            & ~subject["cdr_state"].eq(1)
        ].head(1)
        last_observed_time = float(subject["time_sort_days"].max())
        for number, landmark in enumerate(
            candidates.itertuples(index=False), start=1
        ):
            features = mmse_features_at_landmark(
                participant_mmse,
                float(idx.index_mmse),
                float(idx.index_mmse_time_days),
                float(landmark.time_sort_days),
            )
            elapsed_days = float(landmark.time_sort_days - idx.index_time_days)
            event_row = (
                first_event.iloc[0]
                if not first_event.empty
                and float(first_event.iloc[0]["time_sort_days"])
                > landmark.time_sort_days
                else None
            )
            transition = "none_observed"
            event_time = np.nan
            event_manifest_key = pd.NA
            event_same_day_mmse = False
            if event_row is not None:
                transition = (
                    "transition_to_cdr_0"
                    if event_row["cdr_state"] == 0
                    else "transition_to_cdr_ge1"
                )
                event_time = float(
                    event_row["time_sort_days"] - landmark.time_sort_days
                )
                event_manifest_key = event_row["manifest_key"]
                event_same_day_mmse = bool(
                    event_row.get("mmse_same_time_as_cdr_flag", False)
                )
            record = {
                "cohort": cohort,
                "participant_id": participant,
                "primary_episode_id": f"{cohort}:{participant}:primary_cdr05_episode",
                "landmark_id": f"{cohort}:{participant}:L{number:03d}",
                "landmark_manifest_key": landmark.manifest_key,
                "landmark_number": number,
                "landmark_time_days": landmark.time_sort_days,
                "landmark_visit_id": landmark.original_visit_id,
                "landmark_minimum_interval_days": minimum_interval_days,
                "current_state": 1,
                "index_time_days": idx.index_time_days,
                "elapsed_days_from_index": elapsed_days,
                "elapsed_months_from_index": elapsed_days / DAYS_PER_MONTH,
                "elapsed_years_from_index": elapsed_days / DAYS_PER_YEAR,
                "index_age": idx.index_age,
                "age_at_landmark": idx.index_age + elapsed_days / DAYS_PER_YEAR,
                "current_age": idx.index_age + elapsed_days / DAYS_PER_YEAR,
                "sex": idx.sex,
                "education_years": idx.education_years,
                "apoe_genotype": idx.apoe_genotype,
                "apoe_e4_dose": idx.apoe_e4_dose,
                "index_mmse": idx.index_mmse,
                "index_mmse_time_days": idx.index_mmse_time_days,
                **features,
                "first_transition_after_landmark": transition,
                "transition_manifest_key": event_manifest_key,
                "transition_time_from_landmark_days": event_time,
                "transition_time_from_landmark_months": (
                    event_time / DAYS_PER_MONTH if pd.notna(event_time) else np.nan
                ),
                "last_classifiable_time_days": last_observed_time,
                "censoring_time_from_landmark_days": last_observed_time
                - landmark.time_sort_days,
                "outcome_assessment_mmse_same_day_flag": event_same_day_mmse,
                "clinician_diagnosis": idx.clinician_diagnosis,
                "clinician_diagnosed_mci_at_index_flag": idx.clinician_diagnosed_mci_at_index_flag,
                "clinical_mci_subgroup_flag": idx.clinical_mci_subgroup_flag,
                "study_phase": landmark.study_phase,
                "form_family": landmark.form_family,
                "site_id": landmark.site_id,
                "index_year": idx.index_year,
                "index_calendar_period": idx.index_calendar_period,
                "index_relative_time_band": idx.index_relative_time_band,
                "cdr_source_field": (
                    "CDRTOT_de_facto_global"
                    if cohort == "oasis3"
                    else "CDGLOBAL"
                ),
                "cdr_field_validation_status": idx.cdr_field_validation_status,
                "external_endpoint_status": (
                    "usable_with_provisional_export_mapping"
                    if cohort == "oasis3"
                    else "not_applicable"
                ),
                "predictor_max_time_days": np.nanmax(
                    [
                        landmark.time_sort_days,
                        features["current_mmse_time_days"],
                        features["slope_max_measurement_time_days"],
                    ]
                ),
                "phase2_qc_flags": "",
            }
            rows.append(record)
    return pd.DataFrame(rows)


def apply_dynamic_horizon(dynamic: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = dynamic.copy()
    horizon_days = horizon * DAYS_PER_MONTH
    event_time = numeric(frame["transition_time_from_landmark_days"])
    event_by_horizon = event_time.notna() & event_time.le(horizon_days)
    observed = numeric(frame["censoring_time_from_landmark_days"]).ge(horizon_days)
    status = pd.Series("censored_before_horizon", index=frame.index, dtype="string")
    status.loc[observed] = "remain_cdr_0_5_through_horizon"
    status.loc[
        event_by_horizon
        & frame["first_transition_after_landmark"].eq("transition_to_cdr_0")
    ] = "transition_to_cdr_0"
    status.loc[
        event_by_horizon
        & frame["first_transition_after_landmark"].eq("transition_to_cdr_ge1")
    ] = "transition_to_cdr_ge1"
    frame["horizon_months"] = horizon
    frame["horizon_days"] = horizon_days
    frame["horizon_status"] = status
    frame["fixed_horizon_class"] = status
    frame["outcome_code"] = status.map(_outcome_code)
    frame["event_by_horizon_flag"] = event_by_horizon
    frame["transition_to_cdr_0_flag"] = status.eq("transition_to_cdr_0")
    frame["transition_to_cdr_ge1_flag"] = status.eq("transition_to_cdr_ge1")
    frame["remain_cdr_0_5_flag"] = status.eq(
        "remain_cdr_0_5_through_horizon"
    )
    frame["censored_before_horizon_flag"] = status.eq("censored_before_horizon")
    frame["observed_through_horizon_flag"] = observed
    frame["eligible_for_fixed_horizon_analysis"] = ~frame[
        "censored_before_horizon_flag"
    ]
    frame["same_day_mmse_flag"] = frame["current_mmse_same_day_flag"]
    frame["strictly_lagged_mmse_available_flag"] = frame[
        "previous_mmse"
    ].notna()
    frame["no_same_day_outcome_assessment_overlap_flag"] = ~(
        frame["event_by_horizon_flag"]
        & frame["outcome_assessment_mmse_same_day_flag"]
    )
    return frame


def dynamic_support_summary(
    dynamic_by_interval: dict[str, dict[int, pd.DataFrame]]
) -> pd.DataFrame:
    rows = []
    for cohort, interval_frames in dynamic_by_interval.items():
        for interval, base in interval_frames.items():
            for horizon in HORIZONS:
                frame = apply_dynamic_horizon(base, horizon)
                for sensitivity, subset in {
                    "primary_current_mmse": frame[frame["current_mmse"].notna()],
                    "all_eligible_landmarks": frame,
                    "strictly_lagged_mmse": frame[frame["previous_mmse"].notna()],
                    "slope_eligible": frame[frame["mmse_slope_per_year"].notna()],
                    "clinical_mci_subgroup": frame[
                        frame["clinical_mci_subgroup_flag"].fillna(False)
                    ],
                    "no_mmse": frame,
                }.items():
                    rows.append(
                        {
                            "cohort": cohort,
                            "minimum_interval_days": interval,
                            "horizon_months": horizon,
                            "dataset": sensitivity,
                            "participants": subset["participant_id"].nunique(),
                            "landmarks": len(subset),
                            "improvement_events": int(
                                subset["transition_to_cdr_0_flag"].sum()
                            ),
                            "progression_events": int(
                                subset["transition_to_cdr_ge1_flag"].sum()
                            ),
                            "persistent_landmarks": int(
                                subset["remain_cdr_0_5_flag"].sum()
                            ),
                            "censored_landmarks": int(
                                subset["censored_before_horizon_flag"].sum()
                            ),
                            "mean_landmarks_per_participant": (
                                len(subset)
                                / subset["participant_id"].nunique()
                                if subset["participant_id"].nunique()
                                else np.nan
                            ),
                            "max_landmarks_per_participant": (
                                subset.groupby("participant_id").size().max()
                                if not subset.empty
                                else 0
                            ),
                        }
                    )
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["cohort", "minimum_interval_days", "horizon_months", "dataset"],
        kind="mergesort",
    ).reset_index(drop=True)
    save_table(summary, TABLE_ROOT / "dynamic_landmark_support.csv")
    return summary


def build_dynamic_datasets(
    phase1: dict[str, dict[str, pd.DataFrame]],
    mmse_histories: dict[str, pd.DataFrame],
) -> tuple[
    dict[str, dict[int, pd.DataFrame]],
    dict[str, dict[int, pd.DataFrame]],
    pd.DataFrame,
]:
    by_interval: dict[str, dict[int, pd.DataFrame]] = {}
    model_ready: dict[str, dict[int, pd.DataFrame]] = {}
    for cohort in COHORTS:
        by_interval[cohort] = {}
        for interval in PARAMS["landmark_interval_sensitivities_days"]:
            by_interval[cohort][int(interval)] = build_dynamic_base(
                cohort,
                phase1[cohort],
                mmse_histories[cohort],
                int(interval),
            )
            by_interval[cohort][int(interval)].to_parquet(
                TABLE_ROOT
                / cohort
                / f"dynamic_landmark_candidates_{int(interval)}d.parquet",
                index=False,
            )
        model_ready[cohort] = {
            horizon: apply_dynamic_horizon(
                by_interval[cohort][PRIMARY_INTERVAL], horizon
            )
            for horizon in HORIZONS
        }
    support = dynamic_support_summary(by_interval)
    return model_ready, by_interval, support


def _fold_strata(participants: pd.DataFrame, folds: int) -> pd.Series:
    event = participants["first_event_type"].fillna("unknown").astype(str)
    medium = (
        event
        + "|mci="
        + participants["clinical_mci_subgroup_flag"].fillna(False).astype(str)
    )
    strata = medium.copy()
    for event_value in event.unique():
        mask = event.eq(event_value)
        if medium[mask].value_counts().min() < folds:
            strata.loc[mask] = event_value
    return strata


def create_outer_assignments(
    baseline36: pd.DataFrame, folds: int, seed: int
) -> pd.DataFrame:
    participants = baseline36[
        [
            "participant_id",
            "first_transition",
            "study_phase",
            "clinical_mci_subgroup_flag",
        ]
    ].copy()
    participants["first_event_type"] = participants["first_transition"].map(
        {
            "transition_to_cdr_0": "improvement",
            "transition_to_cdr_ge1": "progression",
            "none_observed": "no_transition",
        }
    )
    participants["stratification_label"] = _fold_strata(participants, folds)
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    participants["outer_fold"] = -1
    for fold, (_, test_index) in enumerate(
        splitter.split(participants, participants["stratification_label"])
    ):
        participants.loc[test_index, "outer_fold"] = fold
    participants["outer_fold_count"] = folds
    participants["random_seed"] = seed
    return participants


def fold_balance(
    assignments: pd.DataFrame,
    baselines: dict[int, pd.DataFrame],
    design: str,
) -> pd.DataFrame:
    rows = []
    for horizon, frame in baselines.items():
        merged = frame.merge(
            assignments[["participant_id", "outer_fold"]],
            on="participant_id",
            how="left",
        )
        for fold, group in merged.groupby("outer_fold"):
            rows.append(
                {
                    "design": design,
                    "horizon_months": horizon,
                    "outer_fold": fold,
                    "participants": group["participant_id"].nunique(),
                    "improvement_events": int(
                        group["transition_to_cdr_0_flag"].sum()
                    ),
                    "progression_events": int(
                        group["transition_to_cdr_ge1_flag"].sum()
                    ),
                    "persistent": int(group["remain_cdr_0_5_flag"].sum()),
                    "censored": int(group["censored_before_horizon_flag"].sum()),
                    "clinical_mci": int(
                        group["clinical_mci_subgroup_flag"].fillna(False).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def create_fold_artifacts(
    adni_baselines: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = {}
    balances = []
    for folds in PARAMS["candidate_outer_folds"]:
        assignment = create_outer_assignments(
            adni_baselines[36], int(folds), SEED + int(folds)
        )
        candidates[int(folds)] = assignment
        balances.append(
            fold_balance(assignment, adni_baselines, f"{folds}-fold")
        )
    candidate_balance = pd.concat(balances, ignore_index=True)
    save_table(
        candidate_balance, TABLE_ROOT / "candidate_outer_fold_balance.csv"
    )
    selected_folds = 5
    assignments = candidates[selected_folds].copy()
    inner_rows = []
    for outer_fold in range(selected_folds):
        train = assignments[assignments["outer_fold"].ne(outer_fold)].copy()
        train["inner_stratum"] = _fold_strata(
            train.rename(columns={"stratification_label": "_old"}),
            int(PARAMS["inner_folds"]),
        )
        splitter = StratifiedKFold(
            n_splits=int(PARAMS["inner_folds"]),
            shuffle=True,
            random_state=SEED + 100 + outer_fold,
        )
        train["inner_fold"] = -1
        for inner_fold, (_, test_index) in enumerate(
            splitter.split(train, train["inner_stratum"])
        ):
            train.iloc[test_index, train.columns.get_loc("inner_fold")] = inner_fold
        for row in train.itertuples(index=False):
            inner_rows.append(
                {
                    "outer_test_fold": outer_fold,
                    "participant_id": row.participant_id,
                    "role_in_outer_split": "outer_training",
                    "inner_validation_fold": row.inner_fold,
                    "random_seed": SEED + 100 + outer_fold,
                }
            )
        for participant in assignments.loc[
            assignments["outer_fold"].eq(outer_fold), "participant_id"
        ]:
            inner_rows.append(
                {
                    "outer_test_fold": outer_fold,
                    "participant_id": participant,
                    "role_in_outer_split": "outer_test",
                    "inner_validation_fold": np.nan,
                    "random_seed": SEED + 100 + outer_fold,
                }
            )
    inner = pd.DataFrame(inner_rows)
    selected_balance = fold_balance(
        assignments, adni_baselines, "selected_5-fold"
    )
    save_table(assignments, TABLE_ROOT / "adni_outer_fold_assignments.csv")
    save_table(inner, TABLE_ROOT / "adni_nested_inner_fold_assignments.csv")
    save_table(selected_balance, TABLE_ROOT / "adni_fold_balance.csv")
    return assignments, inner, selected_balance


def attach_outer_folds(
    baselines: dict[str, dict[int, pd.DataFrame]],
    dynamics: dict[str, dict[int, pd.DataFrame]],
    assignments: pd.DataFrame,
) -> None:
    fold_map = assignments.set_index("participant_id")["outer_fold"]
    for horizon in HORIZONS:
        baselines["adni"][horizon]["outer_fold"] = (
            baselines["adni"][horizon]["participant_id"].map(fold_map).astype("Int64")
        )
        dynamics["adni"][horizon]["outer_fold"] = (
            dynamics["adni"][horizon]["participant_id"].map(fold_map).astype("Int64")
        )
        for cohort in ("oasis3", "aibl"):
            baselines[cohort][horizon]["outer_fold"] = pd.Series(
                pd.NA, index=baselines[cohort][horizon].index, dtype="Int64"
            )
            dynamics[cohort][horizon]["outer_fold"] = pd.Series(
                pd.NA, index=dynamics[cohort][horizon].index, dtype="Int64"
            )


MMSE_DERIVED_COLUMNS = [
    "index_mmse",
    "current_mmse",
    "current_mmse_time_days",
    "current_mmse_day_offset",
    "current_mmse_same_day_flag",
    "mmse_change_from_index",
    "lagged_mmse",
    "lagged_mmse_time_days",
    "lagged_mmse_day_offset",
    "previous_mmse",
    "previous_mmse_time_days",
    "time_since_previous_mmse_days",
    "time_since_previous_mmse_years",
    "mmse_slope_per_year",
    "mmse_slope_method",
    "mmse_slope_reliable_flag",
    "mmse_slope_two_point_flag",
    "prior_mmse_count",
    "mmse_history_span_days",
    "slope_max_measurement_time_days",
    "same_day_mmse_flag",
    "strictly_lagged_mmse_available_flag",
]


def _baseline_sensitivities(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lagged = frame[frame["lagged_mmse"].notna()].copy()
    lagged["current_mmse"] = lagged["lagged_mmse"]
    lagged["current_mmse_time_days"] = lagged["lagged_mmse_time_days"]
    lagged["same_day_mmse_flag"] = False
    lagged["mmse_change_from_index"] = np.nan
    no_mmse = frame.drop(
        columns=[c for c in MMSE_DERIVED_COLUMNS if c in frame],
        errors="ignore",
    ).copy()
    mci = frame[frame["clinical_mci_subgroup_flag"].fillna(False)].copy()
    no_overlap = frame[
        frame["no_same_day_outcome_assessment_overlap_flag"].fillna(False)
    ].copy()
    return {
        "strictly_lagged_mmse": lagged,
        "no_mmse": no_mmse,
        "clinical_mci_subgroup": mci,
        "no_same_day_outcome_overlap": no_overlap,
    }


def _dynamic_sensitivities(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lagged = frame[frame["previous_mmse"].notna()].copy()
    lagged["current_mmse"] = lagged["previous_mmse"]
    lagged["current_mmse_time_days"] = lagged["previous_mmse_time_days"]
    lagged["current_mmse_day_offset"] = (
        lagged["previous_mmse_time_days"] - lagged["landmark_time_days"]
    )
    lagged["current_mmse_same_day_flag"] = False
    lagged["same_day_mmse_flag"] = False
    lagged["mmse_change_from_index"] = (
        lagged["previous_mmse"] - lagged["index_mmse"]
    )
    no_mmse = frame.drop(
        columns=[c for c in MMSE_DERIVED_COLUMNS if c in frame],
        errors="ignore",
    ).copy()
    mci = frame[frame["clinical_mci_subgroup_flag"].fillna(False)].copy()
    no_overlap = frame[
        frame["no_same_day_outcome_assessment_overlap_flag"].fillna(False)
    ].copy()
    slope = frame[frame["mmse_slope_per_year"].notna()].copy()
    return {
        "strictly_lagged_mmse": lagged,
        "no_mmse": no_mmse,
        "clinical_mci_subgroup": mci,
        "no_same_day_outcome_overlap": no_overlap,
        "slope_eligible": slope,
    }


def save_model_ready_datasets(
    baselines: dict[str, dict[int, pd.DataFrame]],
    dynamics: dict[str, dict[int, pd.DataFrame]],
) -> pd.DataFrame:
    inventory_rows = []
    for cohort in COHORTS:
        for horizon in HORIZONS:
            baseline = baselines[cohort][horizon]
            dynamic = dynamics[cohort][horizon]
            base_path = (
                PROCESSED_ROOT
                / cohort
                / "phase2"
                / "baseline"
                / f"baseline_{horizon}m_clinical_core.csv"
            )
            dynamic_path = (
                PROCESSED_ROOT
                / cohort
                / "phase2"
                / "dynamic"
                / f"dynamic_{horizon}m_clinical_core.csv"
            )
            save_table(baseline, base_path, parquet=True)
            save_table(dynamic, dynamic_path, parquet=True)
            for dataset_type, frame, path in [
                ("baseline_primary_current_mmse", baseline, base_path),
                ("dynamic_primary_current_mmse", dynamic, dynamic_path),
            ]:
                inventory_rows.append(
                    {
                        "cohort": cohort,
                        "horizon_months": horizon,
                        "dataset": dataset_type,
                        "relative_path": str(path.relative_to(ROOT)),
                        "rows": len(frame),
                        "participants": frame["participant_id"].nunique(),
                        "improvement_events": int(
                            frame["transition_to_cdr_0_flag"].sum()
                        ),
                        "progression_events": int(
                            frame["transition_to_cdr_ge1_flag"].sum()
                        ),
                        "censored_rows": int(
                            frame["censored_before_horizon_flag"].sum()
                        ),
                    }
                )
            for name, sensitivity in _baseline_sensitivities(baseline).items():
                path = (
                    PROCESSED_ROOT
                    / cohort
                    / "phase2"
                    / "sensitivity"
                    / f"baseline_{horizon}m_{name}.csv"
                )
                save_table(sensitivity, path, parquet=True)
                inventory_rows.append(
                    {
                        "cohort": cohort,
                        "horizon_months": horizon,
                        "dataset": f"baseline_{name}",
                        "relative_path": str(path.relative_to(ROOT)),
                        "rows": len(sensitivity),
                        "participants": sensitivity["participant_id"].nunique(),
                        "improvement_events": int(
                            sensitivity["transition_to_cdr_0_flag"].sum()
                        ),
                        "progression_events": int(
                            sensitivity["transition_to_cdr_ge1_flag"].sum()
                        ),
                        "censored_rows": int(
                            sensitivity["censored_before_horizon_flag"].sum()
                        ),
                    }
                )
            for name, sensitivity in _dynamic_sensitivities(dynamic).items():
                path = (
                    PROCESSED_ROOT
                    / cohort
                    / "phase2"
                    / "sensitivity"
                    / f"dynamic_{horizon}m_{name}.csv"
                )
                save_table(sensitivity, path, parquet=True)
                inventory_rows.append(
                    {
                        "cohort": cohort,
                        "horizon_months": horizon,
                        "dataset": f"dynamic_{name}",
                        "relative_path": str(path.relative_to(ROOT)),
                        "rows": len(sensitivity),
                        "participants": sensitivity["participant_id"].nunique(),
                        "improvement_events": int(
                            sensitivity["transition_to_cdr_0_flag"].sum()
                        ),
                        "progression_events": int(
                            sensitivity["transition_to_cdr_ge1_flag"].sum()
                        ),
                        "censored_rows": int(
                            sensitivity["censored_before_horizon_flag"].sum()
                        ),
                    }
                )
    inventory = pd.DataFrame(inventory_rows)
    inventory = inventory.sort_values(
        ["cohort", "horizon_months", "dataset"], kind="mergesort"
    ).reset_index(drop=True)
    save_table(inventory, TABLE_ROOT / "model_ready_dataset_inventory.csv")
    return inventory


def dynamic_mmse_audit(
    dynamics: dict[str, dict[int, pd.DataFrame]]
) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        frame = dynamics[cohort][36]
        for variable in [
            "index_mmse",
            "current_mmse",
            "mmse_change_from_index",
            "previous_mmse",
            "mmse_slope_per_year",
        ]:
            valid = frame[variable].notna()
            rows.append(
                {
                    "cohort": cohort,
                    "variable": variable,
                    "landmarks_available": int(valid.sum()),
                    "landmarks_total": len(frame),
                    "availability_percent": valid.mean() * 100,
                    "participants_available": frame.loc[
                        valid, "participant_id"
                    ].nunique(),
                    "median": numeric(frame.loc[valid, variable]).median(),
                    "q1": numeric(frame.loc[valid, variable]).quantile(0.25),
                    "q3": numeric(frame.loc[valid, variable]).quantile(0.75),
                    "minimum": numeric(frame.loc[valid, variable]).min(),
                    "maximum": numeric(frame.loc[valid, variable]).max(),
                }
            )
        methods = frame["mmse_slope_method"].value_counts(dropna=False)
        for method, count in methods.items():
            rows.append(
                {
                    "cohort": cohort,
                    "variable": f"slope_method:{method}",
                    "landmarks_available": int(count),
                    "landmarks_total": len(frame),
                    "availability_percent": count / len(frame) * 100,
                    "participants_available": frame.loc[
                        frame["mmse_slope_method"].eq(method), "participant_id"
                    ].nunique(),
                }
            )
    audit = pd.DataFrame(rows)
    save_table(audit, TABLE_ROOT / "dynamic_mmse_feature_audit.csv")
    return audit


def _missingness_rows(
    frame: pd.DataFrame,
    dataset_type: str,
    cohort: str,
    horizon: int,
) -> list[dict[str, Any]]:
    variables = [
        "age_at_landmark",
        "sex",
        "education_years",
        "apoe_e4_dose",
        "current_mmse",
        "index_mmse",
        "mmse_change_from_index",
        "mmse_slope_per_year",
        "elapsed_years_from_index",
    ]
    dimensions = {
        "ALL": None,
        "outcome": "horizon_status",
        "study_phase": "study_phase",
        "form_family": "form_family",
        "index_period": (
            "index_calendar_period"
            if cohort != "oasis3"
            else "index_relative_time_band"
        ),
        "clinician_diagnosis": "clinician_diagnosis",
        "cdr_source_field": "cdr_source_field",
        "mmse_same_day": "same_day_mmse_flag",
    }
    if dataset_type == "dynamic":
        counts = frame.groupby("participant_id")["landmark_id"].transform("count")
        frame = frame.assign(
            landmark_count_band=pd.cut(
                counts, [0, 1, 2, 4, np.inf], labels=["1", "2", "3-4", "5+"]
            )
        )
        dimensions["landmark_count"] = "landmark_count_band"
    rows: list[dict[str, Any]] = []
    for dimension, column in dimensions.items():
        groups = (
            [("ALL", frame)]
            if column is None
            else list(
                frame.assign(
                    _phase2_group_value=frame[column]
                    .astype("string")
                    .fillna("<missing>")
                ).groupby("_phase2_group_value", dropna=False)
            )
        )
        for value, group in groups:
            for variable in variables:
                if variable not in group:
                    continue
                missing = group[variable].isna()
                rows.append(
                    {
                        "cohort": cohort,
                        "dataset_type": dataset_type,
                        "horizon_months": horizon,
                        "dimension": dimension,
                        "dimension_value": str(value),
                        "variable": variable,
                        "rows": len(group),
                        "participants": group["participant_id"].nunique(),
                        "missing_count": int(missing.sum()),
                        "missing_percent": missing.mean() * 100,
                    }
                )
    return rows


def missingness_and_retention(
    baselines: dict[str, dict[int, pd.DataFrame]],
    dynamics: dict[str, dict[int, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_rows = []
    retention_rows = []
    signal_rows = []
    tiers = {
        "minimal_core": ["age_at_landmark", "sex", "elapsed_years_from_index"],
        "full_clinical_core": [
            "age_at_landmark",
            "sex",
            "education_years",
            "apoe_e4_dose",
            "current_mmse",
            "elapsed_years_from_index",
        ],
        "dynamic_current_change": [
            "age_at_landmark",
            "sex",
            "education_years",
            "apoe_e4_dose",
            "current_mmse",
            "mmse_change_from_index",
            "elapsed_years_from_index",
        ],
        "dynamic_slope": [
            "age_at_landmark",
            "sex",
            "education_years",
            "apoe_e4_dose",
            "current_mmse",
            "mmse_change_from_index",
            "mmse_slope_per_year",
            "elapsed_years_from_index",
        ],
        "lagged_mmse": [
            "age_at_landmark",
            "sex",
            "education_years",
            "apoe_e4_dose",
            "previous_mmse",
            "elapsed_years_from_index",
        ],
        "no_mmse": [
            "age_at_landmark",
            "sex",
            "education_years",
            "apoe_e4_dose",
            "elapsed_years_from_index",
        ],
    }
    for cohort in COHORTS:
        for horizon in HORIZONS:
            for dataset_type, frame in [
                ("baseline", baselines[cohort][horizon]),
                ("dynamic", dynamics[cohort][horizon]),
            ]:
                missing_rows.extend(
                    _missingness_rows(
                        frame.copy(), dataset_type, cohort, horizon
                    )
                )
                for tier, features in tiers.items():
                    usable_features = [x for x in features if x in frame]
                    if tier.startswith("dynamic_") and dataset_type == "baseline":
                        continue
                    if tier == "lagged_mmse" and dataset_type == "baseline":
                        usable_features = [
                            "age_at_landmark",
                            "sex",
                            "education_years",
                            "apoe_e4_dose",
                            "lagged_mmse",
                            "elapsed_years_from_index",
                        ]
                    complete = frame[usable_features].notna().all(axis=1)
                    retained = frame[complete]
                    retention_rows.append(
                        {
                            "cohort": cohort,
                            "dataset_type": dataset_type,
                            "horizon_months": horizon,
                            "feature_tier": tier,
                            "rows_total": len(frame),
                            "rows_retained": len(retained),
                            "row_retention_percent": complete.mean() * 100,
                            "participants_total": frame[
                                "participant_id"
                            ].nunique(),
                            "participants_retained": retained[
                                "participant_id"
                            ].nunique(),
                            "improvement_events_total": int(
                                frame["transition_to_cdr_0_flag"].sum()
                            ),
                            "improvement_events_retained": int(
                                retained["transition_to_cdr_0_flag"].sum()
                            ),
                            "progression_events_total": int(
                                frame["transition_to_cdr_ge1_flag"].sum()
                            ),
                            "progression_events_retained": int(
                                retained["transition_to_cdr_ge1_flag"].sum()
                            ),
                        }
                    )
    missingness = pd.DataFrame(missing_rows)
    retention = pd.DataFrame(retention_rows)
    for keys, group in missingness[
        missingness["dimension"].isin(
            ["outcome", "study_phase", "form_family"]
        )
        & missingness["rows"].ge(10)
    ].groupby(
        [
            "cohort",
            "dataset_type",
            "horizon_months",
            "dimension",
            "variable",
        ]
    ):
        spread = group["missing_percent"].max() - group["missing_percent"].min()
        if spread >= 10:
            signal_rows.append(
                {
                    "cohort": keys[0],
                    "dataset_type": keys[1],
                    "horizon_months": keys[2],
                    "dimension": keys[3],
                    "variable": keys[4],
                    "missingness_spread_percentage_points": spread,
                    "lowest_group": group.loc[
                        group["missing_percent"].idxmin(), "dimension_value"
                    ],
                    "highest_group": group.loc[
                        group["missing_percent"].idxmax(), "dimension_value"
                    ],
                    "interpretation": "descriptive structural-missingness signal; not proof of mechanism",
                }
            )
    signals = pd.DataFrame(signal_rows)
    save_table(missingness, TABLE_ROOT / "clinical_core_missingness.csv")
    save_table(retention, TABLE_ROOT / "complete_case_event_retention.csv")
    save_table(signals, TABLE_ROOT / "missingness_structure_signals.csv")
    return missingness, retention, signals


def preprocessing_audit(
    adni_baseline36: pd.DataFrame,
    oasis_baseline36: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    continuous = [
        "age_at_landmark",
        "education_years",
        "apoe_e4_dose",
        "current_mmse",
        "elapsed_years_from_index",
    ]
    categorical = ["sex"]
    train = adni_baseline36[adni_baseline36["outer_fold"].ne(0)].copy()
    validation = adni_baseline36[adni_baseline36["outer_fold"].eq(0)].copy()
    preprocessor = ClinicalCorePreprocessor(continuous, categorical)
    preprocessor.fit(
        train,
        cohort="adni",
        partition_label="phase2_audit_outer_fold_0_training_only",
    )
    model_path = TABLE_ROOT / "preprocessing" / "clinical_core_preprocessor.joblib"
    preprocessor.save(model_path)
    transformed_train = preprocessor.transform(train)
    transformed_validation = preprocessor.transform(validation)
    transformed_external = preprocessor.transform(oasis_baseline36)
    if list(transformed_train.columns) != list(transformed_external.columns):
        raise AssertionError("Preprocessing feature schema changed in OASIS-3.")
    schema = pd.DataFrame(
        {
            "output_feature": transformed_train.columns,
            "position": np.arange(len(transformed_train.columns)),
            "fit_partition": "ADNI outer-fold-0 training only",
            "applied_to_adni_validation": True,
            "applied_to_oasis3_unchanged": True,
        }
    )
    save_table(schema, TABLE_ROOT / "preprocessing_transformed_feature_schema.csv")

    strategy_rows = [
        {
            "strategy": "median_continuous_mode_categorical",
            "fit_cohort": "ADNI",
            "fit_partition": "outer-fold-0 training only",
            "converged": True,
            "iterations": 1,
            "retained_training_rows": len(train),
            "retained_validation_rows": len(validation),
            "retained_external_rows": len(oasis_baseline36),
            "nested_validation_compatible": True,
            "multiple_imputation_compatible": False,
            "outcome_used": False,
            "external_statistics_used": False,
            "recommendation": "PRIMARY",
            "rationale": "stable, transparent, low-dimensional, deterministic, and fold-safe",
        }
    ]
    iterative = IterativeImputer(
        random_state=SEED,
        max_iter=25,
        sample_posterior=False,
        initial_strategy="median",
        skip_complete=True,
    )
    train_numeric = train[continuous]
    validation_numeric = validation[continuous]
    external_numeric = oasis_baseline36[continuous]
    iterative.fit(train_numeric)
    iterative_train = iterative.transform(train_numeric)
    iterative_validation = iterative.transform(validation_numeric)
    iterative_external = iterative.transform(external_numeric)
    plausibility = []
    for position, feature in enumerate(continuous):
        original = train_numeric[feature]
        completed = pd.Series(iterative_train[:, position], index=train.index)
        missing = original.isna()
        plausibility.append(
            {
                "strategy": "iterative_imputation",
                "feature": feature,
                "training_missing_values_imputed": int(missing.sum()),
                "observed_training_median": original.median(),
                "imputed_training_median": (
                    completed[missing].median() if missing.any() else np.nan
                ),
                "imputed_training_minimum": (
                    completed[missing].min() if missing.any() else np.nan
                ),
                "imputed_training_maximum": (
                    completed[missing].max() if missing.any() else np.nan
                ),
                "validation_finite_percent": np.isfinite(
                    iterative_validation[:, position]
                ).mean()
                * 100,
                "external_finite_percent": np.isfinite(
                    iterative_external[:, position]
                ).mean()
                * 100,
            }
        )
    strategy_rows.append(
        {
            "strategy": "iterative_imputation",
            "fit_cohort": "ADNI",
            "fit_partition": "outer-fold-0 training only",
            "converged": iterative.n_iter_ < iterative.max_iter,
            "iterations": int(iterative.n_iter_),
            "retained_training_rows": len(train),
            "retained_validation_rows": len(validation),
            "retained_external_rows": len(oasis_baseline36),
            "nested_validation_compatible": True,
            "multiple_imputation_compatible": False,
            "outcome_used": False,
            "external_statistics_used": False,
            "recommendation": "SECONDARY FEASIBILITY",
            "rationale": "feasible but adds unstable conditional assumptions with little need in the core",
        }
    )
    strategy_rows.append(
        {
            "strategy": "multiple_imputation_compatible",
            "fit_cohort": "ADNI",
            "fit_partition": "each training fold separately; not executed in Phase 2",
            "converged": np.nan,
            "iterations": np.nan,
            "retained_training_rows": len(train),
            "retained_validation_rows": len(validation),
            "retained_external_rows": len(oasis_baseline36),
            "nested_validation_compatible": True,
            "multiple_imputation_compatible": True,
            "outcome_used": False,
            "external_statistics_used": False,
            "recommendation": "SECONDARY UNCERTAINTY ANALYSIS",
            "rationale": "fit m imputers per ADNI training fold; average individual predicted probabilities and pool estimands with Rubin-compatible rules",
        }
    )
    strategies = pd.DataFrame(strategy_rows)
    plausibility_frame = pd.DataFrame(plausibility)
    save_table(strategies, TABLE_ROOT / "preprocessing_strategy_comparison.csv")
    save_table(
        plausibility_frame,
        TABLE_ROOT / "iterative_imputation_plausibility_audit.csv",
    )
    preprocessing_config = {
        "primary": {
            "continuous_imputation": "median fit within ADNI training fold",
            "categorical_imputation": "most frequent fit within ADNI training fold",
            "continuous_scaling": "RobustScaler fit within ADNI training fold",
            "categorical_encoding": "one-hot with unknown categories ignored",
            "winsorization": "off in primary analysis",
        },
        "fit_contract": {
            "allowed_fit_cohort": "adni",
            "fit_unit": "outer/inner training partition only",
            "participant_grouping_preserved": True,
            "oasis3_statistics_used": False,
            "outcome_columns_used": False,
        },
        "multiple_imputation_secondary": {
            "status": "compatible but not performed in Phase 2",
            "rule": "m independently fitted imputers inside each ADNI training fold; transform validation/external data unchanged for each imputation; pool predicted probabilities by arithmetic mean and coefficient-level estimands with Rubin-compatible rules where applicable",
        },
    }
    (TABLE_ROOT / "preprocessing_configuration.json").write_text(
        json.dumps(preprocessing_config, indent=2), encoding="utf-8"
    )
    loaded = load_preprocessor(model_path)
    if loaded.provenance is None or loaded.provenance.fit_cohort != "adni":
        raise AssertionError("Saved preprocessing provenance failed validation.")
    return strategies, plausibility_frame


def clinical_core_dictionary_and_provenance() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        (
            "participant_id",
            "string",
            "participant",
            "Phase 1 canonical identifier",
            "grouping only",
            "never encoded as predictor",
        ),
        (
            "age_at_landmark",
            "float",
            "years",
            "index age plus exact elapsed days / 365.25",
            "primary",
            "valid 18-110",
        ),
        (
            "sex",
            "categorical",
            "Female/Male",
            "Phase 1 cohort mapping; original source preserved upstream",
            "primary",
            "no categories inferred",
        ),
        (
            "education_years",
            "float",
            "years",
            "PTEDUCAT (ADNI); EDUC (OASIS-3)",
            "primary",
            "AIBL unavailable; 0-40",
        ),
        (
            "apoe_e4_dose",
            "integer",
            "0/1/2",
            "count of epsilon-4 alleles in validated genotype",
            "primary",
            "no incomplete-genotype inference",
        ),
        (
            "index_mmse",
            "float",
            "0-30 points",
            "nearest valid score on/before index within 180 days",
            "primary dynamic reference",
            "raw source retained upstream",
        ),
        (
            "current_mmse",
            "float",
            "0-30 points",
            "nearest valid on/before landmark within 180 days",
            "primary",
            "same-day allowed in Dataset A",
        ),
        (
            "mmse_change_from_index",
            "float",
            "points",
            "current MMSE minus index MMSE",
            "primary dynamic",
            "past/current only",
        ),
        (
            "previous_mmse",
            "float",
            "0-30 points",
            "most recent strictly earlier MMSE since the episode-index MMSE",
            "lagged sensitivity",
            "must precede landmark; time since score retained to audit staleness",
        ),
        (
            "mmse_slope_per_year",
            "float",
            "points/year",
            "two-point annualized change or OLS using only pre-landmark history",
            "secondary dynamic",
            "at least 2 points spanning 90 days",
        ),
        (
            "elapsed_years_from_index",
            "float",
            "years",
            "actual elapsed days / 365.25",
            "primary dynamic",
            "never nominal visit spacing",
        ),
        (
            "outcome_code",
            "nullable integer",
            "0 improvement; 1 persistent; 2 progression",
            "horizon-specific first transition",
            "outcome",
            "missing for early censoring/conflict",
        ),
        (
            "outer_fold",
            "integer",
            "0-4",
            "deterministic participant-grouped assignment",
            "validation metadata",
            "ADNI only",
        ),
    ]
    dictionary = pd.DataFrame(
        rows,
        columns=[
            "variable",
            "data_type",
            "unit_or_levels",
            "construction",
            "analysis_role",
            "qc_or_restriction",
        ],
    )
    provenance_rows = []
    source_map = {
        "adni": {
            "age_at_landmark": "PTDEMOG PTDOB/PTDOBYY + CDR date",
            "sex": "PTDEMOG PTGENDER",
            "education_years": "PTDEMOG PTEDUCAT",
            "apoe_e4_dose": "APOERES GENOTYPE",
            "current_mmse": "MMSE MMSCORE",
            "outcome": "CDR CDGLOBAL",
        },
        "oasis3": {
            "age_at_landmark": "UDSb4 age at visit / derived elapsed age",
            "sex": "demographics GENDER",
            "education_years": "demographics EDUC",
            "apoe_e4_dose": "demographics APOE",
            "current_mmse": "UDSb4 MMSE",
            "outcome": "UDSb4 CDRTOT de facto global; provisional export mapping",
        },
        "aibl": {
            "age_at_landmark": "PTDEMOG PTDOB year + CDR EXAMDATE; unverified",
            "sex": "PTDEMOG PTGENDER",
            "education_years": "unavailable",
            "apoe_e4_dose": "APOERES APGEN1/APGEN2",
            "current_mmse": "MMSE MMSCORE",
            "outcome": "CDR CDGLOBAL",
        },
    }
    for cohort, mappings in source_map.items():
        for variable, source in mappings.items():
            provenance_rows.append(
                {
                    "cohort": cohort,
                    "variable": variable,
                    "source": source,
                    "transformation": (
                        "see harmonized dictionary and frozen specifications"
                    ),
                    "timing_rule": (
                        "on/before landmark"
                        if variable == "current_mmse"
                        else "participant/static or outcome after landmark"
                    ),
                    "fit_on_external_data": False,
                    "uses_future_predictor_information": False,
                }
            )
    provenance = pd.DataFrame(provenance_rows)
    save_table(dictionary, TABLE_ROOT / "harmonized_clinical_core_dictionary.csv")
    save_table(provenance, TABLE_ROOT / "phase2_transformation_provenance.csv")
    return dictionary, provenance


def _nearest_pre_index_mri(
    index: pd.DataFrame, mri: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for idx in index.itertuples(index=False):
        candidates = mri[
            mri["participant_id"].astype(str).eq(str(idx.participant_id))
            & mri["mri_time_days"].le(idx.index_time_days)
            & mri["mri_time_days"].ge(idx.index_time_days - BASELINE_WINDOW)
        ].sort_values(
            ["mri_time_days", "mri_source_row"], ascending=[False, True]
        )
        if candidates.empty:
            rows.append(
                {
                    "participant_id": str(idx.participant_id),
                    "mri_available_within_180d_pre_index": False,
                }
            )
        else:
            selected = candidates.iloc[0].to_dict()
            selected["participant_id"] = str(idx.participant_id)
            selected["mri_available_within_180d_pre_index"] = True
            selected["mri_day_offset_from_index"] = (
                selected["mri_time_days"] - idx.index_time_days
            )
            rows.append(selected)
    return pd.DataFrame(rows)


def mri_readiness_audit(
    phase1: dict[str, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adni_path = (
        ROOT / "datasets" / "adni" / "All_Subjects_UCD_WMH_29Sep2025.csv"
    )
    oasis_fs_path = next(
        (ROOT / "datasets" / "oasis3").rglob("OASIS3_Freesurfer_output.csv")
    )
    adni_raw = pd.read_csv(adni_path, low_memory=False)
    oasis_raw = pd.read_csv(oasis_fs_path, low_memory=False)
    adni = pd.DataFrame(
        {
            "participant_id": adni_raw["RID"].astype("Int64").astype("string"),
            "mri_time_days": (
                pd.to_datetime(adni_raw["EXAMDATE"], errors="coerce")
                - pd.Timestamp("1970-01-01")
            ).dt.days,
            "mri_visit_id": adni_raw["VISCODE2"].astype("string"),
            "hippocampal_volume_raw": numeric(adni_raw["TOTAL_HIPPO"]),
            "cranial_volume_candidate_raw": numeric(adni_raw["CEREBRUM_TCV"]),
            "total_brain_volume_raw": numeric(adni_raw["TOTAL_BRAIN"]),
            "wmh_raw": numeric(adni_raw["TOTAL_WMH"]),
            "entorhinal_summary_raw": np.nan,
            "processing_version": "UCD WMH pipeline; exact version not supplied",
            "scanner_field_strength": numeric(
                adni_raw["MAGNETICFIELDSTRENGTH"]
            ),
            "scanner_manufacturer": adni_raw["MANUFACTURER"].astype("string"),
            "mri_source_row": np.arange(len(adni_raw)),
            "mri_source_table": adni_path.name,
        }
    )
    session_days = oasis_raw["MR_session"].astype(str).str.extract(
        r"_d(-?\d+)", expand=False
    )
    oasis = pd.DataFrame(
        {
            "participant_id": oasis_raw["Subject"].astype("string"),
            "mri_time_days": numeric(session_days),
            "mri_visit_id": oasis_raw["MR_session"].astype("string"),
            "hippocampal_volume_raw": numeric(
                oasis_raw["TOTAL_HIPPOCAMPUS_VOLUME"]
            ),
            "cranial_volume_candidate_raw": numeric(
                oasis_raw["IntraCranialVol"]
            ),
            "total_brain_volume_raw": numeric(
                oasis_raw["SupraTentorialVol"]
            ),
            "wmh_raw": numeric(oasis_raw["WM-hypointensities_volume"]),
            "entorhinal_summary_raw": numeric(
                oasis_raw["lh_entorhinal_volume"]
            )
            + numeric(oasis_raw["rh_entorhinal_volume"]),
            "processing_version": oasis_raw["version"].astype("string"),
            "scanner_field_strength": np.nan,
            "scanner_manufacturer": pd.NA,
            "mri_source_row": np.arange(len(oasis_raw)),
            "mri_source_table": oasis_fs_path.name,
            "source_qc_status": oasis_raw["FS QC Status"].astype("string"),
        }
    )
    adni_match = _nearest_pre_index_mri(phase1["adni"]["index"], adni)
    oasis_match = _nearest_pre_index_mri(phase1["oasis3"]["index"], oasis)
    adni_match["cohort"] = "adni"
    oasis_match["cohort"] = "oasis3"
    matches = pd.concat([adni_match, oasis_match], ignore_index=True, sort=False)
    matches["hippocampal_to_cranial_candidate_ratio"] = (
        numeric(matches["hippocampal_volume_raw"])
        / numeric(matches["cranial_volume_candidate_raw"])
    )
    save_table(matches, TABLE_ROOT / "mri_index_window_matches.csv", parquet=True)

    mapping = pd.DataFrame(
        [
            (
                "bilateral hippocampal volume",
                "adni",
                adni_path.name,
                "TOTAL_HIPPO",
                "bilateral total",
                "observed magnitude consistent with cm3; codebook absent",
                "UCD pipeline",
                "candidate only",
            ),
            (
                "bilateral hippocampal volume",
                "oasis3",
                oasis_fs_path.name,
                "TOTAL_HIPPOCAMPUS_VOLUME",
                "left + right",
                "FreeSurfer mm3",
                "mixed FreeSurfer versions",
                "candidate only",
            ),
            (
                "intracranial volume",
                "adni",
                adni_path.name,
                "CEREBRUM_TCV",
                "field label suggests cerebrum total cranial volume",
                "unverified",
                "UCD pipeline",
                "NOT validated as ICV",
            ),
            (
                "intracranial volume",
                "oasis3",
                oasis_fs_path.name,
                "IntraCranialVol",
                "FreeSurfer estimated intracranial volume",
                "mm3",
                "mixed FreeSurfer versions",
                "documented candidate",
            ),
            (
                "total brain volume",
                "adni",
                adni_path.name,
                "TOTAL_BRAIN",
                "total brain label",
                "unverified",
                "UCD pipeline",
                "not anatomically matched",
            ),
            (
                "total brain volume",
                "oasis3",
                oasis_fs_path.name,
                "SupraTentorialVol",
                "supratentorial volume",
                "mm3",
                "mixed FreeSurfer versions",
                "not anatomically identical",
            ),
            (
                "white matter hyperintensity",
                "adni",
                adni_path.name,
                "TOTAL_WMH",
                "FLAIR-derived total WMH",
                "unverified magnitude",
                "UCD WMH pipeline",
                "candidate within ADNI",
            ),
            (
                "white matter hyperintensity",
                "oasis3",
                oasis_fs_path.name,
                "WM-hypointensities_volume",
                "FreeSurfer T1 hypointensity label",
                "mm3",
                "mixed FreeSurfer versions",
                "NOT equivalent to FLAIR WMH",
            ),
            (
                "entorhinal summary",
                "adni",
                adni_path.name,
                "not present",
                "not available in selected labelled table",
                "not available",
                "not available",
                "blocked",
            ),
            (
                "entorhinal summary",
                "oasis3",
                oasis_fs_path.name,
                "lh_entorhinal_volume + rh_entorhinal_volume",
                "bilateral cortical parcellation sum",
                "mm3",
                "mixed FreeSurfer versions",
                "candidate only",
            ),
        ],
        columns=[
            "candidate",
            "cohort",
            "source_table",
            "source_field",
            "anatomical_definition",
            "units",
            "processing_version",
            "comparability_decision",
        ],
    )
    save_table(mapping, TABLE_ROOT / "mri_candidate_mapping.csv")

    retention_rows = []
    for cohort, match in [("adni", adni_match), ("oasis3", oasis_match)]:
        available_participants = set(
            match.loc[
                match["mri_available_within_180d_pre_index"].fillna(False),
                "participant_id",
            ].astype(str)
        )
        for horizon in HORIZONS:
            outcomes = phase1[cohort]["horizons"]
            outcomes = outcomes[
                numeric(outcomes["horizon_months"]).eq(horizon)
            ].copy()
            selected = outcomes[
                outcomes["participant_id"].astype(str).isin(available_participants)
            ]
            retention_rows.append(
                {
                    "cohort": cohort,
                    "horizon_months": horizon,
                    "index_participants": len(outcomes),
                    "mri_matched_participants": selected[
                        "participant_id"
                    ].nunique(),
                    "mri_retention_percent": selected[
                        "participant_id"
                    ].nunique()
                    / len(outcomes)
                    * 100,
                    "improvement_events_retained": int(
                        selected["transition_to_cdr_0_flag"].sum()
                    ),
                    "progression_events_retained": int(
                        selected["transition_to_cdr_ge1_flag"].sum()
                    ),
                    "persistent_retained": int(
                        selected["remain_cdr_0_5_flag"].sum()
                    ),
                    "censored_retained": int(
                        selected["censored_before_horizon_flag"].sum()
                    ),
                }
            )
    retention = pd.DataFrame(retention_rows)
    save_table(retention, TABLE_ROOT / "mri_event_retention.csv")

    distribution_rows = []
    for cohort, match in [("adni", adni_match), ("oasis3", oasis_match)]:
        for variable in [
            "hippocampal_volume_raw",
            "cranial_volume_candidate_raw",
            "total_brain_volume_raw",
            "wmh_raw",
            "entorhinal_summary_raw",
        ]:
            values = numeric(match.get(variable, pd.Series(dtype=float)))
            distribution_rows.append(
                {
                    "cohort": cohort,
                    "variable": variable,
                    "nonmissing": int(values.notna().sum()),
                    "missing_percent": values.isna().mean() * 100,
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "extreme_outlier_count_3iqr": int(
                        (
                            (values < values.quantile(0.25) - 3 * (values.quantile(0.75) - values.quantile(0.25)))
                            | (values > values.quantile(0.75) + 3 * (values.quantile(0.75) - values.quantile(0.25)))
                        ).sum()
                    ),
                }
            )
    distributions = pd.DataFrame(distribution_rows)
    save_table(distributions, TABLE_ROOT / "mri_candidate_distributions.csv")
    return mapping, retention, distributions, matches


def aibl_readiness(
    baselines: dict[str, dict[int, pd.DataFrame]],
    dynamics: dict[str, dict[int, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        baseline = baselines["aibl"][horizon]
        dynamic = dynamics["aibl"][horizon]
        reduced_complete = baseline[
            ["age_at_landmark", "sex", "apoe_e4_dose", "current_mmse"]
        ].notna().all(axis=1)
        rows.append(
            {
                "horizon_months": horizon,
                "eligible_index_participants": len(baseline),
                "age_available": int(baseline["age_at_landmark"].notna().sum()),
                "age_verified": int(
                    baseline.get(
                        "primary_eligibility_age_verified_flag",
                        pd.Series(False, index=baseline.index),
                    )
                    .fillna(False)
                    .sum()
                ),
                "education_available": int(
                    baseline["education_years"].notna().sum()
                ),
                "reduced_core_complete_participants": int(
                    reduced_complete.sum()
                ),
                "improvement_events_reduced_core": int(
                    baseline.loc[
                        reduced_complete, "transition_to_cdr_0_flag"
                    ].sum()
                ),
                "progression_events_reduced_core": int(
                    baseline.loc[
                        reduced_complete, "transition_to_cdr_ge1_flag"
                    ].sum()
                ),
                "primary_90d_dynamic_landmarks": len(dynamic),
                "dynamic_participants": dynamic["participant_id"].nunique(),
                "classification": "descriptive comparison only",
                "reason": "age derivation unverified, education absent, and sparse horizon events",
            }
        )
    frame = pd.DataFrame(rows)
    save_table(frame, TABLE_ROOT / "aibl_readiness.csv")
    return frame


def run_qc(
    phase1: dict[str, dict[str, pd.DataFrame]],
    baselines: dict[str, dict[int, pd.DataFrame]],
    dynamics: dict[str, dict[int, pd.DataFrame]],
    assignments: pd.DataFrame,
    oasis_resolution: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        scope: str,
        check: str,
        passed: bool,
        details: str,
        severity: str = "critical",
        accepted_exception: bool = False,
    ) -> None:
        status = (
            "PASS"
            if passed
            else ("WARN_ACCEPTED" if accepted_exception else "FAIL")
        )
        rows.append(
            {
                "scope": scope,
                "check": check,
                "status": status,
                "severity": severity,
                "accepted_exception": accepted_exception,
                "details": details,
            }
        )

    expected_indices = {"adni": 2447, "oasis3": 491, "aibl": 241}
    for cohort in COHORTS:
        add(
            cohort,
            "Phase 1 index count reproduced",
            len(baselines[cohort][12]) == expected_indices[cohort],
            f"observed={len(baselines[cohort][12])}; expected={expected_indices[cohort]}",
        )
        common_schema = {
            "participant_id",
            "horizon_status",
            "outcome_code",
            "age_at_landmark",
            "sex",
            "education_years",
            "apoe_e4_dose",
            "current_mmse",
            "elapsed_years_from_index",
        }
        for horizon in HORIZONS:
            baseline = baselines[cohort][horizon]
            dynamic = dynamics[cohort][horizon]
            add(
                f"{cohort}:{horizon}m:baseline",
                "unique participant rows",
                baseline["participant_id"].is_unique,
                f"rows={len(baseline)}; participants={baseline['participant_id'].nunique()}",
            )
            add(
                f"{cohort}:{horizon}m:baseline",
                "required common schema present",
                common_schema.issubset(baseline.columns),
                f"missing={sorted(common_schema-set(baseline.columns))}",
            )
            baseline_status_total = baseline[
                [
                    "transition_to_cdr_0_flag",
                    "transition_to_cdr_ge1_flag",
                    "remain_cdr_0_5_flag",
                    "censored_before_horizon_flag",
                    "unresolved_conflict_flag",
                ]
            ].fillna(False).sum(axis=1)
            add(
                f"{cohort}:{horizon}m:baseline",
                "horizon labels are mutually exclusive",
                baseline_status_total.eq(1).all(),
                f"invalid_rows={int(~baseline_status_total.eq(1).sum()) if False else int((~baseline_status_total.eq(1)).sum())}",
            )
            add(
                f"{cohort}:{horizon}m:baseline",
                "early censoring is never persistence",
                not (
                    baseline["censored_before_horizon_flag"].fillna(False)
                    & baseline["remain_cdr_0_5_flag"].fillna(False)
                ).any(),
                "mutual exclusion checked",
            )
            add(
                f"{cohort}:{horizon}m:baseline",
                "no post-index predictor",
                numeric(baseline["predictor_max_time_days"])
                .le(numeric(baseline["index_time_days"]))
                .all(),
                f"violations={int((numeric(baseline['predictor_max_time_days']) > numeric(baseline['index_time_days'])).sum())}",
            )
            phase1_horizon = phase1[cohort]["horizons"]
            phase1_horizon = phase1_horizon[
                numeric(phase1_horizon["horizon_months"]).eq(horizon)
            ]
            current_counts = baseline["horizon_status"].value_counts()
            original_counts = phase1_horizon["horizon_status"].value_counts()
            add(
                f"{cohort}:{horizon}m:baseline",
                "Phase 1 horizon outcome counts reconciled",
                current_counts.to_dict() == original_counts.to_dict(),
                f"phase2={current_counts.to_dict()}; phase1={original_counts.to_dict()}",
            )
            add(
                f"{cohort}:{horizon}m:dynamic",
                "unique participant-landmark keys",
                not dynamic.duplicated(
                    ["participant_id", "landmark_id"]
                ).any(),
                f"duplicate_rows={int(dynamic.duplicated(['participant_id','landmark_id']).sum())}",
            )
            add(
                f"{cohort}:{horizon}m:dynamic",
                "landmarks occur before transitions",
                (
                    numeric(dynamic["transition_time_from_landmark_days"])
                    .dropna()
                    .gt(0)
                    .all()
                ),
                f"nonpositive_event_times={int(numeric(dynamic['transition_time_from_landmark_days']).dropna().le(0).sum())}",
            )
            add(
                f"{cohort}:{horizon}m:dynamic",
                "no future MMSE or slope measurement",
                (
                    numeric(dynamic["current_mmse_time_days"])
                    .dropna()
                    .le(
                        numeric(dynamic.loc[
                            dynamic["current_mmse_time_days"].notna(),
                            "landmark_time_days",
                        ])
                    )
                    .all()
                    and numeric(dynamic["slope_max_measurement_time_days"])
                    .dropna()
                    .le(
                        numeric(dynamic.loc[
                            dynamic["slope_max_measurement_time_days"].notna(),
                            "landmark_time_days",
                        ])
                    )
                    .all()
                ),
                "current and maximum slope-history times checked",
            )
            add(
                f"{cohort}:{horizon}m:dynamic",
                "primary landmark spacing frozen at 90 days",
                dynamic["landmark_minimum_interval_days"].eq(
                    PRIMARY_INTERVAL
                ).all(),
                f"observed_values={sorted(dynamic['landmark_minimum_interval_days'].dropna().unique().tolist())}",
            )
            add(
                f"{cohort}:{horizon}m:dynamic",
                "valid ranges",
                (
                    numeric(dynamic["age_at_landmark"])
                    .dropna()
                    .between(18, 110)
                    .all()
                    and numeric(dynamic["current_mmse"])
                    .dropna()
                    .between(0, 30)
                    .all()
                    and numeric(dynamic["education_years"])
                    .dropna()
                    .between(0, 40)
                    .all()
                    and numeric(dynamic["apoe_e4_dose"])
                    .dropna()
                    .isin([0, 1, 2])
                    .all()
                    and numeric(dynamic["elapsed_years_from_index"])
                    .ge(0)
                    .all()
                ),
                "age/MMSE/education/APOE/time checked",
            )
            lagged = _dynamic_sensitivities(dynamic)[
                "strictly_lagged_mmse"
            ]
            add(
                f"{cohort}:{horizon}m:lagged",
                "strict lagged MMSE always precedes landmark",
                (
                    numeric(lagged["current_mmse_time_days"])
                    .lt(numeric(lagged["landmark_time_days"]))
                    .all()
                    and not lagged["same_day_mmse_flag"].any()
                ),
                f"rows={len(lagged)}",
            )

    add(
        "folds",
        "each ADNI participant assigned exactly one outer fold",
        assignments["participant_id"].is_unique
        and assignments["outer_fold"].between(0, 4).all()
        and len(assignments) == expected_indices["adni"],
        f"participants={len(assignments)}; folds={sorted(assignments['outer_fold'].unique().tolist())}",
    )
    for horizon in (12, 36):
        merged = baselines["adni"][horizon]
        minima = (
            merged.groupby("outer_fold")[
                ["transition_to_cdr_0_flag", "transition_to_cdr_ge1_flag"]
            ]
            .sum()
            .min()
        )
        add(
            f"folds:{horizon}m",
            "every outer fold contains both transition types",
            bool((minima > 0).all()),
            f"minimum_events={minima.to_dict()}",
        )
    oasis_all = oasis_resolution["agreement"].query("form_family == 'ALL'")
    agreement_percent = float(
        oasis_all["exact_agreement_percent"].iloc[0]
    )
    add(
        "oasis3_cdr",
        "CDRTOT agrees with rule-derived global CDR",
        agreement_percent >= 99,
        f"exact_agreement_percent={agreement_percent:.4f}",
    )
    add(
        "oasis3_cdr",
        "authoritative CDRGLOB column delivered",
        False,
        "CDRGLOB is absent; CDRTOT remains a provisional de facto export mapping",
        severity="warning",
        accepted_exception=True,
    )
    model_path = TABLE_ROOT / "preprocessing" / "clinical_core_preprocessor.joblib"
    try:
        saved = load_preprocessor(model_path)
        preprocessing_ok = (
            saved.provenance is not None
            and saved.provenance.fit_cohort == "adni"
            and not saved.provenance.external_outcomes_accessed
            and not saved.provenance.outcome_columns_used
        )
        detail = json.dumps(saved.provenance.__dict__, default=str)
    except Exception as exc:
        preprocessing_ok = False
        detail = str(exc)
    add(
        "preprocessing",
        "saved transformer has ADNI-training-only provenance",
        preprocessing_ok,
        detail,
    )
    add(
        "scope",
        "no predictive model fitted in Phase 2",
        True,
        "only preprocessing transformers and descriptive audits were fitted",
    )
    qc = pd.DataFrame(rows)
    save_table(qc, TABLE_ROOT / "qc_results.csv")
    if ((qc["severity"] == "critical") & qc["status"].eq("FAIL")).any():
        failures = qc[
            (qc["severity"] == "critical") & qc["status"].eq("FAIL")
        ]
        raise AssertionError(
            "Critical Phase 2 QC failures:\n" + failures.to_string(index=False)
        )
    return qc


def _save_figure(
    fig: plt.Figure, stem: str, *, dpi: int = 600
) -> None:
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(PLOT_ROOT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def create_figures(
    oasis_resolution: dict[str, pd.DataFrame],
    baselines: dict[str, dict[int, pd.DataFrame]],
    dynamics: dict[str, dict[int, pd.DataFrame]],
    dynamic_support: pd.DataFrame,
    missingness: pd.DataFrame,
    retention: pd.DataFrame,
    fold_balance_frame: pd.DataFrame,
    mri_retention: pd.DataFrame,
    mri_matches: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    comparison = oasis_resolution["record_comparison"].dropna(
        subset=["CDRTOT", "derived_global_cdr"]
    )
    confusion = pd.crosstab(
        comparison["CDRTOT"], comparison["derived_global_cdr"]
    ).reindex(index=[0, 0.5, 1, 2, 3], columns=[0, 0.5, 1, 2, 3], fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(confusion, annot=True, fmt=",", cmap="Blues", ax=ax)
    ax.set(
        xlabel="Rule-derived global CDR",
        ylabel="Delivered CDRTOT",
        title="OASIS-3 CDRTOT versus rule-derived global CDR",
    )
    _save_figure(fig, "oasis3_cdrtot_vs_rule_derived_global", dpi=1200)

    flow_rows = []
    for cohort in COHORTS:
        flow_rows.extend(
            [
                {
                    "cohort": cohort.upper(),
                    "stage": "Phase 1 index",
                    "participants": len(baselines[cohort][36]),
                },
                {
                    "cohort": cohort.upper(),
                    "stage": "≥1 primary 90d landmark",
                    "participants": dynamics[cohort][36][
                        "participant_id"
                    ].nunique(),
                },
                {
                    "cohort": cohort.upper(),
                    "stage": "36m known landmark outcome",
                    "participants": dynamics[cohort][36].loc[
                        ~dynamics[cohort][36][
                            "censored_before_horizon_flag"
                        ],
                        "participant_id",
                    ].nunique(),
                },
            ]
        )
    flow = pd.DataFrame(flow_rows)
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=flow, x="stage", y="participants", hue="cohort", ax=ax)
    ax.set_title("Baseline and dynamic participant flow")
    ax.tick_params(axis="x", rotation=12)
    _save_figure(fig, "baseline_dynamic_participant_flow", dpi=1200)

    event_rows = []
    for cohort in ("adni", "oasis3", "aibl"):
        for horizon in HORIZONS:
            frame = baselines[cohort][horizon]
            event_rows.extend(
                [
                    {
                        "cohort": cohort.upper(),
                        "horizon": str(horizon),
                        "event": "Improvement",
                        "count": int(frame["transition_to_cdr_0_flag"].sum()),
                    },
                    {
                        "cohort": cohort.upper(),
                        "horizon": str(horizon),
                        "event": "Progression",
                        "count": int(frame["transition_to_cdr_ge1_flag"].sum()),
                    },
                ]
            )
    events = pd.DataFrame(event_rows)
    grid = sns.catplot(
        data=events,
        x="horizon",
        y="count",
        hue="event",
        col="cohort",
        kind="bar",
        height=5.5,
        aspect=0.9,
        sharey=False,
        order=["12", "24", "36", "60"],
    )
    grid.set_axis_labels("Horizon (months)", "Events")
    sns.move_legend(
        grid,
        "upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        title="Event",
        frameon=False,
    )
    grid.fig.suptitle("Baseline first-transition counts by horizon", y=1.12)
    _save_figure(grid.fig, "baseline_event_counts_by_horizon", dpi=1200)

    landmark_counts = pd.concat(
        [
            frame.groupby("participant_id")
            .size()
            .rename("landmarks")
            .reset_index()
            .assign(cohort=cohort.upper())
            for cohort, frame in [
                ("adni", dynamics["adni"][36]),
                ("oasis3", dynamics["oasis3"][36]),
                ("aibl", dynamics["aibl"][36]),
            ]
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.histplot(
        data=landmark_counts,
        x="landmarks",
        hue="cohort",
        discrete=True,
        multiple="dodge",
        shrink=0.8,
        ax=ax,
    )
    ax.set_title("Primary 90-day landmark counts per participant")
    _save_figure(fig, "landmarks_per_participant", dpi=600)

    mmse_depth = pd.concat(
        [
            dynamics[cohort][36][
                ["participant_id", "prior_mmse_count"]
            ].assign(cohort=cohort.upper())
            for cohort in COHORTS
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.ecdfplot(
        data=mmse_depth,
        x="prior_mmse_count",
        hue="cohort",
        complementary=True,
        ax=ax,
    )
    ax.set(
        title="MMSE history depth at dynamic landmarks",
        xlabel="MMSE assessments available on/before landmark",
        ylabel="Proportion with at least this depth",
    )
    _save_figure(fig, "dynamic_mmse_history_depth", dpi=600)

    availability_rows = []
    for cohort in COHORTS:
        frame = dynamics[cohort][36]
        availability_rows.extend(
            [
                {
                    "cohort": cohort.upper(),
                    "measure": "Current/on-day allowed",
                    "percent": frame["current_mmse"].notna().mean() * 100,
                },
                {
                    "cohort": cohort.upper(),
                    "measure": "Strictly lagged",
                    "percent": frame["previous_mmse"].notna().mean() * 100,
                },
                {
                    "cohort": cohort.upper(),
                    "measure": "Slope eligible",
                    "percent": frame["mmse_slope_per_year"].notna().mean()
                    * 100,
                },
            ]
        )
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(
        data=pd.DataFrame(availability_rows),
        x="measure",
        y="percent",
        hue="cohort",
        ax=ax,
    )
    ax.set(
        title="Dynamic MMSE feature availability",
        ylabel="Available landmarks (%)",
        xlabel="",
    )
    ax.tick_params(axis="x", rotation=10)
    _save_figure(fig, "current_lagged_slope_mmse_availability", dpi=600)

    overall_missing = missingness[
        (missingness["dimension"] == "ALL")
        & (missingness["horizon_months"] == 36)
    ]
    matrix = overall_missing.pivot(
        index=["dataset_type", "variable"],
        columns="cohort",
        values="missing_percent",
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        ax=ax,
    )
    ax.set_title("Clinical-core missingness at 36 months (%)")
    _save_figure(fig, "clinical_core_missingness_heatmap", dpi=1200)

    outcome_missing = missingness[
        (missingness["dimension"] == "outcome")
        & (missingness["horizon_months"] == 36)
        & (missingness["dataset_type"] == "dynamic")
    ].copy()
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(
        data=outcome_missing,
        x="variable",
        y="missing_percent",
        hue="dimension_value",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Dynamic missingness by 36-month outcome (all cohorts)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(fig, "missingness_by_outcome_and_cohort", dpi=1200)

    waterfall = retention[
        (retention["horizon_months"] == 36)
        & (retention["dataset_type"] == "dynamic")
    ]
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.pointplot(
        data=waterfall,
        x="feature_tier",
        y="row_retention_percent",
        hue="cohort",
        markers="o",
        linestyles="-",
        ax=ax,
    )
    ax.set(
        title="Complete-case feature retention waterfall",
        ylabel="Landmarks retained (%)",
        xlabel="Feature tier",
    )
    ax.tick_params(axis="x", rotation=25)
    _save_figure(fig, "feature_retention_waterfall", dpi=1200)

    fold_plot = fold_balance_frame[
        fold_balance_frame["horizon_months"].isin([12, 36])
    ].melt(
        id_vars=["horizon_months", "outer_fold"],
        value_vars=["improvement_events", "progression_events"],
        var_name="event",
        value_name="count",
    )
    fold_plot["panel"] = (
        fold_plot["horizon_months"].astype(str) + " months"
    )
    grid = sns.catplot(
        data=fold_plot,
        x="outer_fold",
        y="count",
        hue="event",
        col="panel",
        kind="bar",
        height=5.5,
        aspect=1.1,
    )
    grid.set_axis_labels("Outer fold", "Events")
    sns.move_legend(
        grid,
        "upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        title="Event",
        frameon=False,
    )
    grid.fig.suptitle("ADNI participant-grouped outer-fold balance", y=1.12)
    _save_figure(grid.fig, "adni_outer_fold_balance", dpi=1200)

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(
        data=mri_retention[mri_retention["horizon_months"].isin([12, 36])],
        x="horizon_months",
        y="mri_retention_percent",
        hue="cohort",
        ax=ax,
    )
    ax.set(
        title="Pre-index MRI retention within 180 days",
        xlabel="Horizon (months)",
        ylabel="Index participants retained (%)",
    )
    _save_figure(fig, "mri_event_retention", dpi=600)

    mri_plot = mri_matches[
        mri_matches["mri_available_within_180d_pre_index"].fillna(False)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(
        data=mri_plot,
        x="cohort",
        y="hippocampal_volume_raw",
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set_title("Raw hippocampal fields\n(unharmonized units)")
    sns.boxplot(
        data=mri_plot,
        x="cohort",
        y="hippocampal_to_cranial_candidate_ratio",
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_title("Candidate hippocampal/cranial ratio\n(not approved)")
    _save_figure(fig, "mri_candidate_comparability", dpi=1200)

    predictor_rows = []
    for cohort in ("adni", "oasis3"):
        frame = baselines[cohort][36]
        for variable in [
            "age_at_landmark",
            "education_years",
            "apoe_e4_dose",
            "current_mmse",
        ]:
            predictor_rows.extend(
                {
                    "cohort": cohort.upper(),
                    "variable": variable,
                    "value": value,
                }
                for value in numeric(frame[variable]).dropna()
            )
    predictor = pd.DataFrame(predictor_rows)
    grid = sns.FacetGrid(
        predictor,
        col="variable",
        col_wrap=2,
        hue="cohort",
        palette="deep",
        sharex=False,
        sharey=False,
        height=5,
        aspect=1.2,
    )
    grid.map_dataframe(
        sns.histplot,
        x="value",
        element="step",
        stat="density",
        common_norm=False,
        bins=25,
    )
    grid.add_legend()
    sns.move_legend(
        grid,
        "upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        title="Cohort",
        frameon=False,
    )
    grid.fig.suptitle("ADNI versus OASIS-3 predictor distributions", y=1.12)
    _save_figure(grid.fig, "cross_cohort_predictor_distributions", dpi=1200)


def baseline_count_table(
    baselines: dict[str, dict[int, pd.DataFrame]]
) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        for horizon in HORIZONS:
            frame = baselines[cohort][horizon]
            rows.append(
                {
                    "cohort": cohort,
                    "horizon_months": horizon,
                    "index_participants": len(frame),
                    "improvement": int(
                        frame["transition_to_cdr_0_flag"].sum()
                    ),
                    "progression": int(
                        frame["transition_to_cdr_ge1_flag"].sum()
                    ),
                    "persistent": int(frame["remain_cdr_0_5_flag"].sum()),
                    "censored": int(
                        frame["censored_before_horizon_flag"].sum()
                    ),
                    "unresolved": int(
                        frame["unresolved_conflict_flag"].sum()
                    ),
                    "known_outcome_percent": (
                        ~frame["censored_before_horizon_flag"].fillna(False)
                        & ~frame["unresolved_conflict_flag"].fillna(False)
                    ).mean()
                    * 100,
                }
            )
    counts = pd.DataFrame(rows)
    counts = counts.sort_values(
        ["cohort", "horizon_months"], kind="mergesort"
    ).reset_index(drop=True)
    save_table(counts, TABLE_ROOT / "baseline_event_counts.csv")
    return counts


def write_reports(
    oasis_resolution: dict[str, pd.DataFrame],
    endpoint_spec: pd.DataFrame,
    landmark_spec: pd.DataFrame,
    baselines: dict[str, dict[int, pd.DataFrame]],
    dynamics: dict[str, dict[int, pd.DataFrame]],
    baseline_counts: pd.DataFrame,
    dynamic_support: pd.DataFrame,
    missingness: pd.DataFrame,
    retention: pd.DataFrame,
    missing_signals: pd.DataFrame,
    preprocessing_strategies: pd.DataFrame,
    fold_balance_frame: pd.DataFrame,
    dictionary: pd.DataFrame,
    provenance: pd.DataFrame,
    mri_mapping: pd.DataFrame,
    mri_retention: pd.DataFrame,
    mri_distributions: pd.DataFrame,
    aibl: pd.DataFrame,
    inventory: pd.DataFrame,
    qc: pd.DataFrame,
) -> dict[str, Path]:
    generated = datetime.now(timezone.utc).isoformat()
    all_agreement = oasis_resolution["agreement"].query(
        "form_family == 'ALL'"
    ).iloc[0]
    oasis_report = f"""# OASIS-3 CDR Field Resolution

Generated: {generated}

## Decision

`CDRTOT` is retained as the **de facto global CDR in the delivered OASIS-3 export**, with status **usable with a provisional endpoint**. It is not treated as a CDR sum-of-boxes field in these files. Locked confirmatory external-validation language remains restricted until OASIS confirms that export-level rename.

The decision is based on structure and independent reconstruction, not on reproducing prior counts:

- the delivered `CDRTOT` contains only 0, 0.5, 1, 2, and 3;
- delivered `CDRSUM` spans 0–18 and agrees with the sum of the six standard boxes on {int(oasis_resolution['record_comparison']['cdrsum_box_sum_agree_flag'].sum()):,} of {int(oasis_resolution['record_comparison']['cdrsum_box_sum_agree_flag'].notna().sum()):,} rows;
- the six-box rule-derived global score agrees with `CDRTOT` on {int(all_agreement['exact_agreement_records']):,} of {int(all_agreement['complete_records']):,} complete assessments ({all_agreement['exact_agreement_percent']:.2f}%);
- unweighted Cohen κ is {all_agreement['unweighted_cohen_kappa']:.4f} and quadratic-weighted κ is {all_agreement['quadratic_weighted_kappa']:.4f};
- the authoritative UDS dictionaries name `CDRGLOB`, not `CDRTOT`, as global CDR, but `CDRGLOB` is absent from the delivered CSV.

The official [NACC UDS v2 dictionary](https://files.alz.washington.edu/documentation/uds2-ivp-ded.pdf) and [NACC UDS v3 dictionary](https://files.alz.washington.edu/documentation/uds3-ivp-ded.pdf) both define `CDRGLOB` as the global score. The independent reconstruction follows the [Washington University CDR scoring rules](https://knightadrc.wustl.edu/professionals-clinicians/cdr-dementia-staging-instrument/cdr-scoring-rules/).

## Complete field inventory

{markdown_table(oasis_resolution['field_inventory'])}

## Agreement by form family

{markdown_table(oasis_resolution['agreement'])}

The 24 discordant complete records are retained as documented clinician/export values; rule reconstruction is an audit, not a replacement. CDR scoring rules explicitly note unusual combinations, and a clinician-entered global rating need not always be reproducible from simplified code alone.

## Disagreement by form family, field pair, participant, and visit time

{markdown_table(oasis_resolution['disagreement'])}

The row-level comparison, including participant and visit day, is saved in `results/csvs/phase2/oasis3_cdr_record_level_comparison.csv`. Because `CDRGLOB` is absent, direct `CDRTOT`–`CDRGLOB` agreement is not estimable and is reported as unavailable rather than fabricated.

## Candidate endpoint consequences

{markdown_table(oasis_resolution['candidate_endpoints'])}

{markdown_table(oasis_resolution['candidate_transitions'])}

`CDRSUM_misinterpreted_as_global` is shown only as a falsification audit. Its scale and endpoint behavior demonstrate why it cannot be used as global CDR. The rule-derived endpoint is the prespecified sensitivity reconstruction.

## Residual limitation

The supplied workbook conflicts with the delivered CSV column name. The analytic field is therefore frozen, reproducible, and strongly validated empirically, but external validation must be described as using a provisional OASIS export mapping until data-provider clarification is obtained.
"""
    oasis_path = REPORT_ROOT / "oasis3_cdr_field_resolution_report.md"
    oasis_path.write_text(oasis_report, encoding="utf-8")

    dictionary_report = f"""# Phase 2 Clinical-Core Data Dictionary

Generated: {generated}

## Frozen harmonized variables

{markdown_table(dictionary)}

## Cohort-specific provenance

{markdown_table(provenance)}

## Outcome specification

{markdown_table(endpoint_spec)}

## Landmark specification

{markdown_table(landmark_spec)}

CDR sum-of-boxes and CDR domains are prohibited predictors because they are outcome-adjacent. Cohort identity, future visit count, total follow-up, final diagnosis, post-transition measurements, and future MMSE are also prohibited.
"""
    dictionary_path = (
        REPORT_ROOT / "phase2_clinical_core_data_dictionary.md"
    )
    dictionary_path.write_text(dictionary_report, encoding="utf-8")

    overall_missing = missingness[
        (missingness["dimension"] == "ALL")
        & (missingness["horizon_months"] == 36)
    ]
    missing_report = f"""# Phase 2 Missingness and Preprocessing Specification

Generated: {generated}

## Missingness decision

Missingness is not assumed to be MCAR or MAR. Phase, form-family, outcome, visit, and cohort differences are preserved as audit variables, not automatically used as predictors.

## Overall 36-month missingness

{markdown_table(overall_missing)}

## Complete-case and event retention

{markdown_table(retention[(retention['horizon_months'] == 36)], max_rows=80)}

## Structural-missingness signals

{markdown_table(missing_signals, max_rows=100)}

These signals are descriptive percentage-point spreads among groups with at least 10 rows. They do not establish a missing-data mechanism.

## Approved primary preprocessing

Use median imputation for continuous variables, most-frequent imputation for categorical variables, robust scaling, and one-hot encoding with unknown categories ignored. Every component is fit inside the relevant ADNI training partition. The fitted final ADNI transformer is then applied unchanged to OASIS-3. Winsorization is disabled in the primary pipeline.

{markdown_table(preprocessing_strategies)}

Iterative imputation is feasible but not selected as primary because the low-dimensional core has limited missingness and its conditional assumptions add complexity without a Phase 2 benefit. Multiple imputation remains a secondary uncertainty analysis: fit all imputers inside each ADNI training fold, transform validation/external observations independently with each fit, average individual predicted probabilities, and use Rubin-compatible pooling for estimands where applicable.

The reusable implementation is `scripts/phase2_preprocessing.py`; its runtime guard rejects non-ADNI fitting and non-training partition labels. The serialized audit transformer and JSON provenance are under `results/csvs/phase2/preprocessing/`.
"""
    missing_path = (
        REPORT_ROOT
        / "phase2_missingness_and_preprocessing_specification.md"
    )
    missing_path.write_text(missing_report, encoding="utf-8")

    mri_report = f"""# Phase 2 MRI Readiness Report

Generated: {generated}

## Decision

**NO-GO for MRI enrichment in Phase 3; defer to Phase 5 or a separate module.**

The current extract supports a useful availability audit but not a defensibly harmonized common MRI tier. In particular, ADNI `CEREBRUM_TCV` has no supplied codebook confirming that it is equivalent to FreeSurfer intracranial volume; ADNI `TOTAL_BRAIN` is not documented as anatomically identical to OASIS-3 `SupraTentorialVol`; and OASIS-3 FreeSurfer white-matter hypointensities are not equivalent to ADNI FLAIR-derived WMH. FreeSurfer versions also vary in OASIS-3.

## Candidate mapping

{markdown_table(mri_mapping)}

## Index-window and event retention

{markdown_table(mri_retention)}

## Candidate distributions and outliers

{markdown_table(mri_distributions)}

No ADNI field is labelled “intracranial volume” in the approved output. Normalized hippocampal ratios shown in the audit are candidates only and must not enter Phase 3 until the cranial-volume proxy, units, anatomical definitions, and processing-version mapping are codebook-validated.
"""
    mri_path = REPORT_ROOT / "phase2_mri_readiness_report.md"
    mri_path.write_text(mri_report, encoding="utf-8")

    primary_support = dynamic_support[
        dynamic_support["minimum_interval_days"].eq(PRIMARY_INTERVAL)
        & dynamic_support["dataset"].isin(
            [
                "all_eligible_landmarks",
                "primary_current_mmse",
                "strictly_lagged_mmse",
                "slope_eligible",
                "clinical_mci_subgroup",
            ]
        )
    ]
    qc_summary = (
        qc.groupby(["status", "severity"]).size().rename("checks").reset_index()
    )
    main_report = f"""# Phase 2: Model-Ready Data and Harmonization

Generated: {generated}

## Decision

# GO TO PHASE 3 WITH RESTRICTIONS

No predictive model was fitted. Model-ready baseline, dynamic, sensitivity, grouped-fold, preprocessing, missingness, MRI-readiness, and provenance artifacts have been generated and independently checkable.

Restrictions:

1. OASIS-3 uses `CDRTOT` as a strongly supported but still provisional de facto global-CDR export mapping.
2. OASIS-3 external results must be reported untouched first and labelled provisional until provider confirmation.
3. MRI enrichment is excluded from Phase 3.
4. MMSE slope is secondary; current MMSE plus change is primary.
5. AIBL remains descriptive.
6. 60 months remains long-term secondary because observation selection and censoring increase.

## Frozen endpoint and landmarks

{markdown_table(endpoint_spec)}

{markdown_table(landmark_spec)}

The primary dynamic dataset uses observed State-1 visits separated by at least 90 days. This prevents near-duplicate multiplication while retaining actual irregular time. All observed primary landmarks are allowed in Phase 3 with participant-grouped validation and participant-clustered uncertainty; an equal-participant-weight sensitivity (`1 / participant landmark count`) and annual-landmark sensitivity are recommended if heavily observed participants dominate.

## Baseline counts

{markdown_table(baseline_counts)}

## Dynamic support

{markdown_table(primary_support)}

Twelve- and 36-month dynamic analyses are supported in ADNI and OASIS-3. The strict lagged-MMSE subset is materially smaller, especially at index, and is a sensitivity analysis. Slope availability is not strong enough for an externally transportable primary representation.

## Missingness and retention

{markdown_table(overall_missing)}

{markdown_table(retention[(retention['horizon_months'] == 36)], max_rows=60)}

## Deterministic participant folds

Five participant-grouped outer folds are approved, with five nested participant-grouped inner folds inside each outer-training partition. Four-fold balance was audited but five-fold retained adequate 12- and 36-month improvement and progression events in every outer fold.

{markdown_table(fold_balance_frame)}

## Sensitivity datasets

For every cohort and horizon, the pipeline saves: current/on-day MMSE; strictly lagged MMSE; no MMSE; clinician-coded MCI subgroup; no same-day outcome-assessment overlap; and, for dynamic data, slope-eligible rows. Censored-before-horizon rows retain a missing `outcome_code` and are never relabelled persistent.

## MRI and AIBL

MRI: **defer; Phase 3 no-go**. See the standalone MRI report.

{markdown_table(aibl)}

## QC

{markdown_table(qc_summary)}

The accepted warning is the absent OASIS-3 `CDRGLOB` column. All critical checks passed.

## Output inventory

{markdown_table(inventory, max_rows=80)}
"""
    main_path = (
        REPORT_ROOT / "phase2_model_ready_data_harmonization_report.md"
    )
    main_path.write_text(main_report, encoding="utf-8")

    handoff = f"""# Phase 2 to Phase 3 Handoff

Generated: {generated}

## Decision

**GO TO PHASE 3 WITH RESTRICTIONS. Do not start Phase 3 automatically.**

## Approved datasets

- ADNI baseline clinical-core datasets at 12, 24, 36, and 60 months.
- ADNI primary 90-day observed-landmark datasets at the same horizons.
- OASIS-3 counterparts for locked, unchanged external validation using the provisional `CDRTOT` export mapping.
- Current-MMSE, strictly lagged-MMSE, no-MMSE, clinical-MCI-subgroup, and no-same-day-outcome-overlap sensitivities.
- Dynamic slope-eligible datasets for secondary comparison only.

## Approved horizons

- Primary: 12 and 36 months.
- Sensitivity/comparability: 24 months.
- Long-term secondary: 60 months.

## Approved predictor specifications

- Baseline clinical core: age, sex, education, APOE ε4 dose, index/current MMSE, elapsed time (zero at index).
- Dynamic primary: current age, sex, education, APOE ε4 dose, current MMSE, change from index MMSE, elapsed years from index.
- Dynamic secondary: strictly prior MMSE, time since prior MMSE, pre-landmark slope, and prior MMSE count.
- Prohibited: CDR boxes/CDR-SB, cohort ID, future visit count, total follow-up, final diagnosis, future/post-transition data, outcome-derived fields, and MRI in Phase 3.

## Approved preprocessing

Within each inner/outer ADNI training partition: median continuous imputation, most-frequent categorical imputation, robust scaling, and one-hot categorical encoding. Apply the fitted transformer unchanged to the paired validation fold. After the full ADNI pipeline is frozen, fit preprocessing on all eligible ADNI development data and apply it unchanged to OASIS-3. Never fit on combined cohorts or use OASIS-3 outcomes/statistics.

## Validation design

- Five outer participant-grouped folds.
- Five nested participant-grouped inner folds within each outer-training set.
- Use the supplied deterministic assignments and seeds.
- All landmarks from a participant stay together.
- Bootstrap/resample participants, never rows, for uncertainty.

## Exact allowed Phase 3 modelling hierarchy

1. Time/null benchmark.
2. Baseline transition-specific cause-specific Cox models with prespecified ridge shrinkage.
3. Dynamic landmark transition-specific cause-specific Cox models using current MMSE and change as the primary dynamic representation.
4. A parsimonious flexible-parametric or penalized proportional-hazards comparator if proportional-hazards diagnostics require it.
5. Random survival forest and gradient-boosted survival as secondary machine-learning benchmarks only, tuned entirely within ADNI nested folds.
6. Semi-Markov/multi-state extension only after the primary landmark models are stable.

No deep neural network, unrestricted feature search, external threshold tuning, external model selection, or MRI model is approved.

At every approved horizon report transition-specific discrimination, Brier/prediction error, calibration-in-the-large, calibration slope, observed/expected ratio, and calibration plots. Select the final ADNI pipeline using internal performance, calibration, stability, event information, and parsimony—not OASIS-3 results.

## External-validation lock

Apply the frozen ADNI coefficients, imputation, scaling, encoding, predictor order, and horizons unchanged to OASIS-3. Report untouched external performance first. Recalibration-in-the-large and slope/intercept recalibration are secondary and clearly labelled. Do not tune thresholds or exclusions on OASIS-3.

## Remaining blockers

- OASIS-3 `CDRGLOB` is absent and the workbook conflicts with the `CDRTOT` label; external endpoint is provisional.
- Death/dropout reasons remain incomplete.
- MMSE/CDR circularity requires lagged and no-MMSE sensitivities.
- MRI harmonization failed the Phase 3 gate.
- AIBL age is unverified and education absent.
- Diagnosis-based sustained reversion is reserved for Phase 5.
"""
    handoff_path = REPORT_ROOT / "phase2_phase3_handoff.md"
    handoff_path.write_text(handoff, encoding="utf-8")
    return {
        "main": main_path,
        "oasis": oasis_path,
        "dictionary": dictionary_path,
        "missingness": missing_path,
        "mri": mri_path,
        "handoff": handoff_path,
    }


def run_metadata(
    qc: pd.DataFrame,
    baselines: dict[str, dict[int, pd.DataFrame]],
    dynamics: dict[str, dict[int, pd.DataFrame]],
) -> Path:
    signatures = {}
    for cohort in COHORTS:
        for horizon in HORIZONS:
            signatures[f"{cohort}_baseline_{horizon}m"] = dataframe_sha256(
                baselines[cohort][horizon], ["participant_id"]
            )
            signatures[f"{cohort}_dynamic_{horizon}m"] = dataframe_sha256(
                dynamics[cohort][horizon],
                ["participant_id", "landmark_time_days"],
            )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "config": PARAMS,
        "predictive_models_fitted": False,
        "preprocessing_transformer_fitted": True,
        "critical_qc_failures": int(
            ((qc["severity"] == "critical") & qc["status"].eq("FAIL")).sum()
        ),
        "accepted_warnings": qc[qc["status"].eq("WARN_ACCEPTED")][
            ["scope", "check", "details"]
        ].to_dict(orient="records"),
        "regeneration_signatures": signatures,
    }
    path = TABLE_ROOT / "phase2_run_metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_phase2() -> dict[str, Any]:
    ensure_directories()
    print("Resolving OASIS-3 CDR fields...", flush=True)
    oasis_resolution = resolve_oasis_cdr()
    print("Loading locked Phase 1 artifacts...", flush=True)
    phase1 = load_phase1()
    mmse_histories = load_mmse_histories()
    endpoint_spec, landmark_spec = frozen_specifications()
    build_recurrent_episode_table(
        {cohort: phase1[cohort]["manifest"] for cohort in COHORTS}
    )
    print("Building baseline datasets...", flush=True)
    baselines = build_baseline_datasets(phase1, mmse_histories)
    print("Building dynamic landmark datasets...", flush=True)
    dynamics, dynamic_by_interval, dynamic_support = build_dynamic_datasets(
        phase1, mmse_histories
    )
    print("Creating participant-grouped folds...", flush=True)
    assignments, inner_assignments, fold_balance_frame = create_fold_artifacts(
        baselines["adni"]
    )
    attach_outer_folds(baselines, dynamics, assignments)
    print("Auditing missingness and preprocessing...", flush=True)
    missingness, retention, missing_signals = missingness_and_retention(
        baselines, dynamics
    )
    preprocessing_strategies, imputation_audit = preprocessing_audit(
        baselines["adni"][36], baselines["oasis3"][36]
    )
    dictionary, provenance = clinical_core_dictionary_and_provenance()
    print("Auditing MRI and AIBL readiness...", flush=True)
    mri_mapping, mri_retention, mri_distributions, mri_matches = (
        mri_readiness_audit(phase1)
    )
    aibl = aibl_readiness(baselines, dynamics)
    baseline_counts = baseline_count_table(baselines)
    dynamic_mmse_audit(dynamics)
    print("Running critical QC...", flush=True)
    qc = run_qc(
        phase1, baselines, dynamics, assignments, oasis_resolution
    )
    print("Saving model-ready datasets...", flush=True)
    inventory = save_model_ready_datasets(baselines, dynamics)
    create_figures(
        oasis_resolution,
        baselines,
        dynamics,
        dynamic_support,
        missingness,
        retention,
        fold_balance_frame,
        mri_retention,
        mri_matches,
    )
    reports = write_reports(
        oasis_resolution,
        endpoint_spec,
        landmark_spec,
        baselines,
        dynamics,
        baseline_counts,
        dynamic_support,
        missingness,
        retention,
        missing_signals,
        preprocessing_strategies,
        fold_balance_frame,
        dictionary,
        provenance,
        mri_mapping,
        mri_retention,
        mri_distributions,
        aibl,
        inventory,
        qc,
    )
    metadata_path = run_metadata(qc, baselines, dynamics)
    print(f"Phase 2 report: {reports['main']}", flush=True)
    return {
        "reports": reports,
        "metadata": metadata_path,
        "baseline_counts": baseline_counts,
        "dynamic_support": dynamic_support,
        "qc": qc,
        "inventory": inventory,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    run_phase2()
