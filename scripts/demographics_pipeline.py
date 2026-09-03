"""Reproducible demographics and data-availability audit for ADNI, OASIS-3, and AIBL.

The module recursively discovers CSV/XLSX files, profiles every table/column, builds a
cohort-specific subject-level baseline table, creates figures, and synthesizes a
cross-cohort Markdown report. Raw source files are never modified.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets"
RESULTS_ROOT = PROJECT_ROOT / "results"
REPORT_ROOT = PROJECT_ROOT / "reports" / "demographics"
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}

COHORT_LABELS = {"adni": "ADNI", "oasis3": "OASIS-3", "aibl": "AIBL"}

ID_ALIASES = [
    "RID", "PTID", "OASISID", "OASIS_ID", "Subject", "subject_id",
    "participant_id", "Subject_accession",
]
VISIT_ALIASES = [
    "VISCODE2", "VISCODE", "OASIS_session_label", "OASIS_session_id",
    "session_id", "session", "MR_session", "PET Session ID", "label",
]

DOMAIN_PATTERNS = [
    ("identifiers_timing", r"(^|_)(rid|ptid|oasisid|oasis_id|subject|subject_id|participant|siteid|viscode|session|visit|examdate|visdate|date|days_to_visit|accession|imageuid|filename|id)($|_)"),
    ("demographics", r"age|birth|dob|yob|gender|(^|_)sex|m/f|educ|race|ethni|marri|hand|language|ses|living|residen"),
    ("diagnosis_staging", r"(^|_)(dx|diag|diagnosis)|dement|mci|normcog|prob(ad)?|poss(ad)?|alz|cdr|cdglobal|cdsum|cdtot"),
    ("cognition", r"mmse|mmscore|adas|moca|memory|orient|cogn|trail|digit|logical|limm|ldel|bnt|fluency|psy|craft|mint"),
    ("mood_behavior", r"gds|depress|npi|anx|apathy|agitat|halluc|delusion|behavior|mood|psychi"),
    ("function", r"faq|functional|independ|bills|shopping|meal|travel|stove|games|remdates"),
    ("genetics", r"apoe|genotyp|genetic|(^|_)dna|(^|_)rna|mutation|(^|_)gene"),
    ("fluid_biomarkers", r"abeta|amyloid.*csf|csf.*amyloid|ptau|p-tau|(^|_)tau($|_)|csftau|biomk|plasma|serum|elecsys"),
    ("pet_amyloid_tau", r"pet_|pup|suvr|centiloid|braak|tauopathy|tracer|av45|av1451|pib|flute|amyloid|taupet"),
    ("imaging_mri", r"hippo|freesurfer|(^|_)mri|(^|_)mr_|cortex|cortical|volume|thickness|surface|wmh|white.matter|intracranial|icv|fsver"),
    ("vascular_metabolic", r"vascular|stroke|tia|cvd|hypert|diabet|chol|card|blood.pressure|bpsys|bpdias|bmi|weight|height"),
    ("medical_history", r"medhist|history|seiz|traum|tbi|cancer|thyroid|apnea|tobac|alcohol|parkin|health"),
    ("medications", r"(^|_)(drug|meds?|medication|anymeds)($|_)"),
    ("qc_metadata", r"(^|_)(qc|status|phase|project|version|manufacturer|model|modality|update|userdate|creator|license|software|protocol)($|_)"),
]

CONCEPT_PATTERNS = {
    "participant_identifier": r"^rid$|^ptid$|^oasisid$|^oasis_id$|^subject$|^subject_id$|participant_id",
    "visit_or_session": r"viscode|session|days_to_visit|time(months)?",
    "age": r"^age$|age.at.visit|ageatentry",
    "sex_or_gender": r"ptgender|^gender$|^m/f$|(^|_)sex$",
    "education": r"pteducat|^educ$|education",
    "race": r"race",
    "ethnicity": r"ethnic|ethcat",
    "diagnosis": r"diagnosis|dxcurren|dxnorm|dxmci|dxad|normcog|demented|mciamem|dx1$",
    "cdr_global": r"cdglobal|cdrtot",
    "cdr_sum_of_boxes": r"cdrsb|cdrsum",
    "mmse": r"mmscore|^mmse$",
    "adas_cog": r"totscore|total13|adas",
    "moca": r"mocatots",
    "depression_gds": r"gdtotal|^gds$|gdscale",
    "neuropsychiatric_npi": r"npiscore|npiq",
    "functional_faq": r"faq|bills|taxes|shopping|mealprep|remdates",
    "apoe": r"apoe|genotype|apgen",
    "amyloid_beta": r"abeta|amyloid|centiloid|pib|av45",
    "tau_or_p_tau": r"ptau|p-tau|(^|_)tau($|_)|braak|tauopathy|av1451",
    "hippocampal_volume": r"hippo|hippocampus",
    "white_matter_hyperintensity": r"wmh|white.matter.hyper",
    "intracranial_volume": r"intracranial|icv",
    "vascular_risk": r"stroke|tia|hypert|diabet|chol|cvd|bpsys|bpdias",
}

METADATA_NAME_PATTERN = re.compile(
    r"(^id$|siteid|userdate|update_stamp|filename|accession|license|creator|"
    r"software|manufacturer|serial|qc|status|project|phase|version)", re.I
)
KNOWN_MISSING_STRINGS = {"", " ", "na", "n/a", "nan", "null", "none", ".", "-4"}
SENTINEL_CANDIDATES = {-9, -4, -1, 88, 99, 888, 999, 9999}
PARSE_WARNINGS: list[dict] = []


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return value or "table"


def _safe_text(value: object, limit: int = 80) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text[:limit]


def discover_data_files(cohort: str) -> list[Path]:
    """Recursively find genuine CSV/XLS/XLSX files while skipping OS/lock artifacts."""
    root = DATASET_ROOT / cohort
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if "__MACOSX" in path.parts or path.name.startswith((".", "~")):
            continue
        files.append(path)
    return sorted(files)


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a CSV using common clinical missing tokens and an encoding fallback."""
    kwargs = dict(low_memory=False, na_values=list(KNOWN_MISSING_STRINGS), keep_default_na=True)
    encoding = "utf-8-sig"
    try:
        return pd.read_csv(path, encoding=encoding, **kwargs)
    except UnicodeDecodeError:
        encoding = "latin-1"
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except pd.errors.ParserError:
            pass
    except pd.errors.ParserError:
        pass

    # A small number of legacy exports contain isolated unquoted delimiters. Retain
    # all valid records, skip only malformed records, and write an explicit audit row.
    expected_fields = len(pd.read_csv(path, encoding=encoding, nrows=0).columns)

    def record_bad_line(fields: list[str]) -> list[str] | None:
        can_repair_flutemeta = (
            path.name.lower().startswith("aibl_flutemeta_") and len(fields) == expected_fields + 1
        )
        action = "row_repaired_unquoted_comma_in_penultimate_field" if can_repair_flutemeta else "row_skipped"
        PARSE_WARNINGS.append({
            "relative_path": str(path),
            "warning": "Malformed delimited record",
            "expected_fields": expected_fields,
            "fields_seen": len(fields),
            "action": action,
        })
        if can_repair_flutemeta:
            return fields[: expected_fields - 2] + [",".join(fields[expected_fields - 2 : -1]), fields[-1]]
        return None

    return pd.read_csv(
        path, encoding=encoding, engine="python", on_bad_lines=record_bad_line,
        na_values=list(KNOWN_MISSING_STRINGS), keep_default_na=True,
    )


def read_data_file(path: Path) -> dict[str, pd.DataFrame]:
    """Read one file. Workbooks return one named DataFrame per sheet."""
    if path.suffix.lower() == ".csv":
        return {path.stem: read_csv_robust(path)}
    return pd.read_excel(path, sheet_name=None)


def find_column(columns: Iterable[object], aliases: Iterable[str]) -> str | None:
    lookup = {str(c).casefold(): str(c) for c in columns}
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    return None


def classify_domain(column: str, file_name: str = "") -> str:
    column_text = str(column).lower()
    text = f"{file_name} {column}".lower()
    for domain, pattern in DOMAIN_PATTERNS:
        if re.search(pattern, column_text, re.I) or re.search(pattern, text, re.I):
            return domain
    return "other"


def matching_concepts(column: str, file_name: str = "") -> str:
    column_text = str(column).lower()
    text = f"{file_name} {column}".lower()
    matches = [
        name for name, pattern in CONCEPT_PATTERNS.items()
        if re.search(pattern, column_text, re.I) or re.search(pattern, text, re.I)
    ]
    return ";".join(matches)


def _potential_sentinels(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return ""
    counts = numeric[numeric.isin(SENTINEL_CANDIDATES)].value_counts().sort_index()
    return "; ".join(f"{value:g}:{count}" for value, count in counts.items())


def _is_usable(column: str, non_missing: int, unique: int) -> tuple[bool, str]:
    if non_missing == 0:
        return False, "all_missing"
    if unique <= 1:
        return False, "constant"
    if str(column).lower().startswith("unnamed:"):
        return False, "unnamed_export_artifact"
    if str(column).casefold() in {alias.casefold() for alias in ID_ALIASES}:
        return False, "participant_identifier_not_predictor"
    if METADATA_NAME_PATTERN.search(str(column)):
        return False, "identifier_or_metadata"
    return True, "candidate_feature_or_outcome"


def profile_dataframe(
    cohort: str,
    relative_path: str,
    table_name: str,
    df: pd.DataFrame,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    id_col = find_column(df.columns, ID_ALIASES)
    visit_col = find_column(df.columns, VISIT_ALIASES)
    unique_subjects = int(df[id_col].nunique(dropna=True)) if id_col else np.nan
    duplicate_keys = np.nan
    if id_col:
        key = [id_col] + ([visit_col] if visit_col and visit_col != id_col else [])
        duplicate_keys = int(df.duplicated(key, keep=False).sum())

    file_row = {
        "cohort": cohort,
        "relative_path": relative_path,
        "table_name": table_name,
        "rows": len(df),
        "columns": len(df.columns),
        "cells": int(df.shape[0] * df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(float(df.isna().mean().mean() * 100), 3) if len(df) else np.nan,
        "duplicate_rows": int(df.duplicated().sum()),
        "id_column": id_col or "",
        "visit_column": visit_col or "",
        "unique_subjects": unique_subjects,
        "duplicate_subject_visit_rows": duplicate_keys,
    }

    column_rows: list[dict] = []
    numeric_rows: list[dict] = []
    categorical_rows: list[dict] = []
    coverage_rows: list[dict] = []

    if id_col:
        visits = df.groupby(id_col, dropna=True).size()
        coverage_rows.append({
            "cohort": cohort,
            "relative_path": relative_path,
            "table_name": table_name,
            "id_column": id_col,
            "subjects": int(visits.size),
            "records": int(visits.sum()),
            "median_records_per_subject": float(visits.median()) if len(visits) else np.nan,
            "max_records_per_subject": int(visits.max()) if len(visits) else np.nan,
            "subjects_with_multiple_records": int((visits > 1).sum()),
        })

    for column in df.columns:
        series = df[column]
        non_missing = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        usable, reason = _is_usable(str(column), non_missing, unique)
        sample_values = " | ".join(_safe_text(v) for v in series.dropna().unique()[:5])
        column_rows.append({
            "cohort": cohort,
            "relative_path": relative_path,
            "table_name": table_name,
            "column": str(column),
            "dtype": str(series.dtype),
            "domain": classify_domain(str(column), relative_path),
            "concepts": matching_concepts(str(column), relative_path),
            "non_missing": non_missing,
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 3) if len(series) else np.nan,
            "unique_values": unique,
            "is_usable_candidate": usable,
            "usability_reason": reason,
            "potential_sentinel_values_and_counts": _potential_sentinels(series),
            "sample_values": sample_values,
        })

        numeric = pd.to_numeric(series, errors="coerce")
        numeric_fraction = float(numeric.notna().sum() / max(non_missing, 1))
        if non_missing and numeric_fraction >= 0.80 and numeric.notna().sum() >= 2:
            desc = numeric.describe(percentiles=[0.25, 0.5, 0.75])
            numeric_rows.append({
                "cohort": cohort, "relative_path": relative_path, "table_name": table_name,
                "column": str(column), "domain": classify_domain(str(column), relative_path),
                "count": int(desc.get("count", 0)), "mean": desc.get("mean", np.nan),
                "std": desc.get("std", np.nan), "min": desc.get("min", np.nan),
                "q25": desc.get("25%", np.nan), "median": desc.get("50%", np.nan),
                "q75": desc.get("75%", np.nan), "max": desc.get("max", np.nan),
            })
        elif non_missing and unique <= min(50, max(15, int(len(series) * 0.1))):
            counts = series.value_counts(dropna=False).head(20)
            for rank, (value, count) in enumerate(counts.items(), start=1):
                categorical_rows.append({
                    "cohort": cohort, "relative_path": relative_path, "table_name": table_name,
                    "column": str(column), "domain": classify_domain(str(column), relative_path),
                    "rank": rank, "value": _safe_text(value), "count": int(count),
                    "pct_of_rows": round(float(count / max(len(series), 1) * 100), 3),
                })
    return file_row, column_rows, numeric_rows, categorical_rows, coverage_rows


def _baseline_adni(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    preferred = next((v for k, v in frames.items() if k.lower() == "baseline_subj"), None)
    if preferred is not None:
        base = preferred.drop_duplicates("RID").copy()
        # baseline_subj's supplied Age is referenced to the 2025 export year, not
        # the participant's baseline visit. Preserve it and derive approximate
        # baseline age because only birth year (not full DOB) is available.
        if "Age" in base:
            base = base.rename(columns={"Age": "AGE_SOURCE_ORIGINAL"})
        if {"PTDOBYY", "BL_DATE"}.issubset(base.columns):
            baseline_year = pd.to_datetime(base["BL_DATE"], errors="coerce").dt.year
            base["AGE_AT_BASELINE_APPROX"] = baseline_year - pd.to_numeric(base["PTDOBYY"], errors="coerce")

        cdr = next((v for k, v in frames.items() if "all_subjects_cdr" in k.lower()), pd.DataFrame())
        if not cdr.empty:
            cdr = cdr.copy()
            baseline_flag = cdr.get("VISCODE2", pd.Series(index=cdr.index, dtype=object)).astype(str).str.lower().eq("bl")
            cdr["_baseline_priority"] = (~baseline_flag).astype(int)
            cdr["_visit_date"] = pd.to_datetime(cdr.get("VISDATE"), errors="coerce")
            cdr = cdr.sort_values(["RID", "_baseline_priority", "_visit_date"], na_position="last").drop_duplicates("RID")
            keep = [c for c in ["RID", "CDGLOBAL", "CDRSB"] if c in cdr]
            base = base.merge(cdr[keep], on="RID", how="left", validate="one_to_one")

        apoe = next((v for k, v in frames.items() if "all_subjects_apoeres" in k.lower()), pd.DataFrame())
        if not apoe.empty and "GENOTYPE" in apoe:
            apoe = apoe.dropna(subset=["GENOTYPE"]).drop_duplicates("RID")[["RID", "GENOTYPE"]]
            base = base.merge(apoe.rename(columns={"GENOTYPE": "_GENOTYPE_COMPLETE"}), on="RID", how="left", validate="one_to_one")
            if "GENOTYPE" in base:
                base["GENOTYPE"] = base["GENOTYPE"].combine_first(base["_GENOTYPE_COMPLETE"])
            else:
                base["GENOTYPE"] = base["_GENOTYPE_COMPLETE"]
            base = base.drop(columns="_GENOTYPE_COMPLETE")
        return base
    demo = next((v for k, v in frames.items() if "ptdemog" in k.lower()), pd.DataFrame())
    return demo.drop_duplicates("RID") if "RID" in demo else demo


def _baseline_oasis3(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    demo = next((v for k, v in frames.items() if k.lower() == "oasis3_demographics"), pd.DataFrame())
    cdr = next((v for k, v in frames.items() if "udsb4_cdr" in k.lower()), pd.DataFrame())
    if demo.empty:
        return pd.DataFrame()
    base = demo.drop_duplicates("OASISID").copy()
    if not cdr.empty:
        cdr = cdr.copy()
        cdr["_valid_age"] = pd.to_numeric(cdr.get("age at visit"), errors="coerce").between(18, 110)
        cdr["_distance_from_baseline"] = pd.to_numeric(cdr.get("days_to_visit"), errors="coerce").abs()
        cdr = cdr.sort_values(
            ["OASISID", "_valid_age", "_distance_from_baseline"],
            ascending=[True, False, True], na_position="last",
        ).drop_duplicates("OASISID")
        if "age at visit" in cdr:
            cdr.loc[~cdr["_valid_age"], "age at visit"] = np.nan
        keep = [c for c in ["OASISID", "OASIS_session_label", "days_to_visit", "age at visit", "MMSE", "CDRSUM", "CDRTOT", "dx1"] if c in cdr]
        base = base.merge(cdr[keep], on="OASISID", how="left", validate="one_to_one")
    return base


def _baseline_aibl(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tokens = ("ptdemog", "cdr", "mmse", "pdxconv", "apoeres")
    selected = [(k, v) for k, v in frames.items() if any(t in k.lower() for t in tokens) and "RID" in v]
    base: pd.DataFrame | None = None
    for name, frame in selected:
        part = frame.copy()
        if "VISCODE" in part:
            baseline_mask = part["VISCODE"].astype(str).str.lower().isin(["bl", "baseline", "0"])
            if baseline_mask.any():
                part = part.loc[baseline_mask]
        part = part.drop_duplicates("RID")
        keep = [c for c in part if c == "RID" or (c not in {"SITEID", "VISCODE"} and not str(c).startswith("Unnamed"))]
        part = part[keep]
        if base is None:
            base = part
        else:
            overlap = [c for c in part if c in base and c != "RID"]
            part = part.drop(columns=overlap)
            base = base.merge(part, on="RID", how="outer", validate="one_to_one")
    return base if base is not None else pd.DataFrame()


def build_baseline(cohort: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    builders = {"adni": _baseline_adni, "oasis3": _baseline_oasis3, "aibl": _baseline_aibl}
    baseline = builders[cohort](frames)
    if not baseline.empty:
        baseline.insert(0, "source_cohort", COHORT_LABELS[cohort])
    return baseline


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_bar(series: pd.Series, title: str, xlabel: str, path: Path, top: int = 20) -> None:
    values = series.dropna().astype(str).value_counts().head(top)
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(values) * 0.34)))
    sns.barplot(x=values.values, y=values.index, ax=ax, color="#3264a8")
    ax.set(title=title, xlabel=xlabel, ylabel="")
    _save_plot(fig, path)


def create_cohort_plots(
    cohort: str,
    file_inventory: pd.DataFrame,
    column_inventory: pd.DataFrame,
    baseline: pd.DataFrame,
) -> list[Path]:
    sns.set_theme(style="whitegrid", context="notebook")
    label = COHORT_LABELS[cohort]
    out = RESULTS_ROOT / "plots" / cohort
    out.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    if not file_inventory.empty:
        plot_path = out / "file_record_counts.png"
        view = file_inventory.sort_values("rows", ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(11, max(5, len(view) * 0.34)))
        sns.barplot(data=view, x="rows", y="table_name", ax=ax, color="#3c8d7b")
        ax.set(title=f"{label}: records per table", xlabel="Rows", ylabel="")
        _save_plot(fig, plot_path); created.append(plot_path)

    if not column_inventory.empty:
        plot_path = out / "highest_column_missingness.png"
        view = column_inventory.query("non_missing > 0").sort_values("missing_pct", ascending=False).head(30)
        labels = view["table_name"].astype(str) + " :: " + view["column"].astype(str)
        fig, ax = plt.subplots(figsize=(11, max(6, len(view) * 0.31)))
        sns.barplot(x=view["missing_pct"], y=labels, ax=ax, color="#c65f5f")
        ax.set(title=f"{label}: highest missingness among non-empty columns", xlabel="Missing (%)", ylabel="", xlim=(0, 100))
        _save_plot(fig, plot_path); created.append(plot_path)

        domain = column_inventory.groupby("domain").agg(columns=("column", "size"), median_missing_pct=("missing_pct", "median")).sort_values("columns")
        plot_path = out / "data_domains.png"
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=domain.reset_index(), x="columns", y="domain", ax=ax, color="#7357a6")
        ax.set(title=f"{label}: available columns by clinical domain", xlabel="Column instances", ylabel="")
        _save_plot(fig, plot_path); created.append(plot_path)

    if baseline.empty:
        return created

    miss = baseline.isna().mean().mul(100).sort_values(ascending=False).head(30)
    plot_path = out / "baseline_missingness.png"
    fig, ax = plt.subplots(figsize=(10, max(5, len(miss) * 0.3)))
    sns.barplot(x=miss.values, y=miss.index, ax=ax, color="#d39035")
    ax.set(title=f"{label}: baseline table missingness", xlabel="Missing (%)", ylabel="", xlim=(0, 100))
    _save_plot(fig, plot_path); created.append(plot_path)

    plot_specs = {
        "age": (["AGE_AT_BASELINE_APPROX", "AgeatEntry", "age at visit", "Age"], "hist"),
        "sex_gender": (["PTGENDER", "GENDER", "M/F"], "bar"),
        "education": (["PTEDUCAT", "EDUC", "Education"], "hist"),
        "diagnosis": (["DIAGNOSIS", "dx1", "DXCURREN"], "bar"),
        "cdr_global": (["CDGLOBAL", "CDRTOT"], "bar"),
        "mmse": (["MMSCORE", "MMSE"], "hist"),
        "apoe": (["GENOTYPE", "APOE", "APGEN1"], "bar"),
    }
    for concept, (aliases, kind) in plot_specs.items():
        column = find_column(baseline.columns, aliases)
        if not column or baseline[column].notna().sum() == 0:
            continue
        plot_path = out / f"baseline_{concept}.png"
        if kind == "hist":
            values = pd.to_numeric(baseline[column], errors="coerce").dropna()
            if values.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(values, kde=len(values) >= 30, ax=ax, color="#3264a8")
            ax.set(title=f"{label}: baseline {concept.replace('_', ' ')}", xlabel=column, ylabel="Participants")
            _save_plot(fig, plot_path)
        else:
            _plot_bar(baseline[column], f"{label}: baseline {concept.replace('_', ' ')}", "Participants", plot_path)
        created.append(plot_path)
    return created


def baseline_variable_summary(cohort: str, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in baseline.columns:
        series = baseline[column]
        non_missing = int(series.notna().sum())
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_fraction = numeric.notna().sum() / max(non_missing, 1)
        row = {
            "cohort": cohort, "column": column, "domain": classify_domain(column),
            "n": len(series), "non_missing": non_missing,
            "missing_pct": round(float(series.isna().mean() * 100), 3),
            "unique_values": int(series.nunique(dropna=True)), "mean": np.nan,
            "std": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan,
            "min": np.nan, "max": np.nan, "top_values": "",
        }
        if non_missing and numeric_fraction >= 0.8 and numeric.notna().sum() >= 2:
            row.update({
                "mean": numeric.mean(), "std": numeric.std(), "median": numeric.median(),
                "q25": numeric.quantile(.25), "q75": numeric.quantile(.75),
                "min": numeric.min(), "max": numeric.max(),
            })
        else:
            counts = series.value_counts(dropna=True).head(10)
            row["top_values"] = "; ".join(f"{_safe_text(v)}: {n}" for v, n in counts.items())
        rows.append(row)
    return pd.DataFrame(rows)


def run_cohort_analysis(cohort: str, verbose: bool = True) -> dict:
    """Run the complete recursive profile and write cohort CSVs/plots."""
    if cohort not in COHORT_LABELS:
        raise ValueError(f"Unknown cohort {cohort!r}; choose one of {sorted(COHORT_LABELS)}")
    PARSE_WARNINGS.clear()
    files = discover_data_files(cohort)
    csv_dir = RESULTS_ROOT / "csvs" / cohort
    csv_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "plots" / cohort).mkdir(parents=True, exist_ok=True)

    file_rows, column_rows, numeric_rows, categorical_rows, coverage_rows = [], [], [], [], []
    workbook_rows, errors = [], []
    frames: dict[str, pd.DataFrame] = {}
    if verbose:
        print(f"[{COHORT_LABELS[cohort]}] Found {len(files)} supported data files recursively.")

    for index, path in enumerate(files, start=1):
        relative = str(path.relative_to(DATASET_ROOT / cohort))
        try:
            tables = read_data_file(path)
            is_dictionary = "dictionar" in path.name.lower()
            if verbose:
                print(f"  {index:>2}/{len(files)} read {relative} ({len(tables)} table/sheet(s))")
            for sheet, df in tables.items():
                table_name = path.stem if len(tables) == 1 else f"{path.stem}__{sheet}"
                workbook_rows.append({
                    "cohort": cohort, "relative_path": relative, "sheet": sheet,
                    "rows": len(df), "columns": len(df.columns), "is_data_dictionary": is_dictionary,
                })
                if is_dictionary:
                    # The workbook is fully read and inventoried by sheet, but dictionary rows
                    # are not treated as participant observations.
                    continue
                frames[table_name] = df
                profile = profile_dataframe(cohort, relative, table_name, df)
                file_rows.append(profile[0]); column_rows.extend(profile[1])
                numeric_rows.extend(profile[2]); categorical_rows.extend(profile[3]); coverage_rows.extend(profile[4])
        except Exception as exc:
            errors.append({"cohort": cohort, "relative_path": relative, "error_type": type(exc).__name__, "message": str(exc)})
            warnings.warn(f"Could not read {path}: {exc}")

    file_inventory = pd.DataFrame(file_rows)
    column_inventory = pd.DataFrame(column_rows)
    numeric_summary = pd.DataFrame(numeric_rows)
    categorical_summary = pd.DataFrame(categorical_rows)
    subject_coverage = pd.DataFrame(coverage_rows)
    workbook_summary = pd.DataFrame(workbook_rows)
    load_errors = pd.DataFrame(errors, columns=["cohort", "relative_path", "error_type", "message"])
    parse_warnings = pd.DataFrame(
        [{"cohort": cohort, **row} for row in PARSE_WARNINGS],
        columns=["cohort", "relative_path", "warning", "expected_fields", "fields_seen", "action"],
    )
    baseline = build_baseline(cohort, frames)
    baseline_summary = baseline_variable_summary(cohort, baseline) if not baseline.empty else pd.DataFrame()
    concept_inventory = column_inventory.loc[column_inventory["concepts"].ne("")].copy() if not column_inventory.empty else pd.DataFrame()

    outputs = {
        "file_inventory.csv": file_inventory,
        "column_inventory.csv": column_inventory,
        "numeric_summary.csv": numeric_summary,
        "categorical_summary.csv": categorical_summary,
        "subject_coverage.csv": subject_coverage,
        "workbook_sheet_inventory.csv": workbook_summary,
        "load_errors.csv": load_errors,
        "parse_warnings.csv": parse_warnings,
        "baseline_demographics.csv": baseline,
        "baseline_variable_summary.csv": baseline_summary,
        "clinical_concept_inventory.csv": concept_inventory,
    }
    for name, frame in outputs.items():
        frame.to_csv(csv_dir / name, index=False)

    plots = create_cohort_plots(cohort, file_inventory, column_inventory, baseline)
    if verbose:
        print(f"[{COHORT_LABELS[cohort]}] Profiled {len(file_inventory):,} tables and {len(column_inventory):,} column instances.")
        print(f"[{COHORT_LABELS[cohort]}] Baseline analysis table: {len(baseline):,} participants x {len(baseline.columns):,} columns.")
        print(f"[{COHORT_LABELS[cohort]}] Wrote {len(outputs)} CSV outputs and {len(plots)} plots.")
        if len(load_errors):
            print(f"[{COHORT_LABELS[cohort]}] WARNING: {len(load_errors)} file(s) failed to load; see load_errors.csv.")
        else:
            print(f"[{COHORT_LABELS[cohort]}] All discovered data files loaded successfully.")
        if len(parse_warnings):
            repaired = parse_warnings["action"].str.startswith("row_repaired").sum()
            skipped = (parse_warnings["action"] == "row_skipped").sum()
            print(f"[{COHORT_LABELS[cohort]}] Handled {len(parse_warnings)} malformed row(s): {repaired} repaired, {skipped} skipped; details are in parse_warnings.csv.")
    return {
        "cohort": cohort, "files": len(files), "tables": len(file_inventory),
        "columns": len(column_inventory), "baseline_participants": len(baseline),
        "plots": [str(p.relative_to(PROJECT_ROOT)) for p in plots], "load_errors": len(load_errors),
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No data available._"
    view = frame.copy()
    if columns:
        view = view[[c for c in columns if c in view]]
    if max_rows:
        view = view.head(max_rows)
    view = view.replace({np.nan: ""})
    return view.to_markdown(index=False)


def _concept_availability(inventories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for concept in CONCEPT_PATTERNS:
        row = {"concept": concept}
        for cohort, inventory in inventories.items():
            match = inventory["concepts"].fillna("").str.split(";").apply(lambda x: concept in x)
            subset = inventory.loc[match]
            row[cohort] = int(len(subset))
            row[f"{cohort}_example"] = "; ".join(
                (subset["table_name"].astype(str) + "." + subset["column"].astype(str)).head(4)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _baseline_key_metrics(cohort: str, baseline: pd.DataFrame) -> list[dict]:
    aliases = {
        "age": ["AGE_AT_BASELINE_APPROX", "AgeatEntry", "age at visit", "Age"],
        "sex_gender": ["PTGENDER", "GENDER", "M/F"],
        "education": ["PTEDUCAT", "EDUC", "Education"],
        "diagnosis": ["DIAGNOSIS", "dx1", "DXCURREN"],
        "cdr_global": ["CDGLOBAL", "CDRTOT"],
        "mmse": ["MMSCORE", "MMSE"],
        "apoe": ["GENOTYPE", "APOE", "APGEN1"],
    }
    rows = []
    categorical_concepts = {"sex_gender", "diagnosis", "apoe"}
    valid_ranges = {"age": (18, 110), "education": (0, 30), "cdr_global": (0, 3), "mmse": (0, 30)}
    for concept, candidates in aliases.items():
        column = find_column(baseline.columns, candidates)
        if not column:
            rows.append({"cohort": COHORT_LABELS[cohort], "concept": concept, "column": "not available", "n": 0, "missing_pct": 100.0, "summary": ""})
            continue
        series = baseline[column]
        numeric = pd.to_numeric(series, errors="coerce")
        invalid_count = 0
        if concept in valid_ranges:
            low, high = valid_ranges[concept]
            valid = numeric.between(low, high)
            invalid_count = int((numeric.notna() & ~valid).sum())
            numeric = numeric.where(valid)
        if concept not in categorical_concepts and numeric.notna().sum() / max(series.notna().sum(), 1) >= .8 and numeric.notna().sum() >= 2:
            summary = f"mean {numeric.mean():.2f}; SD {numeric.std():.2f}; median {numeric.median():.2f}; range {numeric.min():.2f}–{numeric.max():.2f}"
            if invalid_count:
                summary += f"; {invalid_count} out-of-range value(s) excluded"
            metric_n = int(numeric.notna().sum())
            missing_pct = round(float((1 - metric_n / max(len(series), 1)) * 100), 2)
        else:
            summary = "; ".join(f"{_safe_text(v)}={n}" for v, n in series.value_counts().head(8).items())
            metric_n = int(series.notna().sum())
            missing_pct = round(float(series.isna().mean() * 100), 2)
        rows.append({
            "cohort": COHORT_LABELS[cohort], "concept": concept, "column": column,
            "n": metric_n, "missing_pct": missing_pct, "summary": summary,
        })
    return rows


def synthesize_report(verbose: bool = True) -> Path:
    """Combine completed cohort outputs into cross-cohort CSVs and one Markdown report."""
    inventories, files, baselines, coverage = {}, {}, {}, {}
    for cohort in COHORT_LABELS:
        base = RESULTS_ROOT / "csvs" / cohort
        required = [base / "column_inventory.csv", base / "file_inventory.csv", base / "baseline_demographics.csv", base / "subject_coverage.csv"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Run {cohort} analysis first. Missing: {missing}")
        inventories[cohort] = pd.read_csv(required[0], low_memory=False)
        files[cohort] = pd.read_csv(required[1], low_memory=False)
        baselines[cohort] = pd.read_csv(required[2], low_memory=False)
        coverage[cohort] = pd.read_csv(required[3], low_memory=False)

    combined_dir = RESULTS_ROOT / "csvs" / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    concept_matrix = _concept_availability(inventories)
    concept_matrix.to_csv(combined_dir / "cross_cohort_concept_availability.csv", index=False)
    all_key_metrics = pd.DataFrame([
        row for cohort, baseline in baselines.items() for row in _baseline_key_metrics(cohort, baseline)
    ])
    all_key_metrics.to_csv(combined_dir / "cross_cohort_baseline_metrics.csv", index=False)
    cohort_overview = pd.DataFrame([
        {
            "cohort": COHORT_LABELS[c], "tables": len(files[c]), "column_instances": len(inventories[c]),
            "distinct_column_names": inventories[c]["column"].nunique(),
            "baseline_participants": len(baselines[c]),
            "usable_candidate_columns": int(inventories[c]["is_usable_candidate"].astype(str).str.lower().eq("true").sum()),
            "median_table_missing_pct": round(float(files[c]["missing_pct"].median()), 2),
            "tables_with_subject_id": int(files[c]["id_column"].fillna("").ne("").sum()),
            "tables_with_repeated_subjects": int((coverage[c]["subjects_with_multiple_records"] > 0).sum()) if len(coverage[c]) else 0,
        } for c in COHORT_LABELS
    ])
    cohort_overview.to_csv(combined_dir / "cohort_overview.csv", index=False)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_ROOT / "demographics_report.md"
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        "# Clinical Cohort Demographics and Data-Availability Report",
        "",
        f"Generated: **{generated}**",
        "",
        "## Executive summary",
        "",
        "This report audits every readable CSV/XLS/XLSX file currently supplied for ADNI, OASIS-3, and AIBL. "
        "It describes the available participant records, demographics, clinical outcomes, biomarkers, imaging variables, missingness, and longitudinal structure. "
        "It is a planning document—not a final statistical analysis—and coded source values must be decoded against each study's data dictionary before modeling.",
        "",
        _markdown_table(cohort_overview),
        "",
        "### Immediate study-design conclusions",
        "",
        "- **Common clinical core:** participant identifier, sex/gender, diagnostic status, MMSE, CDR, and APOE-related fields are represented in all three supplied cohorts.",
        "- **External validation is feasible only after harmonization:** names, coding, visit definitions, recruitment criteria, and measurement protocols differ. A model should be developed in one cohort and tested unchanged in another only after a documented harmonization step.",
        "- **Within-cohort validation must be participant-grouped:** repeated visits occur in multiple tables. Random row splitting would leak the same participant across training and validation folds.",
        "- **AIBL limitation in this extract:** demographics are sparse and quantified MRI/PET regional measures are not present; most supplied imaging tables describe acquisition metadata. Do not claim imaging-biomarker comparability from these files alone.",
        "- **Missingness is structural as well as random:** many measures are protocol-, phase-, modality-, or visit-specific. Missingness indicators may encode study design and must not be used without checking for cohort/visit leakage.",
        "",
        "## Cross-cohort baseline demographics and key measures",
        "",
        _markdown_table(all_key_metrics),
        "",
        "The table above uses one subject-level baseline record per cohort. ADNI starts from `baseline_subj.csv`, derives approximate age at baseline from baseline year minus birth year, and adds baseline/first CDR plus the most complete APOE genotype record. "
        "OASIS-3 combines its demographics table with the valid CDR visit closest to day 0. AIBL outer-joins baseline/earliest demographic, CDR, MMSE, diagnosis, and APOE records by RID; its birth/date coding requires dictionary-assisted derivation before age is used.",
        "",
        "## Cross-cohort concept availability",
        "",
        "Counts are column instances, not participants. Multiple instruments or repeated exports can contribute more than one instance.",
        "",
        _markdown_table(concept_matrix, ["concept", "adni", "oasis3", "aibl"]),
        "",
        "Detailed examples are saved in `results/csvs/combined/cross_cohort_concept_availability.csv`.",
        "",
        "## Validation and modeling recommendations",
        "",
        "1. Define the prediction time point, outcome, and allowable predictor window before joining files. Never include post-outcome visits or future conversion information.",
        "2. Create a canonical participant/visit manifest per cohort, then join instrument tables on participant plus visit/date—not participant alone.",
        "3. Use `GroupKFold`, `StratifiedGroupKFold`, or a participant-held-out temporal split for internal validation. Keep every record from a participant in one fold.",
        "4. Reserve a whole cohort for external validation where possible. Fit preprocessing, imputation, scaling, feature selection, and thresholds using training cohorts only.",
        "5. Harmonize measurement direction and units (for example ADAS higher=worse, MMSE higher=better; MRI volumes may need intracranial-volume normalization).",
        "6. Report eligibility and complete-case attrition at each join. Use multiple imputation only when its assumptions are clinically defensible; keep a missingness audit by cohort and visit.",
        "7. Minimum adjustment set for many analyses should consider age, sex/gender, education, diagnosis/stage, APOE, site/scanner, and visit time, subject to the causal question.",
        "8. Avoid direct race/ethnicity pooling until codebooks and category mappings are reviewed. Preserve unknown/not-reported categories and report them explicitly.",
        "9. Evaluate calibration, discrimination, confidence intervals, and subgroup performance—not accuracy alone. For conversion/time-to-event work, account for censoring and time origin.",
        "",
        "## Biomarker and instrument interpretation",
        "",
        "- **Cognitive/staging:** MMSE and CDR are the strongest common measures. ADNI also supplies ADAS-Cog; OASIS-3 supplies a broad psychometric battery; AIBL supplies logical-memory measures.",
        "- **Genetics:** APOE is present across cohorts but encoded differently (combined genotype versus allele fields). Derive ε4 carrier count (0/1/2) only after validating codes.",
        "- **Fluid biomarkers:** ADNI explicitly includes Aβ40, Aβ42, total tau, and p-tau Elecsys values. OASIS-3 supplied diagnosis fields indicate biomarker evidence but the current clinical files are not equivalent quantitative assays. AIBL lab fields are coded and need its dictionary.",
        "- **MRI:** ADNI supplies FreeSurfer and WMH measures; OASIS-3 supplies FreeSurfer regional volumes/thickness and MRI metadata; AIBL currently supplies MRI acquisition metadata only.",
        "- **PET:** OASIS-3 includes amyloid Centiloid, tau Braak/tauopathy, and high-dimensional PUP outputs. AIBL includes PET acquisition metadata, not the regional quantitative results needed for direct SUVR/Centiloid analysis. The supplied ADNI selection has no comparable PET quantification table.",
        "- **Mood/function/behavior:** GDS and NPI-Q are available in ADNI and OASIS-3. Functional items are strong in OASIS-3; verify equivalent ADNI/AIBL files before cross-cohort functional modeling.",
        "",
    ]

    for cohort in COHORT_LABELS:
        label = COHORT_LABELS[cohort]
        file_df, inv, baseline, cov = files[cohort], inventories[cohort], baselines[cohort], coverage[cohort]
        domain_summary = inv.groupby("domain").agg(
            column_instances=("column", "size"), usable_candidates=("is_usable_candidate", lambda x: x.astype(str).str.lower().eq("true").sum()),
            median_missing_pct=("missing_pct", "median"),
        ).reset_index().sort_values("column_instances", ascending=False)
        high_missing = inv[(inv["non_missing"] > 0) & (inv["missing_pct"] >= 50)].sort_values("missing_pct", ascending=False)
        sentinel = inv[inv["potential_sentinel_values_and_counts"].fillna("").ne("")]
        lines.extend([
            f"## {label}", "",
            f"The recursive audit found **{len(file_df):,} participant/measurement tables**, **{len(inv):,} column instances**, and **{len(baseline):,} rows** in the constructed subject-level baseline table.",
            "", "### Source tables", "",
            _markdown_table(file_df.sort_values("rows", ascending=False), ["table_name", "rows", "columns", "missing_pct", "id_column", "visit_column", "unique_subjects", "duplicate_subject_visit_rows"]),
            "", "### Clinical domains and missingness", "", _markdown_table(domain_summary), "",
            f"There are **{len(high_missing):,} non-empty column instances with at least 50% missingness** and **{len(sentinel):,} columns containing potential sentinel codes** such as -1, 88, 99, 888, or 999. "
            f"Review `results/csvs/{cohort}/column_inventory.csv` before recoding; these values can be legitimate for some instruments.",
            "", "### Longitudinal structure", "",
            _markdown_table(cov.sort_values("subjects", ascending=False), ["table_name", "subjects", "records", "median_records_per_subject", "max_records_per_subject", "subjects_with_multiple_records"]),
            "", "### Baseline key measures", "",
            _markdown_table(all_key_metrics[all_key_metrics["cohort"] == label]),
            "", "### Generated figures", "",
        ])
        for plot in sorted((RESULTS_ROOT / "plots" / cohort).glob("*.png")):
            relative = plot.relative_to(REPORT_ROOT).as_posix() if False else Path("../../results/plots") / cohort / plot.name
            lines.append(f"![{label} {plot.stem.replace('_', ' ')}]({relative.as_posix()})")
            lines.append("")

        lines.extend(["### Complete usable-column appendix", ""])
        lines.append("The following lists every column automatically classed as a candidate feature/outcome, grouped by source table. Missing percentages are shown in parentheses. Identifiers, export artifacts, constants, and all-missing fields remain documented in the full column-inventory CSV.")
        lines.append("")
        usable_mask = inv["is_usable_candidate"].astype(str).str.lower().eq("true")
        usable = inv[usable_mask].sort_values(["table_name", "domain", "column"])
        for table_name, group in usable.groupby("table_name", sort=True):
            items = [f"`{row.column}` ({row.missing_pct:.1f}%; {row.domain})" for row in group.itertuples()]
            lines.extend(["<details>", f"<summary>{table_name}: {len(items)} usable candidates</summary>", "", ", ".join(items), "", "</details>", ""])
        lines.extend([
            f"Full machine-readable outputs: `results/csvs/{cohort}/`. Raw data were read only and were not modified.", "",
        ])

    lines.extend([
        "## Data-quality caveats", "",
        "- Automatic profiling cannot determine clinical validity, assay batch effects, scanner harmonization, or whether a coded value means missing/unknown without the source data dictionary.",
        "- Potential sentinel values are flagged, but only blank/common textual missing tokens and `-4` were parsed as missing during loading. Review every instrument codebook before broader recoding.",
        "- Distinct subject counts are table-specific. Do not sum them across tables; tables overlap in participants.",
        "- Duplicate participant-visit keys may represent repeated forms, amendments, acquisitions, or true duplicates. Resolve them using protocol/date/QC fields before joining.",
        "- The baseline tables are exploratory conveniences. They are not analysis-ready until inclusion criteria, visit windows, units, code mappings, provenance, and outcome definitions are finalized.",
        "- Twenty AIBL flutemeta rows contained one unquoted comma. The reader reconstructed the affected penultimate field, retained every row, and documented each repair in `results/csvs/aibl/parse_warnings.csv`.",
        "", "## Reproducibility and output map", "",
        "- Executed cohort notebooks: `notebooks/DataExploration/<cohort>/`",
        "- Reusable pipeline: `scripts/demographics_pipeline.py`",
        "- Cohort CSV audits: `results/csvs/<cohort>/`",
        "- Cross-cohort CSVs: `results/csvs/combined/`",
        "- Figures: `results/plots/<cohort>/`",
        "- This report: `reports/demographics/demographics_report.md`",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    if verbose:
        print(f"Wrote synthesized report: {report_path.relative_to(PROJECT_ROOT)}")
        print(f"Report size: {report_path.stat().st_size / 1024:.1f} KiB")
    return report_path


def run_all() -> Path:
    for cohort in COHORT_LABELS:
        run_cohort_analysis(cohort)
    return synthesize_report()


if __name__ == "__main__":
    run_all()
