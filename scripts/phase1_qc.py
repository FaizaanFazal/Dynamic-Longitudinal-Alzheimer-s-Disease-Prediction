#!/usr/bin/env python3
"""Independent on-disk verifier for Phase 1 generated artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("adni", "oasis3", "aibl")
TABLE_ROOT = ROOT / "results" / "csvs" / "phase1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_saved_artifact_qc() -> None:
    for cohort in COHORTS:
        manifest_path = ROOT / "ProcessedData" / cohort / "phase1_canonical_visit_manifest.csv"
        index_path = ROOT / "ProcessedData" / cohort / "phase1_primary_index_cohort.csv"
        transition_path = TABLE_ROOT / cohort / "cdr_transition_events.csv"
        summary_path = TABLE_ROOT / cohort / "cdr_transition_summary.csv"
        horizon_path = TABLE_ROOT / cohort / "participant_horizon_outcomes.csv"
        for path in [manifest_path, index_path, transition_path, summary_path, horizon_path]:
            require(path.exists(), f"Missing required artifact: {path}")

        manifest = pd.read_csv(manifest_path, low_memory=False)
        index = pd.read_csv(index_path, low_memory=False)
        transitions = pd.read_csv(transition_path, low_memory=False)
        summary = pd.read_csv(summary_path, low_memory=False)
        horizon = pd.read_csv(horizon_path, low_memory=False)

        require(manifest["manifest_key"].is_unique, f"{cohort}: manifest key is not unique")
        require(manifest["cohort"].eq(cohort).all(), f"{cohort}: incorrect cohort label in manifest")
        valid_cdr = manifest["cdr_global"].dropna().isin([0, 0.5, 1, 2, 3]).all()
        require(bool(valid_cdr), f"{cohort}: invalid harmonized CDR")
        valid_mmse = manifest["mmse"].dropna().between(0, 30).all()
        require(bool(valid_mmse), f"{cohort}: invalid harmonized MMSE")
        valid_age = manifest["age_at_visit"].dropna().between(18, 110).all()
        require(bool(valid_age), f"{cohort}: invalid age")
        require((index["index_mmse_time_days"] <= index["index_time_days"]).all(), f"{cohort}: future index MMSE")
        if not transitions.empty:
            require((transitions["transition_interval_days"] > 0).all(), f"{cohort}: nonpositive transition interval")
        summary_events = int(summary["events"].sum()) if not summary.empty else 0
        require(summary_events == len(transitions), f"{cohort}: transition summary mismatch")
        for horizon_month, group in horizon.groupby("horizon_months"):
            require(group["participant_id"].nunique() == len(index), f"{cohort}: horizon {horizon_month} denominator mismatch")
            statuses = group[
                ["transition_to_cdr_0_flag", "transition_to_cdr_ge1_flag", "remain_cdr_0_5_flag", "censored_before_horizon_flag", "unresolved_conflict_flag"]
            ].fillna(False).sum(axis=1)
            require(statuses.eq(1).all(), f"{cohort}: nonexclusive horizon status")

    qc = pd.read_csv(TABLE_ROOT / "qc_results.csv")
    require(not ((qc["severity"] == "critical") & (qc["status"] != "PASS")).any(), "Saved QC table contains a critical failure")
    print("All saved Phase 1 artifact QC checks passed.")


if __name__ == "__main__":
    run_saved_artifact_qc()
