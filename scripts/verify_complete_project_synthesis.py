#!/usr/bin/env python3
"""Independent integrity checks for the complete project evidence book.

This verifier deliberately does not import the synthesis builder.  It reads
the finished evidence book, the artifact inventory, and the original
machine-readable results, then reconstructs exact Markdown representations
of embedded CSVs for byte-for-byte comparison.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SYNTHESIS = ROOT / "evidenceBOOK/Complete_Project_Results_Synthesis.md"
INVENTORY = (
    ROOT / "results/csvs/phase5/complete_project_synthesis_inventory.csv"
)
QC_CSV = (
    ROOT / "results/csvs/phase5/"
    "complete_project_synthesis_verification.csv"
)
QC_REPORT = (
    ROOT / "evidenceBOOK/"
    "Complete_Project_Results_Synthesis_Verification.md"
)

EMBEDDED_RE = re.compile(
    r'<!-- EMBEDDED_CSV_START path="([^"]+)" '
    r'sha256="([0-9a-f]{64})" -->\n'
    r"(.*?)\n<!-- EMBEDDED_CSV_END -->",
    flags=re.DOTALL,
)
SELECTED_FIGURE_RE = re.compile(
    r'<!-- SELECTED_FIGURE path="([^"]+)" -->'
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_markdown_table(path: Path) -> str:
    """Reconstruct the builder's lossless string-table representation."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in frame.columns:
        frame[column] = (
            frame[column]
            .astype(str)
            .str.replace("|", "&#124;", regex=False)
            .str.replace("\r\n", "<br>", regex=False)
            .str.replace("\n", "<br>", regex=False)
        )
    return frame.to_markdown(index=False, disable_numparse=True)


class Recorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def add(
        self,
        test_id: str,
        category: str,
        passed: bool,
        details: str,
        evidence: str,
    ) -> None:
        self.rows.append(
            {
                "test_id": test_id,
                "category": category,
                "status": "PASS" if passed else "FAIL",
                "details": details,
                "evidence": evidence,
            }
        )


def numeric_row(
    path: str,
    filters: dict[str, object],
    column: str,
) -> float:
    frame = pd.read_csv(ROOT / path)
    selected = frame.copy()
    for key, expected in filters.items():
        series = selected[key]
        if isinstance(expected, (int, float)) and not isinstance(
            expected, bool
        ):
            selected = selected[
                np.isclose(
                    pd.to_numeric(series, errors="coerce"),
                    float(expected),
                    rtol=0,
                    atol=1e-12,
                )
            ]
        else:
            selected = selected[series.astype(str) == str(expected)]
    if len(selected) != 1:
        raise ValueError(
            f"{path}: filters {filters} yielded {len(selected)} rows"
        )
    return float(selected.iloc[0][column])


def check_numeric_claims(recorder: Recorder, text: str) -> None:
    """Recalculate the principal values quoted in narrative prose."""
    specs = [
        (
            "claim_d1_12_auc_progression",
            "results/csvs/phase4/untouched_external_metrics.csv",
            {"model_id": "D1", "horizon_months": 12},
            "auc_progression",
            0.7212213600954863,
            "0.7212",
        ),
        (
            "claim_d1_36_auc_improvement",
            "results/csvs/phase4/untouched_external_metrics.csv",
            {"model_id": "D1", "horizon_months": 36},
            "auc_improvement",
            0.7472072098936181,
            "0.7472",
        ),
        (
            "claim_d1_12_improvement_oe",
            "results/csvs/phase4/untouched_external_metrics.csv",
            {"model_id": "D1", "horizon_months": 12},
            "observed_expected_improvement",
            2.1535230709543924,
            "2.1535",
        ),
        (
            "claim_d1_36_improvement_oe",
            "results/csvs/phase4/untouched_external_metrics.csv",
            {"model_id": "D1", "horizon_months": 36},
            "observed_expected_improvement",
            2.282595305020406,
            "2.2826",
        ),
        (
            "claim_d1_d0_12_progression_auc",
            "results/csvs/phase4/d1_vs_d0_paired_bootstrap_ci.csv",
            {"horizon_months": 12, "metric": "auc_progression"},
            "bootstrap_median_difference",
            0.07061716782750316,
            "0.0706",
        ),
        (
            "claim_d1_d0_36_multistate_brier",
            "results/csvs/phase4/d1_vs_d0_paired_bootstrap_ci.csv",
            {"horizon_months": 36, "metric": "multistate_brier"},
            "bootstrap_median_difference",
            -0.0143586622366206,
            "−0.0144",
        ),
        (
            "claim_overlap_jaccard",
            "results/csvs/phase5/cdr_diagnosis_agreement_summary.csv",
            {},
            "jaccard_similarity",
            0.25405405405405407,
            "0.2541",
        ),
    ]
    for test_id, path, filters, column, expected, prose in specs:
        try:
            actual = numeric_row(path, filters, column)
            passed = bool(
                np.isclose(actual, expected, rtol=0, atol=1e-12)
                and prose in text
            )
            detail = (
                f"source={actual:.16g}; expected={expected:.16g}; "
                f"narrative token={prose!r}"
            )
        except Exception as error:
            passed = False
            detail = str(error)
        recorder.add(test_id, "principal numerical claim", passed, detail, path)

    # Count claims with important unit/scope distinctions.
    count_specs = [
        (
            "claim_final_index_counts",
            "results/csvs/phase1/index_cohort_flow_all_cohorts.csv",
            "Final index counts are ADNI/OASIS-3/AIBL = 2,447/491/241.",
            (
                "final CDR 0.5 eligibility yields "
                "2,447/491/241"
            ),
            {"adni": 2447, "oasis3": 491, "aibl": 241},
        ),
        (
            "claim_reversion_group_counts",
            "results/csvs/phase5/reversion_group_counts.csv",
            "Baseline-MCI groups include sustained=60, progression=399, "
            "persistent=746, strongest nested=50.",
            "60 sustained and 50 strongest reverters, 399 progressors, "
            "and 746 persistent-MCI participants",
            {
                "sustained_reversion": 60,
                "progression": 399,
                "persistent_mci": 746,
                "strongest_reversion_nested": 50,
            },
        ),
    ]
    for test_id, path, detail, prose, expected in count_specs:
        try:
            frame = pd.read_csv(ROOT / path)
            if "cohort" in frame.columns:
                frame = frame[
                    frame["flow_step"]
                    == "Final first eligible index participants"
                ]
                actual = dict(
                    zip(
                        frame["cohort"],
                        frame["participants_remaining"].astype(int),
                    )
                )
            else:
                actual = dict(
                    zip(
                        frame["trajectory_group"],
                        frame["participants"].astype(int),
                    )
                )
            passed = all(actual.get(k) == v for k, v in expected.items())
            passed = passed and prose in text
        except Exception as error:
            passed = False
            detail = f"{detail} Error: {error}"
        recorder.add(test_id, "principal count claim", passed, detail, path)


def refresh_inventory() -> None:
    """Record verifier outputs without creating a self-checksum paradox."""
    frame = pd.read_csv(INVENTORY, dtype=str, keep_default_na=False)
    outputs = [
        (
            "evidence book",
            "project-wide",
            SYNTHESIS,
            "authoritative synthesis",
        ),
        (
            "evidence book",
            "project-wide",
            QC_REPORT,
            "independent synthesis verification",
        ),
        (
            "script",
            "project-wide",
            Path(__file__),
            "independent verifier",
        ),
        (
            "table",
            "phase5",
            QC_CSV,
            "independent verification results",
        ),
        (
            "table",
            "phase5",
            INVENTORY,
            "referenced-artifact inventory",
        ),
    ]
    for artifact_type, phase, path, role in outputs:
        relative = path.relative_to(ROOT).as_posix()
        values = {
            "artifact_type": artifact_type,
            "phase": phase,
            "relative_path": relative,
            "exists": "True",
            "bytes": str(path.stat().st_size),
            "sha256": "" if path == INVENTORY else sha256(path),
            "rows": "",
            "columns": "",
            "participant_count_or_note": "",
            "embedded_full_table": "False",
            "manuscript_recommendation": (
                "not recommended (metadata/provenance only)"
                if path.suffix == ".csv"
                else ""
            ),
            "referenced_role": role,
        }
        match = frame["relative_path"] == relative
        if match.any():
            for column, value in values.items():
                frame.loc[match, column] = value
        else:
            frame = pd.concat(
                [frame, pd.DataFrame([values])],
                ignore_index=True,
            )
    frame = frame.sort_values(
        ["artifact_type", "phase", "relative_path"]
    ).reset_index(drop=True)
    frame.to_csv(INVENTORY, index=False)
    # Stabilize the inventory's own byte count. Its checksum remains blank
    # because adding that checksum would change the file being hashed.
    size = str(INVENTORY.stat().st_size)
    match = (
        frame["relative_path"]
        == "results/csvs/phase5/complete_project_synthesis_inventory.csv"
    )
    frame.loc[match, "bytes"] = size
    frame.to_csv(INVENTORY, index=False)


def write_report(results: pd.DataFrame) -> None:
    passed = int((results["status"] == "PASS").sum())
    failed = int((results["status"] == "FAIL").sum())
    status = "PASS" if failed == 0 else "FAIL"
    failures = results[results["status"] == "FAIL"]
    failure_text = (
        "None."
        if failures.empty
        else failures[
            ["test_id", "details", "evidence"]
        ].to_markdown(index=False)
    )
    report = f"""# Complete Project Synthesis — Independent Verification

**Run:** {datetime.now(timezone.utc).isoformat()}  
**Overall status:** **{status}**  
**Checks:** {passed} passed; {failed} failed.

This verifier is separate from the synthesis builder. It did not fit models,
rerun analyses, access raw cohort records, or modify frozen model artifacts.
It reconstructed every embedded CSV table independently and compared the
rendered content exactly.

## Verification coverage

- all inventory source paths exist;
- all phase 1–5 and lock-repair CSV outputs are represented in the inventory;
- the embedded-table set equals the inventory-declared embedded set;
- every embedded table matches its source CSV exactly as a string table;
- embedded-table source SHA-256 values are current;
- participant-, landmark-, prediction-, manifest-, fold-, bootstrap-
  replicate-, and other row-level tables are not reproduced in full;
- all catalogued figure files and all six selected main figures exist;
- principal numerical and count claims match machine-readable sources.

## Failed checks

{failure_text}

## Complete check results

{results.to_markdown(index=False)}

## Decision

**{status} — authoritative synthesis integrity
{"confirmed" if status == "PASS" else "not confirmed; correct failures before use"}.**
"""
    QC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    QC_REPORT.write_text(report, encoding="utf-8")


def main() -> int:
    recorder = Recorder()
    required = [SYNTHESIS, INVENTORY]
    recorder.add(
        "required_inputs",
        "setup",
        all(path.is_file() for path in required),
        "Synthesis and inventory are present.",
        "; ".join(path.relative_to(ROOT).as_posix() for path in required),
    )
    if not all(path.is_file() for path in required):
        results = pd.DataFrame(recorder.rows)
        QC_CSV.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(QC_CSV, index=False)
        write_report(results)
        return 1

    text = SYNTHESIS.read_text(encoding="utf-8")
    inventory = pd.read_csv(INVENTORY)
    embedded_matches = EMBEDDED_RE.findall(text)
    embedded_paths = [match[0] for match in embedded_matches]
    embedded_map = {match[0]: (match[1], match[2]) for match in embedded_matches}

    recorder.add(
        "embedded_paths_unique",
        "embedded tables",
        len(embedded_paths) == len(set(embedded_paths)),
        f"{len(embedded_paths)} markers; {len(set(embedded_paths))} unique.",
        "evidenceBOOK/Complete_Project_Results_Synthesis.md",
    )

    declared = set(
        inventory.loc[
            inventory["embedded_full_table"].astype(str).str.lower()
            == "true",
            "relative_path",
        ]
    )
    found = set(embedded_paths)
    recorder.add(
        "embedded_set_matches_inventory",
        "embedded tables",
        found == declared,
        f"found={len(found)}, declared={len(declared)}, "
        f"missing={sorted(declared-found)[:3]}, "
        f"unexpected={sorted(found-declared)[:3]}",
        "results/csvs/phase5/complete_project_synthesis_inventory.csv",
    )

    table_failures: list[str] = []
    hash_failures: list[str] = []
    missing_sources: list[str] = []
    for relative, (recorded_hash, rendered) in embedded_map.items():
        source = ROOT / relative
        if not source.is_file():
            missing_sources.append(relative)
            continue
        if sha256(source) != recorded_hash:
            hash_failures.append(relative)
        if exact_markdown_table(source) != rendered:
            table_failures.append(relative)
    recorder.add(
        "embedded_sources_exist",
        "embedded tables",
        not missing_sources,
        f"missing={missing_sources[:5]}",
        "embedded CSV markers",
    )
    recorder.add(
        "embedded_sha256_current",
        "embedded tables",
        not hash_failures,
        f"checked={len(embedded_map)}; mismatches={hash_failures[:5]}",
        "embedded CSV markers",
    )
    recorder.add(
        "embedded_tables_exact",
        "embedded tables",
        not table_failures,
        f"checked={len(embedded_map)}; mismatches={table_failures[:5]}",
        "source CSV versus independently rendered Markdown",
    )

    forbidden_tokens = [
        "/predictions/",
        "manifest",
        "fold_assignments",
        "bootstrap_metric_replicates",
        "bootstrap_replicates",
        "participant_landmark",
        "participant_level",
        "participant_horizon",
        "participant_transition",
        "primary_episode_time_to_event",
        "all_adjacent_cdr_pairs",
        "cdr_transition_events",
        "inter_assessment_intervals",
        "recalibrated_predictions",
        "outcome_merged_predictions",
        "representative_dynamic_risk_trajectories",
    ]
    forbidden_embedded = sorted(
        path
        for path in found
        if any(token in path.lower() for token in forbidden_tokens)
    )
    recorder.add(
        "no_row_level_tables_embedded",
        "privacy and document size",
        not forbidden_embedded,
        f"forbidden embedded paths={forbidden_embedded[:10]}",
        "embedded CSV markers",
    )

    phase_csvs: set[str] = set()
    for directory in [
        "phase1",
        "phase2",
        "phase3",
        "phase3_lock_repair",
        "phase4",
        "phase5",
    ]:
        phase_csvs.update(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "results/csvs" / directory).rglob("*.csv")
            if path not in {INVENTORY, QC_CSV}
        )
    inventory_paths = set(inventory["relative_path"].astype(str))
    recorder.add(
        "all_phase_csvs_in_inventory",
        "inventory",
        phase_csvs <= inventory_paths,
        f"phase CSVs={len(phase_csvs)}; "
        f"missing={sorted(phase_csvs-inventory_paths)[:10]}",
        "results/csvs/phase1 through phase5",
    )

    planned_outputs = {
        "evidenceBOOK/Complete_Project_Results_Synthesis_Verification.md",
        "results/csvs/phase5/complete_project_synthesis_verification.csv",
        "scripts/verify_complete_project_synthesis.py",
    }
    missing_inventory_paths = [
        relative
        for relative in inventory_paths
        if not (ROOT / relative).is_file() and relative not in planned_outputs
    ]
    recorder.add(
        "inventory_source_paths_exist",
        "inventory",
        not missing_inventory_paths,
        f"checked={len(inventory_paths)}; "
        f"missing={sorted(missing_inventory_paths)[:10]}",
        "complete_project_synthesis_inventory.csv",
    )

    figures = inventory[
        inventory["artifact_type"].astype(str) == "figure"
    ]["relative_path"].astype(str)
    missing_figures = [
        relative for relative in figures if not (ROOT / relative).is_file()
    ]
    uncatalogued_figures = [
        relative for relative in figures if f"`{relative}`" not in text
    ]
    recorder.add(
        "all_figure_files_exist",
        "figures",
        not missing_figures,
        f"figure files={len(figures)}; missing={missing_figures[:5]}",
        "inventory figure rows",
    )
    recorder.add(
        "all_figure_files_catalogued",
        "figures",
        not uncatalogued_figures,
        f"figure files={len(figures)}; "
        f"not referenced={uncatalogued_figures[:5]}",
        "Section 6 complete plot catalog",
    )

    selected = SELECTED_FIGURE_RE.findall(text)
    selected_missing = [
        relative for relative in selected if not (ROOT / relative).is_file()
    ]
    recorder.add(
        "six_selected_figures_exist",
        "figures",
        len(selected) == 6 and not selected_missing,
        f"selected={len(selected)}; missing={selected_missing}",
        "SELECTED_FIGURE markers",
    )

    required_sections = [
        "## Implementation and pipeline overview",
        "### Model-key quick reference",
        "## 1. Executive summary",
        "## 2. Original plan versus completed study",
        "## 3. Literature-gap alignment",
        "## 4. Phase-by-phase synthesis",
        "## 5. Master result tables",
        "## 6. Complete plot catalog",
        "## 7. Primary finding synthesis",
        "## 8. Claim-to-evidence matrix",
        "## 9. Consistency audit",
        "## 10. Manuscript selection recommendation",
        "## 11. Journal positioning",
        "## 12. Remaining manuscript tasks",
    ]
    missing_sections = [
        section for section in required_sections if section not in text
    ]
    recorder.add(
        "required_sections_present",
        "document structure",
        not missing_sections,
        f"missing={missing_sections}",
        "authoritative synthesis headings",
    )

    check_numeric_claims(recorder, text)

    results = pd.DataFrame(recorder.rows)
    QC_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(QC_CSV, index=False)
    write_report(results)
    refresh_inventory()

    failures = int((results["status"] == "FAIL").sum())
    print(
        f"Complete synthesis verification: "
        f"{len(results)-failures}/{len(results)} PASS; "
        f"{failures} FAIL"
    )
    print(f"Report: {QC_REPORT.relative_to(ROOT)}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
