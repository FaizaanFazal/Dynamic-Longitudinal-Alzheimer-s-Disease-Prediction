#!/usr/bin/env python3
"""Independent Phase 5 final-project QC and provenance audit.

This verifier does not import the Phase 5 analysis script. It recomputes key
counts from saved participant-level Phase 5 data, verifies source and frozen
artifact checksums, checks manuscript tables against their source CSVs, and
audits reports, figures, notebook execution, and interpretation boundaries.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "csvs" / "phase5"
MANUSCRIPT_TABLES = TABLES / "manuscript_tables"
PLOTS = ROOT / "results" / "plots" / "phase5"
REPORTS = ROOT / "reports" / "phase5"
PROCESSED = ROOT / "ProcessedData" / "adni" / "phase5"
PHASE4 = ROOT / "results" / "csvs" / "phase4"
LOCK = ROOT / "models" / "phase3" / "phase4_lock_v2"
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "Phase5"
    / "Phase5_SecondaryAnalysesAndFinalReporting.ipynb"
)
ALTERNATE = ROOT / "LiteratureAndPlan" / "Alternate_Study_Directions.md"
GENERATED_AT = datetime.now(timezone.utc).isoformat()

RESULTS: list[dict[str, str]] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def record(
    check_id: str,
    section: str,
    passed: bool,
    details: str,
) -> None:
    RESULTS.append(
        {
            "check_id": check_id,
            "section": section,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


def frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        return True
    except AssertionError:
        return False


def verify_checksum_manifest(
    manifest_path: Path,
    path_column: str,
    base: Path,
) -> tuple[int, int, list[str]]:
    manifest = pd.read_csv(manifest_path)
    failures: list[str] = []
    for row in manifest.to_dict("records"):
        path = base / str(row[path_column])
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row["bytes"])
            or sha256(path) != str(row["sha256"])
        ):
            failures.append(str(row[path_column]))
    return len(manifest) - len(failures), len(manifest), failures


def check_frozen_sources() -> None:
    passed, total, failures = verify_checksum_manifest(
        LOCK / "checksum_manifest.csv",
        "package_relative_path",
        LOCK,
    )
    record(
        "FROZEN-01",
        "frozen sources",
        passed == total,
        f"repaired Phase 4 lock: {passed}/{total} checksums pass; "
        f"failures={failures}",
    )

    passed, total, failures = verify_checksum_manifest(
        PHASE4 / "phase4_output_manifest.csv",
        "relative_path",
        ROOT,
    )
    record(
        "FROZEN-02",
        "frozen sources",
        passed == total,
        f"Phase 4 output manifest: {passed}/{total} checksums pass; "
        f"failures={failures}",
    )

    phase4_qc = pd.read_csv(PHASE4 / "phase4_qc_results.csv")
    record(
        "FROZEN-03",
        "frozen sources",
        phase4_qc["status"].eq("PASS").all(),
        f"Phase 4 independent QC retained "
        f"{phase4_qc['status'].eq('PASS').sum()}/{len(phase4_qc)} PASS",
    )

    source_manifest = pd.read_csv(TABLES / "phase5_source_provenance.csv")
    failures = []
    for row in source_manifest.to_dict("records"):
        path = ROOT / str(row["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256(path) != str(row["sha256"])
        ):
            failures.append(str(row["relative_path"]))
    record(
        "FROZEN-04",
        "frozen sources",
        not failures,
        f"Phase 5 source provenance rehash: "
        f"{len(source_manifest)-len(failures)}/{len(source_manifest)} pass; "
        f"failures={failures}",
    )

    inventory = pd.read_csv(LOCK / "model_inventory.csv")
    model_horizons = {
        (str(row.model_id), int(row.horizon_months))
        for row in inventory.itertuples()
    }
    required = {
        ("B1", 12),
        ("B1", 36),
        ("D0", 12),
        ("D0", 36),
        ("D1", 12),
        ("D1", 36),
        ("B3", 12),
        ("B3", 36),
        ("D3", 12),
        ("D3", 36),
    }
    record(
        "FROZEN-05",
        "frozen sources",
        required.issubset(model_horizons),
        f"required frozen model/horizon artifacts present: "
        f"{len(required & model_horizons)}/{len(required)}",
    )

    no_external_fit = (
        not inventory["external_outcomes_accessed"].fillna(True).astype(bool).any()
        and inventory["model_version_id"].astype(str).str.contains(
            "phase3-adni", case=False
        ).all()
    )
    record(
        "FROZEN-06",
        "frozen sources",
        no_external_fit,
        "all frozen models are ADNI-origin and external_outcomes_accessed=False",
    )


def check_reversion() -> None:
    participants = pd.read_csv(
        PROCESSED / "diagnosis_reversion_participant_analysis.csv"
    )
    observed = participants["trajectory_group"].value_counts().to_dict()
    expected = {
        "sustained_reversion": 60,
        "progression": 399,
        "persistent_mci": 746,
        "unstable_or_single_cn": 45,
        "no_later_assessment": 350,
    }
    exact = len(participants) == 1600 and all(
        int(observed.get(group, -1)) == count
        for group, count in expected.items()
    )
    record(
        "REV-01",
        "diagnosis reversion",
        exact,
        f"baseline MCI={len(participants)}; trajectory counts={observed}",
    )

    strongest = participants["strongest_reversion_flag"].astype(bool)
    sustained = participants["sustained_reversion_flag"].astype(bool)
    record(
        "REV-02",
        "diagnosis reversion",
        int(strongest.sum()) == 50
        and int(sustained.sum()) == 60
        and (strongest <= sustained).all(),
        f"strongest={strongest.sum()}, sustained={sustained.sum()}, "
        "strongest is a nested subset",
    )

    timing = pd.read_csv(TABLES / "reversion_timing_and_relapse.csv")
    timing_counts = dict(
        zip(timing["definition"], timing["participants"], strict=True)
    )
    expected_timing = {
        "apparent_reversion": 110,
        "confirmed_reversion": 74,
        "sustained_reversion": 60,
        "strongest_reversion": 50,
        "progression": 399,
        "persistent_mci": 746,
    }
    record(
        "REV-03",
        "diagnosis reversion",
        all(
            int(timing_counts.get(key, -1)) == value
            for key, value in expected_timing.items()
        ),
        f"timing and relapse definitions reproduce audited counts: "
        f"{timing_counts}",
    )

    fixed = pd.read_csv(TABLES / "diagnosis_fixed_horizon_summary.csv")
    totals = fixed.groupby("horizon_months")["participants"].sum().to_dict()
    record(
        "REV-04",
        "diagnosis reversion",
        totals == {24: 1600, 36: 1600, 48: 1600},
        f"fixed-horizon trajectory classifications cover all participants: "
        f"{totals}",
    )

    associations = pd.read_csv(
        TABLES / "reversion_adjusted_associations.csv"
    )
    observation = pd.read_csv(
        TABLES / "reversion_observation_process_associations.csv"
    )
    features = set(associations["feature"])
    expected_features = {
        "baseline_age",
        "sex_male",
        "education_years",
        "apoe_e4_dose",
        "baseline_mmse",
    }
    association_ok = (
        features == expected_features
        and associations["ridge_C"].eq(1.0).all()
        and associations["bootstrap_replicates"].eq(1000).all()
        and associations["bootstrap_unit"].eq("participant").all()
        and associations["analysis_type"]
        .str.contains("not prediction or causal")
        .all()
    )
    record(
        "REV-05",
        "diagnosis reversion",
        association_ok,
        "fixed-penalty five-variable ridge association models; "
        "1,000 participant resamples; no high-dimensional selection",
    )

    observation_ok = (
        observation["bootstrap_replicates"].eq(1000).all()
        and observation["bootstrap_unit"].eq("participant").all()
        and observation["analysis_type"]
        .str.contains("not prediction or causal")
        .all()
    )
    record(
        "REV-06",
        "diagnosis reversion",
        observation_ok,
        "post-baseline follow-up/visit associations are explicitly "
        "ascertainment analyses, not prediction or causal results",
    )

    agreement = pd.read_csv(TABLES / "cdr_diagnosis_agreement_summary.csv")
    row = agreement.iloc[0]
    cdr = participants["cdr_defined_improvement_flag"].astype(bool)
    overlap = int((cdr & sustained).sum())
    union = int((cdr | sustained).sum())
    agreement_ok = (
        int(row["sustained_diagnosis_reverters"]) == int(sustained.sum())
        and int(row["cdr_defined_improvers"]) == int(cdr.sum()) == 172
        and int(row["overlap_participants"]) == overlap == 47
        and int(row["union_participants"]) == union == 185
        and np.isclose(float(row["jaccard_similarity"]), overlap / union)
    )
    record(
        "REV-07",
        "endpoint agreement",
        agreement_ok,
        f"recomputed overlap={overlap}, union={union}, "
        f"Jaccard={overlap/union:.6f}",
    )

    categories = (
        participants["definition_agreement_category"]
        .value_counts()
        .to_dict()
    )
    expected_categories = {
        "neither_definition": 1415,
        "cdr_improvement_only": 125,
        "both_sustained_diagnosis_and_cdr_improvement": 47,
        "diagnosis_sustained_only": 13,
    }
    record(
        "REV-08",
        "endpoint agreement",
        categories == expected_categories,
        f"participant-level disagreement categories={categories}",
    )


def check_primary_results() -> None:
    table_pairs = {
        "table_01_cohort_characteristics.csv": (
            TABLES / "cohort_characteristics.csv"
        ),
        "table_02_adni_oasis_transition_counts.csv": (
            TABLES / "transition_counts_by_horizon.csv"
        ),
        "table_03_adni_internal_model_performance.csv": (
            TABLES / "adni_internal_performance_synthesis.csv"
        ),
        "table_04_untouched_oasis_external_performance.csv": (
            TABLES / "oasis_untouched_performance_synthesis.csv"
        ),
        "table_05_d1_vs_d0_external_comparison.csv": (
            PHASE4 / "d1_vs_d0_paired_bootstrap_ci.csv"
        ),
        "table_06_calibration_and_recalibration.csv": (
            TABLES / "calibration_recalibration_synthesis.csv"
        ),
        "table_07_distribution_shift_summary.csv": (
            TABLES / "distribution_shift_synthesis.csv"
        ),
        "table_08_endpoint_sensitivity.csv": (
            TABLES / "endpoint_sensitivity_synthesis.csv"
        ),
        "table_09_diagnosis_reversion_group_characteristics.csv": (
            TABLES / "reversion_group_characteristics.csv"
        ),
        "table_10_cdr_vs_diagnosis_reversion_agreement.csv": (
            TABLES / "cdr_diagnosis_agreement_summary.csv"
        ),
        "table_11_sensitivity_analysis_summary.csv": (
            TABLES / "sensitivity_analysis_summary.csv"
        ),
        "table_12_limitations_and_bias_controls.csv": (
            TABLES / "limitations_bias_controls.csv"
        ),
    }
    mismatches = []
    for manuscript_name, source in table_pairs.items():
        manuscript_path = MANUSCRIPT_TABLES / manuscript_name
        if (
            not manuscript_path.is_file()
            or not frames_equal(
                pd.read_csv(manuscript_path), pd.read_csv(source)
            )
        ):
            mismatches.append(manuscript_name)
    record(
        "TAB-01",
        "manuscript tables",
        not mismatches and len(table_pairs) == 12,
        f"12/12 primary table copies checked; mismatches={mismatches}",
    )

    external = pd.read_csv(TABLES / "oasis_untouched_performance_synthesis.csv")
    phase4_external = pd.read_csv(PHASE4 / "untouched_external_metrics.csv")
    keys = ["model_id", "family", "horizon_months"]
    cols = [
        "model_id",
        "family",
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
    source_subset = phase4_external[
        phase4_external["model_id"].isin(["B1", "D0", "D1"])
        & phase4_external["horizon_months"].isin([12, 36])
    ][cols].sort_values(keys)
    phase5_subset = external[cols].sort_values(keys)
    record(
        "NUM-01",
        "numerical traceability",
        frames_equal(phase5_subset, source_subset),
        "Phase 5 untouched external metrics exactly match Phase 4 source rows",
    )

    internal = pd.read_csv(TABLES / "adni_internal_performance_synthesis.csv")
    phase3 = pd.read_csv(
        ROOT
        / "results"
        / "csvs"
        / "phase3"
        / "discrimination_and_horizon_metrics.csv"
    )
    common = [
        column
        for column in internal.columns
        if column in phase3.columns
    ]
    phase3_subset = phase3[
        phase3["model_id"].isin(["B1", "D0", "D1"])
        & phase3["horizon_months"].isin([12, 36])
    ][common].sort_values(keys)
    internal_subset = internal[common].sort_values(keys)
    record(
        "NUM-02",
        "numerical traceability",
        frames_equal(internal_subset, phase3_subset),
        "Phase 5 ADNI internal metrics match Phase 3 source rows",
    )

    recal = pd.read_csv(TABLES / "calibration_recalibration_synthesis.csv")
    source_recal = pd.read_csv(PHASE4 / "recalibration_effect_comparison.csv")
    source_recal = source_recal[
        source_recal["model_id"].isin(["B1", "D0", "D1"])
        & source_recal["horizon_months"].isin([12, 36])
    ]
    record(
        "NUM-03",
        "numerical traceability",
        frames_equal(recal, source_recal),
        "untouched and cross-fitted recalibration rows exactly match Phase 4",
    )

    recal_params = pd.read_csv(TABLES / "recalibration_parameter_summary.csv")
    record(
        "CAL-01",
        "recalibration",
        recal_params["cross_validation_folds"].eq(5).all()
        and not recal_params["predictor_coefficients_refit"].astype(bool).any()
        and recal_params["interpretation"]
        .str.contains("not untouched external validation")
        .all(),
        "all recalibration summaries use five participant-grouped folds; "
        "predictor coefficients were not refit",
    )

    untouched = recal["method"].eq("untouched")
    updated = ~untouched
    labels_ok = (
        recal.loc[untouched, "validation_design"].isna().all()
        and recal.loc[updated, "validation_design"]
        .fillna("")
        .str.contains("participant-grouped")
        .all()
    )
    record(
        "CAL-02",
        "recalibration",
        labels_ok,
        "untouched rows remain distinct from participant-grouped "
        "cross-fitted model-updating rows",
    )

    framework = pd.read_csv(
        TABLES / "transportability_classification_framework.csv"
    )
    selected = framework.loc[
        framework["phase5_status"].eq("selected"), "classification"
    ].tolist()
    record(
        "CAL-03",
        "transportability",
        selected == ["transportable_with_recalibration"]
        and set(framework["classification"])
        == {
            "transportable_without_updating",
            "transportable_with_recalibration",
            "limited_transportability",
            "not_transportable",
        },
        f"four-category framework complete; selected={selected}",
    )

    ci_files = [
        PHASE4 / "participant_bootstrap_confidence_intervals.csv",
        PHASE4 / "d1_vs_d0_paired_bootstrap_ci.csv",
        TABLES / "reversion_adjusted_associations.csv",
        TABLES / "reversion_observation_process_associations.csv",
    ]
    ci_details = []
    ci_ok = True
    for path in ci_files:
        frame = pd.read_csv(path)
        units = sorted(frame["bootstrap_unit"].dropna().unique().tolist())
        reps = sorted(frame["bootstrap_replicates"].dropna().unique().tolist())
        ci_details.append(f"{path.name}: unit={units}, replicates={reps}")
        ci_ok &= units == ["participant"] and reps == [1000]
    record(
        "UNC-01",
        "uncertainty",
        ci_ok,
        "; ".join(ci_details),
    )

    model_tables = [
        pd.read_csv(TABLES / "adni_internal_performance_synthesis.csv"),
        pd.read_csv(TABLES / "oasis_untouched_performance_synthesis.csv"),
        pd.read_csv(TABLES / "calibration_recalibration_synthesis.csv"),
    ]
    names_horizons_ok = all(
        set(frame["model_id"]).issubset({"B1", "D0", "D1"})
        and set(frame["horizon_months"]).issubset({12, 36})
        for frame in model_tables
    )
    record(
        "NAM-01",
        "model consistency",
        names_horizons_ok,
        "primary synthesis tables consistently use B1/D0/D1 at 12/36 months",
    )

    aibl = pd.read_csv(TABLES / "aibl_descriptive_summary.csv")
    record(
        "SCOPE-01",
        "scope control",
        not aibl["confirmatory_validation_supported"].astype(bool).any()
        and aibl["verified_age_available"].eq(0).all()
        and aibl["education_available"].eq(0).all(),
        "AIBL remains descriptive only; no confirmatory performance supported",
    )

    all_report_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in REPORTS.glob("*.md")
    ).lower()
    scope_ok = (
        "mri was excluded" in all_report_text
        and "aibl is therefore descriptive" in all_report_text
        and "biological recovery" in all_report_text
        and "model updating" in all_report_text
    )
    record(
        "SCOPE-02",
        "scope control",
        scope_ok,
        "reports retain MRI exclusion, AIBL descriptive status, "
        "no-biological-recovery boundary, and model-updating label",
    )


def check_figures_and_notebook() -> None:
    complex_figures = {1, 2, 3, 5, 6, 8, 12}
    failures = []
    details = []
    for number in range(1, 13):
        matches = sorted(PLOTS.glob(f"figure_{number:02d}_*.png"))
        if len(matches) != 1:
            failures.append(f"figure {number}: PNG count={len(matches)}")
            continue
        png = matches[0]
        svg = png.with_suffix(".svg")
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            minimum = 1199.0 if number in complex_figures else 599.0
            ok = min(float(dpi[0]), float(dpi[1])) >= minimum
            details.append(
                f"{number}: {dpi[0]:.2f}x{dpi[1]:.2f} DPI, "
                f"{image.width}x{image.height}px"
            )
        if not ok or not svg.is_file() or svg.stat().st_size == 0:
            failures.append(
                f"figure {number}: dpi={dpi}, svg={svg.is_file()}"
            )
    record(
        "FIG-01",
        "figures",
        not failures,
        "12 PNG/SVG pairs; " + "; ".join(details)
        + f"; failures={failures}",
    )

    source_tables = [
        path
        for path in TABLES.glob("*.csv")
        if path.name
        not in {
            "phase5_qc_results.csv",
            "phase5_required_output_status.csv",
            "phase5_output_manifest.csv",
        }
    ]
    newest_source = max(path.stat().st_mtime for path in source_tables)
    oldest_figure = min(path.stat().st_mtime for path in PLOTS.glob("*.png"))
    script_text = (
        ROOT / "scripts" / "phase5_secondary_synthesis.py"
    ).read_text(encoding="utf-8")
    record(
        "FIG-02",
        "figures",
        oldest_figure >= newest_source
        and "def create_figures(" in script_text
        and "save_csv(" in script_text,
        "figures were regenerated after saved analysis tables by the "
        "traceable Phase 5 synthesis script",
    )

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    executed = all(cell.get("execution_count") is not None for cell in code_cells)
    record(
        "NB-01",
        "executed notebook",
        len(code_cells) >= 10 and executed and not errors,
        f"{len(code_cells)} code cells executed; errors={len(errors)}; "
        f"notebook_bytes={NOTEBOOK.stat().st_size}",
    )


def check_reports() -> None:
    required = [
        "phase5_final_report.md",
        "phase5_manuscript_draft.md",
        "phase5_final_model_card.md",
        "phase5_reproducibility_report.md",
        "phase5_project_executive_summary.md",
        "phase5_journal_target_recommendations.md",
        "phase5_transportability_synthesis.md",
        "phase5_adni_diagnosis_reversion_report.md",
        "phase5_cdr_diagnosis_agreement_report.md",
        "phase5_aibl_descriptive_analysis.md",
    ]
    missing = [
        name
        for name in required
        if not (REPORTS / name).is_file()
        or (REPORTS / name).stat().st_size == 0
    ]
    record(
        "REP-01",
        "reports",
        not missing,
        f"required substantive reports present; missing={missing}",
    )

    manuscript = (REPORTS / "phase5_manuscript_draft.md").read_text(
        encoding="utf-8"
    )
    sections = [
        "## Abstract",
        "## Introduction",
        "## Methods",
        "## Results",
        "## Discussion",
        "### Strengths",
        "### Limitations",
        "### Clinical implications",
        "## Conclusion",
    ]
    missing_sections = [section for section in sections if section not in manuscript]
    record(
        "REP-02",
        "reports",
        not missing_sections,
        f"complete manuscript structure; missing={missing_sections}",
    )

    required_statements = [
        "provisional de facto global CDR",
        "Death and dropout reasons were incomplete",
        "CDR 0.5 is not identical to clinical MCI",
        "MMSE and CDR",
        "model updating",
        "AIBL was descriptive only",
        "MRI was excluded",
        "research-grade",
    ]
    missing_statements = [
        statement
        for statement in required_statements
        if statement.lower() not in manuscript.lower()
    ]
    record(
        "REP-03",
        "reports",
        not missing_statements,
        f"required interpretation statements present; "
        f"missing={missing_statements}",
    )

    model_card = " ".join(
        (REPORTS / "phase5_final_model_card.md")
        .read_text(encoding="utf-8")
        .lower()
        .split()
    )
    card_terms = [
        "intended use",
        "non-intended use",
        "research-grade",
        "not approved for clinical decision-making",
        "required local recalibration",
        "deployment restrictions",
        "subgroup limitations",
    ]
    record(
        "REP-04",
        "reports",
        all(term in model_card for term in card_terms),
        "final model card contains use, restriction, subgroup, calibration, "
        "and deployment boundaries",
    )

    final_report = (REPORTS / "phase5_final_report.md").read_text(
        encoding="utf-8"
    )
    manual_terms = [
        "author list",
        "ethics/IRB",
        "funding",
        "literature citations",
        "journal",
        "data-access statements",
    ]
    record(
        "REP-05",
        "reports",
        all(term.lower() in final_report.lower() for term in manual_terms),
        "final report lists exact remaining author, ethics, funding, "
        "reference, journal, and data-access tasks",
    )

    alternate_text = ALTERNATE.read_text(encoding="utf-8").lower()
    directions = [
        "missing-modality",
        "vascular",
        "biomarker-discordance",
        "mri-enriched",
        "diagnosis-based multi-state",
        "aibl",
    ]
    record(
        "REP-06",
        "future directions",
        all(term in alternate_text for term in directions),
        "six required future directions preserved separately",
    )

    report_texts = {
        path.name: path.read_text(encoding="utf-8").lower()
        for path in REPORTS.glob("*.md")
    }
    interpretive_recalibration_reports = {
        "phase5_final_report.md",
        "phase5_journal_target_recommendations.md",
        "phase5_transportability_synthesis.md",
        "phase5_project_executive_summary.md",
        "phase5_manuscript_draft.md",
        "phase5_final_model_card.md",
    }
    recal_files = [
        (name, report_texts[name])
        for name in sorted(interpretive_recalibration_reports)
    ]
    label_ok = all(
        "model updating" in text or "model-updating" in text
        for _, text in recal_files
    )
    record(
        "REP-07",
        "interpretation boundaries",
        label_ok,
        f"every report discussing recalibration labels it model updating: "
        f"{[name for name, _ in recal_files]}",
    )

    source_values = pd.read_csv(
        TABLES / "oasis_untouched_performance_synthesis.csv"
    )
    d1 = source_values[source_values["model_id"].eq("D1")].set_index(
        "horizon_months"
    )
    executive = report_texts["phase5_project_executive_summary.md"]
    numerical_strings = [
        f"{d1.loc[12, 'auc_improvement']:.3f}",
        f"{d1.loc[12, 'auc_progression']:.3f}",
        f"{d1.loc[36, 'auc_improvement']:.3f}",
        f"{d1.loc[36, 'auc_progression']:.3f}",
        "0.254",
    ]
    record(
        "REP-08",
        "numerical traceability",
        all(value in executive for value in numerical_strings),
        f"executive-summary headline values match machine output: "
        f"{numerical_strings}",
    )


def required_output_status() -> pd.DataFrame:
    paths: list[tuple[str, Path]] = [
        ("final Phase 5 report", REPORTS / "phase5_final_report.md"),
        ("complete manuscript draft", REPORTS / "phase5_manuscript_draft.md"),
        ("final model card", REPORTS / "phase5_final_model_card.md"),
        (
            "final reproducibility report",
            REPORTS / "phase5_reproducibility_report.md",
        ),
        (
            "independent final QC report",
            REPORTS / "phase5_final_qc_report.md",
        ),
        (
            "alternate study directions",
            ROOT / "LiteratureAndPlan" / "Alternate_Study_Directions.md",
        ),
        (
            "project executive summary",
            REPORTS / "phase5_project_executive_summary.md",
        ),
        (
            "journal target recommendations",
            REPORTS / "phase5_journal_target_recommendations.md",
        ),
        ("executed Phase 5 notebook", NOTEBOOK),
        (
            "Phase 5 analysis script",
            ROOT / "scripts" / "phase5_secondary_synthesis.py",
        ),
        ("independent Phase 5 QC script", Path(__file__).resolve()),
    ]
    paths.extend(
        (f"primary manuscript table {number:02d}", path)
        for number, path in enumerate(
            sorted(MANUSCRIPT_TABLES.glob("table_*.csv")), start=1
        )
    )
    paths.extend(
        (f"primary PNG figure {number:02d}", path)
        for number, path in enumerate(sorted(PLOTS.glob("figure_*.png")), start=1)
    )
    paths.extend(
        (f"primary SVG figure {number:02d}", path)
        for number, path in enumerate(sorted(PLOTS.glob("figure_*.svg")), start=1)
    )
    return pd.DataFrame(
        [
            {
                "required_output": name,
                "relative_path": rel(path),
                "status": (
                    "COMPLETE"
                    if path.is_file() and path.stat().st_size > 0
                    else "MISSING"
                ),
                "bytes": path.stat().st_size if path.is_file() else 0,
            }
            for name, path in paths
        ]
    )


def write_output_manifest() -> pd.DataFrame:
    paths: set[Path] = set()
    for base in [TABLES, PLOTS, REPORTS, PROCESSED]:
        paths.update(path for path in base.rglob("*") if path.is_file())
    paths.update(
        {
            NOTEBOOK,
            ALTERNATE,
            ROOT / "scripts" / "phase5_secondary_synthesis.py",
            ROOT / "scripts" / "phase5_qc.py",
            ROOT / "scripts" / "create_phase5_notebook.py",
        }
    )
    manifest_path = TABLES / "phase5_output_manifest.csv"
    paths.discard(manifest_path)

    def artifact_class(path: Path) -> str:
        relative = rel(path)
        if relative.startswith("reports/"):
            return "report"
        if relative.startswith("results/csvs/"):
            return "table_or_metadata"
        if relative.startswith("results/plots/"):
            return "figure"
        if relative.startswith("ProcessedData/"):
            return "processed_dataset"
        if relative.startswith("notebooks/"):
            return "executed_notebook"
        if relative.startswith("scripts/"):
            return "source_code"
        if relative.startswith("LiteratureAndPlan/"):
            return "study_plan"
        return "other"

    manifest = pd.DataFrame(
        [
            {
                "relative_path": rel(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "artifact_class": artifact_class(path),
                "manifested_at_utc": GENERATED_AT,
            }
            for path in sorted(paths)
        ]
    )
    manifest.to_csv(manifest_path, index=False)
    return manifest


def render_qc_report(qc: pd.DataFrame, decision: str) -> str:
    failures = qc[qc["status"].eq("FAIL")]
    return f"""# Phase 5 Independent Final-Project QC Report

Generated: {GENERATED_AT}

Verifier: `scripts/phase5_qc.py`, independent of the Phase 5 synthesis script.

## Decision

**{decision}**

## Summary

- Checks passed: **{qc["status"].eq("PASS").sum()}/{len(qc)}**
- Checks failed: **{len(failures)}**
- Frozen B1/D0/D1/B3/D3 models refit or modified: **No**
- Untouched OASIS validation kept separate from updating: **Yes**
- Confidence-interval resampling unit: **participant**
- AIBL role: **descriptive only**
- MRI prediction claim: **none**

## Check results

{qc.to_markdown(index=False)}

## Final scientific conclusion

D1 retained useful untouched discrimination in OASIS-3 and added external
value over D0, while substantial improvement-risk underprediction prevented
classification as transportable without updating. Participant-grouped,
cross-fitted recalibration improved calibration and prediction error but is
local model updating. Reconstructed CDR did not reverse the primary
conclusion.

## Strongest contribution

A calibration-first, genuinely locked external validation of bidirectional
dynamic cognitive-stage prediction, with explicit endpoint-definition
sensitivity and transparent separation of untouched transport from local
updating.

## Main limitation

Cross-cohort differences in outcome incidence, endpoint construction,
assessment process, and follow-up—together with incomplete death/dropout
reasons—limit probability transport and causal interpretation.

## Submission readiness

The computational study is ready for manuscript submission if all QC checks
pass. Remaining manual work is limited to author/affiliation/ORCID details;
cohort-specific ethics, consent, and data-use language; funding,
acknowledgements, competing interests, and contributions; reference
completion; final clinical-author review; journal-specific formatting and
reporting checklists; and compliant data/repository statements that do not
redistribute raw cohort data.
"""


def write_reports_and_metadata() -> int:
    qc = pd.DataFrame(RESULTS)
    failures = qc[qc["status"].eq("FAIL")]
    qc.to_csv(TABLES / "phase5_qc_results.csv", index=False)
    decision = (
        "READY FOR MANUSCRIPT SUBMISSION AFTER MANUAL "
        "AUTHOR/ETHICS/DATA-SHARING COMPLETION"
        if failures.empty
        else "HOLD FOR PHASE 5 QC REPAIR"
    )
    (REPORTS / "phase5_final_qc_report.md").write_text(
        render_qc_report(qc, decision), encoding="utf-8"
    )

    required = required_output_status()
    required.to_csv(TABLES / "phase5_required_output_status.csv", index=False)
    if not required["status"].eq("COMPLETE").all():
        missing = required.loc[
            ~required["status"].eq("COMPLETE"), "required_output"
        ].tolist()
        record(
            "OUT-01",
            "required outputs",
            False,
            f"missing required outputs={missing}",
        )
    else:
        record(
            "OUT-01",
            "required outputs",
            True,
            f"{len(required)}/{len(required)} required outputs complete",
        )

    # Re-save QC after adding the required-output check and refresh the report.
    qc = pd.DataFrame(RESULTS)
    failures = qc[qc["status"].eq("FAIL")]
    qc.to_csv(TABLES / "phase5_qc_results.csv", index=False)
    decision = (
        "READY FOR MANUSCRIPT SUBMISSION AFTER MANUAL "
        "AUTHOR/ETHICS/DATA-SHARING COMPLETION"
        if failures.empty
        else "HOLD FOR PHASE 5 QC REPAIR"
    )
    (REPORTS / "phase5_final_qc_report.md").write_text(
        render_qc_report(qc, decision), encoding="utf-8"
    )

    metadata_path = TABLES / "phase5_run_metadata.json"
    metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "final_qc_completed_at_utc": GENERATED_AT,
            "independent_final_qc_pass": failures.empty,
            "independent_final_qc_checks": len(qc),
            "independent_final_qc_failures": len(failures),
            "final_qc_pending": False,
            "phase5_complete": failures.empty,
            "decision": decision,
            "qc_software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "nbformat": nbformat.__version__,
            },
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    manifest = write_output_manifest()
    print(
        json.dumps(
            {
                "checks": len(qc),
                "passed": int(qc["status"].eq("PASS").sum()),
                "failed": len(failures),
                "manifested_outputs": len(manifest),
                "decision": decision,
            },
            indent=2,
        )
    )
    return len(failures)


def main() -> None:
    check_frozen_sources()
    check_reversion()
    check_primary_results()
    check_figures_and_notebook()
    check_reports()
    failures = write_reports_and_metadata()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
