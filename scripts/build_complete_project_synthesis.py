#!/usr/bin/env python3
"""Build the authoritative project evidence book without rerunning analyses."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidenceBOOK"
OUTPUT = EVIDENCE_DIR / "Complete_Project_Results_Synthesis.md"
POINTER = ROOT / "LiteratureAndPlan" / "Complete_Project_Results_Synthesis.md"
INVENTORY = (
    ROOT
    / "results"
    / "csvs"
    / "phase5"
    / "complete_project_synthesis_inventory.csv"
)
VERIFICATION_REPORT = (
    EVIDENCE_DIR / "Complete_Project_Results_Synthesis_Verification.md"
)
GENERATED_AT = datetime.now(timezone.utc).isoformat()
Image.MAX_IMAGE_PIXELS = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def code_path(relative: str) -> str:
    return f"`{relative}`"


def escaped_string_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in frame.columns:
        frame[column] = (
            frame[column]
            .astype(str)
            .str.replace("|", "&#124;", regex=False)
            .str.replace("\r\n", "<br>", regex=False)
            .str.replace("\n", "<br>", regex=False)
        )
    return frame


def exact_markdown_table(path: Path) -> str:
    return escaped_string_frame(path).to_markdown(
        index=False, disable_numparse=True
    )


def csv_shape(path: Path) -> tuple[int, int, list[str], str]:
    try:
        frame = pd.read_csv(path)
        participant_count = ""
        for column in [
            "participant_id",
            "subject_id",
            "RID",
            "record_id",
        ]:
            if column in frame.columns:
                participant_count = str(frame[column].nunique(dropna=True))
                break
        if (
            not participant_count
            and "participants" in frame.columns
            and len(frame)
        ):
            participant_count = (
                "reported values: "
                + "|".join(
                    map(
                        str,
                        sorted(
                            {
                                int(value)
                                for value in pd.to_numeric(
                                    frame["participants"], errors="coerce"
                                ).dropna()
                            }
                        ),
                    )
                )
            )
        return (
            len(frame),
            len(frame.columns),
            list(map(str, frame.columns)),
            participant_count or "not directly encoded",
        )
    except Exception as error:
        return 0, 0, [], f"unreadable: {error}"


def metadata_only_csv(path: Path) -> bool:
    relative = rel(path).lower()
    name = path.name.lower()
    if "manifest" in name:
        return True
    if "/predictions/" in relative or "prediction" in name and any(
        token in name
        for token in [
            "out_of_fold",
            "benchmark",
            "matched_dynamic",
            "temporal_validation",
            "recalibrated",
            "outcome_merged",
        ]
    ):
        return True
    exact_names = {
        "all_adjacent_cdr_pairs.csv",
        "cdr_transition_events.csv",
        "cdr_transition_events_all_cohorts.csv",
        "duplicate_and_conflict_records.csv",
        "duplicate_and_conflict_records_all_cohorts.csv",
        "inter_assessment_intervals.csv",
        "inter_assessment_intervals_all_cohorts.csv",
        "invalid_and_missing_cdr_source_records.csv",
        "invalid_and_missing_cdr_records_all_cohorts.csv",
        "participant_horizon_outcomes.csv",
        "participant_horizon_outcomes_all_cohorts.csv",
        "participant_longitudinal_depth.csv",
        "participant_longitudinal_depth_all_cohorts.csv",
        "participant_transition_summary.csv",
        "participant_transition_summary_all_cohorts.csv",
        "primary_episode_time_to_event.csv",
        "primary_episode_time_to_event_all_cohorts.csv",
        "adni_nested_inner_fold_assignments.csv",
        "adni_outer_fold_assignments.csv",
        "mri_index_window_matches.csv",
        "oasis3_cdr_record_level_comparison.csv",
        "recurrent_episode_identifiers.csv",
        "bootstrap_metric_replicates.csv",
        "influential_participants.csv",
        "participant_landmark_contribution_details.csv",
        "participant_level_event_information.csv",
        "schoenfeld_residuals.csv",
        "d1_vs_d0_paired_bootstrap_replicates.csv",
        "participant_bootstrap_metric_replicates.csv",
        "untouched_outcome_merged_predictions.csv",
        "cross_fitted_recalibrated_predictions.csv",
        "cdr_diagnosis_timing_comparison.csv",
        "representative_dynamic_risk_trajectories.csv",
        "phase5_output_manifest.csv",
        "phase5_source_provenance.csv",
        "complete_project_synthesis_inventory.csv",
        "complete_project_synthesis_verification.csv",
    }
    return name in exact_names


def csv_title(path: Path) -> str:
    return path.stem.replace("_", " ").strip().title()


def csv_purpose(path: Path) -> str:
    name = path.name.lower()
    rules = [
        ("qc", "Quality-control checks and phase-gate evidence."),
        ("calibration", "Calibration estimates, bins, curves, or updating evidence."),
        ("brier", "Prediction-error estimates based on Brier scoring."),
        ("discrimination", "Transition-specific discrimination estimates."),
        ("event_count", "Outcome-event support at prespecified horizons."),
        ("transition", "Longitudinal CDR/diagnosis transition evidence."),
        ("missingness", "Missing-data availability and structural-selection audit."),
        ("fold", "Participant-grouped fold construction or tuning audit."),
        ("coefficient", "Saved model coefficient or stability evidence."),
        ("penalt", "Ridge-penalty tuning, selection, or recovery evidence."),
        ("preprocess", "Leakage-safe preprocessing specification or verification."),
        ("horizon", "Outcome support and censoring by prediction horizon."),
        ("reversion", "Diagnosis-based reversion, persistence, or progression evidence."),
        ("agreement", "Endpoint-definition agreement or field-resolution evidence."),
        ("shift", "ADNI-to-OASIS case-mix, predictor, or event-rate shift."),
        ("model", "Model specification, comparison, inventory, or performance evidence."),
        ("flow", "Cohort inclusion and analysis-population flow."),
        ("software", "Recorded computational environment or version evidence."),
        ("access", "Outcome-blindness or file-access audit."),
    ]
    for token, purpose in rules:
        if token in name:
            return purpose
    return "Machine-readable project result or audit output."


KEY_INTERPRETATIONS = {
    "results/csvs/phase1/index_cohort_flow_all_cohorts.csv": (
        "Final first-index cohorts were ADNI 2,447, OASIS-3 491, and AIBL 241."
    ),
    "results/csvs/phase1/horizon_feasibility_all_cohorts.csv": (
        "ADNI and OASIS support 12/36-month CDR analyses; AIBL outcome "
        "observation is sparse."
    ),
    "results/csvs/phase2/oasis3_cdr_agreement_statistics.csv": (
        "Delivered OASIS CDRTOT agreed with rule-derived global CDR on "
        "99.72% of complete comparisons but remained provisional."
    ),
    "results/csvs/phase3/final_model_selection_summary.csv": (
        "The prespecified calibration/Brier hierarchy selected D1 over D0."
    ),
    "results/csvs/phase3/discrimination_and_horizon_metrics.csv": (
        "Authoritative ADNI out-of-fold performance for all model variants."
    ),
    "results/csvs/phase4/untouched_external_metrics.csv": (
        "Primary untouched OASIS external-validation evidence."
    ),
    "results/csvs/phase4/d1_vs_d0_paired_bootstrap_ci.csv": (
        "Paired participant-bootstrap evidence for the external value of "
        "updated MMSE."
    ),
    "results/csvs/phase4/recalibration_effect_comparison.csv": (
        "Untouched and OASIS-updated estimates are deliberately separated."
    ),
    "results/csvs/phase4/transportability_classification.csv": (
        "The prespecified result is TRANSPORTABLE WITH RECALIBRATION."
    ),
    "results/csvs/phase5/cdr_diagnosis_agreement_summary.csv": (
        "Only 47 participants met both improvement definitions; Jaccard 0.254."
    ),
    "results/csvs/phase5/reversion_group_counts.csv": (
        "The audited ADNI groups include 60 sustained and 50 strongest "
        "reverters, 399 progressors, and 746 persistent-MCI participants."
    ),
}


def csv_interpretation(path: Path) -> str:
    relative = rel(path)
    if relative in KEY_INTERPRETATIONS:
        return KEY_INTERPRETATIONS[relative]
    name = path.name.lower()
    if "qc" in name:
        return "Audit evidence; it is not a scientific-effect estimate."
    if "manifest" in name or "inventory" in name:
        return "Provenance/inventory evidence; not a manuscript result."
    if "replicate" in name or "prediction" in name:
        return "Supporting row-level or resampling output; use only through aggregate summaries."
    return "Interpret with the corresponding phase report; no additional claim is inferred here."


MAIN_TABLE_PATHS = {
    "results/csvs/phase5/manuscript_tables/table_01_cohort_characteristics.csv",
    "results/csvs/phase5/manuscript_tables/table_02_adni_oasis_transition_counts.csv",
    "results/csvs/phase5/manuscript_tables/table_03_adni_internal_model_performance.csv",
    "results/csvs/phase5/manuscript_tables/table_04_untouched_oasis_external_performance.csv",
    "results/csvs/phase5/manuscript_tables/table_05_d1_vs_d0_external_comparison.csv",
    "results/csvs/phase5/manuscript_tables/table_06_calibration_and_recalibration.csv",
    "results/csvs/phase5/manuscript_tables/table_10_cdr_vs_diagnosis_reversion_agreement.csv",
}


def manuscript_use(path: Path, metadata_only: bool) -> str:
    relative = rel(path)
    name = path.name.lower()
    if metadata_only:
        return "not recommended (metadata/provenance only)"
    if relative in MAIN_TABLE_PATHS:
        return "manuscript-main (some files form panels of one display table)"
    if "/manuscript_tables/" in relative:
        return "supplementary"
    if any(
        token in name
        for token in [
            "qc",
            "manifest",
            "access_audit",
            "verification",
            "software",
            "file_inventory",
            "checksum",
            "checklist",
        ]
    ):
        return "audit-only"
    if "phase5" in relative or any(
        token in name
        for token in [
            "sensitivity",
            "shift",
            "missingness",
            "diagnostic",
            "coefficient",
            "calibration",
        ]
    ):
        return "supplementary"
    return "audit-only or supplementary"


def report_catalog() -> str:
    paths = sorted(
        [
            *ROOT.glob("reports/feasibility/*.md"),
            *ROOT.glob("reports/demographics/*.md"),
            *ROOT.glob("reports/phase1/*.md"),
            *ROOT.glob("reports/phase2/*.md"),
            *ROOT.glob("reports/phase3/*.md"),
            *ROOT.glob("reports/phase3_lock_repair/*.md"),
            *ROOT.glob("reports/phase4/*.md"),
            *ROOT.glob("reports/phase5/*.md"),
        ]
    )
    rows = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        decision_match = re.search(
            r"(GO TO PHASE \d WITH RESTRICTIONS|"
            r"GO WITH NARROWED SCOPE|"
            r"RESUME PHASE 4 WITH RESTRICTIONS|"
            r"TRANSPORTABLE WITH RECALIBRATION|"
            r"READY FOR MANUSCRIPT SUBMISSION[^*\n]*|"
            r"^# PASS$|^\*\*PASS\*\*)",
            text,
            flags=re.MULTILINE,
        )
        rows.append(
            {
                "path": f"`{rel(path)}`",
                "title": (
                    title_match.group(1)
                    if title_match
                    else path.stem.replace("_", " ")
                ),
                "lines": len(text.splitlines()),
                "bytes": path.stat().st_size,
                "decision_or_role": (
                    decision_match.group(1)
                    if decision_match
                    else "supporting report/data dictionary"
                ),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def phase_csv_paths() -> dict[str, list[Path]]:
    return {
        "Phase 1": sorted((ROOT / "results/csvs/phase1").rglob("*.csv")),
        "Phase 2": sorted((ROOT / "results/csvs/phase2").rglob("*.csv")),
        "Phase 3": sorted((ROOT / "results/csvs/phase3").rglob("*.csv")),
        "Phase 3 lock repair": sorted(
            (ROOT / "results/csvs/phase3_lock_repair").rglob("*.csv")
        ),
        "Phase 4": sorted((ROOT / "results/csvs/phase4").rglob("*.csv")),
        "Phase 5": sorted((ROOT / "results/csvs/phase5").rglob("*.csv")),
    }


def master_result_tables() -> tuple[str, dict[str, bool]]:
    sections = []
    embedded: dict[str, bool] = {}
    for phase, paths in phase_csv_paths().items():
        sections.append(f"### {phase}")
        sections.append(
            f"{len(paths)} CSV files were inspected. Aggregate tables are "
            "embedded exactly as source strings; excluded row-level files are "
            "described without reproducing their rows."
        )
        for path in paths:
            relative = rel(path)
            metadata_only = metadata_only_csv(path)
            embedded[relative] = not metadata_only
            rows, columns, schema, participants = csv_shape(path)
            sections.append(f"#### {csv_title(path)}")
            sections.append(f"- **Source:** `{relative}`")
            sections.append(f"- **Purpose:** {csv_purpose(path)}")
            sections.append(
                f"- **Dimensions:** {rows:,} rows × {columns:,} columns"
            )
            sections.append(f"- **Schema:** `{', '.join(schema)}`")
            sections.append(f"- **Participant count:** {participants}")
            sections.append(
                f"- **Recommended use:** {manuscript_use(path, metadata_only)}"
            )
            sections.append(
                f"- **Key interpretation:** {csv_interpretation(path)}"
            )
            if metadata_only:
                privacy = (
                    "Derived participant/landmark/prediction-level data remain "
                    "subject to cohort data-use agreements and are not "
                    "redistributable."
                    if participants != "not directly encoded"
                    or any(
                        token in path.name.lower()
                        for token in [
                            "participant",
                            "prediction",
                            "landmark",
                            "record",
                            "event",
                        ]
                    )
                    else "High-volume audit, manifest, or resampling output; "
                    "cataloged rather than embedded."
                )
                sections.append(
                    f"- **Checksum:** `{sha256(path)}`\n"
                    f"- **Privacy/licensing note:** {privacy}"
                )
            else:
                table = exact_markdown_table(path)
                sections.append(
                    "<details>\n"
                    f"<summary>Embedded exact source table ({rows:,} rows)</summary>\n\n"
                    f'<!-- EMBEDDED_CSV_START path="{relative}" '
                    f'sha256="{sha256(path)}" -->\n'
                    f"{table}\n"
                    "<!-- EMBEDDED_CSV_END -->\n\n"
                    "</details>"
                )
        sections.append("")
    return "\n\n".join(sections), embedded


FIGURE_TEXT = {
    "study_and_cohort_flow": (
        "Study populations and analysis roles",
        "Cohort eligibility narrows to ADNI development, OASIS validation, "
        "and descriptive AIBL.",
    ),
    "cdr_state_transition_diagram": (
        "Bidirectional CDR state structure",
        "Improvement and progression are both observable from CDR 0.5.",
    ),
    "dynamic_external_validation_workflow": (
        "Locked dynamic-validation workflow",
        "ADNI fitting is separated from outcome-hidden OASIS prediction and "
        "outcome opening.",
    ),
    "internal_vs_external_discrimination": (
        "ADNI versus OASIS discrimination",
        "D1 retains useful external rank discrimination.",
    ),
    "untouched_calibration": (
        "Untouched OASIS calibration",
        "Improvement risk is materially underpredicted.",
    ),
    "untouched_vs_recalibrated": (
        "Untouched versus locally updated calibration",
        "Local updating reduces error but is not untouched validation.",
    ),
    "d1_vs_d0_external_comparison": (
        "External D1 versus D0 value",
        "Updated MMSE improves progression discrimination and 36-month error.",
    ),
    "predictor_and_event_rate_shift": (
        "Cross-cohort distribution shift",
        "Event incidence and predictor/assessment distributions differ.",
    ),
    "cdr_vs_diagnosis_reversion_overlap": (
        "Endpoint-definition overlap",
        "CDR improvement and sustained diagnosis reversion overlap modestly.",
    ),
    "representative_dynamic_risk_trajectories": (
        "Illustrative longitudinal probability trajectories",
        "Risk estimates change as elapsed time and MMSE are updated.",
    ),
    "reversion_trajectory_and_relapse": (
        "Reversion timing and relapse",
        "Apparent and confirmed reversions can later relapse.",
    ),
    "final_graphical_summary": (
        "Graphical summary",
        "Dynamic value, miscalibration, updating, and endpoint sensitivity "
        "are summarized together.",
    ),
}

MAIN_FIGURES = {
    "figure_01_study_and_cohort_flow": 1,
    "figure_03_dynamic_external_validation_workflow": 2,
    "figure_04_internal_vs_external_discrimination": 3,
    "figure_06_untouched_vs_recalibrated": 4,
    "figure_07_d1_vs_d0_external_comparison": 5,
    "figure_09_cdr_vs_diagnosis_reversion_overlap": 6,
}

PHASE5_FIGURE_SOURCES = {
    "figure_01_study_and_cohort_flow": "results/csvs/phase5/cohort_characteristics.csv",
    "figure_02_cdr_state_transition_diagram": "results/csvs/phase1/cdr_transition_summary_all_cohorts.csv",
    "figure_03_dynamic_external_validation_workflow": "phase reports and execution-gate audit",
    "figure_04_internal_vs_external_discrimination": "results/csvs/phase5/internal_external_transportability.csv",
    "figure_05_untouched_calibration": "results/csvs/phase4/untouched_calibration_curves.csv",
    "figure_06_untouched_vs_recalibrated": "results/csvs/phase5/calibration_recalibration_synthesis.csv",
    "figure_07_d1_vs_d0_external_comparison": "results/csvs/phase4/d1_vs_d0_paired_bootstrap_ci.csv",
    "figure_08_predictor_and_event_rate_shift": "results/csvs/phase5/distribution_shift_synthesis.csv",
    "figure_09_cdr_vs_diagnosis_reversion_overlap": "results/csvs/phase5/cdr_diagnosis_agreement_summary.csv",
    "figure_10_representative_dynamic_risk_trajectories": "results/csvs/phase5/representative_dynamic_risk_trajectories.csv",
    "figure_11_reversion_trajectory_and_relapse": "results/csvs/phase5/reversion_timing_and_relapse.csv",
    "figure_12_final_graphical_summary": "multiple Phase 4/5 aggregate tables",
}


def figure_catalog() -> str:
    rows = []
    selected_markers = []
    for phase in ["phase1", "phase2", "phase3", "phase4", "phase5"]:
        directory = ROOT / "results" / "plots" / phase
        groups: dict[str, dict[str, Path]] = {}
        for path in sorted(directory.glob("*")):
            if path.suffix.lower() not in {".png", ".svg", ".pdf"}:
                continue
            groups.setdefault(path.stem, {})[path.suffix.lower()] = path
        for stem, variants in groups.items():
            png = variants.get(".png")
            vector = variants.get(".svg") or variants.get(".pdf")
            width = height = 0
            dpi_text = "unavailable"
            if png:
                with Image.open(png) as image:
                    width, height = image.size
                    dpi = image.info.get("dpi", (0, 0))
                    dpi_text = f"{dpi[0]:.2f} × {dpi[1]:.2f}"
            lookup = re.sub(r"^figure_\d+_", "", stem)
            display, finding = FIGURE_TEXT.get(
                lookup,
                (
                    stem.replace("_", " ").title(),
                    "Supporting visualization; interpret with its phase report.",
                ),
            )
            if phase == "phase5" and stem in MAIN_FIGURES:
                recommendation = "main figure"
                proposed = f"Figure {MAIN_FIGURES[stem]}"
                selected_markers.append(
                    f'<!-- SELECTED_FIGURE path="{rel(png)}" -->'
                )
            elif phase == "phase5" and stem == "figure_12_final_graphical_summary":
                recommendation = "graphical abstract"
                proposed = "Graphical abstract"
            elif phase == "phase5":
                recommendation = "supplementary figure"
                proposed = "Supplementary"
            elif phase in {"phase3", "phase4"}:
                recommendation = (
                    "supplementary figure or duplicate of Phase 5 synthesis"
                )
                proposed = "Supplementary/audit"
            else:
                recommendation = "audit only or supplementary"
                proposed = "Supplementary/audit"
            source = (
                PHASE5_FIGURE_SOURCES.get(stem, "")
                if phase == "phase5"
                else f"corresponding {phase} aggregate CSV tables"
            )
            quality = (
                "Publication-resolution PNG; vector companion available."
                if png
                and float(dpi_text.split()[0]) >= 599
                and vector is not None
                else "Review resolution/vector availability before submission."
            )
            rows.append(
                {
                    "phase": phase,
                    "figure": stem,
                    "PNG path": f"`{rel(png)}`" if png else "unavailable",
                    "SVG/PDF path": (
                        f"`{rel(vector)}`" if vector else "unavailable"
                    ),
                    "pixels": f"{width}×{height}" if png else "unavailable",
                    "DPI": dpi_text,
                    "data source": source,
                    "displays": display,
                    "principal finding": finding,
                    "quality/readability": quality,
                    "recommendation": recommendation,
                    "proposed number": proposed,
                    "caption summary": f"{display}. {finding}",
                }
            )
    frame = pd.DataFrame(rows)
    return (
        "\n".join(selected_markers)
        + "\n\n"
        + frame.to_markdown(index=False)
    )


def compact_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".4f")


def original_plan_table() -> str:
    rows = [
        [
            "Calibration-first external validation",
            "External validation and calibration are rare.",
            "ADNI development; unchanged OASIS evaluation.",
            "Completed with outcome-hidden prediction freezing and untouched evaluation.",
            "completed",
            "None.",
            "Core novelty and strongest journal-positioning element.",
        ],
        [
            "Dynamic visit updating",
            "Baseline-only conversion dominates the literature.",
            "Update risk using repeated cognition and elapsed time.",
            "D1 landmark model uses current MMSE, MMSE change, and elapsed time.",
            "completed",
            "Landmarking used instead of a computationally heavier joint model.",
            "Maintains clinical interpretability and transportability.",
        ],
        [
            "Bidirectional/multi-state outcome",
            "Reversion and competing transitions are under-modelled.",
            "Predict CDR 0 improvement, CDR 0.5 persistence, and CDR ≥1 progression.",
            "Cause-specific ridge hazards combined into coherent three-state probabilities.",
            "completed",
            "Primary prediction begins at CDR 0.5 rather than a full recurrent CN↔MCI↔AD process.",
            "Novel bidirectional framing without unsupported rare transitions.",
        ],
        [
            "Clinical-core versus enriched ladder",
            "Few studies compare portable clinical data with expensive modalities.",
            "Clinical core, MRI, and possible fluid enrichment.",
            "Clinical core completed; B3 no-APOE sensitivity completed.",
            "modified",
            "MRI harmonization and retained event support failed the prespecified gate.",
            "Strengthens portability but removes a multimodal incremental-value claim.",
        ],
        [
            "MRI enrichment",
            "Imaging gains may not transport.",
            "Prespecified hippocampal/brain/WMH increment.",
            "Readiness audit only; no MRI prediction model.",
            "excluded",
            "Inadequate harmonized sample retention and scanner/processing comparability.",
            "Transparent negative feasibility result; future-study opportunity.",
        ],
        [
            "Fluid biomarkers",
            "Biomarker richness trades off against selection and missingness.",
            "Secondary ADNI enrichment if support allowed.",
            "Feasibility/descriptive only.",
            "excluded",
            "Only 33 sustained events in the clinical-plus-fluid tier; strong protocol selection.",
            "Avoids overfit and complete-case bias.",
        ],
        [
            "AIBL validation",
            "A third cohort could strengthen generalizability.",
            "Reduced-core exploratory validation.",
            "Descriptive transition and availability analysis only.",
            "modified",
            "Verified age unavailable, education entirely missing, sparse MMSE/follow-up.",
            "Shows real cross-cohort data-availability limits.",
        ],
        [
            "Diagnosis-based MCI reversion",
            "Clinical reversion is often ignored.",
            "ADNI sustained/strongest reversion secondary study.",
            "Completed as descriptive and five-variable ridge association analysis.",
            "secondary",
            "External diagnosis endpoints were incompatible.",
            "Adds endpoint-definition sensitivity without contaminating primary CDR validation.",
        ],
        [
            "Reconstructed OASIS CDR",
            "Endpoint differences threaten transport.",
            "Audit or sensitivity if feasible.",
            "Completed using six-box reconstructed global CDR after untouched freeze.",
            "secondary",
            "Delivered CDRTOT remains provisional.",
            "Demonstrates robustness while preserving endpoint caveat.",
        ],
        [
            "Recalibration",
            "Calibration often fails across cohorts.",
            "Secondary calibration-in-the-large and intercept/slope updating.",
            "Completed with five-fold participant-grouped cross-fitting.",
            "secondary",
            "Uses OASIS outcomes and therefore cannot be called untouched validation.",
            "Supports the precise classification 'transportable with recalibration'.",
        ],
        [
            "Missing-modality robustness",
            "Cross-cohort modality patterns differ structurally.",
            "Availability-aware comparisons if support allowed.",
            "Missingness/retention audits and no-APOE/no-MMSE sensitivities only.",
            "future",
            "No sufficiently aligned multimodal feature space.",
            "Preserved as a distinct future methods project.",
        ],
        [
            "Subgroup analyses",
            "Performance heterogeneity is often unreported.",
            "Sex, age, APOE, diagnosis, and phase subgroups.",
            "Phase/time and clinical-MCI sensitivities completed; broad demographic fairness not established.",
            "modified",
            "Small transition counts in several strata.",
            "Prevents unstable subgroup claims.",
        ],
    ]
    columns = [
        "planned component",
        "literature-gap motivation",
        "planned implementation",
        "final implementation",
        "status",
        "reason for deviation",
        "effect on novelty/journal positioning",
    ]
    return pd.DataFrame(rows, columns=columns).to_markdown(index=False)


def literature_gap_table() -> str:
    rows = [
        [
            "External validation and calibration",
            "§4.1",
            "fully addressed",
            "Unchanged ADNI models evaluated in OASIS; D1 AUC 0.680/0.721 at 12m and 0.747/0.729 at 36m; improvement O/E 2.154/2.283.",
            "OASIS endpoint remains provisional and death/dropout incomplete.",
        ],
        [
            "Diagnostic reversion and competing risks",
            "§4.2",
            "partially addressed",
            "Both CDR improvement/progression modelled; ADNI diagnosis analysis found 60 sustained reverters and 399 progressors.",
            "Death was not comparably available; no full death competing-risk endpoint.",
        ],
        [
            "Missing modality patterns",
            "§4.3",
            "partially addressed",
            "Structural missingness, complete-case retention, B3 no-APOE, and no-MMSE sensitivities were audited.",
            "No availability-aware multimodal model was supported.",
        ],
        [
            "Clinical core versus expensive multimodal models",
            "§4.4",
            "partially addressed",
            "Portable clinical-core models completed; MRI/fluid retention quantified.",
            "No harmonized MRI or fluid increment was estimable.",
        ],
        [
            "Harmonization without test leakage",
            "§4.5",
            "fully addressed for the clinical core",
            "All ADNI preprocessing/tuning was participant-grouped and applied unchanged to OASIS; outcome-hidden predictions were frozen first.",
            "Does not establish leakage-free imaging harmonization.",
        ],
        [
            "Dynamic prediction with irregular visits",
            "§4.6",
            "fully addressed within cohort constraints",
            "Observed-time 90-day landmarks, elapsed time, repeat-MMSE updates, annual-window and participant-normalized sensitivities.",
            "Research-cohort visit processes differ from routine clinical care.",
        ],
        [
            "Biomarker-discordance phenotyping",
            "§4.7",
            "not addressed",
            "Preserved in Alternate_Study_Directions.md.",
            "Requires harmonized tau/amyloid/MRI support.",
        ],
        [
            "Confidence intervals and uncertainty",
            "§4.8",
            "partially addressed",
            "1,000 participant-clustered bootstrap intervals for performance and associations.",
            "No individual-level predictive interval or conformal uncertainty.",
        ],
        [
            "Strong clinical baselines",
            "§4.9",
            "fully addressed",
            "B0 demographic, B1 clinical core, B3 no-APOE, D0 static dynamic comparator, plus RSF/GBSA baseline benchmarks.",
            "Dynamic ML benchmarks were infeasible with required clustered weighting.",
        ],
        [
            "Reproducibility and code availability",
            "§4.10",
            "fully addressed within data-use constraints",
            "Executed notebooks, scripts, checksums, immutable lock, deterministic inference, independent QC.",
            "Raw cohort data and participant-level derived files cannot be redistributed.",
        ],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "gap",
            "literature-review section",
            "status",
            "exact supporting result",
            "remaining limitation",
        ],
    ).to_markdown(index=False)


def model_glossary() -> str:
    rows = [
        ["B0", "Baseline benchmark", "Age and sex only.", "Comparator", "Not frozen for Phase 4."],
        ["B1", "Baseline clinical core", "Age, sex, education, APOE ε4 dose, index MMSE.", "Selected baseline model", "Frozen at 12/24/36/60m."],
        ["B2", "Baseline no-MMSE", "B1 clinical variables without MMSE.", "Sensitivity", "Tests MMSE contribution; not required in repaired lock."],
        ["B3", "Baseline no-APOE", "Age, sex, education, index MMSE.", "Sensitivity", "Frozen/repaired at 12/36m; tests portability without APOE."],
        ["B4", "Baseline complete case", "Same predictors as B1, without imputation.", "Sensitivity", "Tests imputation/complete-case effects."],
        ["D0", "Dynamic static-information comparator", "Age, sex, education, APOE, index MMSE, elapsed time; no updated MMSE.", "Primary comparator", "Same landmark framework as D1; frozen/repaired at 12/36m."],
        ["D1", "Dynamic updated-cognition model", "Age, sex, education, APOE, current MMSE, MMSE change from index, elapsed time.", "Selected primary dynamic model", "Frozen at 12/24/36/60m."],
        ["D2", "Dynamic no-MMSE", "Age, sex, education, APOE, elapsed time.", "Sensitivity", "Tests the contribution of updated cognition."],
        ["D3", "Strictly lagged-MMSE dynamic model", "Age, sex, education, APOE, previous MMSE strictly before landmark, time since previous MMSE, elapsed time.", "Temporal sensitivity", "Frozen/repaired at 12/36m; no same-day MMSE."],
        ["D4", "Slope-enriched dynamic model", "D1 predictors plus pre-landmark MMSE slope, count, and reliability flag.", "Sensitivity", "Slope support was too limited for the primary model."],
        ["D5", "Clinical-MCI subgroup D1", "D1 within the clinician-diagnosed MCI subgroup.", "Exploratory", "Small external event counts."],
        ["D1PN", "Participant-normalized D1", "D1 with participant-normalized landmark weights.", "Sensitivity", "Tests domination by frequently observed participants."],
        ["D1Annual", "Annual-window D1", "D1 restricted to one landmark per participant-year window.", "Sensitivity", "Tests landmark-frequency dependence."],
        ["D1LagRef", "Lag-eligible D1 reference", "D1 representation fitted/evaluated on the same lag-eligible rows as D3.", "Matched reference", "Required for a fair D1-versus-D3 comparison."],
        ["D1SlopeRef", "Slope-eligible D1 reference", "D1 representation on the same slope-eligible rows as D4.", "Matched reference", "Required for a fair D1-versus-D4 comparison."],
        ["RSF_B1", "Random survival forest B1", "Survival forest using B1 predictors.", "ML benchmark", "Baseline only; not a dynamic repeated-landmark benchmark."],
        ["GBSA_B1", "Gradient-boosted survival B1", "Boosted survival model using B1 predictors.", "ML benchmark", "Baseline only; tuned inside ADNI."],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "ID",
            "plain-language name",
            "information used",
            "role",
            "important note",
        ],
    ).to_markdown(index=False)


def term_glossary() -> str:
    rows = [
        ["Landmark", "A visit/time at which risk is recalculated using information available by that time."],
        ["Baseline row", "One index row per participant; ADNI B1 has 2,447 rows/participants."],
        ["Dynamic row", "One eligible landmark, not one independent person; ADNI has 4,517 rows from 1,412 participants."],
        ["Horizon", "Future window over which transition probabilities are estimated; 12 and 36 months are primary."],
        ["Cause-specific hazard", "Instantaneous rate of one transition while accounting for the alternative transition."],
        ["AUC", "Horizon-specific ranking discrimination; higher is better."],
        ["C-index", "Time-to-event concordance; higher is better."],
        ["Brier score", "Squared probability error; lower is better."],
        ["Integrated Brier score", "Prediction error accumulated over time; lower is better."],
        ["Calibration slope", "Spread of predicted risk; ideal approximately 1."],
        ["Calibration intercept/CITL", "Systematic risk-level offset; ideal approximately 0."],
        ["O/E", "Observed divided by expected risk; >1 means underprediction, <1 overprediction."],
        ["Untouched external validation", "Frozen ADNI predictions evaluated in OASIS without fitting any OASIS parameter."],
        ["Recalibration/model updating", "OASIS-derived intercept/slope adjustment after untouched results were frozen."],
        ["Participant-grouped", "All rows from one person remain in one fold/resample unit."],
        ["IPCW", "Inverse probability of censoring weighting used to account for horizon censoring under stated assumptions."],
        ["Ridge alpha", "Shrinkage penalty; larger alpha means stronger coefficient shrinkage."],
        ["CDR 0/0.5/≥1", "Improvement-like, index/persistent, and dementia-stage CDR states; not identical to CN/MCI/AD diagnoses."],
    ]
    return pd.DataFrame(rows, columns=["term", "meaning in this project"]).to_markdown(index=False)


def phase_sections() -> str:
    rows: list[tuple[str, dict[str, str]]] = [
        (
            "Feasibility and reversion audit",
            {
                "Objective": "Test whether sustained clinical-MCI reversion could support the originally considered study.",
                "Inputs": "Raw longitudinal diagnosis, demographics, MMSE/CDR, APOE, MRI/WMH, and fluid availability across ADNI/OASIS/AIBL.",
                "Methods": "Trajectory reconstruction; apparent/confirmed/sustained/strongest definitions; fixed-horizon and retention audits.",
                "Key counts": "ADNI baseline MCI 1,600; later assessment 1,250; sustained 60; strongest 50; progression 399; persistent with follow-up 746.",
                "Main result": "ADNI supports a low-dimensional secondary study, but guarantee-time bias and cross-cohort diagnosis incompatibility block a three-cohort reversion prediction model.",
                "QC": "Counts reconciled against demographics and later Phase 5 hard assertions.",
                "Decision": "GO WITH NARROWED SCOPE.",
                "Restrictions": "No high-dimensional reversion prediction; fixed-horizon/follow-up sensitivity required; no biological-recovery claim.",
                "Reports": "`reports/feasibility/sustained_mci_reversion_feasibility_audit.md`; `reports/demographics/demographics_report.md`.",
                "Scripts/notebooks": "`scripts/demographics_pipeline.py`; `notebooks/DataExploration/04_synthesize_demographics.ipynb`.",
                "Key CSV/plots": "Pre-phase cohort CSV/plot directories plus Phase 5 reversion outputs.",
                "Effect on next phase": "Moved CDR 0.5 transitions to the primary project and reserved diagnosis reversion for Phase 5.",
            },
        ),
        (
            "Phase 1 — data foundation and endpoint validation",
            {
                "Objective": "Build canonical visits, construct CDR states, define index cohorts, and verify event support.",
                "Inputs": "Cohort clinical/CDR/MMSE source tables and provisional code mappings.",
                "Methods": "Deterministic duplicate resolution, actual-time ordering, first eligible CDR 0.5 index, transition and censoring audits.",
                "Key counts": "Canonical rows: ADNI 14,524; OASIS 8,626; AIBL 1,688. Index participants: 2,447; 491; 241.",
                "Main result": "ADNI/OASIS support 12/36-month bidirectional CDR prediction; AIBL does not support confirmation.",
                "QC": "40 critical PASS and 2 documented warnings.",
                "Decision": "GO TO PHASE 2 WITH RESTRICTIONS.",
                "Restrictions": "OASIS CDRTOT provisional; AIBL age unverified; death/dropout incomplete; preserve MMSE timing sensitivities.",
                "Reports": "`reports/phase1/phase1_data_foundation_endpoint_validation_report.md`; `reports/phase1/canonical_manifest_data_dictionary.md`.",
                "Scripts/notebooks": "`scripts/phase1_data_foundation.py`; `scripts/phase1_qc.py`; `notebooks/Phase1/Phase1_DataFoundationAndEndpointValidation.ipynb`.",
                "Key CSV/plots": "`results/csvs/phase1/index_cohort_flow_all_cohorts.csv`; `results/csvs/phase1/horizon_feasibility_all_cohorts.csv`; `results/plots/phase1/cohort_flow_diagram.png`.",
                "Effect on next phase": "Froze endpoint populations and exposed OASIS/AIBL source limitations for harmonization.",
            },
        ),
        (
            "Phase 2 — landmarks, predictors, missingness, and harmonization",
            {
                "Objective": "Create leakage-safe baseline/dynamic datasets, predictor windows, folds, and preprocessing rules.",
                "Inputs": "Phase 1 canonical/index manifests and cohort clinical-core sources.",
                "Methods": "90-day observed landmarks; current/lagged/slope MMSE variants; participant folds; training-only imputation/scaling; MRI gate.",
                "Key counts": "ADNI primary dynamic 4,517 landmarks/1,412 participants; OASIS D0 support 858 landmarks/437 participants; D1 observed-MMSE eligibility later yields 845 rows.",
                "Main result": "Current MMSE/change supported the primary dynamic representation; slope was secondary; MRI failed readiness; AIBL remained descriptive.",
                "QC": "153 critical PASS; one accepted OASIS field warning.",
                "Decision": "GO TO PHASE 3 WITH RESTRICTIONS.",
                "Restrictions": "No MRI in Phase 3; no combined-cohort preprocessing; preserve no-MMSE and strictly lagged sensitivities.",
                "Reports": "`reports/phase2/phase2_model_ready_data_harmonization_report.md`; `reports/phase2/phase2_phase3_handoff.md`; `reports/phase2/oasis3_cdr_field_resolution_report.md`.",
                "Scripts/notebooks": "`scripts/phase2_model_ready_data.py`; `scripts/phase2_preprocessing.py`; `scripts/phase2_qc.py`; `notebooks/Phase2/Phase2_LandmarksMissingnessAndHarmonization.ipynb`.",
                "Key CSV/plots": "`results/csvs/phase2/dynamic_landmark_support.csv`; `results/csvs/phase2/model_ready_dataset_inventory.csv`; `results/plots/phase2/baseline_dynamic_participant_flow.png`.",
                "Effect on next phase": "Froze predictors, horizons, folds, and transformations for ADNI-only model selection.",
            },
        ),
        (
            "Phase 3 — ADNI development and internal validation",
            {
                "Objective": "Select B1/D1 using nested participant-grouped ADNI validation without OASIS access.",
                "Inputs": "Frozen Phase 2 ADNI baseline/dynamic datasets and deterministic folds.",
                "Methods": "Ridge cause-specific hazards, nested 5×5 participant folds, IPCW metrics, 1,000 participant bootstraps, ML baseline benchmarks.",
                "Key counts": "B1: 2,447 participants. D0/D1: 4,517 landmarks from 1,412 participants.",
                "Main result": "D1 selected over D0 by the prespecified integrated-Brier/calibration rule; B1 selected baseline.",
                "QC": "Independent 23/23 critical PASS; no OASIS participant-level access.",
                "Decision": "GO TO PHASE 4 WITH RESTRICTIONS.",
                "Restrictions": "Primary horizons 12/36m; 24/60m sensitivity; no retrospective OASIS selection; B1 12m improvement information limited.",
                "Reports": "`reports/phase3/phase3_model_comparison_and_selection.md`; `reports/phase3/phase3_internal_validation_report.md`; `reports/phase3/phase3_model_card.md`; `reports/phase3/phase3_phase4_handoff.md`.",
                "Scripts/notebooks": "`scripts/phase3_adni_model_development.py`; `scripts/phase3_survival_core.py`; `scripts/phase3_qc.py`; `notebooks/Phase3/Phase3_ADNIModelDevelopmentAndInternalValidation.ipynb`.",
                "Key CSV/plots": "`results/csvs/phase3/final_model_selection_summary.csv`; `results/csvs/phase3/discrimination_and_horizon_metrics.csv`; `results/plots/phase3/baseline_vs_dynamic_performance.png`.",
                "Effect on next phase": "Created the immutable ADNI B1/D1 package and exact external-validation order.",
            },
        ),
        (
            "Phase 3 outcome-blind lock repair",
            {
                "Objective": "Add missing D0/B3/D3 frozen bundles without examining OASIS data or changing B1/D1.",
                "Inputs": "Phase 3 ADNI data/folds/results, original lock, reports, and machine-readable tuning decisions.",
                "Methods": "Recovered modal-fold penalties; ADNI-only final fitting; byte-copy B1/D1; independent deterministic fixtures.",
                "Key counts": "14 frozen bundles total; D3 3,134 rows/1,016 participants; 55 package files hashed.",
                "Main result": "B1/D1 predictions remained exactly identical; D0/B3/D3 deterministic; D3 strictly pre-landmark.",
                "QC": "Independent 18/18 PASS; original lock 16/16 intact; repaired lock 55/55 hashes.",
                "Decision": "RESUME PHASE 4 WITH RESTRICTIONS using `models/phase3/phase4_lock_v2`.",
                "Restrictions": "No retuning; no OASIS access; original lock unchanged; Phase 4 not automatically resumed.",
                "Reports": "`reports/phase3_lock_repair/phase3_outcome_blind_lock_repair_report.md`; `reports/phase3_lock_repair/phase3_lock_repair_independent_qc_report.md`; `reports/phase3_lock_repair/phase4_restart_handoff.md`.",
                "Scripts/notebooks": "`scripts/phase3_lock_repair.py`; `scripts/phase3_lock_repair_qc.py`; `notebooks/Phase3/Phase3_OutcomeBlindLockRepair.ipynb`.",
                "Key CSV/plots": "`results/csvs/phase3_lock_repair/model_inventory.csv`; `models/phase3/phase4_lock_v2/checksum_manifest.csv`; no scientific plot was generated.",
                "Effect on next phase": "Made Phase 4 comparator/sensitivity validation self-contained and auditable.",
            },
        ),
        (
            "Phase 4 — locked OASIS-3 external validation",
            {
                "Objective": "Evaluate frozen ADNI models unchanged in OASIS, then run separate endpoint/updating analyses.",
                "Inputs": "Repaired lock v2 and frozen Phase 2 OASIS predictor/outcome definitions.",
                "Methods": "Lock verification → outcome-hidden features → frozen predictions/hash → independent regeneration → outcome opening → untouched evaluation → secondary analyses.",
                "Key counts": "B1 491 participants; D0 858 landmarks/437 participants; D1 845 landmarks/437 participants; D3 421 landmarks/207 participants.",
                "Main result": "D1 retained useful discrimination and exceeded D0; improvement risk was underpredicted; classification TRANSPORTABLE WITH RECALIBRATION.",
                "QC": "Independent final 28/28 PASS; repaired lock 55/55; original lock 16/16.",
                "Decision": "GO TO PHASE 5 WITH RESTRICTIONS.",
                "Restrictions": "Untouched results primary; OASIS recalibration is updating; CDRTOT provisional; B1 12m progression only 18 events.",
                "Reports": "`reports/phase4/phase4_locked_external_validation_report.md`; `reports/phase4/phase4_d1_vs_d0_transportability_report.md`; `reports/phase4/phase4_recalibration_report.md`; `reports/phase4/phase4_phase5_handoff.md`.",
                "Scripts/notebooks": "`scripts/phase4_generate_predictions.py`; `scripts/phase4_external_validation.py`; `scripts/phase4_qc.py`; `notebooks/Phase4/Phase4_LockedOASIS3ExternalValidation.ipynb`.",
                "Key CSV/plots": "`results/csvs/phase4/untouched_external_metrics.csv`; `results/csvs/phase4/d1_vs_d0_paired_bootstrap_ci.csv`; `results/plots/phase4/untouched_oasis_calibration_panels.png`.",
                "Effect on next phase": "Fixed the final scientific conclusion and supplied transport, recalibration, shift, and endpoint evidence.",
            },
        ),
        (
            "Phase 5 — secondary analyses and final reporting",
            {
                "Objective": "Complete reversion/endpoint/AIBL analyses and synthesize the manuscript package.",
                "Inputs": "Frozen Phase 1–4 outputs plus audited ADNI diagnosis records.",
                "Methods": "Trajectory reconstruction, five-variable ridge associations, endpoint overlap, transport synthesis, manuscript figures/tables, independent QC.",
                "Key counts": "60 sustained, 50 strongest, 399 progression, 746 persistent with follow-up; CDR/diagnosis overlap 47, Jaccard 0.254.",
                "Main result": "Reversion is definition-dependent; D1 is transportable with recalibration; AIBL remains descriptive.",
                "QC": "Independent final 37/37 PASS; 47/47 required outputs complete; 86 outputs manifested.",
                "Decision": "READY FOR MANUSCRIPT SUBMISSION AFTER MANUAL AUTHOR/ETHICS/DATA-SHARING COMPLETION.",
                "Restrictions": "No clinical deployment; no biological-recovery or causal claim; no AIBL performance; no MRI prediction claim.",
                "Reports": "`reports/phase5/phase5_final_report.md`; `reports/phase5/phase5_manuscript_draft.md`; `reports/phase5/phase5_final_qc_report.md`.",
                "Scripts/notebooks": "`scripts/phase5_secondary_synthesis.py`; `scripts/phase5_qc.py`; `notebooks/Phase5/Phase5_SecondaryAnalysesAndFinalReporting.ipynb`.",
                "Key CSV/plots": "`results/csvs/phase5/manuscript_tables/`; `results/plots/phase5/`.",
                "Effect on next phase": "No new phase; computational work is complete and the remaining work is manuscript governance/formatting.",
            },
        ),
    ]
    output = []
    for title, fields in rows:
        output.append(f"### {title}")
        output.append(
            pd.DataFrame(
                [{"item": key, "evidence": value} for key, value in fields.items()]
            ).to_markdown(index=False)
        )
    output.append("### Complete report index")
    output.append(report_catalog())
    return "\n\n".join(output)


def primary_findings() -> str:
    internal = pd.read_csv(
        ROOT / "results/csvs/phase5/adni_internal_performance_synthesis.csv"
    )
    external = pd.read_csv(
        ROOT / "results/csvs/phase5/oasis_untouched_performance_synthesis.csv"
    )
    paired = pd.read_csv(
        ROOT / "results/csvs/phase4/d1_vs_d0_paired_bootstrap_ci.csv"
    )
    recal = pd.read_csv(
        ROOT / "results/csvs/phase5/calibration_recalibration_synthesis.csv"
    )
    sensitivity = pd.read_csv(
        ROOT / "results/csvs/phase5/sensitivity_analysis_summary.csv"
    )
    agreement = pd.read_csv(
        ROOT / "results/csvs/phase5/cdr_diagnosis_agreement_summary.csv"
    )
    reversion = pd.read_csv(
        ROOT / "results/csvs/phase5/reversion_group_counts.csv"
    )
    aibl = pd.read_csv(
        ROOT / "results/csvs/phase5/aibl_descriptive_summary.csv"
    )
    endpoint = pd.read_csv(
        ROOT / "results/csvs/phase5/endpoint_sensitivity_synthesis.csv"
    )
    internal_columns = [
        "model_id",
        "horizon_months",
        "rows",
        "participants",
        "auc_improvement",
        "auc_progression",
        "multistate_brier",
        "calibration_slope_improvement",
        "observed_expected_improvement",
        "calibration_slope_progression",
        "observed_expected_progression",
    ]
    external_columns = [
        "model_id",
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
    paired_selected = paired[
        paired["metric"].isin(
            ["auc_improvement", "auc_progression", "multistate_brier"]
        )
    ]
    d1_recal = recal[recal["model_id"].eq("D1")][
        [
            "horizon_months",
            "method",
            "auc_improvement",
            "auc_progression",
            "multistate_brier",
            "calibration_slope_improvement",
            "observed_expected_improvement",
            "calibration_slope_progression",
            "observed_expected_progression",
        ]
    ]
    d1_endpoint = endpoint[
        endpoint["model_id"].eq("D1")
        & endpoint["horizon_months"].isin([12, 36])
        & endpoint["metric"].isin(
            ["auc_improvement", "auc_progression", "multistate_brier"]
        )
    ]
    return f"""
### 7.1 ADNI internal B1/D0/D1 performance

Source CSV: `results/csvs/phase5/adni_internal_performance_synthesis.csv`.
Interpretation report: `reports/phase3/phase3_internal_validation_report.md`.

{compact_table(internal, internal_columns)}

The valid same-landmark longitudinal comparison is D1 versus D0. B1 and D1
use different eligible populations and are not a paired baseline-versus-
dynamic experiment.

### 7.2 Untouched OASIS B1/D0/D1 performance

<!-- PRINCIPAL_CLAIM id="D1_EXTERNAL_AUC" source="results/csvs/phase4/untouched_external_metrics.csv" -->

Source CSV: `results/csvs/phase4/untouched_external_metrics.csv`.
Report: `reports/phase4/phase4_locked_external_validation_report.md`.

{compact_table(external, external_columns)}

D1 external AUCs were **0.6800/0.7212** for improvement/progression at 12
months and **0.7472/0.7293** at 36 months. D1 improvement O/E was
**2.1535** and **2.2826**, respectively: observed improvement was more than
twice predicted. B1 12-month progression uses only 18 events.

### 7.3 External D1 versus D0

<!-- PRINCIPAL_CLAIM id="D1_D0_EXTERNAL" source="results/csvs/phase4/d1_vs_d0_paired_bootstrap_ci.csv" -->

{paired_selected.to_markdown(index=False, floatfmt=".4f")}

At 12 months, progression AUC improved by 0.0706 (95% participant-bootstrap
CI 0.0261–0.1260). At 36 months, improvement/progression AUC gains were
0.0252 (0.0026–0.0500) and 0.0283 (0.0041–0.0526); multistate Brier changed
by −0.0144 (−0.0273 to −0.0014).

### 7.4 Calibration and recalibration

<!-- PRINCIPAL_CLAIM id="D1_RECALIBRATION" source="results/csvs/phase4/recalibration_effect_comparison.csv" -->

{d1_recal.to_markdown(index=False, floatfmt=".4f")}

The 36-month D1 multistate Brier score changed from 0.7071 untouched to
0.6291 after cross-fitted intercept-and-slope updating; improvement O/E
changed from 2.2826 to 1.2795. This is OASIS model updating, not untouched
validation. Slight cross-fitted AUC changes arise because participants
receive transforms from different training folds; predictor coefficients
were not refit.

### 7.5 Reconstructed-CDR endpoint sensitivity

{d1_endpoint.to_markdown(index=False, floatfmt=".4f")}

At primary horizons, reconstructed-CDR D1 AUC changes were small (12m:
−0.0092 improvement/−0.0002 progression; 36m: −0.0122/−0.0052) and did not
reverse the conclusion. Source:
`results/csvs/phase4/reconstructed_cdr_metric_comparison.csv`; report:
`reports/phase4/phase4_endpoint_sensitivity_report.md`.

### 7.6 B3 and D3 sensitivities

{sensitivity[sensitivity["model_id"].isin(["B3", "D3"])].to_markdown(index=False, floatfmt=".4f")}

B3 was similar to B1 and did not improve transport by removing APOE. D3 was
directionally usable but evaluated on only 421 landmarks/207 participants;
matched D1 generally discriminated better. D3's MMSE was independently
verified strictly pre-landmark.

### 7.7 Diagnosis-based reversion

<!-- PRINCIPAL_CLAIM id="REVERSION_COUNTS" source="results/csvs/phase5/reversion_group_counts.csv" -->

{reversion.to_markdown(index=False)}

Apparent reversion occurred in 110 participants, with 41 later relapses;
confirmed reversion occurred in 74, with 15 later relapses. Sustained and
strongest labels prohibit later relapse by definition. The unrestricted label
is observation-dependent: sustained reverters had median follow-up 84.5
months versus 25.3 months for persistent MCI. Sources:
`results/csvs/phase5/reversion_timing_and_relapse.csv`,
`results/csvs/phase5/reversion_group_characteristics.csv`, and
`reports/phase5/phase5_adni_diagnosis_reversion_report.md`.

### 7.8 CDR versus diagnosis improvement

<!-- PRINCIPAL_CLAIM id="ENDPOINT_OVERLAP" source="results/csvs/phase5/cdr_diagnosis_agreement_summary.csv" -->

{agreement.to_markdown(index=False, floatfmt=".4f")}

Only 47 participants met both definitions (Jaccard 0.2541); 78.3% of
sustained diagnosis reverters showed CDR improvement, while 27.3% of CDR
improvers met sustained diagnosis reversion. Neither endpoint demonstrates
biological recovery.

### 7.9 AIBL descriptive evidence

{aibl.to_markdown(index=False)}

AIBL has 241 eligible index participants, no verified ages, no education,
only 66 participants with at least two MMSE observations, and heavy
horizon censoring. No discrimination or calibration estimate is supported.
"""


def claim_matrix() -> str:
    rows = [
        [
            "Longitudinal updating adds value.",
            "External D1−D0 progression AUC +0.0706 at 12m; +0.0283 at 36m; 36m multistate Brier −0.0144.",
            "`results/csvs/phase4/d1_vs_d0_paired_bootstrap_ci.csv`",
            "`results/plots/phase5/figure_07_d1_vs_d0_external_comparison.png`",
            "`reports/phase4/phase4_d1_vs_d0_transportability_report.md`",
            "strong for specified external comparisons",
            "Updated MMSE added externally reproducible predictive value over D0.",
            "D1 is universally superior or clinically proven.",
        ],
        [
            "D1 transports with recalibration.",
            "Minimum primary-horizon/cause D1 AUC 0.680; untouched calibration outside bounds.",
            "`results/csvs/phase4/transportability_classification.csv`",
            "`results/plots/phase5/figure_04_internal_vs_external_discrimination.png`",
            "`reports/phase5/phase5_transportability_synthesis.md`",
            "strong within ADNI→OASIS design",
            "D1 retained useful discrimination but requires local recalibration.",
            "Transportable unchanged; validated after recalibration.",
        ],
        [
            "Local updating improves error.",
            "D1 36m Brier 0.7071 untouched versus 0.6291 cross-fitted intercept/slope.",
            "`results/csvs/phase4/recalibration_effect_comparison.csv`",
            "`results/plots/phase5/figure_06_untouched_vs_recalibrated.png`",
            "`reports/phase4/phase4_recalibration_report.md`",
            "secondary model-updating evidence",
            "Cross-fitted local updating improved prediction error.",
            "Recalibrated OASIS performance is untouched external validation.",
        ],
        [
            "Improvement was underpredicted.",
            "D1 improvement O/E 2.1535 at 12m and 2.2826 at 36m.",
            "`results/csvs/phase4/untouched_external_metrics.csv`",
            "`results/plots/phase5/figure_05_untouched_calibration.png`",
            "`reports/phase4/phase4_untouched_calibration_report.md`",
            "strong",
            "Untouched D1 substantially underpredicted OASIS improvement.",
            "The model was well calibrated externally.",
        ],
        [
            "Reversion depends on endpoint definition.",
            "47 overlap; union 185; Jaccard 0.2541.",
            "`results/csvs/phase5/cdr_diagnosis_agreement_summary.csv`",
            "`results/plots/phase5/figure_09_cdr_vs_diagnosis_reversion_overlap.png`",
            "`reports/phase5/phase5_cdr_diagnosis_agreement_report.md`",
            "strong descriptive evidence",
            "Observed reversion was definition-dependent.",
            "Either endpoint demonstrates biological recovery.",
        ],
        [
            "A low-dimensional clinical core is practical.",
            "B1 used 2,447 ADNI participants and applied to 491 OASIS participants; B3 remained similar.",
            "`results/csvs/phase3/discrimination_and_horizon_metrics.csv`; `results/csvs/phase5/sensitivity_analysis_summary.csv`",
            "`results/plots/phase5/figure_01_study_and_cohort_flow.png`",
            "`reports/phase5/phase5_final_model_card.md`",
            "moderate",
            "The clinical core was portable enough for locked validation.",
            "Clinical usefulness or cost-effectiveness was proven.",
        ],
        [
            "MRI was not supportable in this project.",
            "Harmonized event retention/comparability failed the Phase 3 readiness gate.",
            "`results/csvs/phase2/mri_event_retention.csv`",
            "`results/plots/phase2/mri_event_retention.png`",
            "`reports/phase2/phase2_mri_readiness_report.md`",
            "strong feasibility evidence",
            "MRI enrichment was excluded because harmonized support was inadequate.",
            "MRI has no predictive value.",
        ],
        [
            "AIBL cannot confirm performance under the current extract.",
            "241 index participants; verified age 0; education 0; ≥2 MMSE 66.",
            "`results/csvs/phase5/aibl_descriptive_summary.csv`",
            "`results/plots/phase5/figure_01_study_and_cohort_flow.png`",
            "`reports/phase5/phase5_aibl_descriptive_analysis.md`",
            "strong feasibility evidence",
            "AIBL was descriptive only under the supplied data.",
            "The model failed in AIBL.",
        ],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "proposed manuscript claim",
            "exact supporting result",
            "table source",
            "figure source",
            "report source",
            "strength",
            "wording allowed",
            "wording to avoid",
        ],
    ).to_markdown(index=False)


def consistency_audit() -> str:
    rows = [
        [
            "OASIS strict diagnosis MCI→AD: 61 versus 60 events",
            "Earlier feasibility matrix reported 61; row-level frozen Phase 1 reconstruction produced 60.",
            "`results/csvs/phase1/diagnosis_vs_cdr_transition_comparison.csv` and Phase 1 report",
            "Use 60 for the documented frozen strict-diagnosis reconstruction; this does not affect primary CDR validation.",
        ],
        [
            "Strongest ADNI reverters: 52 versus 50",
            "52 includes all ADNI trajectories; 50 is restricted to baseline clinical-MCI participants.",
            "`reports/feasibility/sustained_mci_reversion_feasibility_audit.md`",
            "Use 50 for the Phase 5 baseline-MCI analysis; state the scope.",
        ],
        [
            "Literature-review cohort sizes versus final index cohorts",
            "Review estimates (~4,746/~1,378/~862) describe broad cohorts; final CDR 0.5 eligibility yields 2,447/491/241.",
            "`results/csvs/phase1/index_cohort_flow_all_cohorts.csv`",
            "Use final index counts in Results; broad counts only in data-source context.",
        ],
        [
            "ADNI D1 4,517 rows versus 1,412 participants",
            "Repeated landmark rows are not independent people.",
            "`results/csvs/phase3/discrimination_and_horizon_metrics.csv`",
            "Always write '4,517 landmarks from 1,412 participants'.",
        ],
        [
            "OASIS D0 858 rows versus D1 845 rows",
            "D1 requires observed current-MMSE eligibility; both involve 437 participants.",
            "`results/csvs/phase4/outcome_hidden_population_verification.csv`",
            "Use 845 matched rows for D1-versus-D0 comparisons.",
        ],
        [
            "Training event rows versus unique internal event participants",
            "Frozen full-fit metadata counts event rows (e.g., D1 154/303 at 12m); OOF metric tables report unique event participants (149/278).",
            "`models/phase3/phase4_lock_v2/model_inventory.csv`; `results/csvs/phase3/discrimination_and_horizon_metrics.csv`",
            "Name the unit explicitly; do not treat these as contradictory.",
        ],
        [
            "AIBL age available versus verified age unavailable",
            "Phase 1 parsed year-only PTDOB; Phase 5 classifies all 241 as unverified proxies.",
            "`results/csvs/phase5/aibl_descriptive_summary.csv`",
            "Say 'unverified year-only age proxy available; verified age unavailable'.",
        ],
        [
            "CDR transition counts versus diagnosis-reversion counts",
            "ADNI 392 all-history CDR 0.5→0 events/317 people differs from 60 sustained diagnosis reverters.",
            "`results/csvs/phase1/cdr_transition_summary_all_cohorts.csv`; `results/csvs/phase5/reversion_group_counts.csv`",
            "Never merge or relabel these endpoints.",
        ],
        [
            "Untouched versus recalibrated OASIS results",
            "Recalibration uses OASIS outcomes after untouched results were frozen.",
            "`results/csvs/phase4/recalibration_effect_comparison.csv`",
            "Call only method='untouched' rows external validation; call others model updating.",
        ],
        [
            "B1/D1 original lock versus repaired v2 lock",
            "B1/D1 are byte-identical copies; D0/B3/D3 were ADNI-only final fits using recovered specifications.",
            "`results/csvs/phase3_lock_repair/b1_d1_original_copy_equivalence.csv`",
            "Use v2 for inference; do not say all five families were original-lock artifacts.",
        ],
        [
            "Rounding",
            "Reports commonly show four decimals; CSVs contain greater precision.",
            "All machine-readable phase CSVs",
            "Use consistent three decimals plus 95% CI in prose, and retain full precision in supplementary tables.",
        ],
        [
            "Biological recovery wording",
            "Both CDR and diagnosis outcomes are observed clinical-state changes.",
            "`results/csvs/phase5/cdr_diagnosis_agreement_summary.csv`",
            "Use improvement/reversion/reclassification; avoid recovery or disease reversal.",
        ],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "discrepancy/risk",
            "what differs",
            "authoritative evidence",
            "required manuscript handling",
        ],
    ).to_markdown(index=False)


def manuscript_selection() -> str:
    return """
### Recommended six main display tables

1. **Table 1 — Cohort characteristics:** Phase 5 table 01.
2. **Table 2 — Cohort flow and transition support:** Phase 5 table 02,
   with detailed censoring in the supplement.
3. **Table 3 — Internal and untouched external model performance:** combine
   Phase 5 tables 03 and 04 as panels A/B.
4. **Table 4 — Paired external D1-versus-D0 comparison:** Phase 5 table 05.
5. **Table 5 — Untouched calibration and cross-fitted updating:** Phase 5
   table 06; include updating labels in every row.
6. **Table 6 — CDR-versus-diagnosis endpoint agreement:** Phase 5 table 10.

### Recommended six main figures

1. Phase 5 Figure 1 — study/cohort flow.
2. Phase 5 Figure 3 — locked dynamic external-validation workflow.
3. Phase 5 Figure 4 — internal versus external discrimination.
4. Phase 5 Figure 6 — untouched versus recalibrated calibration.
5. Phase 5 Figure 7 — paired D1-versus-D0 external comparison.
6. Phase 5 Figure 9 — CDR-versus-diagnosis improvement overlap.

Use Phase 5 Figure 12 as the graphical abstract rather than an additional main
results figure.

### Supplementary figures

- Phase 5 Figures 2, 5, 8, 10, and 11.
- Phase 3 nested-validation, penalty, diagnostics, coefficient-stability,
  lagged-MMSE, slope, complete-case, and benchmark figures.
- Phase 4 reconstructed-CDR and detailed distribution-shift figures.
- Phase 1/2 source, missingness, longitudinal-depth, MRI-readiness, and
  fold-balance figures.

### Supplementary tables

Use Phase 5 tables 07–09 and 11–12; participant-bootstrap CIs; B3/D3,
24/60-month and clinical-MCI sensitivities; recalibration parameters;
endpoint reconstruction; diagnosis timing/relapse; missingness; and full QC
tables.

### Results not recommended for the primary manuscript

- Individual predictions, landmark rows, participant-level transition files,
  bootstrap replicates, fold assignments, and influential-participant lists.
- Unstable subgroup estimates with very few transition events.
- AIBL discrimination/calibration (not calculated).
- MRI or fluid-biomarker prediction claims (models not supported).
- Recalibrated estimates presented as external validation.
- Direct B1-versus-D1 superiority as if populations were paired.
- Causal, clinical-deployment, treatment-threshold, or biological-recovery
  language.

### Best narrative order

Clinical problem and literature gap → bidirectional CDR endpoint → cohort
roles and locked workflow → ADNI internal selection → untouched OASIS
performance → paired D1-versus-D0 value → calibration failure and shift →
secondary recalibration → reconstructed endpoint → diagnosis-reversion
definition comparison → AIBL/MRI limitations → implications and
research-grade restrictions.
"""


def journal_positioning() -> str:
    rows = [
        [
            "Alzheimer’s & Dementia: Diagnosis, Assessment & Disease Monitoring",
            "Excellent: diagnostic/monitoring instrument and algorithm development/validation.",
            "Genuine locked external validation plus calibration-first bidirectional dynamic prediction.",
            "Provisional OASIS endpoint and lack of immediate clinical deployment.",
            "Lead with algorithm validation, calibration, endpoint sensitivity, and transparent negative transport findings.",
            "1 — primary",
        ],
        [
            "Alzheimer’s Research & Therapy",
            "Good but more translationally demanding; official scope emphasizes translational AD research.",
            "Practical local recalibration/endpoint-audit framework.",
            "Clinical actionability and biomarker/therapy connection may be viewed as limited.",
            "Stress research-to-practice transport lessons without overstating readiness.",
            "3 — ambitious alternative",
        ],
        [
            "Brain Communications",
            "Strong neurological-methods fit; accepts rigorous replication and important negative findings.",
            "External reproducibility, bidirectional staging, and transparent calibration failure.",
            "Broad neurological readership; eight-display-item limit requires compression.",
            "Frame as rigorous translational neuroscience and use six main displays plus supplement.",
            "2 — strong alternative",
        ],
        [
            "BMC Medical Research Methodology",
            "Moderate: official scope includes healthcare study design, data quality, statistics, and modelling.",
            "Outcome-blind lock, participant-grouped dynamic validation, transport/recalibration separation.",
            "Single disease application may seem insufficiently general as a methods contribution.",
            "Emphasize generalizable validation architecture and empirically show how methodological choices change conclusions.",
            "4 — methods-focused fallback",
        ],
    ]
    return (
        pd.DataFrame(
            rows,
            columns=[
                "journal",
                "scope fit",
                "strongest selling point",
                "likely editorial concern",
                "required framing",
                "priority",
            ],
        ).to_markdown(index=False)
        + "\n\nOfficial scope links checked 2026-07-31: "
        "[DADM product information](https://alz-journals.onlinelibrary.wiley.com/hub/journal/23528729/homepage/productinformation.html); "
        "[Alzheimer's Research & Therapy aims and scope](https://link.springer.com/journal/13195/aims-and-scope); "
        "[Brain Communications instructions](https://academic.oup.com/BRAINCOMMS/pages/General_Instructions); "
        "[BMC Medical Research Methodology aims and scope](https://link.springer.com/journal/12874/aims-and-scope)."
    )


def artifact_paths() -> list[Path]:
    paths: set[Path] = {
        ROOT / "LiteratureAndPlan/Plan.md",
        ROOT / "LiteratureAndPlan/alzheimer_literature_review_2022_2026.md",
        ROOT / "LiteratureAndPlan/Alternate_Study_Directions.md",
        OUTPUT,
        POINTER,
        INVENTORY,
        VERIFICATION_REPORT,
        ROOT / "scripts/build_complete_project_synthesis.py",
        ROOT / "scripts/verify_complete_project_synthesis.py",
    }
    for pattern in [
        "reports/feasibility/*.md",
        "reports/demographics/*.md",
        "reports/phase1/*.md",
        "reports/phase2/*.md",
        "reports/phase3/*.md",
        "reports/phase3_lock_repair/*.md",
        "reports/phase4/*.md",
        "reports/phase5/*.md",
        "results/csvs/phase1/**/*.csv",
        "results/csvs/phase2/**/*.csv",
        "results/csvs/phase3/**/*.csv",
        "results/csvs/phase3_lock_repair/**/*.csv",
        "results/csvs/phase4/**/*.csv",
        "results/csvs/phase5/**/*.csv",
        "results/plots/phase1/*",
        "results/plots/phase2/*",
        "results/plots/phase3/*",
        "results/plots/phase4/*",
        "results/plots/phase5/*",
        "models/phase3/phase4_lock/**/*",
        "models/phase3/phase4_lock_v2/**/*",
        "notebooks/Phase1/*.ipynb",
        "notebooks/Phase2/*.ipynb",
        "notebooks/Phase3/*.ipynb",
        "notebooks/Phase4/*.ipynb",
        "notebooks/Phase5/*.ipynb",
        "notebooks/DataExploration/**/*.ipynb",
        "scripts/phase1*.py",
        "scripts/phase2*.py",
        "scripts/phase3*.py",
        "scripts/phase4*.py",
        "scripts/phase5*.py",
        "scripts/create_phase*.py",
        "scripts/demographics_pipeline.py",
        "scripts/notebook_phase_review.py",
    ]:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(paths)


def artifact_phase(path: Path) -> str:
    relative = rel(path).lower()
    for phase in [
        "phase3_lock_repair",
        "phase1",
        "phase2",
        "phase3",
        "phase4",
        "phase5",
    ]:
        if phase in relative:
            return phase
    if "demographic" in relative or "feasibility" in relative:
        return "feasibility"
    return "project-wide"


def artifact_type(path: Path) -> str:
    relative = rel(path)
    if relative.startswith("reports/"):
        return "report"
    if relative.startswith("results/csvs/"):
        return "table"
    if relative.startswith("results/plots/"):
        return "figure"
    if relative.startswith("models/"):
        return "model artifact"
    if relative.startswith("notebooks/"):
        return "notebook"
    if relative.startswith("scripts/"):
        return "script"
    if relative.startswith("LiteratureAndPlan/"):
        return "plan/literature"
    if relative.startswith("evidenceBOOK/"):
        return "evidence book"
    return "other"


def write_inventory(embedded: dict[str, bool]) -> pd.DataFrame:
    rows = []
    for path in artifact_paths():
        exists = path.is_file()
        row_count = column_count = ""
        participants = ""
        if exists and path.suffix.lower() == ".csv":
            row_count, column_count, _, participants = csv_shape(path)
        relative = rel(path)
        rows.append(
            {
                "artifact_type": artifact_type(path),
                "phase": artifact_phase(path),
                "relative_path": relative,
                "exists": exists,
                "bytes": path.stat().st_size if exists else 0,
                "sha256": (
                    sha256(path)
                    if exists and path != INVENTORY
                    else ""
                ),
                "rows": row_count,
                "columns": column_count,
                "participant_count_or_note": participants,
                "embedded_full_table": embedded.get(relative, False),
                "manuscript_recommendation": (
                    manuscript_use(path, metadata_only_csv(path))
                    if path.suffix.lower() == ".csv"
                    else ""
                ),
                "referenced_role": (
                    "authoritative synthesis"
                    if path == OUTPUT
                    else "independent synthesis verification"
                    if path == VERIFICATION_REPORT
                    else "source or supporting artifact"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(INVENTORY, index=False)
    return frame


def build_document(master_tables: str) -> str:
    return f"""# Complete Project Results Synthesis

**Authoritative location:** `evidenceBOOK/Complete_Project_Results_Synthesis.md`  
**Generated:** {GENERATED_AT}  
**Scope:** read-only synthesis of completed Phases 1–5 and the outcome-blind
Phase 3 lock repair. No model was fitted, no analysis was rerun, and no frozen
artifact was modified.

This is the project evidence ledger for manuscript selection. Numerical
claims defer to machine-readable CSVs; reports provide interpretation. The
short file at `LiteratureAndPlan/Complete_Project_Results_Synthesis.md` points
here so there is only one authoritative synthesis.

## Implementation and pipeline overview

```text
ANALYSIS CHARTER + LITERATURE GAPS
                 ↓
     ┌───────────┼───────────┐
     ↓           ↓           ↓
 ADNI raw    OASIS-3 raw   AIBL raw
     └───────────┼───────────┘
                 ↓
 PHASE 1: codebooks → canonical visits → CDR states → first CDR-0.5 index
                 ↓
 PHASE 2: baseline rows + 90-day landmarks
          → temporal predictor windows
          → participant-grouped folds
          → ADNI-only preprocessing
          ├→ MRI readiness gate → EXCLUDED
          └→ AIBL readiness gate → DESCRIPTIVE ONLY
                 ↓
 PHASE 3 (ADNI ONLY):
   B0/B1/B2/B3/B4 baseline branch
                  ╲
                   → nested 5×5 validation → B1 selected
                  ╱
   D0/D1/D2/D3/D4/D5 dynamic branch
                   → D1 selected over D0
                 ↓
 IMMUTABLE B1/D1 LOCK
                 ↓
 OUTCOME-BLIND LOCK REPAIR:
   byte-identical B1/D1 + ADNI-only D0/B3/D3
                 ↓
        phase4_lock_v2 (55/55 hashes)
                 ↓
 PHASE 4 STAGE GATES:
   verify lock
      → build outcome-hidden OASIS features
      → freeze/hash predictions
      → independent prediction regeneration
      → open OASIS outcomes
      → UNTOUCHED EXTERNAL VALIDATION (primary)
              ├→ endpoint reconstruction (secondary)
              ├→ recalibration/model updating (secondary)
              └→ distribution-shift audit
                 ↓
 PHASE 5 PARALLEL SECONDARY BRANCHES:
   diagnosis reversion ─┐
   CDR/diagnosis overlap ├→ final synthesis + manuscript assets
   AIBL descriptive ────┘
                 ↓
 INDEPENDENT FINAL QC (37/37 PASS)
                 ↓
 EVIDENCE BOOK + MANUSCRIPT SELECTION
```

### Model-key quick reference

The leading letter means **B = baseline** or **D = dynamic landmark**. The
number identifies a prespecified model specification; it is not a performance
rank. In particular, D0 is the static-information dynamic comparator and D1
is the updated-cognition model.

{model_glossary()}

### Statistical and data terms

{term_glossary()}

## 1. Executive summary

- **Final research question:** among participants observed at CDR 0.5, can a
  low-dimensional longitudinal clinical model update probabilities of CDR 0
  improvement, CDR 0.5 persistence, and CDR ≥1 progression over 12 and 36
  months, and do those probabilities transport from ADNI to OASIS-3?
- **Cohort roles:** ADNI development/internal validation; OASIS-3 locked
  untouched external validation; AIBL descriptive only.
- **Primary outcomes:** first transition from CDR 0.5 to CDR 0 or CDR ≥1,
  with remaining CDR 0.5 as the third horizon state.
- **Selected models:** B1 baseline clinical core and D1 updated-cognition
  dynamic model; D0 comparator; B3/D3 sensitivities.
- **Primary horizons:** 12 and 36 months; 24/60 months secondary.
- **Internal finding:** D1 met the prespecified selection rule versus D0,
  improving mean integrated multistate Brier by 0.0050 without material
  calibration-deviation worsening.
- **External finding:** D1 retained AUC 0.6800/0.7212 at 12 months and
  0.7472/0.7293 at 36 months for improvement/progression, and improved
  externally over D0.
- **Calibration finding:** D1 improvement O/E was 2.1535/2.2826 at 12/36
  months, showing substantial underprediction.
- **Transportability:** **TRANSPORTABLE WITH RECALIBRATION**; untouched
  validation and OASIS model updating remain separate.
- **Strongest contribution:** genuinely locked, calibration-first external
  validation of bidirectional dynamic risk with explicit endpoint sensitivity.
- **Main limitation:** cross-cohort event-incidence, endpoint, assessment, and
  follow-up differences, with incomplete death/dropout reasons.
- **Final project decision:** **READY FOR MANUSCRIPT SUBMISSION AFTER MANUAL
  AUTHOR/ETHICS/DATA-SHARING COMPLETION.**

## 2. Original plan versus completed study

{original_plan_table()}

## 3. Literature-gap alignment

{literature_gap_table()}

## 4. Phase-by-phase synthesis

{phase_sections()}

## 5. Master result tables

Rules for this ledger:

- Exact CSV strings are embedded, including source precision.
- Long tables are collapsed with `<details>`.
- Participant-level, landmark-level, prediction-level, bootstrap-replicate,
  fold-assignment, manifest, and individual residual/influence files are never
  embedded in full.
- The independent verifier reconstructs every embedded Markdown table from its
  CSV and checks exact equality.

{master_tables}

## 6. Complete plot catalog

PNG/SVG pairs are treated as one unique figure. DPI is read from the PNG
metadata. Phase 5 recreations are preferred over conceptually duplicate
earlier-phase figures.

{figure_catalog()}

## 7. Primary finding synthesis

{primary_findings()}

## 8. Claim-to-evidence matrix

{claim_matrix()}

## 9. Consistency audit

Machine-readable CSVs are authoritative when prose and tables differ. Known
scope/unit differences are retained rather than silently rewritten.

{consistency_audit()}

## 10. Manuscript selection recommendation

{manuscript_selection()}

## 11. Journal positioning

{journal_positioning()}

## 12. Remaining manuscript tasks

Only the following tasks remain:

1. Verify every literature citation against the original paper and update the
   review's July 2026 search claims if submission occurs materially later.
2. Complete the applicable TRIPOD/TRIPOD+AI checklist and map every item to
   manuscript or supplement.
3. Complete a PROBAST-oriented risk-of-bias/applicability assessment, with
   particular attention to participants, endpoint construction, censoring,
   repeated landmarks, and analysis.
4. Insert cohort-specific ethics/IRB, informed-consent, and data-use language
   from the current agreements.
5. Add funding, cohort/consortium acknowledgements, competing interests, and
   CRediT author contributions.
6. Finalize author names, affiliations, ORCIDs, and corresponding-author
   details.
7. Finalize the data/code availability statement without redistributing raw or
   participant-level cohort data.
8. Obtain clinical-author review of CDR terminology, provisional OASIS
   `CDRTOT`, diagnosis reversion, and non-recovery wording.
9. Select the target journal and apply its current word, abstract, display,
   graphical-abstract, reporting-checklist, and file-format requirements.
10. Decide whether the diagnosis-reversion analysis is retained as a major
    secondary section or moved primarily to the supplement.

## Provenance and independent verification

- Referenced-artifact inventory:
  `results/csvs/phase5/complete_project_synthesis_inventory.csv`.
- Independent verifier: `scripts/verify_complete_project_synthesis.py`.
- Verification report:
  `evidenceBOOK/Complete_Project_Results_Synthesis_Verification.md`.

## Final completeness checklist

- [x] Implementation pipeline and parallel/sequential gates documented.
- [x] B0/B1/B2/B3/B4 and D0/D1/D2/D3/D4/D5 terms explained.
- [x] Executive conclusion, contribution, limitation, and decision recorded.
- [x] Original plan reconciled with the completed study.
- [x] All ten literature gaps classified.
- [x] Feasibility, Phases 1–5, and lock repair synthesized.
- [x] Every phase CSV inventoried; aggregate tables embedded exactly.
- [x] Participant/prediction/replicate/manifest tables excluded from full embedding.
- [x] Every generated phase figure cataloged as a PNG/vector pair.
- [x] Primary findings linked to machine-readable CSVs and reports.
- [x] Claim-to-evidence and wording guardrails provided.
- [x] Known discrepancies and unit/scope traps retained explicitly.
- [x] Main/supplementary manuscript tables and figures recommended.
- [x] Four target journals compared.
- [x] Remaining manuscript tasks limited to genuinely unfinished work.
- [ ] Independent synthesis verifier pass recorded in the separate verification report.
"""


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    master_tables, embedded = master_result_tables()
    document = build_document(master_tables)
    OUTPUT.write_text(document, encoding="utf-8")
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    POINTER.write_text(
        "# Complete Project Results Synthesis\n\n"
        "The single authoritative synthesis is:\n\n"
        "`evidenceBOOK/Complete_Project_Results_Synthesis.md`\n\n"
        "This pointer is retained because the original requested location was "
        "`LiteratureAndPlan/Complete_Project_Results_Synthesis.md`.\n",
        encoding="utf-8",
    )
    inventory = write_inventory(embedded)
    print(
        json.dumps(
            {
                "authoritative_synthesis": rel(OUTPUT),
                "pointer": rel(POINTER),
                "inventory": rel(INVENTORY),
                "inventory_rows": len(inventory),
                "embedded_aggregate_tables": int(
                    inventory["embedded_full_table"].sum()
                ),
                "synthesis_bytes": OUTPUT.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
