#!/usr/bin/env python3
"""Create the review notebook for the completed Phase 3 lock repair."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "notebooks"
    / "Phase3"
    / "Phase3_OutcomeBlindLockRepair.ipynb"
)


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (clinical_env)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10.20",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    },
}

notebook["cells"] = [
    markdown(
        """
# Phase 3 Outcome-Blind Lock Repair

This notebook reviews the completed, separately versioned Phase 3 lock repair.
It does **not** fit models, open OASIS-3 participant-level data, merge external
outcomes, or resume Phase 4.

The independent verifier is implemented separately in
`scripts/phase3_lock_repair_qc.py`.
"""
    ),
    code(
        """
from pathlib import Path
import json
import platform

import pandas as pd
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
if not (ROOT / "models" / "phase3").exists():
    ROOT = Path.cwd().resolve().parents[1]

LOCK = ROOT / "models" / "phase3" / "phase4_lock_v2"
ORIGINAL_LOCK = ROOT / "models" / "phase3" / "phase4_lock"
TABLES = ROOT / "results" / "csvs" / "phase3_lock_repair"
REPORTS = ROOT / "reports" / "phase3_lock_repair"

print(f"Python: {platform.python_version()}")
print(f"Original immutable lock: {ORIGINAL_LOCK}")
print(f"Repaired lock: {LOCK}")
print(f"Repaired lock exists: {LOCK.exists()}")
"""
    ),
    markdown("## Final run status"),
    code(
        """
run_metadata = json.loads(
    (TABLES / "phase3_lock_repair_run_metadata.json").read_text()
)
display(pd.Series(run_metadata, name="value").to_frame())
"""
    ),
    markdown("## Complete frozen-model inventory"),
    code(
        """
inventory = pd.read_csv(LOCK / "model_inventory.csv")
display(
    inventory[
        [
            "model_id", "family", "horizon_months", "model_origin",
            "training_rows", "training_participants",
            "improvement_event_rows", "progression_event_rows",
            "improvement_alpha", "progression_alpha",
            "refit_during_repair", "sha256", "bytes",
        ]
    ].sort_values(["model_id", "horizon_months"]).reset_index(drop=True)
)
print(f"Frozen bundles: {len(inventory)}")
"""
    ),
    code(
        """
required = pd.MultiIndex.from_tuples(
    [
        ("B1", 12), ("B1", 24), ("B1", 36), ("B1", 60),
        ("D0", 12), ("D0", 36),
        ("D1", 12), ("D1", 24), ("D1", 36), ("D1", 60),
        ("B3", 12), ("B3", 36),
        ("D3", 12), ("D3", 36),
    ],
    names=["model_id", "horizon_months"],
)
present = pd.MultiIndex.from_frame(
    inventory[["model_id", "horizon_months"]]
)
coverage = (
    pd.DataFrame(index=required)
    .assign(present=[item in present for item in required])
    .reset_index()
)
display(coverage)
print(f"Every required model present: {coverage['present'].all()}")
"""
    ),
    markdown("## Recovered Phase 3 penalties — no retuning"),
    code(
        """
penalties = pd.read_csv(LOCK / "penalty_source_recovery.csv")
display(
    penalties[
        [
            "model_id", "horizon_months", "cause",
            "recovered_final_alpha", "outer_fold_selected_alphas",
            "recovery_rule", "source_path", "retuned_during_repair",
        ]
    ]
)
"""
    ),
    markdown("## D0, B3, and D3 ADNI training metadata"),
    code(
        """
training = pd.read_csv(LOCK / "repaired_model_training_metadata.csv")
display(
    training[
        [
            "model_id", "family", "horizon_months", "dataset_variant",
            "training_rows", "training_participants",
            "improvement_event_rows", "progression_event_rows",
            "improvement_alpha", "progression_alpha",
            "continuous_features", "categorical_features",
            "endpoint_construction", "baseline_hazard_construction",
            "retuned_during_repair",
        ]
    ]
)
"""
    ),
    markdown("## Per-model software versions"),
    code(
        """
software = pd.read_csv(LOCK / "model_software_versions.csv")
display(software.reset_index(drop=True))
"""
    ),
    markdown("## Original versus repaired B1/D1 equivalence"),
    code(
        """
equivalence = pd.read_csv(
    TABLES / "independent_b1_d1_equivalence.csv"
)
display(equivalence)
print(
    "All artifact hashes and predictions identical:",
    equivalence["status"].eq("PASS").all(),
)
"""
    ),
    markdown("## Independent D0/B3/D3 deterministic inference"),
    code(
        """
fixtures = pd.read_csv(
    TABLES / "independent_inference_fixture_verification.csv"
)
display(fixtures)
print(
    "All held-out ADNI fixtures reproduced:",
    fixtures["status"].eq("PASS").all(),
)
"""
    ),
    markdown("## D3 strictly pre-landmark MMSE verification"),
    code(
        """
d3_temporal = pd.read_csv(
    TABLES / "independent_d3_temporal_verification.csv"
)
display(d3_temporal)
print(
    "No same-day/post-landmark MMSE used:",
    d3_temporal["status"].eq("PASS").all(),
)
"""
    ),
    markdown("## Dynamic participant-grouping verification"),
    code(
        """
grouping = pd.read_csv(
    TABLES / "independent_participant_grouping_verification.csv"
)
display(grouping)
print(
    "Every dynamic grouping check passed:",
    grouping["status"].eq("PASS").all(),
)
"""
    ),
    markdown("## Preprocessing, features, coefficients, and baseline hazards"),
    code(
        """
preprocessing = pd.read_csv(LOCK / "frozen_preprocessing_parameters.csv")
schema = pd.read_csv(LOCK / "frozen_feature_schema.csv")
coefficients = pd.read_csv(LOCK / "frozen_model_coefficients.csv")
hazards = pd.read_csv(LOCK / "frozen_baseline_hazards.csv")

coverage_tables = pd.DataFrame(
    {
        "table": [
            "frozen_feature_schema",
            "frozen_preprocessing_parameters",
            "frozen_model_coefficients",
            "frozen_baseline_hazards",
        ],
        "rows": [
            len(schema), len(preprocessing), len(coefficients), len(hazards)
        ],
        "models": [
            schema["model_id"].nunique(),
            preprocessing["model_id"].nunique(),
            coefficients["model_id"].nunique(),
            hazards["model_id"].nunique(),
        ],
        "model_horizon_pairs": [
            schema[["model_id", "horizon_months"]].drop_duplicates().shape[0],
            preprocessing[["model_id", "horizon_months"]].drop_duplicates().shape[0],
            coefficients[["model_id", "horizon_months"]].drop_duplicates().shape[0],
            hazards[["model_id", "horizon_months"]].drop_duplicates().shape[0],
        ],
    }
)
display(coverage_tables)
display(schema.sort_values(["model_id", "horizon_months"]).reset_index(drop=True))
"""
    ),
    markdown("## Original and repaired checksum verification"),
    code(
        """
original_checks = pd.read_csv(
    TABLES / "independent_original_lock_verification.csv"
)
package_checks = pd.read_csv(
    TABLES / "independent_package_checksum_verification.csv"
)
display(
    pd.DataFrame(
        [
            {
                "scope": "original immutable lock",
                "files_checked": len(original_checks),
                "failed": int(original_checks["status"].ne("PASS").sum()),
                "status": (
                    "PASS"
                    if original_checks["status"].eq("PASS").all()
                    else "FAIL"
                ),
            },
            {
                "scope": "repaired v2 lock",
                "files_checked": len(package_checks),
                "failed": int(package_checks["status"].ne("PASS").sum()),
                "status": (
                    "PASS"
                    if package_checks["status"].eq("PASS").all()
                    else "FAIL"
                ),
            },
        ]
    )
)
"""
    ),
    markdown("## Outcome-blind access audit"),
    code(
        """
access = pd.read_csv(LOCK / "access_audit.csv")
forbidden_fragments = [
    "/processeddata/oasis3/",
    "/datasets/oasis3/",
    "/datasets/oasis/",
    "/results/csvs/phase1/oasis3/",
    "/results/csvs/phase2/oasis3/",
]
paths = "/" + access["path"].astype(str).str.lower()
forbidden = paths.map(
    lambda value: any(fragment in value for fragment in forbidden_fragments)
)
display(
    access.groupby("access_class", dropna=False)
    .size()
    .rename("records")
    .reset_index()
)
print(f"Input-read records: {len(access)}")
print(f"Forbidden OASIS participant-file paths: {int(forbidden.sum())}")
print(
    "OASIS participant-level features accessed:",
    access["oasis_participant_level_features_accessed"].astype(bool).any(),
)
print(
    "OASIS participant-level outcomes accessed:",
    access["oasis_participant_level_outcomes_accessed"].astype(bool).any(),
)
"""
    ),
    markdown("## Independent QC checklist"),
    code(
        """
qc = pd.read_csv(TABLES / "independent_qc_results.csv")
display(qc)
print(f"Critical checks: {int(qc['critical'].sum())}")
print(
    "Critical failures:",
    int((qc["critical"] & qc["status"].ne("PASS")).sum()),
)
"""
    ),
    markdown("## Reports and Phase 4 restart handoff"),
    code(
        """
for report_name in [
    "phase3_outcome_blind_lock_repair_report.md",
    "phase3_lock_repair_inference_compatibility_report.md",
    "phase3_lock_repair_independent_qc_report.md",
    "phase4_restart_handoff.md",
]:
    display(Markdown((REPORTS / report_name).read_text()))
"""
    ),
    markdown("## Final decision"),
    code(
        """
decision = run_metadata["final_decision"]
lock_path = ROOT / run_metadata["phase4_must_use_lock"]
display(
    Markdown(
        f"# {decision}\\n\\n"
        f"Phase 4 must use: `{lock_path}`\\n\\n"
        "**Phase 4 external validation has not been resumed.**"
    )
)
"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
