"""Read-only helpers for the five phase review notebooks.

These functions deliberately summarize existing, frozen artifacts.  They do
not run modelling pipelines, overwrite results, or read raw cohort data.
"""

from __future__ import annotations

import json
import math
import platform
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from IPython.display import Markdown, display
from PIL import Image, ImageDraw, ImageFont

# The project figures are known local 1200-DPI exports, not untrusted uploads.
# Their pixel counts exceed Pillow's generic decompression-bomb threshold.
Image.MAX_IMAGE_PIXELS = None


PHASE_TITLES = {
    1: "Data Foundation and Endpoint Validation",
    2: "Landmark Datasets, Predictors, Missingness, and Harmonization",
    3: "ADNI Model Development and Internal Validation",
    4: "Locked OASIS-3 External Validation",
    5: "Secondary Analyses and Final Reporting",
}

PHASE_STATES = {
    1: ("COMPLETE", "Existing Phase 1 artifacts are reviewed below."),
    2: ("COMPLETE", "Existing Phase 2 artifacts are reviewed below."),
    3: (
        "COMPLETE",
        "Existing Phase 3 artifacts are reviewed below; the Phase 4 lock remains active.",
    ),
    4: (
        "ON HOLD — STAGE A LOCK AUDIT",
        "The mandatory frozen-model completeness check failed. This notebook does not inspect OASIS-3 record-level features or outcomes.",
    ),
    5: (
        "NOT STARTED",
        "Secondary analyses are intentionally deferred until locked external validation is complete.",
    ),
}

KEY_TABLES = {
    1: [
        "index_cohort_flow_all_cohorts.csv",
        "cdr_transition_summary_all_cohorts.csv",
        "horizon_feasibility_all_cohorts.csv",
        "longitudinal_depth_all_cohorts.csv",
        "clinical_core_missingness.csv",
        "qc_results.csv",
    ],
    2: [
        "baseline_event_counts.csv",
        "dynamic_landmark_support.csv",
        "model_ready_dataset_inventory.csv",
        "clinical_core_missingness.csv",
        "complete_case_event_retention.csv",
        "adni_fold_balance.csv",
        "qc_results.csv",
    ],
    3: [
        "final_model_selection_summary.csv",
        "model_comparison_table.csv",
        "discrimination_and_horizon_metrics.csv",
        "brier_score_table.csv",
        "outer_fold_event_information.csv",
        "probability_coherence_checks.csv",
        "phase3_qc_results.csv",
    ],
}


def find_project_root(start: Path | str | None = None) -> Path:
    """Find the project root from a notebook or command-line working directory."""
    start_path = Path(start or Path.cwd()).expanduser().resolve()
    candidates = [start_path, *start_path.parents]
    for candidate in candidates:
        if (
            (candidate / "scripts").is_dir()
            and (candidate / "notebooks").is_dir()
            and (candidate / "ProcessedData").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find the project root containing scripts/, notebooks/, and ProcessedData/."
    )


def phase_heading(phase: int, root: Path) -> None:
    """Display execution context and the phase completion boundary."""
    title = PHASE_TITLES[phase]
    state, note = PHASE_STATES[phase]
    display(
        Markdown(
            f"""
## Phase {phase} execution status

**{state}**

{note}
"""
        )
    )
    context = pd.DataFrame(
        {
            "field": [
                "project_root",
                "phase",
                "phase_title",
                "notebook_executed_at",
                "python",
                "review_mode",
            ],
            "value": [
                str(root),
                phase,
                title,
                datetime.now().astimezone().isoformat(timespec="seconds"),
                platform.python_version(),
                "read-only artifact review",
            ],
        }
    )
    display(context)


def _phase_paths(root: Path, phase: int) -> list[Path]:
    """Return every artifact directly attributable to a phase."""
    paths: set[Path] = set()
    phase_name = f"phase{phase}"

    for base in [
        root / "reports" / phase_name,
        root / "results" / "csvs" / phase_name,
        root / "results" / "plots" / phase_name,
        root / "models" / phase_name,
    ]:
        if base.exists():
            paths.update(path for path in base.rglob("*") if path.is_file())

    for pattern_root, pattern in [
        (root / "scripts", f"{phase_name}_*.py"),
        (root / "config", f"{phase_name}_*"),
    ]:
        if pattern_root.exists():
            paths.update(path for path in pattern_root.glob(pattern) if path.is_file())

    processed_root = root / "ProcessedData"
    if processed_root.exists():
        paths.update(
            path
            for path in processed_root.rglob("*")
            if path.is_file()
            and (
                path.name.lower().startswith(f"{phase_name}_")
                or phase_name.lower() in {part.lower() for part in path.parts}
            )
        )
    return sorted(paths, key=lambda path: str(path.relative_to(root)).lower())


def _category(relative_path: Path) -> str:
    parts = relative_path.parts
    if parts[0] == "reports":
        return "report"
    if parts[:2] == ("results", "csvs"):
        return "result_table"
    if parts[:2] == ("results", "plots"):
        return "figure"
    if parts[0] == "ProcessedData":
        return "processed_dataset"
    if parts[0] == "models":
        return "model_artifact"
    if parts[0] == "scripts":
        return "source_script"
    if parts[0] == "config":
        return "configuration"
    return "other"


def build_phase_inventory(root: Path, phase: int) -> pd.DataFrame:
    """Build a complete file-level artifact inventory for a phase."""
    rows = []
    for path in _phase_paths(root, phase):
        stat = path.stat()
        relative = path.relative_to(root)
        rows.append(
            {
                "category": _category(relative),
                "path": relative.as_posix(),
                "format": path.suffix.lower().lstrip(".") or "(none)",
                "size_kib": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(
                    timespec="seconds"
                ),
            }
        )
    return pd.DataFrame(
        rows, columns=["category", "path", "format", "size_kib", "modified"]
    )


def display_artifact_inventory(root: Path, phase: int) -> pd.DataFrame:
    """Display the complete artifact inventory and category totals."""
    inventory = build_phase_inventory(root, phase)
    display(Markdown("## Complete artifact inventory"))
    if inventory.empty:
        display(Markdown(f"No Phase {phase} output artifacts exist yet."))
        return inventory

    summary = (
        inventory.groupby("category", as_index=False)
        .agg(files=("path", "size"), total_kib=("size_kib", "sum"))
        .sort_values("category")
    )
    summary["total_mib"] = (summary.pop("total_kib") / 1024).round(2)
    display(summary)
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_colwidth",
        140,
        "display.width",
        220,
    ):
        display(inventory)
    print(f"Inventory total: {len(inventory):,} files")
    return inventory


def _read_table(path: Path, max_rows: int = 20) -> tuple[pd.DataFrame, dict]:
    """Read a bounded preview and compute useful metadata."""
    suffix = path.suffix.lower()
    size_bytes = path.stat().st_size
    full_read_limit = 2 * 1024 * 1024

    if suffix == ".csv":
        if size_bytes <= full_read_limit:
            full = pd.read_csv(path, low_memory=False)
            preview = full if len(full) <= max_rows else full.head(max_rows)
            metadata = {
                "rows": len(full),
                "columns": len(full.columns),
                "shown_rows": len(preview),
                "display_mode": "complete" if len(full) <= max_rows else "first rows",
            }
            return preview, metadata
        preview = pd.read_csv(path, nrows=max_rows, low_memory=False)
        line_count = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                line_count += chunk.count(b"\n")
        metadata = {
            "rows": max(0, line_count - 1),
            "columns": len(preview.columns),
            "shown_rows": len(preview),
            "display_mode": "first rows (large table)",
        }
        return preview, metadata

    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        preview = pd.read_parquet(path).head(max_rows)
        metadata = {
            "rows": parquet.metadata.num_rows,
            "columns": parquet.metadata.num_columns,
            "shown_rows": len(preview),
            "display_mode": "first rows",
        }
        return preview, metadata

    raise ValueError(f"Unsupported table format: {suffix}")


def display_key_results(root: Path, phase: int) -> None:
    """Display the most decision-relevant phase tables in full or bounded previews."""
    display(Markdown("## Decision-relevant results"))
    table_dir = root / "results" / "csvs" / f"phase{phase}"
    for name in KEY_TABLES.get(phase, []):
        path = table_dir / name
        if not path.exists():
            display(Markdown(f"- Missing expected key table: `{path.relative_to(root)}`"))
            continue
        preview, metadata = _read_table(path, max_rows=100)
        display(Markdown(f"### `{path.relative_to(root).as_posix()}`"))
        display(pd.DataFrame([metadata]))
        with pd.option_context(
            "display.max_rows", 100, "display.max_columns", None, "display.max_colwidth", 100
        ):
            display(preview)


def _display_json(path: Path) -> None:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        rendered = json.dumps(content, indent=2, ensure_ascii=False, default=str)
    except Exception as exc:
        rendered = f"Unable to parse JSON: {exc}"
    display(Markdown(f"```json\n{rendered}\n```"))


def display_all_result_tables(root: Path, phase: int, max_rows: int = 20) -> None:
    """Represent every saved result table in notebook output."""
    table_dir = root / "results" / "csvs" / f"phase{phase}"
    display(Markdown("## All saved result tables"))
    if not table_dir.exists():
        display(Markdown(f"No result table directory exists at `{table_dir.relative_to(root)}`."))
        return

    paths = sorted(
        (path for path in table_dir.rglob("*") if path.is_file()),
        key=lambda path: path.name.lower(),
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        display(Markdown(f"### `{relative}`"))
        if path.suffix.lower() == ".json":
            _display_json(path)
            continue
        if path.suffix.lower() not in {".csv", ".parquet"}:
            print("Listed in the artifact inventory; no inline renderer is configured.")
            continue
        try:
            preview, metadata = _read_table(path, max_rows=max_rows)
            metadata.update(
                {
                    "file_size_mib": round(path.stat().st_size / (1024**2), 3),
                    "path": relative,
                }
            )
            display(pd.DataFrame([metadata]))
            with pd.option_context(
                "display.max_rows",
                max_rows,
                "display.max_columns",
                None,
                "display.max_colwidth",
                80,
            ):
                display(preview)
        except Exception as exc:
            display(Markdown(f"**Preview error:** `{type(exc).__name__}: {exc}`"))


def display_qc_summary(root: Path, phase: int) -> None:
    """Display all QC checks and a concise pass/fail summary."""
    candidates = [
        root / "results" / "csvs" / f"phase{phase}" / f"phase{phase}_qc_results.csv",
        root / "results" / "csvs" / f"phase{phase}" / "qc_results.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    display(Markdown("## Quality-control results"))
    if path is None:
        display(Markdown("No phase QC result table has been created."))
        return

    qc = pd.read_csv(path, low_memory=False)
    status_col = next(
        (column for column in qc.columns if column.lower() in {"status", "result"}), None
    )
    if status_col:
        counts = qc[status_col].fillna("(missing)").astype(str).value_counts()
        display(counts.rename_axis("status").reset_index(name="checks"))
        failures = qc[~qc[status_col].astype(str).str.upper().isin({"PASS", "PASSED"})]
        print(
            f"QC decision: {'PASS' if failures.empty else 'REVIEW REQUIRED'} "
            f"({len(qc):,} checks; {len(failures):,} non-pass)"
        )
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 120
    ):
        display(qc)


def display_reports(root: Path, phase: int) -> None:
    """Render every Markdown report for the phase into notebook output."""
    report_dir = root / "reports" / f"phase{phase}"
    display(Markdown("## Full saved reports"))
    if not report_dir.exists():
        display(Markdown(f"No Phase {phase} report directory exists yet."))
        return
    reports = sorted(report_dir.rglob("*.md"), key=lambda path: path.name.lower())
    if not reports:
        display(Markdown(f"No Markdown reports exist in `{report_dir.relative_to(root)}`."))
        return
    for path in reports:
        display(Markdown(f"---\n\n### Source: `{path.relative_to(root).as_posix()}`"))
        display(Markdown(path.read_text(encoding="utf-8", errors="replace")))


def _gallery_page(paths: Iterable[Path], root: Path, columns: int = 2) -> Image.Image:
    """Build a memory-bounded thumbnail sheet from high-resolution PNG files."""
    paths = list(paths)
    tile_width, tile_height = 660, 500
    rows = math.ceil(len(paths) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)

    for index, path in enumerate(paths):
        column = index % columns
        row = index // columns
        x0, y0 = column * tile_width, row * tile_height
        label = "\n".join(
            textwrap.wrap(path.relative_to(root).as_posix(), width=65)
        )
        draw.multiline_text((x0 + 12, y0 + 10), label, fill="black", font=font)
        try:
            with Image.open(path) as image:
                image.thumbnail((tile_width - 24, tile_height - 75), Image.Resampling.LANCZOS)
                rendered = image.convert("RGB")
                x = x0 + (tile_width - rendered.width) // 2
                y = y0 + 65 + (tile_height - 65 - rendered.height) // 2
                canvas.paste(rendered, (x, y))
        except Exception as exc:
            draw.text(
                (x0 + 12, y0 + 90),
                f"Unable to render: {type(exc).__name__}: {exc}",
                fill="red",
                font=font,
            )
        draw.rectangle(
            (x0, y0, x0 + tile_width - 1, y0 + tile_height - 1),
            outline="#B0B0B0",
            width=1,
        )
    return canvas


def display_plot_gallery(root: Path, phase: int, plots_per_page: int = 6) -> None:
    """Display every unique PNG figure as compact gallery pages."""
    plot_dir = root / "results" / "plots" / f"phase{phase}"
    display(Markdown("## Complete figure gallery"))
    if not plot_dir.exists():
        display(Markdown(f"No Phase {phase} plot directory exists yet."))
        return
    paths = sorted(plot_dir.rglob("*.png"), key=lambda path: path.name.lower())
    if not paths:
        display(Markdown("No PNG figures have been created."))
        return
    print(
        f"Displaying {len(paths):,} PNG figures. Matching SVG files remain listed "
        "in the complete artifact inventory."
    )
    for offset in range(0, len(paths), plots_per_page):
        page_paths = paths[offset : offset + plots_per_page]
        display(
            Markdown(
                f"### Figure page {offset // plots_per_page + 1} "
                f"({offset + 1}–{offset + len(page_paths)} of {len(paths)})"
            )
        )
        page = _gallery_page(page_paths, root=root, columns=2)
        display(page)


def display_phase4_lock(root: Path) -> None:
    """Review the frozen handoff without accessing any OASIS-3 record-level data."""
    display(Markdown("## Frozen Phase 4 handoff (Phase 3 artifacts only)"))
    lock_dir = root / "models" / "phase3" / "phase4_lock"
    allowed = {
        ".csv",
        ".json",
    }
    paths = sorted(
        path
        for path in lock_dir.glob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )
    for path in paths:
        display(Markdown(f"### `{path.relative_to(root).as_posix()}`"))
        if path.suffix.lower() == ".json":
            _display_json(path)
        else:
            frame = pd.read_csv(path, low_memory=False)
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 120
            ):
                display(frame)
    print(
        "Safety confirmation: this cell read only models/phase3/phase4_lock metadata. "
        "No OASIS-3 record-level or outcome file was opened."
    )


def display_future_phase_plan(phase: int) -> None:
    """Display an explicit, non-executing plan for an unstarted phase."""
    if phase == 4:
        steps = [
            "Verify frozen model checksums and inference verification rows.",
            "Map OASIS-3 predictors using the frozen schema without outcome-guided changes.",
            "Run unchanged primary external discrimination and calibration evaluation.",
            "Quantify dataset shift and only then run labelled secondary recalibration.",
            "Write the locked external-validation decision and transportability report.",
        ]
        guardrail = (
            "Do not inspect OASIS-3 outcome-linked performance until the Phase 4 "
            "execution is explicitly authorized."
        )
    elif phase == 5:
        steps = [
            "Complete diagnosis-based MCI reversion sensitivity analyses.",
            "Evaluate MRI and fluid-biomarker enrichment only where coverage supports it.",
            "Run clearly labelled AIBL exploratory evaluation.",
            "Complete prespecified sensitivity analyses.",
            "Assemble final figures, tables, reporting, and manuscript-ready outputs.",
        ]
        guardrail = (
            "Do not interpret Phase 5 analyses before the locked Phase 4 primary "
            "external-validation result is complete."
        )
    else:
        raise ValueError("Future-phase plan is only defined for Phase 4 or 5.")

    lines = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    display(
        Markdown(
            f"""
## Planned work — not executed

{lines}

**Guardrail:** {guardrail}
"""
        )
    )
