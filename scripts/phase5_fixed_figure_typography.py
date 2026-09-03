"""Render the six overflow-safe Phase 5 figures with larger non-title text.

Title font sizes intentionally match the previously QC-approved fixed figures.
Only legends, axes, ticks, annotations, and diagram/body labels are enlarged.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "results" / "plots" / "phase5"


def _save_and_show(fig: plt.Figure, stem: str) -> None:
    """Overwrite only the approved ``*_fixed`` outputs at 600 DPI."""
    png = PLOT_DIR / f"{stem}.png"
    svg = PLOT_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    print(f"Saved {png.relative_to(ROOT)} at 600 DPI with enlarged non-title text")
    plt.show()


def render_figure_01() -> None:
    """Figure 1: enlarge cohort-box text while retaining the 18-point title."""
    table = pd.read_csv(ROOT / "results/csvs/phase5/cohort_characteristics.csv")
    stem = "figure_01_study_and_cohort_flow_fixed"

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = ["#d7e9f7", "#e1f2e8", "#f7ead6"]
    y_positions = [0.64, 0.36, 0.08]
    for color, y, row in zip(colors, y_positions, table.itertuples()):
        box = patches.FancyBboxPatch(
            (0.055, y), 0.89, 0.20,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=color, edgecolor="#34495e", linewidth=2.0,
        )
        ax.add_patch(box)
        verification = "verified age" if bool(row.age_verified) else "age proxy unverified"
        ax.text(
            0.5, y + 0.135,
            f"{row.cohort}: {row.index_participants:,} CDR 0.5 index participants",
            ha="center", va="center", fontsize=15.5, fontweight="bold",
        )
        ax.text(
            0.5, y + 0.065, f"{row.role}  •  {verification}",
            ha="center", va="center", fontsize=13.0,
        )

    # Title size is intentionally unchanged from the prior fixed figure.
    ax.text(
        0.5, 0.965, "Study cohorts and analysis roles",
        ha="center", va="top", fontsize=18, fontweight="bold",
    )
    fig.subplots_adjust(left=0.025, right=0.975, top=0.99, bottom=0.02)
    _save_and_show(fig, stem)


def render_figure_02() -> None:
    """Figure S1: enlarge node and transition-count labels, not titles."""
    transition = pd.read_csv(
        ROOT / "results/csvs/phase1/cdr_transition_summary_all_cohorts.csv"
    )
    transition = transition[
        transition["transition_label"].isin(["CDR 0.5 -> 0", "CDR 0.5 -> >=1"])
    ]
    pivot = transition.pivot(index="cohort", columns="transition_label", values="events")
    stem = "figure_02_cdr_state_transition_diagram_fixed"

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0))
    positions = {"CDR 0": (0.14, 0.48), "CDR 0.5": (0.50, 0.48), "CDR ≥1": (0.86, 0.48)}
    for ax, cohort in zip(axes, ["adni", "oasis3", "aibl"]):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        for label, (x, y) in positions.items():
            circle = patches.Circle(
                (x, y), 0.14, facecolor="#eef3f7",
                edgecolor="#2c3e50", linewidth=2.0,
            )
            ax.add_patch(circle)
            ax.text(x, y, label, ha="center", va="center", fontsize=13, fontweight="bold")

        improvement = int(pivot.loc[cohort, "CDR 0.5 -> 0"])
        progression = int(pivot.loc[cohort, "CDR 0.5 -> >=1"])
        ax.annotate(
            "", xy=(0.285, 0.54), xytext=(0.355, 0.54),
            arrowprops=dict(arrowstyle="->", lw=2.8, color="#2a9d8f"),
        )
        ax.annotate(
            "", xy=(0.715, 0.42), xytext=(0.645, 0.42),
            arrowprops=dict(arrowstyle="->", lw=2.8, color="#d95f02"),
        )
        ax.text(0.32, 0.67, f"n={improvement}", ha="center", color="#167c6b", fontsize=13.5)
        ax.text(0.68, 0.28, f"n={progression}", ha="center", color="#a94600", fontsize=13.5)
        title = "OASIS-3" if cohort == "oasis3" else cohort.upper()
        # Cohort-title size is intentionally unchanged.
        ax.set_title(title, fontsize=15, fontweight="bold", pad=10)

    # Global-title size is intentionally unchanged.
    fig.suptitle("Observed CDR transitions from state 0.5", fontsize=18, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.07, wspace=0.08)
    _save_and_show(fig, stem)


def render_figure_03() -> None:
    """Figure 2: enlarge workflow labels and explanatory annotation only."""
    stem = "figure_03_dynamic_external_validation_workflow_fixed"
    stages = [
        "ADNI\nlandmarks",
        "Participant-grouped\nnested validation",
        "Frozen D1\nmodel",
        "Outcome-hidden\nOASIS prediction",
        "Untouched\nexternal validation",
        "Separate\nrecalibration",
    ]
    positions = [
        (0.18, 0.62), (0.50, 0.62), (0.82, 0.62),
        (0.82, 0.30), (0.50, 0.30), (0.18, 0.30),
    ]
    colors = ["#d9eef2", "#d9eef2", "#d9eef2", "#f4e5cf", "#f4e5cf", "#eadcf2"]

    fig, ax = plt.subplots(figsize=(13.4, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_w, box_h = 0.25, 0.18
    for (x, y), label, color in zip(positions, stages, colors):
        box = patches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=color, edgecolor="#34495e", linewidth=2.0,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=14, linespacing=1.18)

    for start, end in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        direction = 1 if x2 > x1 else -1
        ax.annotate(
            "", xy=(x2 - direction * box_w / 2 - direction * 0.01, y2),
            xytext=(x1 + direction * box_w / 2 + direction * 0.01, y1),
            arrowprops=dict(arrowstyle="->", lw=2.3, color="#34495e"),
        )
    ax.annotate(
        "", xy=(positions[3][0], positions[3][1] + box_h / 2 + 0.015),
        xytext=(positions[2][0], positions[2][1] - box_h / 2 - 0.015),
        arrowprops=dict(arrowstyle="->", lw=2.3, color="#34495e"),
    )

    # Title size is intentionally unchanged.
    ax.text(
        0.5, 0.945, "Dynamic model development and locked external-validation workflow",
        ha="center", va="top", fontsize=18, fontweight="bold",
    )
    ax.text(
        0.5, 0.07,
        "Untouched validation remains primary; OASIS-based recalibration is model updating",
        ha="center", va="center", fontsize=13.5, color="#6c3483",
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
    _save_and_show(fig, stem)


def render_figure_08() -> None:
    """Figure S3: enlarge axes and tick labels while preserving both title sizes."""
    shift = pd.read_csv(ROOT / "results/csvs/phase5/distribution_shift_synthesis.csv")
    continuous = shift[shift["shift_type"].eq("continuous_predictor")].head(8).copy()
    event = shift[
        shift["shift_type"].eq("event_rate") & shift["model_id"].eq("D1")
    ].copy()
    continuous["display_item"] = continuous["item"].str.replace("_", " ").map(
        lambda value: "\n".join(textwrap.wrap(value, width=20))
    )
    event["display_item"] = (
        event["horizon_months"].astype(int).astype(str)
        + "-month " + event["item"].str.replace("_", " ")
    )
    stem = "figure_08_predictor_and_event_rate_shift_fixed"

    fig, axes = plt.subplots(
        1, 2, figsize=(13.6, 7.3),
        gridspec_kw={"width_ratios": [1.40, 1.0]},
    )
    axes[0].barh(
        continuous["display_item"], continuous["difference_metric"],
        color=np.where(continuous["difference_metric"].ge(0), "#3182bd", "#e6550d"),
    )
    axes[0].axvline(0, color="#555555", linewidth=1.2)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Standardized mean difference\n(OASIS-3 − ADNI)", fontsize=14.5, labelpad=9)
    # Panel-title size is intentionally unchanged.
    axes[0].set_title("Predictor shift", fontsize=14, pad=10)

    axes[1].barh(event["display_item"], event["difference_metric"], color="#756bb1")
    axes[1].axvline(1, color="#555555", linestyle="--", linewidth=1.4)
    axes[1].set_xlabel("OASIS-3 / ADNI\nobserved-rate ratio", fontsize=14.5, labelpad=9)
    # Panel-title size is intentionally unchanged.
    axes[1].set_title("Event-rate shift", fontsize=14, pad=10)
    axes[1].invert_yaxis()

    for ax in axes:
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)

    # Global-title size is intentionally unchanged.
    fig.suptitle("Development-to-external dataset shift", fontsize=18, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.25, right=0.975, top=0.82, bottom=0.20, wspace=0.80)
    _save_and_show(fig, stem)


def render_figure_10() -> None:
    """Figure S4: enlarge legend/axes while retaining approved title typography."""
    trajectories = pd.read_csv(
        ROOT / "results/csvs/phase5/representative_dynamic_risk_trajectories.csv"
    )
    stem = "figure_10_representative_dynamic_risk_trajectories_fixed"
    order = ["improvement-risk-increase", "progression-risk-increase", "stable-risk-example"]
    titles = {
        "improvement-risk-increase": "Improvement risk increases",
        "progression-risk-increase": "Progression risk increases",
        "stable-risk-example": "Stable-risk example",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.6), sharey=True)
    for panel, (ax, label) in enumerate(zip(axes, order)):
        subset = trajectories.loc[
            trajectories["trajectory_label"].eq(label)
        ].sort_values("elapsed_years_from_index")
        ax.plot(
            subset["elapsed_years_from_index"], subset["probability_improvement"],
            "o-", label="Improvement", color="#2a9d8f", linewidth=2.3, markersize=6,
        )
        ax.plot(
            subset["elapsed_years_from_index"], subset["probability_progression"],
            "s-", label="Progression", color="#d95f02", linewidth=2.3, markersize=6,
        )
        # Panel-heading size is intentionally unchanged.
        ax.text(
            0.5, 1.035, titles[label], transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11, fontweight="semibold",
        )
        ax.set_xlabel("Years from index", fontsize=13.5, labelpad=8)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=11.5)
        ax.grid(alpha=0.20)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.02, 0.96, chr(65 + panel), transform=ax.transAxes,
            ha="left", va="top", fontsize=13.5, fontweight="bold",
        )

    axes[0].set_ylabel("Frozen D1 36-month probability", fontsize=13.5, labelpad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.84),
        ncol=2, columnspacing=2.5, handlelength=2.7, frameon=False, fontsize=13,
    )
    # Global-title size is intentionally unchanged from the second-pass fixed figure.
    fig.suptitle(
        "Representative out-of-fold dynamic risk trajectories",
        fontsize=17, fontweight="bold", y=0.965,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.66, bottom=0.17, wspace=0.17)
    _save_and_show(fig, stem)


def render_figure_11() -> None:
    """Figure S5: enlarge all non-title annotations, axes, ticks, and legend."""
    timing = pd.read_csv(ROOT / "results/csvs/phase5/reversion_timing_and_relapse.csv")
    definitions = [
        "apparent_reversion", "confirmed_reversion",
        "sustained_reversion", "strongest_reversion",
    ]
    subset = timing.set_index("definition").loc[definitions].reset_index()
    labels = ["Apparent", "Confirmed", "Sustained", "Strongest"]
    stem = "figure_11_reversion_trajectory_and_relapse_fixed"

    fig, ax1 = plt.subplots(figsize=(10.6, 6.6))
    x = np.arange(len(subset))
    bars = ax1.bar(
        x, subset["participants"], width=0.72,
        color=["#b3cde3", "#8c96c6", "#8856a7", "#810f7c"],
    )
    ax1.set_xticks(x, labels, fontsize=12.5)
    ax1.set_ylabel("Participants", fontsize=15, labelpad=10)
    ax1.set_ylim(0, 126)
    ax1.tick_params(axis="y", labelsize=12.5)
    ax1.grid(axis="y", alpha=0.18)
    ax1.spines[["top", "right"]].set_visible(False)

    for bar, count, relapse in zip(
        bars, subset["participants"], subset["later_relapse_or_lower_state"]
    ):
        center = bar.get_x() + bar.get_width() / 2
        ax1.text(
            center, bar.get_height() + 3, str(int(count)),
            ha="center", va="bottom", fontsize=13.5, fontweight="bold",
        )
        ax1.text(
            center, 7, f"later relapse: {int(relapse)}",
            ha="center", va="bottom", rotation=90, fontsize=10.5, color="#333333",
        )

    ax2 = ax1.twinx()
    line = ax2.plot(
        x, subset["time_median_months"], "o-",
        color="#e6550d", linewidth=2.5, markersize=8,
        label="Median qualifying time",
    )
    ax2.set_ylim(32, 65)
    ax2.set_ylabel("Median qualifying time (months)", fontsize=15, labelpad=14)
    ax2.tick_params(axis="y", labelsize=12.5)
    ax2.spines["top"].set_visible(False)

    # Title size is intentionally unchanged; the legend has its own reserved band.
    fig.suptitle(
        "Diagnosis-reversion confirmation and relapse",
        fontsize=18, fontweight="bold", y=0.965,
    )
    fig.legend(
        line, ["Median qualifying time"], loc="upper center",
        bbox_to_anchor=(0.5, 0.87), frameon=False, fontsize=12.5,
        handlelength=2.7,
    )
    fig.subplots_adjust(left=0.12, right=0.80, top=0.70, bottom=0.17)
    _save_and_show(fig, stem)

