#!/usr/bin/env python3
"""Locked OASIS-3 external validation after the pre-outcome prediction freeze.

The script refuses to open outcomes unless the prediction manifest and
independent regeneration gate pass. It freezes untouched CDRTOT results before
running reconstructed-CDR sensitivity or participant-grouped recalibration.
No frozen predictor coefficient, preprocessing object, penalty, or baseline
hazard is modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "models" / "phase3" / "phase4_lock_v2"
RUNTIME = LOCK / "inference_runtime"
CONFIG_PATH = ROOT / "config" / "phase4_analysis_parameters.json"
TABLE_ROOT = ROOT / "results" / "csvs" / "phase4"
PLOT_ROOT = ROOT / "results" / "plots" / "phase4"
REPORT_ROOT = ROOT / "reports" / "phase4"
FEATURE_ROOT = (
    ROOT
    / "ProcessedData"
    / "oasis3"
    / "phase4"
    / "outcome_hidden_features"
)
PREDICTION_ROOT = (
    ROOT
    / "ProcessedData"
    / "oasis3"
    / "phase4"
    / "frozen_predictions"
)
PHASE1_EPISODE = (
    ROOT
    / "results"
    / "csvs"
    / "phase1"
    / "oasis3"
    / "primary_episode_time_to_event.csv"
)
CDR_COMPARISON = (
    ROOT
    / "results"
    / "csvs"
    / "phase2"
    / "oasis3_cdr_record_level_comparison.csv"
)
GENERATED_AT = datetime.now(timezone.utc).isoformat()

TABLE_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUNTIME))

from phase3_survival_core import (
    EPS,
    KaplanMeierCensoring,
    calibration_bins,
    fixed_horizon_ipcw,
    horizon_metric_record,
    weighted_auc,
    weighted_logistic_calibration,
)


CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
DAYS_PER_MONTH = 30.4375
PRIMARY_MODELS = {"B1", "D0", "D1"}
PRIMARY_HORIZONS = {12, 36}
BOOTSTRAPS = int(CONFIG["participant_bootstrap_replicates"])
SEED = int(CONFIG["random_seed"])
PNG_DPI = int(CONFIG["primary_figure_dpi"])
COMPLEX_DPI = int(CONFIG["complex_figure_dpi"])
TOLERANCE = float(CONFIG["probability_sum_tolerance"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def save_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def save_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def md(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    shown = frame if columns is None else frame.loc[:, columns]
    return shown.to_markdown(index=False, floatfmt=".4f")


def verify_outcome_gate() -> tuple[pd.DataFrame, dict[str, Any]]:
    freeze_path = TABLE_ROOT / "prediction_freeze_metadata.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / freeze["prediction_manifest_path"]
    manifest = pd.read_csv(manifest_path)
    independent = pd.read_csv(
        TABLE_ROOT / "phase4_independent_prediction_qc_results.csv"
    )
    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, details: str) -> None:
        rows.append(
            {
                "check": check,
                "critical": True,
                "status": "PASS" if passed else "FAIL",
                "details": details,
            }
        )

    add(
        "prediction manifest matches its pre-outcome freeze hash",
        sha256(manifest_path) == freeze["prediction_manifest_sha256"],
        f"manifest={rel(manifest_path)}",
    )
    file_matches = []
    for record in manifest.itertuples(index=False):
        path = ROOT / str(record.relative_path)
        file_matches.append(
            path.exists()
            and sha256(path) == str(record.sha256)
            and path.stat().st_size == int(record.bytes)
        )
    add(
        "all 28 frozen prediction files retain their hashes",
        all(file_matches) and len(file_matches) == 28,
        f"matched={sum(file_matches)}/{len(file_matches)}",
    )
    add(
        "independent point/curve regeneration passed",
        independent["status"].eq("PASS").all()
        and bool_value(freeze.get("independent_prediction_qc_pass")),
        (
            f"checks={len(independent)}; "
            f"failures={int(independent['status'].ne('PASS').sum())}"
        ),
    )
    add(
        "outcomes were not accessed before prediction freeze",
        not bool_value(freeze["oasis_outcomes_accessed_before_freeze"]),
        f"freeze_timestamp={freeze['freeze_timestamp_utc']}",
    )
    gate = pd.DataFrame(rows)
    save_csv(gate, TABLE_ROOT / "phase4_outcome_gate_verification.csv")
    if not gate["status"].eq("PASS").all():
        raise RuntimeError("Phase 4 outcome gate failed; outcomes remain closed.")
    return manifest, freeze


def primary_source(model_id: str, family: str, horizon: int) -> Path:
    if model_id == "D3":
        return (
            ROOT
            / "ProcessedData"
            / "oasis3"
            / "phase2"
            / "sensitivity"
            / f"dynamic_{horizon}m_strictly_lagged_mmse.csv"
        )
    return (
        ROOT
        / "ProcessedData"
        / "oasis3"
        / "phase2"
        / family
        / f"{family}_{horizon}m_clinical_core.csv"
    )


def add_primary_outcomes(
    point: pd.DataFrame,
    *,
    model_id: str,
    family: str,
    horizon: int,
    episode: pd.DataFrame,
) -> pd.DataFrame:
    horizon_days = horizon * DAYS_PER_MONTH
    if family == "baseline":
        metadata = pd.read_csv(
            primary_source(model_id, family, horizon),
            usecols=[
                "participant_id",
                "clinical_mci_subgroup_flag",
                "clinician_diagnosed_mci_at_index_flag",
                "study_phase",
                "form_family",
                "site_id",
            ],
        )
        outcome = episode[
            [
                "participant_id",
                "first_event_type",
                "first_event_time_days",
                "analysis_endpoint_time_days",
            ]
        ].merge(metadata, on="participant_id", how="left", validate="one_to_one")
        merged = point.merge(
            outcome,
            on="participant_id",
            how="left",
            validate="one_to_one",
        )
        raw_event_time = pd.to_numeric(
            merged["first_event_time_days"], errors="coerce"
        )
        raw_analysis_time = pd.to_numeric(
            merged["analysis_endpoint_time_days"], errors="coerce"
        )
        event_within = (
            merged["first_event_type"].isin(["improvement", "progression"])
            & raw_event_time.le(horizon_days + 1e-10)
        )
        merged["event_type"] = np.where(
            event_within, merged["first_event_type"], "none"
        )
        merged["analysis_time_days"] = np.minimum(
            raw_analysis_time, horizon_days
        )
        merged["censoring_event"] = (
            merged["first_event_type"].eq("no_transition_observed")
            & raw_analysis_time.lt(horizon_days - 1e-10)
        )
    else:
        source = primary_source(model_id, family, horizon)
        outcome = pd.read_csv(
            source,
            usecols=[
                "participant_id",
                "landmark_manifest_key",
                "first_transition_after_landmark",
                "transition_time_from_landmark_days",
                "censoring_time_from_landmark_days",
                "clinical_mci_subgroup_flag",
                "clinician_diagnosed_mci_at_index_flag",
                "study_phase",
                "form_family",
                "site_id",
            ],
        ).rename(columns={"landmark_manifest_key": "phase4_row_key"})
        outcome["phase4_row_key"] = outcome["phase4_row_key"].astype(str)
        merged = point.merge(
            outcome.drop(columns=["participant_id"]),
            on="phase4_row_key",
            how="left",
            validate="one_to_one",
        )
        raw_event_time = pd.to_numeric(
            merged["transition_time_from_landmark_days"], errors="coerce"
        )
        raw_censor_time = pd.to_numeric(
            merged["censoring_time_from_landmark_days"], errors="coerce"
        )
        mapped = merged["first_transition_after_landmark"].map(
            {
                "transition_to_cdr_0": "improvement",
                "transition_to_cdr_ge1": "progression",
            }
        )
        event_within = raw_event_time.notna() & raw_event_time.le(
            horizon_days + 1e-10
        )
        merged["event_type"] = np.where(event_within, mapped, "none")
        raw_analysis_time = raw_event_time.where(
            raw_event_time.notna(), raw_censor_time
        )
        merged["analysis_time_days"] = np.minimum(
            raw_analysis_time, horizon_days
        )
        merged["censoring_event"] = (
            raw_event_time.isna()
            & raw_censor_time.lt(horizon_days - 1e-10)
        )
    if merged["analysis_time_days"].isna().any():
        raise RuntimeError(f"Missing outcome time for {model_id} {horizon}m.")
    merged["horizon_months"] = horizon
    merged["family"] = family
    merged["model_id"] = model_id
    merged = merged.rename(
        columns={
            "improvement_probability": "probability_improvement",
            "progression_probability": "probability_progression",
            "remaining_cdr_0_5_probability": "probability_remaining",
        }
    )
    return merged


def censoring_weights(frame: pd.DataFrame, horizon_days: float) -> np.ndarray:
    censoring_model = KaplanMeierCensoring.fit(
        frame["analysis_time_days"].to_numpy(dtype=float),
        frame["censoring_event"].to_numpy(dtype=bool),
    )
    return fixed_horizon_ipcw(
        frame["analysis_time_days"].to_numpy(dtype=float),
        frame["event_type"].to_numpy(dtype=str),
        horizon_days,
        censoring_model,
    )


def evaluate_group(
    point: pd.DataFrame,
    curve: pd.DataFrame,
    *,
    model_label: str,
    endpoint: str,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
]:
    family = str(point["family"].iloc[0])
    horizon = int(point["horizon_months"].iloc[0])
    horizon_days = horizon * DAYS_PER_MONTH
    point = point.copy()
    point["ipcw_weight"] = censoring_weights(point, horizon_days)
    metrics = horizon_metric_record(
        probability_improvement=point["probability_improvement"].to_numpy(),
        probability_progression=point["probability_progression"].to_numpy(),
        probability_remaining=point["probability_remaining"].to_numpy(),
        time=point["analysis_time_days"].to_numpy(),
        event_type=point["event_type"].to_numpy(),
        horizon=horizon_days,
        ipcw=point["ipcw_weight"].to_numpy(),
    )
    y_remaining = (
        point["event_type"].eq("none")
        & point["analysis_time_days"].ge(horizon_days - 1e-10)
    ).to_numpy()
    (
        remaining_intercept,
        remaining_slope,
        remaining_oe,
        remaining_observed,
        remaining_expected,
    ) = weighted_logistic_calibration(
        y_remaining,
        point["probability_remaining"].to_numpy(),
        point["ipcw_weight"].to_numpy(),
    )
    metrics.update(
        {
            "calibration_intercept_remaining": remaining_intercept,
            "calibration_slope_remaining": remaining_slope,
            "observed_expected_remaining": remaining_oe,
            "observed_rate_remaining": remaining_observed,
            "expected_rate_remaining": remaining_expected,
        }
    )
    record = {
        "endpoint": endpoint,
        "model_id": model_label,
        "source_model_id": point["model_id"].iloc[0],
        "family": family,
        "horizon_months": horizon,
        "rows": len(point),
        "participants": point["participant_id"].nunique(),
        "improvement_events": int(point["event_type"].eq("improvement").sum()),
        "progression_events": int(point["event_type"].eq("progression").sum()),
        "censored_before_horizon": int(point["censoring_event"].sum()),
        **metrics,
    }

    calibration_parts = []
    for cause, outcome, probability in [
        (
            "improvement",
            point["event_type"].eq("improvement").to_numpy(),
            point["probability_improvement"].to_numpy(),
        ),
        (
            "progression",
            point["event_type"].eq("progression").to_numpy(),
            point["probability_progression"].to_numpy(),
        ),
        ("remaining", y_remaining, point["probability_remaining"].to_numpy()),
    ]:
        bins = calibration_bins(
            outcome,
            probability,
            point["ipcw_weight"].to_numpy(),
            bins=10,
        )
        if not bins.empty:
            bins.insert(0, "endpoint", endpoint)
            bins.insert(1, "model_id", model_label)
            bins.insert(2, "family", family)
            bins.insert(3, "horizon_months", horizon)
            bins.insert(4, "cause", cause)
            calibration_parts.append(bins)
    calibration = (
        pd.concat(calibration_parts, ignore_index=True)
        if calibration_parts
        else pd.DataFrame()
    )

    probabilities = point[
        [
            "probability_improvement",
            "probability_progression",
            "probability_remaining",
        ]
    ]
    sums = probabilities.sum(axis=1)
    coherence = {
        "endpoint": endpoint,
        "model_id": model_label,
        "family": family,
        "horizon_months": horizon,
        "rows": len(point),
        "minimum_probability": float(probabilities.min().min()),
        "maximum_probability": float(probabilities.max().max()),
        "mean_probability_sum": float(sums.mean()),
        "maximum_absolute_sum_error": float(np.max(np.abs(sums - 1))),
        "negative_probability_rows": int((probabilities < -TOLERANCE).any(axis=1).sum()),
        "greater_than_one_rows": int(
            (probabilities > 1 + TOLERANCE).any(axis=1).sum()
        ),
        "status": (
            "PASS"
            if np.max(np.abs(sums - 1)) <= TOLERANCE
            and not (probabilities < -TOLERANCE).any(axis=1).any()
            and not (probabilities > 1 + TOLERANCE).any(axis=1).any()
            else "FAIL"
        ),
    }

    outcome_columns = point[
        [
            "phase4_row_key",
            "analysis_time_days",
            "event_type",
            "censoring_event",
        ]
    ]
    curve_eval = curve.merge(
        outcome_columns,
        on="phase4_row_key",
        how="inner",
        validate="many_to_one",
    ).rename(
        columns={
            "improvement_probability": "probability_improvement",
            "progression_probability": "probability_progression",
            "remaining_cdr_0_5_probability": "probability_remaining",
        }
    )
    censoring_model = KaplanMeierCensoring.fit(
        point["analysis_time_days"].to_numpy(dtype=float),
        point["censoring_event"].to_numpy(dtype=bool),
    )
    brier_rows: list[dict[str, Any]] = []
    for evaluation_time, at_time in curve_eval.groupby(
        "evaluation_time_days", sort=True
    ):
        weight = fixed_horizon_ipcw(
            at_time["analysis_time_days"].to_numpy(dtype=float),
            at_time["event_type"].to_numpy(dtype=str),
            float(evaluation_time),
            censoring_model,
        )
        curve_eval.loc[at_time.index, "ipcw_weight"] = weight
        yi = (
            at_time["event_type"].eq("improvement")
            & at_time["analysis_time_days"].le(evaluation_time + 1e-10)
        ).to_numpy()
        yp = (
            at_time["event_type"].eq("progression")
            & at_time["analysis_time_days"].le(evaluation_time + 1e-10)
        ).to_numpy()
        yr = at_time["analysis_time_days"].ge(
            evaluation_time - 1e-10
        ).to_numpy()
        denominator = max(float(weight.sum()), EPS)
        bi = float(
            np.sum(
                weight
                * (yi - at_time["probability_improvement"].to_numpy()) ** 2
            )
            / denominator
        )
        bp = float(
            np.sum(
                weight
                * (yp - at_time["probability_progression"].to_numpy()) ** 2
            )
            / denominator
        )
        br = float(
            np.sum(
                weight
                * (yr - at_time["probability_remaining"].to_numpy()) ** 2
            )
            / denominator
        )
        brier_rows.append(
            {
                "endpoint": endpoint,
                "model_id": model_label,
                "family": family,
                "horizon_months": horizon,
                "evaluation_time_days": evaluation_time,
                "brier_improvement": bi,
                "brier_progression": bp,
                "brier_remaining": br,
                "multistate_brier": bi + bp + br,
            }
        )
    brier = pd.DataFrame(brier_rows).sort_values("evaluation_time_days")
    x = np.concatenate([[0.0], brier["evaluation_time_days"].to_numpy()])
    integrated = {
        "endpoint": endpoint,
        "model_id": model_label,
        "family": family,
        "horizon_months": horizon,
    }
    for metric in [
        "brier_improvement",
        "brier_progression",
        "brier_remaining",
        "multistate_brier",
    ]:
        y = np.concatenate([[0.0], brier[metric].to_numpy()])
        integrated[f"integrated_{metric}"] = float(
            np.trapezoid(y, x) / horizon_days
        )
    return record, calibration, coherence, integrated, curve_eval


def _fast_calibration(
    outcome: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> tuple[float, float, float]:
    use = weight > 0
    y = outcome[use].astype(float)
    p = np.clip(prediction[use].astype(float), 1e-6, 1 - 1e-6)
    w = weight[use].astype(float)
    if len(y) == 0 or y.min() == y.max():
        return np.nan, np.nan, np.nan
    observed = float(np.average(y, weights=w))
    expected = float(np.average(p, weights=w))
    citl = float(
        np.log(np.clip(observed, 1e-6, 1 - 1e-6) / np.clip(1 - observed, 1e-6, 1))
        - np.log(np.clip(expected, 1e-6, 1 - 1e-6) / np.clip(1 - expected, 1e-6, 1))
    )
    lp = np.log(p / (1 - p))
    design = np.column_stack([np.ones(len(lp)), lp])
    theta = np.array([0.0, 1.0])
    for _ in range(30):
        mu = expit(np.clip(design @ theta, -30, 30))
        gradient = design.T @ (w * (y - mu))
        hessian = design.T @ ((w * mu * (1 - mu))[:, None] * design)
        step = np.linalg.pinv(hessian) @ gradient
        theta += step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return citl, float(theta[1]), observed / max(expected, EPS)


def participant_ibs_components(
    curve: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    participants = np.sort(curve["participant_id"].unique())
    participant_index = {value: i for i, value in enumerate(participants)}
    times = np.sort(curve["evaluation_time_days"].unique())
    denominators = np.zeros((len(participants), len(times)), dtype=float)
    numerators = {
        "improvement": np.zeros_like(denominators),
        "progression": np.zeros_like(denominators),
        "remaining": np.zeros_like(denominators),
        "multistate": np.zeros_like(denominators),
    }
    for time_index, eval_time in enumerate(times):
        at_time = curve[np.isclose(curve["evaluation_time_days"], eval_time)]
        weights = at_time["ipcw_weight"].to_numpy(dtype=float)
        yi = (
            at_time["event_type"].eq("improvement")
            & at_time["analysis_time_days"].le(eval_time + 1e-10)
        ).to_numpy(dtype=float)
        yp = (
            at_time["event_type"].eq("progression")
            & at_time["analysis_time_days"].le(eval_time + 1e-10)
        ).to_numpy(dtype=float)
        yr = at_time["analysis_time_days"].ge(
            eval_time - 1e-10
        ).to_numpy(dtype=float)
        errors = {
            "improvement": weights
            * (yi - at_time["probability_improvement"].to_numpy()) ** 2,
            "progression": weights
            * (yp - at_time["probability_progression"].to_numpy()) ** 2,
            "remaining": weights
            * (yr - at_time["probability_remaining"].to_numpy()) ** 2,
        }
        errors["multistate"] = (
            errors["improvement"] + errors["progression"] + errors["remaining"]
        )
        codes = at_time["participant_id"].map(participant_index).to_numpy()
        np.add.at(denominators[:, time_index], codes, weights)
        for key in numerators:
            np.add.at(numerators[key][:, time_index], codes, errors[key])
    return participants, times, numerators, denominators


def bootstrap_metrics(
    point_groups: dict[tuple[str, int], pd.DataFrame],
    curve_groups: dict[tuple[str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicate_rows: list[dict[str, Any]] = []
    for model_id, horizon in [
        ("B1", 12),
        ("B1", 36),
        ("D0", 12),
        ("D0", 36),
        ("D1", 12),
        ("D1", 36),
    ]:
        frame = point_groups[(model_id, horizon)]
        curve = curve_groups[(model_id, horizon)]
        participants, eval_times, ibs_num, ibs_den = participant_ibs_components(
            curve
        )
        participant_index = {value: i for i, value in enumerate(participants)}
        row_codes = frame["participant_id"].map(participant_index).to_numpy()
        family_seed = 0 if frame["family"].iloc[0] == "baseline" else 100_000
        generator = np.random.default_rng(SEED + family_seed + horizon * 1009)
        counts = generator.multinomial(
            len(participants),
            np.full(len(participants), 1 / len(participants)),
            size=BOOTSTRAPS,
        )
        yi = frame["event_type"].eq("improvement").to_numpy()
        yp = frame["event_type"].eq("progression").to_numpy()
        yr = (
            frame["event_type"].eq("none")
            & frame["analysis_time_days"].ge(horizon * DAYS_PER_MONTH - 1e-10)
        ).to_numpy()
        base_weight = frame["ipcw_weight"].to_numpy()
        pi = frame["probability_improvement"].to_numpy()
        pp = frame["probability_progression"].to_numpy()
        pr = frame["probability_remaining"].to_numpy()
        for bootstrap_index in range(BOOTSTRAPS):
            row_weight = base_weight * counts[bootstrap_index, row_codes]
            denominator = max(float(row_weight.sum()), EPS)
            ci_i, slope_i, oe_i = _fast_calibration(yi, pi, row_weight)
            ci_p, slope_p, oe_p = _fast_calibration(yp, pp, row_weight)
            aggregated_den = counts[bootstrap_index] @ ibs_den
            ibs_values: dict[str, float] = {}
            for key, numerator in ibs_num.items():
                brier_time = (counts[bootstrap_index] @ numerator) / np.maximum(
                    aggregated_den, EPS
                )
                x = np.concatenate([[0.0], eval_times])
                y = np.concatenate([[0.0], brier_time])
                ibs_values[f"integrated_brier_{key}"] = float(
                    np.trapezoid(y, x) / (horizon * DAYS_PER_MONTH)
                )
            replicate_rows.append(
                {
                    "bootstrap_index": bootstrap_index,
                    "bootstrap_unit": "participant",
                    "model_id": model_id,
                    "family": frame["family"].iloc[0],
                    "horizon_months": horizon,
                    "sampled_participants": len(participants),
                    "auc_improvement": weighted_auc(yi, pi, row_weight),
                    "auc_progression": weighted_auc(yp, pp, row_weight),
                    "brier_improvement": float(
                        np.sum(row_weight * (yi - pi) ** 2) / denominator
                    ),
                    "brier_progression": float(
                        np.sum(row_weight * (yp - pp) ** 2) / denominator
                    ),
                    "multistate_brier": float(
                        np.sum(
                            row_weight
                            * (
                                (yi - pi) ** 2
                                + (yp - pp) ** 2
                                + (yr - pr) ** 2
                            )
                        )
                        / denominator
                    ),
                    "calibration_intercept_improvement": ci_i,
                    "calibration_slope_improvement": slope_i,
                    "observed_expected_improvement": oe_i,
                    "calibration_intercept_progression": ci_p,
                    "calibration_slope_progression": slope_p,
                    "observed_expected_progression": oe_p,
                    **ibs_values,
                }
            )
        print(
            f"[Phase 4 bootstrap] {model_id} {horizon}m: {BOOTSTRAPS}",
            flush=True,
        )
    replicates = pd.DataFrame(replicate_rows)
    index_columns = {
        "bootstrap_index",
        "bootstrap_unit",
        "model_id",
        "family",
        "horizon_months",
        "sampled_participants",
    }
    metric_columns = [c for c in replicates if c not in index_columns]
    interval_rows: list[dict[str, Any]] = []
    for keys, group in replicates.groupby(
        ["model_id", "family", "horizon_months"]
    ):
        for metric in metric_columns:
            values = group[metric].dropna()
            interval_rows.append(
                {
                    "model_id": keys[0],
                    "family": keys[1],
                    "horizon_months": keys[2],
                    "metric": metric,
                    "bootstrap_replicates": len(group),
                    "valid_replicates": len(values),
                    "bootstrap_median": values.median(),
                    "ci_low": values.quantile(0.025),
                    "ci_high": values.quantile(0.975),
                    "bootstrap_unit": "participant",
                }
            )
    return replicates, pd.DataFrame(interval_rows)


def metrics_on_common_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    comparison: str,
) -> pd.DataFrame:
    common = sorted(set(left["phase4_row_key"]) & set(right["phase4_row_key"]))
    left_common = left[left["phase4_row_key"].isin(common)].sort_values(
        "phase4_row_key"
    )
    right_common = right[right["phase4_row_key"].isin(common)].sort_values(
        "phase4_row_key"
    )
    if not left_common[
        ["phase4_row_key", "event_type", "analysis_time_days"]
    ].reset_index(drop=True).equals(
        right_common[
            ["phase4_row_key", "event_type", "analysis_time_days"]
        ].reset_index(drop=True)
    ):
        raise RuntimeError(f"Outcome mismatch in {comparison} common rows.")
    horizon = int(left_common["horizon_months"].iloc[0])
    horizon_days = horizon * DAYS_PER_MONTH
    weights = censoring_weights(left_common, horizon_days)
    rows = []
    metric_names = [
        "auc_improvement",
        "auc_progression",
        "brier_improvement",
        "brier_progression",
        "multistate_brier",
        "calibration_intercept_improvement",
        "calibration_slope_improvement",
        "observed_expected_improvement",
        "calibration_intercept_progression",
        "calibration_slope_progression",
        "observed_expected_progression",
    ]
    records = {}
    for name, frame in [(left_name, left_common), (right_name, right_common)]:
        record = horizon_metric_record(
            probability_improvement=frame["probability_improvement"].to_numpy(),
            probability_progression=frame["probability_progression"].to_numpy(),
            probability_remaining=frame["probability_remaining"].to_numpy(),
            time=frame["analysis_time_days"].to_numpy(),
            event_type=frame["event_type"].to_numpy(),
            horizon=horizon_days,
            ipcw=weights,
        )
        records[name] = record
    for metric in metric_names:
        rows.append(
            {
                "comparison": comparison,
                "horizon_months": horizon,
                "common_rows": len(common),
                "common_participants": left_common["participant_id"].nunique(),
                "left_model": left_name,
                "right_model": right_name,
                "metric": metric,
                "left_estimate": records[left_name][metric],
                "right_estimate": records[right_name][metric],
                "right_minus_left": (
                    records[right_name][metric] - records[left_name][metric]
                ),
            }
        )
    return pd.DataFrame(rows)


def participant_paired_bootstrap_d1_d0(
    d0: pd.DataFrame, d1: pd.DataFrame
) -> pd.DataFrame:
    common = sorted(set(d0["phase4_row_key"]) & set(d1["phase4_row_key"]))
    d0 = d0[d0["phase4_row_key"].isin(common)].sort_values("phase4_row_key")
    d1 = d1[d1["phase4_row_key"].isin(common)].sort_values("phase4_row_key")
    participants = np.sort(d1["participant_id"].unique())
    participant_index = {value: i for i, value in enumerate(participants)}
    row_codes = d1["participant_id"].map(participant_index).to_numpy()
    horizon = int(d1["horizon_months"].iloc[0])
    base_weight = censoring_weights(d1, horizon * DAYS_PER_MONTH)
    generator = np.random.default_rng(SEED + 900_000 + horizon * 1009)
    counts = generator.multinomial(
        len(participants),
        np.full(len(participants), 1 / len(participants)),
        size=BOOTSTRAPS,
    )
    yi = d1["event_type"].eq("improvement").to_numpy()
    yp = d1["event_type"].eq("progression").to_numpy()
    yr = (
        d1["event_type"].eq("none")
        & d1["analysis_time_days"].ge(horizon * DAYS_PER_MONTH - 1e-10)
    ).to_numpy()
    rows = []
    for bootstrap_index in range(BOOTSTRAPS):
        weight = base_weight * counts[bootstrap_index, row_codes]
        denominator = max(float(weight.sum()), EPS)
        model_metrics: dict[str, dict[str, float]] = {}
        for name, frame in [("D0", d0), ("D1", d1)]:
            pi = frame["probability_improvement"].to_numpy()
            pp = frame["probability_progression"].to_numpy()
            pr = frame["probability_remaining"].to_numpy()
            model_metrics[name] = {
                "auc_improvement": weighted_auc(yi, pi, weight),
                "auc_progression": weighted_auc(yp, pp, weight),
                "brier_improvement": float(
                    np.sum(weight * (yi - pi) ** 2) / denominator
                ),
                "brier_progression": float(
                    np.sum(weight * (yp - pp) ** 2) / denominator
                ),
                "multistate_brier": float(
                    np.sum(
                        weight
                        * (
                            (yi - pi) ** 2
                            + (yp - pp) ** 2
                            + (yr - pr) ** 2
                        )
                    )
                    / denominator
                ),
            }
        for metric in model_metrics["D0"]:
            rows.append(
                {
                    "horizon_months": horizon,
                    "bootstrap_index": bootstrap_index,
                    "bootstrap_unit": "participant",
                    "metric": metric,
                    "d0_estimate": model_metrics["D0"][metric],
                    "d1_estimate": model_metrics["D1"][metric],
                    "d1_minus_d0": (
                        model_metrics["D1"][metric]
                        - model_metrics["D0"][metric]
                    ),
                }
            )
    replicate = pd.DataFrame(rows)
    summary_rows = []
    for metric, group in replicate.groupby("metric"):
        values = group["d1_minus_d0"].dropna()
        summary_rows.append(
            {
                "comparison": "D1_minus_D0_matched_rows",
                "horizon_months": horizon,
                "metric": metric,
                "bootstrap_replicates": BOOTSTRAPS,
                "bootstrap_median_difference": values.median(),
                "ci_low": values.quantile(0.025),
                "ci_high": values.quantile(0.975),
                "bootstrap_unit": "participant",
            }
        )
    return pd.DataFrame(summary_rows), replicate


def distribution_shift(
    inventory: pd.DataFrame,
    feature_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows: list[dict[str, Any]] = []
    categorical_rows: list[dict[str, Any]] = []
    for record in inventory[
        inventory["horizon_months"].isin([12, 36])
    ].itertuples(index=False):
        model_id = str(record.model_id)
        horizon = int(record.horizon_months)
        bundle = joblib.load(ROOT / str(record.artifact_path))
        if model_id == "D3":
            adni_path = (
                ROOT
                / "ProcessedData"
                / "adni"
                / "phase2"
                / "sensitivity"
                / f"dynamic_{horizon}m_strictly_lagged_mmse.csv"
            )
        else:
            family = str(record.family)
            adni_path = (
                ROOT
                / "ProcessedData"
                / "adni"
                / "phase2"
                / family
                / f"{family}_{horizon}m_clinical_core.csv"
            )
        oasis_record = feature_manifest[
            feature_manifest["model_id"].eq(model_id)
            & feature_manifest["horizon_months"].eq(horizon)
        ].iloc[0]
        oasis = pd.read_csv(ROOT / str(oasis_record.relative_path))
        required = list(bundle["continuous_features"]) + list(
            bundle["categorical_features"]
        )
        adni = pd.read_csv(adni_path, usecols=["participant_id"] + required)
        if model_id == "D1" and CONFIG[
            "d1_requires_observed_current_and_change_mmse"
        ]:
            oasis = oasis.dropna(
                subset=["current_mmse", "mmse_change_from_index"]
            )
        for feature in bundle["continuous_features"]:
            left = pd.to_numeric(adni[feature], errors="coerce")
            right = pd.to_numeric(oasis[feature], errors="coerce")
            pooled_sd = math.sqrt(
                (float(left.std(ddof=1)) ** 2 + float(right.std(ddof=1)) ** 2)
                / 2
            )
            smd = (
                (float(right.mean()) - float(left.mean())) / pooled_sd
                if pooled_sd > 0
                else np.nan
            )
            numeric_rows.append(
                {
                    "model_id": model_id,
                    "family": record.family,
                    "horizon_months": horizon,
                    "feature": feature,
                    "adni_rows": len(adni),
                    "oasis_rows": len(oasis),
                    "adni_missing_percent": left.isna().mean() * 100,
                    "oasis_missing_percent": right.isna().mean() * 100,
                    "adni_mean": left.mean(),
                    "oasis_mean": right.mean(),
                    "adni_sd": left.std(ddof=1),
                    "oasis_sd": right.std(ddof=1),
                    "adni_median": left.median(),
                    "oasis_median": right.median(),
                    "standardized_mean_difference": smd,
                    "absolute_standardized_mean_difference": abs(smd),
                    "oasis_outside_adni_range_percent": (
                        ((right < left.min()) | (right > left.max())).mean()
                        * 100
                    ),
                }
            )
        for feature in bundle["categorical_features"]:
            levels = sorted(
                set(adni[feature].dropna().astype(str))
                | set(oasis[feature].dropna().astype(str))
            )
            for level in levels:
                adni_rate = adni[feature].astype(str).eq(level).mean()
                oasis_rate = oasis[feature].astype(str).eq(level).mean()
                categorical_rows.append(
                    {
                        "model_id": model_id,
                        "family": record.family,
                        "horizon_months": horizon,
                        "feature": feature,
                        "level": level,
                        "adni_percent": adni_rate * 100,
                        "oasis_percent": oasis_rate * 100,
                        "oasis_minus_adni_percentage_points": (
                            oasis_rate - adni_rate
                        )
                        * 100,
                    }
                )
    return pd.DataFrame(numeric_rows), pd.DataFrame(categorical_rows)


def reconstructed_outcomes(
    point: pd.DataFrame,
    cdr: pd.DataFrame,
) -> pd.DataFrame:
    family = str(point["family"].iloc[0])
    horizon = int(point["horizon_months"].iloc[0])
    horizon_days = horizon * DAYS_PER_MONTH
    state = pd.to_numeric(cdr["derived_global_cdr"], errors="coerce")
    cdr = cdr.assign(
        reconstructed_state=np.select(
            [state.eq(0), state.eq(0.5), state.ge(1)],
            [0, 1, 2],
            default=np.nan,
        )
    )
    groups = {
        participant: group.sort_values("time_sort_days", kind="mergesort")
        for participant, group in cdr[
            cdr["reconstructed_state"].notna()
        ].groupby("participant_id")
    }
    rows: list[dict[str, Any]] = []
    reference_column = (
        "index_time_days" if family == "baseline" else "landmark_time_days"
    )
    for record in point.itertuples(index=False):
        participant = str(record.participant_id)
        reference = float(getattr(record, reference_column))
        group = groups.get(participant)
        if group is None:
            continue
        at_reference = group[np.isclose(group["time_sort_days"], reference)]
        eligible = (
            not at_reference.empty
            and float(at_reference.iloc[0]["reconstructed_state"]) == 1
        )
        if not eligible:
            continue
        later = group[group["time_sort_days"].gt(reference)]
        event = later[~later["reconstructed_state"].eq(1)].head(1)
        last_time = float(group["time_sort_days"].max() - reference)
        event_type = "none"
        event_time = np.nan
        if not event.empty:
            event_time = float(event.iloc[0]["time_sort_days"] - reference)
            destination = int(event.iloc[0]["reconstructed_state"])
            if event_time <= horizon_days + 1e-10:
                event_type = "improvement" if destination == 0 else "progression"
        raw_analysis_time = event_time if pd.notna(event_time) else last_time
        rows.append(
            {
                "phase4_row_key": str(record.phase4_row_key),
                "reconstructed_event_type": event_type,
                "reconstructed_analysis_time_days": min(
                    raw_analysis_time, horizon_days
                ),
                "reconstructed_censoring_event": (
                    pd.isna(event_time) and last_time < horizon_days - 1e-10
                ),
            }
        )
    alt = pd.DataFrame(rows)
    merged = point.merge(
        alt,
        on="phase4_row_key",
        how="inner",
        validate="one_to_one",
    )
    merged["event_type"] = merged["reconstructed_event_type"]
    merged["analysis_time_days"] = merged[
        "reconstructed_analysis_time_days"
    ]
    merged["censoring_event"] = merged["reconstructed_censoring_event"]
    return merged.drop(
        columns=[
            "reconstructed_event_type",
            "reconstructed_analysis_time_days",
            "reconstructed_censoring_event",
        ]
    )


def fit_logistic_calibration(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    *,
    slope_free: bool,
) -> tuple[float, float]:
    use = np.isfinite(y) & np.isfinite(p) & (w > 0)
    y = y[use].astype(float)
    p = np.clip(p[use].astype(float), 1e-6, 1 - 1e-6)
    w = w[use].astype(float)
    lp = np.log(p / (1 - p))
    if len(y) == 0 or y.min() == y.max():
        return 0.0, 1.0
    if slope_free:
        design = np.column_stack([np.ones(len(lp)), lp])
        theta = np.array([0.0, 1.0])
    else:
        design = np.ones((len(lp), 1))
        theta = np.array([0.0])
    for _ in range(50):
        eta = (
            design @ theta
            if slope_free
            else theta[0] + lp
        )
        mu = expit(np.clip(eta, -30, 30))
        gradient = design.T @ (w * (y - mu))
        hessian = design.T @ ((w * mu * (1 - mu))[:, None] * design)
        step = np.linalg.pinv(hessian) @ gradient
        theta += step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return (
        (float(theta[0]), float(theta[1]))
        if slope_free
        else (float(theta[0]), 1.0)
    )


def cross_fitted_recalibration(
    point: pd.DataFrame,
    model_id: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participants = np.sort(point["participant_id"].unique())
    generator = np.random.default_rng(SEED + horizon * 101 + sum(map(ord, model_id)))
    shuffled = participants.copy()
    generator.shuffle(shuffled)
    fold_map = {
        participant: index % int(CONFIG["recalibration_cross_validation_folds"])
        for index, participant in enumerate(shuffled)
    }
    work = point.copy()
    work["recalibration_fold"] = work["participant_id"].map(fold_map)
    parameter_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    for method, slope_free in [
        ("calibration_in_the_large", False),
        ("intercept_and_slope", True),
    ]:
        method_parts = []
        for fold in range(int(CONFIG["recalibration_cross_validation_folds"])):
            train = work[work["recalibration_fold"].ne(fold)]
            test = work[work["recalibration_fold"].eq(fold)].copy()
            calibrated: dict[str, np.ndarray] = {}
            for cause, event_label, probability_column in [
                ("improvement", "improvement", "probability_improvement"),
                ("progression", "progression", "probability_progression"),
            ]:
                y_train = train["event_type"].eq(event_label).to_numpy()
                intercept, slope = fit_logistic_calibration(
                    y_train,
                    train[probability_column].to_numpy(),
                    train["ipcw_weight"].to_numpy(),
                    slope_free=slope_free,
                )
                p_test = np.clip(
                    test[probability_column].to_numpy(), 1e-6, 1 - 1e-6
                )
                calibrated[cause] = expit(
                    intercept + slope * np.log(p_test / (1 - p_test))
                )
                parameter_rows.append(
                    {
                        "model_id": model_id,
                        "horizon_months": horizon,
                        "method": method,
                        "fold": fold,
                        "cause": cause,
                        "training_rows": len(train),
                        "training_participants": train["participant_id"].nunique(),
                        "training_events": int(y_train.sum()),
                        "intercept": intercept,
                        "slope": slope,
                        "predictor_coefficients_refit": False,
                    }
                )
            remaining = test["probability_remaining"].to_numpy()
            total = calibrated["improvement"] + calibrated["progression"] + remaining
            test["probability_improvement"] = calibrated["improvement"] / total
            test["probability_progression"] = calibrated["progression"] / total
            test["probability_remaining"] = remaining / total
            test["recalibration_method"] = method
            method_parts.append(test)
        prediction_parts.append(pd.concat(method_parts, ignore_index=True))
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_rows = []
    for method, group in predictions.groupby("recalibration_method"):
        record = horizon_metric_record(
            probability_improvement=group["probability_improvement"].to_numpy(),
            probability_progression=group["probability_progression"].to_numpy(),
            probability_remaining=group["probability_remaining"].to_numpy(),
            time=group["analysis_time_days"].to_numpy(),
            event_type=group["event_type"].to_numpy(),
            horizon=horizon * DAYS_PER_MONTH,
            ipcw=group["ipcw_weight"].to_numpy(),
        )
        metric_rows.append(
            {
                "model_id": model_id,
                "horizon_months": horizon,
                "method": method,
                "validation_design": "5-fold participant-grouped cross-fitting",
                "rows": len(group),
                "participants": group["participant_id"].nunique(),
                **record,
            }
        )
    return predictions, pd.DataFrame(parameter_rows), pd.DataFrame(metric_rows)


def save_figure(fig: plt.Figure, stem: str, *, complex_panel: bool = False) -> None:
    fig.savefig(
        PLOT_ROOT / f"{stem}.png",
        dpi=COMPLEX_DPI if complex_panel else PNG_DPI,
        bbox_inches="tight",
    )
    fig.savefig(PLOT_ROOT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def create_figures(
    calibration: pd.DataFrame,
    external_internal: pd.DataFrame,
    shift: pd.DataFrame,
    comparisons: pd.DataFrame,
    endpoint_comparison: pd.DataFrame,
    recalibration: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    selected = calibration[
        calibration["model_id"].isin(["B1", "D0", "D1"])
        & calibration["horizon_months"].isin([12, 36])
    ]
    fig, axes = plt.subplots(3, 2, figsize=(8, 9), sharex=True, sharey=True)
    colors = {"B1": "#1f77b4", "D0": "#ff7f0e", "D1": "#2ca02c"}
    for row, cause in enumerate(["improvement", "progression", "remaining"]):
        for col, horizon in enumerate([12, 36]):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1.5)
            panel = selected[
                selected["cause"].eq(cause)
                & selected["horizon_months"].eq(horizon)
            ]
            for model_id, group in panel.groupby("model_id"):
                ax.plot(
                    group["mean_prediction"],
                    group["observed_probability"],
                    marker="o",
                    linewidth=2,
                    label=model_id,
                    color=colors[model_id],
                )
            ax.set_title(f"{cause.title()} — {horizon} months")
            ax.set_xlabel("Frozen predicted probability")
            ax.set_ylabel("IPCW observed probability")
            ax.grid(alpha=0.25)
            ax.legend()
    fig.suptitle("Untouched OASIS-3 calibration", fontsize=22, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "untouched_oasis_calibration_panels", complex_panel=True)

    auc = external_internal[
        external_internal["metric"].isin(["auc_improvement", "auc_progression"])
        & external_internal["model_id"].isin(["B1", "D0", "D1"])
        & external_internal["horizon_months"].isin([12, 36])
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = (
        auc["model_id"]
        + "-"
        + auc["horizon_months"].astype(str)
        + "-"
        + auc["metric"].str.replace("auc_", "")
    )
    x = np.arange(len(auc))
    width = 0.38
    ax.bar(x - width / 2, auc["adni_internal"], width, label="ADNI OOF")
    ax.bar(x + width / 2, auc["oasis_external"], width, label="OASIS untouched")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Time-dependent AUC")
    ax.set_title("Internal-to-external discrimination transport")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "adni_oasis_auc_transport", complex_panel=True)

    heat = shift[
        shift["horizon_months"].eq(12)
    ].pivot_table(
        index="model_id",
        columns="feature",
        values="standardized_mean_difference",
        aggfunc="first",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(heat.fillna(0), cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set_xticks(np.arange(len(heat.columns)), heat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat.index)), heat.index)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            value = heat.iloc[i, j]
            ax.text(
                j,
                i,
                "" if pd.isna(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=11,
            )
    fig.colorbar(image, ax=ax, label="Standardized mean difference (OASIS − ADNI)")
    ax.set_title("Frozen-predictor distribution shift")
    fig.tight_layout()
    save_figure(fig, "oasis_adni_distribution_shift_heatmap", complex_panel=True)

    d1d0 = comparisons[
        comparisons["comparison"].eq("D1_vs_D0_matched_rows")
        & comparisons["metric"].isin(
            ["auc_improvement", "auc_progression", "multistate_brier"]
        )
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = (
        d1d0["horizon_months"].astype(str)
        + "m "
        + d1d0["metric"].str.replace("_", " ")
    )
    ax.barh(labels, d1d0["right_minus_left"], color="#4c78a8")
    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_xlabel("D1 minus D0 (AUC higher is better; Brier lower is better)")
    ax.set_title("Incremental transport value of updated MMSE")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "d1_vs_d0_transport_value")

    endpoint_auc = endpoint_comparison[
        endpoint_comparison["metric"].isin(
            ["auc_improvement", "auc_progression", "multistate_brier"]
        )
        & endpoint_comparison["model_id"].isin(["B1", "D0", "D1"])
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = (
        endpoint_auc["model_id"]
        + "-"
        + endpoint_auc["horizon_months"].astype(str)
        + "-"
        + endpoint_auc["metric"].str.replace("_", " ")
    )
    ax.barh(labels, endpoint_auc["reconstructed_minus_cdrtot"])
    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_xlabel("Reconstructed-CDR minus CDRTOT estimate")
    ax.set_title("Endpoint reconstruction sensitivity")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "reconstructed_cdr_sensitivity")

    recal_selected = recalibration[
        recalibration["model_id"].isin(["B1", "D0", "D1"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, cause in zip(axes, ["improvement", "progression"]):
        column = f"observed_expected_{cause}"
        labels = (
            recal_selected["model_id"]
            + "-"
            + recal_selected["horizon_months"].astype(str)
            + "-"
            + recal_selected["method"]
        )
        ax.barh(labels, recal_selected[column])
        ax.axvline(1, color="black", linestyle="--", linewidth=1.5)
        ax.set_title(f"{cause.title()} O/E after cross-fitted updating")
        ax.set_xlabel("Observed / expected")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "cross_fitted_recalibration_effect", complex_panel=True)


def main() -> None:
    manifest, freeze = verify_outcome_gate()
    outcomes_first_accessed_at = GENERATED_AT
    access_path = TABLE_ROOT / "phase4_access_audit.csv"
    access = pd.read_csv(access_path)
    access = pd.concat(
        [
            access,
            pd.DataFrame(
                [
                    {
                        "sequence": len(access) + 1,
                        "timestamp_utc": outcomes_first_accessed_at,
                        "stage": "D",
                        "action": "first permitted OASIS outcome access after frozen predictions and independent QC",
                        "resource_scope": (
                            "Phase 1 primary episode outcomes and Phase 2 "
                            "CDRTOT event/censoring fields"
                        ),
                        "loaded_columns": "event type/time, censoring time, subgroup metadata",
                        "oasis_record_level_features_accessed": True,
                        "oasis_record_level_outcomes_accessed": True,
                        "outcome_hidden_predictions_created": True,
                        "outcomes_first_accessed_at_utc": outcomes_first_accessed_at,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    save_csv(access, access_path)

    episode = pd.read_csv(PHASE1_EPISODE)
    inventory = pd.read_csv(LOCK / "model_inventory.csv").sort_values(
        ["model_id", "horizon_months"]
    )
    feature_manifest = pd.read_csv(
        TABLE_ROOT / "outcome_hidden_feature_manifest.csv"
    )
    point_groups: dict[tuple[str, int], pd.DataFrame] = {}
    curve_groups: dict[tuple[str, int], pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    calibration_parts: list[pd.DataFrame] = []
    coherence_rows: list[dict[str, Any]] = []
    integrated_rows: list[dict[str, Any]] = []
    merged_parts: list[pd.DataFrame] = []

    for record in inventory.itertuples(index=False):
        model_id = str(record.model_id)
        horizon = int(record.horizon_months)
        point_record = manifest[
            manifest["model_id"].eq(model_id)
            & manifest["horizon_months"].eq(horizon)
            & manifest["file_role"].eq("point_prediction")
        ].iloc[0]
        curve_record = manifest[
            manifest["model_id"].eq(model_id)
            & manifest["horizon_months"].eq(horizon)
            & manifest["file_role"].eq("time_grid_prediction")
        ].iloc[0]
        point = pd.read_csv(ROOT / str(point_record.relative_path))
        curve = pd.read_csv(ROOT / str(curve_record.relative_path))
        point = add_primary_outcomes(
            point,
            model_id=model_id,
            family=str(record.family),
            horizon=horizon,
            episode=episode,
        )
        result, calibration, coherence, integrated, curve_eval = evaluate_group(
            point,
            curve,
            model_label=model_id,
            endpoint="CDRTOT_de_facto_global",
        )
        point["ipcw_weight"] = censoring_weights(
            point, horizon * DAYS_PER_MONTH
        )
        point_groups[(model_id, horizon)] = point
        curve_groups[(model_id, horizon)] = curve_eval
        metric_rows.append(result)
        calibration_parts.append(calibration)
        coherence_rows.append(coherence)
        integrated_rows.append(integrated)
        merged_parts.append(point)

    untouched_metrics = pd.DataFrame(metric_rows).sort_values(
        ["model_id", "horizon_months"]
    )
    untouched_calibration = pd.concat(calibration_parts, ignore_index=True)
    untouched_coherence = pd.DataFrame(coherence_rows)
    untouched_integrated = pd.DataFrame(integrated_rows)
    untouched_merged = pd.concat(merged_parts, ignore_index=True)
    if not untouched_coherence["status"].eq("PASS").all():
        raise RuntimeError("Untouched probability coherence failed.")

    expected_counts = {12: (45, 18), 24: (143, 58), 36: (170, 101), 60: (187, 150)}
    count_rows = []
    for horizon, expected in expected_counts.items():
        row = untouched_metrics[
            untouched_metrics["model_id"].eq("B1")
            & untouched_metrics["horizon_months"].eq(horizon)
        ].iloc[0]
        observed = (
            int(row.improvement_events),
            int(row.progression_events),
        )
        count_rows.append(
            {
                "horizon_months": horizon,
                "expected_improvement": expected[0],
                "observed_improvement": observed[0],
                "expected_progression": expected[1],
                "observed_progression": observed[1],
                "status": "PASS" if observed == expected else "FAIL",
            }
        )
    baseline_event_count_verification = pd.DataFrame(count_rows)
    if not baseline_event_count_verification["status"].eq("PASS").all():
        save_csv(
            baseline_event_count_verification,
            TABLE_ROOT / "baseline_event_count_verification.csv",
        )
        raise RuntimeError("Baseline event counts differ from the frozen expectation.")

    bootstrap_replicates, bootstrap_intervals = bootstrap_metrics(
        point_groups, curve_groups
    )
    comparison_parts = []
    paired_summary_parts = []
    paired_replicate_parts = []
    for horizon in [12, 36]:
        comparison_parts.append(
            metrics_on_common_rows(
                point_groups[("D0", horizon)],
                point_groups[("D1", horizon)],
                left_name="D0",
                right_name="D1",
                comparison="D1_vs_D0_matched_rows",
            )
        )
        summary, replicate = participant_paired_bootstrap_d1_d0(
            point_groups[("D0", horizon)],
            point_groups[("D1", horizon)],
        )
        paired_summary_parts.append(summary)
        paired_replicate_parts.append(replicate)
        comparison_parts.append(
            metrics_on_common_rows(
                point_groups[("B1", horizon)],
                point_groups[("B3", horizon)],
                left_name="B1",
                right_name="B3",
                comparison="B3_vs_B1_no_APOE",
            )
        )
        comparison_parts.append(
            metrics_on_common_rows(
                point_groups[("D1", horizon)],
                point_groups[("D3", horizon)],
                left_name="D1",
                right_name="D3",
                comparison="D3_vs_D1_strictly_lagged_MMSE",
            )
        )
    model_comparisons = pd.concat(comparison_parts, ignore_index=True)
    paired_d1_d0_ci = pd.concat(paired_summary_parts, ignore_index=True)
    paired_d1_d0_replicates = pd.concat(
        paired_replicate_parts, ignore_index=True
    )

    clinical_mci_rows = []
    for model_id in ["B1", "D1"]:
        for horizon in [12, 36]:
            point = point_groups[(model_id, horizon)]
            subset = point[
                point["clinical_mci_subgroup_flag"].map(bool_value)
            ].copy()
            if subset.empty:
                continue
            subset["ipcw_weight"] = censoring_weights(
                subset, horizon * DAYS_PER_MONTH
            )
            result = horizon_metric_record(
                probability_improvement=subset[
                    "probability_improvement"
                ].to_numpy(),
                probability_progression=subset[
                    "probability_progression"
                ].to_numpy(),
                probability_remaining=subset[
                    "probability_remaining"
                ].to_numpy(),
                time=subset["analysis_time_days"].to_numpy(),
                event_type=subset["event_type"].to_numpy(),
                horizon=horizon * DAYS_PER_MONTH,
                ipcw=subset["ipcw_weight"].to_numpy(),
            )
            clinical_mci_rows.append(
                {
                    "model_id": model_id,
                    "horizon_months": horizon,
                    "rows": len(subset),
                    "participants": subset["participant_id"].nunique(),
                    "improvement_events": int(
                        subset["event_type"].eq("improvement").sum()
                    ),
                    "progression_events": int(
                        subset["event_type"].eq("progression").sum()
                    ),
                    **result,
                }
            )
    clinical_mci = pd.DataFrame(clinical_mci_rows)

    internal = pd.read_csv(
        ROOT
        / "results"
        / "csvs"
        / "phase3"
        / "discrimination_and_horizon_metrics.csv"
    )
    internal_ibs = pd.read_csv(
        ROOT / "results" / "csvs" / "phase3" / "integrated_brier_scores.csv"
    )
    internal = internal.merge(
        internal_ibs,
        on=["model_id", "family", "horizon_months"],
        how="left",
        validate="one_to_one",
    )
    external_full = untouched_metrics.merge(
        untouched_integrated,
        on=["endpoint", "model_id", "family", "horizon_months"],
        how="left",
        validate="one_to_one",
    )
    external_internal_rows = []
    comparison_metrics = [
        "auc_improvement",
        "auc_progression",
        "c_index_improvement",
        "c_index_progression",
        "brier_improvement",
        "brier_progression",
        "multistate_brier",
        "calibration_slope_improvement",
        "calibration_slope_progression",
        "observed_expected_improvement",
        "observed_expected_progression",
        "integrated_brier_improvement",
        "integrated_brier_progression",
        "integrated_multistate_brier",
    ]
    for row in external_full.itertuples(index=False):
        match = internal[
            internal["model_id"].eq(row.model_id)
            & internal["horizon_months"].eq(row.horizon_months)
        ]
        if match.empty:
            continue
        internal_row = match.iloc[0]
        for metric in comparison_metrics:
            external_value = getattr(row, metric)
            internal_value = internal_row[metric]
            external_internal_rows.append(
                {
                    "model_id": row.model_id,
                    "family": row.family,
                    "horizon_months": row.horizon_months,
                    "metric": metric,
                    "adni_internal": internal_value,
                    "oasis_external": external_value,
                    "oasis_minus_adni": external_value - internal_value,
                }
            )
    external_internal = pd.DataFrame(external_internal_rows)

    numeric_shift, categorical_shift = distribution_shift(
        inventory, feature_manifest
    )
    event_rate_shift_rows = []
    for row in external_full.itertuples(index=False):
        match = internal[
            internal["model_id"].eq(row.model_id)
            & internal["horizon_months"].eq(row.horizon_months)
        ]
        if match.empty:
            continue
        for cause in ["improvement", "progression"]:
            internal_rate = match.iloc[0][f"observed_rate_{cause}"]
            external_rate = getattr(row, f"observed_rate_{cause}")
            event_rate_shift_rows.append(
                {
                    "model_id": row.model_id,
                    "family": row.family,
                    "horizon_months": row.horizon_months,
                    "cause": cause,
                    "adni_ipcw_observed_rate": internal_rate,
                    "oasis_ipcw_observed_rate": external_rate,
                    "oasis_minus_adni": external_rate - internal_rate,
                    "oasis_to_adni_ratio": external_rate
                    / max(internal_rate, EPS),
                }
            )
    event_rate_shift = pd.DataFrame(event_rate_shift_rows)

    untouched_paths = [
        save_csv(
            untouched_metrics,
            TABLE_ROOT / "untouched_external_metrics.csv",
        ),
        save_csv(
            untouched_integrated,
            TABLE_ROOT / "untouched_integrated_brier_scores.csv",
        ),
        save_csv(
            untouched_calibration,
            TABLE_ROOT / "untouched_calibration_curves.csv",
        ),
        save_csv(
            untouched_coherence,
            TABLE_ROOT / "untouched_probability_coherence.csv",
        ),
        save_csv(
            baseline_event_count_verification,
            TABLE_ROOT / "baseline_event_count_verification.csv",
        ),
        save_csv(
            untouched_merged,
            TABLE_ROOT / "untouched_outcome_merged_predictions.csv",
        ),
        save_csv(
            bootstrap_replicates,
            TABLE_ROOT / "participant_bootstrap_metric_replicates.csv",
        ),
        save_csv(
            bootstrap_intervals,
            TABLE_ROOT / "participant_bootstrap_confidence_intervals.csv",
        ),
        save_csv(
            model_comparisons,
            TABLE_ROOT / "untouched_model_comparisons.csv",
        ),
        save_csv(
            paired_d1_d0_ci,
            TABLE_ROOT / "d1_vs_d0_paired_bootstrap_ci.csv",
        ),
        save_csv(
            paired_d1_d0_replicates,
            TABLE_ROOT / "d1_vs_d0_paired_bootstrap_replicates.csv",
        ),
        save_csv(
            clinical_mci,
            TABLE_ROOT / "clinical_mci_subgroup_results.csv",
        ),
        save_csv(
            external_internal,
            TABLE_ROOT / "adni_oasis_performance_comparison.csv",
        ),
        save_csv(
            numeric_shift,
            TABLE_ROOT / "predictor_distribution_shift.csv",
        ),
        save_csv(
            categorical_shift,
            TABLE_ROOT / "categorical_distribution_shift.csv",
        ),
        save_csv(
            event_rate_shift,
            TABLE_ROOT / "event_rate_shift.csv",
        ),
    ]
    untouched_artifacts = pd.DataFrame(
        [
            {
                "relative_path": rel(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in untouched_paths
        ]
    )
    untouched_manifest_path = save_csv(
        untouched_artifacts,
        TABLE_ROOT / "untouched_result_artifact_manifest.csv",
    )
    untouched_freeze = {
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "CDRTOT de facto global CDR (provisional)",
        "prediction_manifest_sha256": freeze["prediction_manifest_sha256"],
        "untouched_artifact_manifest_path": rel(untouched_manifest_path),
        "untouched_artifact_manifest_sha256": sha256(untouched_manifest_path),
        "artifacts": len(untouched_artifacts),
        "endpoint_sensitivity_started_after_freeze": False,
        "recalibration_started_after_freeze": False,
    }
    untouched_freeze_path = save_json(
        untouched_freeze,
        TABLE_ROOT / "untouched_results_freeze_metadata.json",
    )

    # Secondary endpoint sensitivity starts only after the untouched freeze.
    cdr = pd.read_csv(CDR_COMPARISON)
    endpoint_metric_rows = []
    endpoint_count_rows = []
    endpoint_calibration_parts = []
    for (model_id, horizon), primary_point in point_groups.items():
        reconstructed = reconstructed_outcomes(primary_point, cdr)
        curve = pd.read_csv(
            ROOT
            / str(
                manifest[
                    manifest["model_id"].eq(model_id)
                    & manifest["horizon_months"].eq(horizon)
                    & manifest["file_role"].eq("time_grid_prediction")
                ].iloc[0].relative_path
            )
        )
        result, calibration, _, integrated, _ = evaluate_group(
            reconstructed,
            curve,
            model_label=model_id,
            endpoint="derived_global_from_boxes",
        )
        result.update(
            {
                key: value
                for key, value in integrated.items()
                if key.startswith("integrated_")
            }
        )
        endpoint_metric_rows.append(result)
        endpoint_calibration_parts.append(calibration)
        primary = untouched_metrics[
            untouched_metrics["model_id"].eq(model_id)
            & untouched_metrics["horizon_months"].eq(horizon)
        ].iloc[0]
        endpoint_count_rows.append(
            {
                "model_id": model_id,
                "family": primary.family,
                "horizon_months": horizon,
                "primary_rows": int(primary.rows),
                "reconstructed_eligible_rows": len(reconstructed),
                "eligibility_change_rows": len(reconstructed) - int(primary.rows),
                "primary_participants": int(primary.participants),
                "reconstructed_participants": reconstructed[
                    "participant_id"
                ].nunique(),
                "primary_improvement_events": int(primary.improvement_events),
                "reconstructed_improvement_events": int(
                    reconstructed["event_type"].eq("improvement").sum()
                ),
                "primary_progression_events": int(primary.progression_events),
                "reconstructed_progression_events": int(
                    reconstructed["event_type"].eq("progression").sum()
                ),
            }
        )
    endpoint_metrics = pd.DataFrame(endpoint_metric_rows)
    endpoint_counts = pd.DataFrame(endpoint_count_rows)
    endpoint_calibration = pd.concat(
        endpoint_calibration_parts, ignore_index=True
    )
    endpoint_comparison_rows = []
    for row in endpoint_metrics.itertuples(index=False):
        primary = external_full[
            external_full["model_id"].eq(row.model_id)
            & external_full["horizon_months"].eq(row.horizon_months)
        ].iloc[0]
        for metric in comparison_metrics:
            if not hasattr(row, metric):
                continue
            endpoint_comparison_rows.append(
                {
                    "model_id": row.model_id,
                    "family": row.family,
                    "horizon_months": row.horizon_months,
                    "metric": metric,
                    "cdrtot_primary": primary[metric],
                    "reconstructed_cdr": getattr(row, metric),
                    "reconstructed_minus_cdrtot": (
                        getattr(row, metric) - primary[metric]
                    ),
                }
            )
    endpoint_comparison = pd.DataFrame(endpoint_comparison_rows)
    save_csv(
        endpoint_metrics,
        TABLE_ROOT / "reconstructed_cdr_metrics.csv",
    )
    save_csv(
        endpoint_counts,
        TABLE_ROOT / "reconstructed_cdr_eligibility_event_counts.csv",
    )
    save_csv(
        endpoint_calibration,
        TABLE_ROOT / "reconstructed_cdr_calibration_curves.csv",
    )
    save_csv(
        endpoint_comparison,
        TABLE_ROOT / "reconstructed_cdr_metric_comparison.csv",
    )

    # Cross-fitted recalibration starts after untouched CDRTOT results are frozen.
    recalibration_prediction_parts = []
    recalibration_parameter_parts = []
    recalibration_metric_parts = []
    for model_id in ["B1", "D0", "D1"]:
        for horizon in [12, 36]:
            predictions, parameters, metrics = cross_fitted_recalibration(
                point_groups[(model_id, horizon)],
                model_id,
                horizon,
            )
            recalibration_prediction_parts.append(predictions)
            recalibration_parameter_parts.append(parameters)
            recalibration_metric_parts.append(metrics)
    recalibration_predictions = pd.concat(
        recalibration_prediction_parts, ignore_index=True
    )
    recalibration_parameters = pd.concat(
        recalibration_parameter_parts, ignore_index=True
    )
    recalibration_metrics = pd.concat(
        recalibration_metric_parts, ignore_index=True
    )
    untouched_for_recalibration = untouched_metrics[
        untouched_metrics["model_id"].isin(["B1", "D0", "D1"])
        & untouched_metrics["horizon_months"].isin([12, 36])
    ].copy()
    untouched_for_recalibration["method"] = "untouched"
    recalibration_comparison = pd.concat(
        [
            untouched_for_recalibration,
            recalibration_metrics,
        ],
        ignore_index=True,
        sort=False,
    )
    save_csv(
        recalibration_predictions,
        TABLE_ROOT / "cross_fitted_recalibrated_predictions.csv",
    )
    save_csv(
        recalibration_parameters,
        TABLE_ROOT / "cross_fitted_recalibration_parameters.csv",
    )
    save_csv(
        recalibration_metrics,
        TABLE_ROOT / "cross_fitted_recalibration_metrics.csv",
    )
    save_csv(
        recalibration_comparison,
        TABLE_ROOT / "recalibration_effect_comparison.csv",
    )

    create_figures(
        untouched_calibration,
        external_internal,
        numeric_shift,
        model_comparisons,
        endpoint_comparison,
        recalibration_comparison,
    )

    rules = CONFIG["transportability_rules"]
    d1_auc = external_internal[
        external_internal["model_id"].eq("D1")
        & external_internal["horizon_months"].isin([12, 36])
        & external_internal["metric"].isin(
            ["auc_improvement", "auc_progression"]
        )
    ]
    median_auc_degradation = float(
        np.median(d1_auc["adni_internal"] - d1_auc["oasis_external"])
    )
    minimum_external_auc = float(d1_auc["oasis_external"].min())
    d1_calibration = untouched_metrics[
        untouched_metrics["model_id"].eq("D1")
        & untouched_metrics["horizon_months"].isin([12, 36])
    ]
    calibration_acceptable = bool(
        d1_calibration[
            [
                "observed_expected_improvement",
                "observed_expected_progression",
            ]
        ]
        .ge(rules["calibration_oe_acceptable_lower"])
        .all(axis=None)
        and d1_calibration[
            [
                "observed_expected_improvement",
                "observed_expected_progression",
            ]
        ]
        .le(rules["calibration_oe_acceptable_upper"])
        .all(axis=None)
        and d1_calibration[
            [
                "calibration_slope_improvement",
                "calibration_slope_progression",
            ]
        ]
        .ge(rules["calibration_slope_acceptable_lower"])
        .all(axis=None)
        and d1_calibration[
            [
                "calibration_slope_improvement",
                "calibration_slope_progression",
            ]
        ]
        .le(rules["calibration_slope_acceptable_upper"])
        .all(axis=None)
    )
    if (
        minimum_external_auc < rules["not_transportable_auc_floor"]
        or median_auc_degradation
        > rules["maximum_acceptable_median_auc_degradation"]
    ):
        transportability = "LIMITED TRANSPORTABILITY"
    elif (
        minimum_external_auc >= rules["minimum_acceptable_external_auc"]
        and calibration_acceptable
    ):
        transportability = "TRANSPORTABLE WITHOUT UPDATING"
    elif minimum_external_auc >= rules["minimum_acceptable_external_auc"]:
        transportability = "TRANSPORTABLE WITH RECALIBRATION"
    else:
        transportability = "LIMITED TRANSPORTABILITY"
    classification = pd.DataFrame(
        [
            {
                "selected_model": "D1",
                "primary_horizons": "12|36",
                "minimum_external_auc_across_causes_horizons": minimum_external_auc,
                "median_internal_minus_external_auc": median_auc_degradation,
                "untouched_calibration_within_prespecified_bounds": calibration_acceptable,
                "classification": transportability,
                "classification_rule_source": rel(CONFIG_PATH),
            }
        ]
    )
    save_csv(
        classification,
        TABLE_ROOT / "transportability_classification.csv",
    )

    untouched_freeze["endpoint_sensitivity_started_after_freeze"] = True
    untouched_freeze["recalibration_started_after_freeze"] = True
    untouched_freeze["secondary_analyses_completed_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )
    save_json(untouched_freeze, untouched_freeze_path)
    freeze["phase4_outcomes_merged"] = True
    freeze["outcomes_first_accessed_at_utc"] = outcomes_first_accessed_at
    save_json(freeze, TABLE_ROOT / "prediction_freeze_metadata.json")

    access = pd.read_csv(access_path)
    access = pd.concat(
        [
            access,
            pd.DataFrame(
                [
                    {
                        "sequence": len(access) + 1,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "stage": "E",
                        "action": "reconstructed-CDR endpoint sensitivity after untouched freeze",
                        "resource_scope": rel(CDR_COMPARISON),
                        "loaded_columns": "participant, visit time, six-box reconstructed global CDR",
                        "oasis_record_level_features_accessed": True,
                        "oasis_record_level_outcomes_accessed": True,
                        "outcome_hidden_predictions_created": True,
                        "outcomes_first_accessed_at_utc": outcomes_first_accessed_at,
                    },
                    {
                        "sequence": len(access) + 2,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "stage": "F",
                        "action": "participant-grouped cross-fitted recalibration after untouched freeze",
                        "resource_scope": "frozen predictions plus CDRTOT outcomes",
                        "loaded_columns": "frozen probabilities and merged evaluation outcomes",
                        "oasis_record_level_features_accessed": True,
                        "oasis_record_level_outcomes_accessed": True,
                        "outcome_hidden_predictions_created": True,
                        "outcomes_first_accessed_at_utc": outcomes_first_accessed_at,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    save_csv(access, access_path)

    primary_summary = external_full[
        external_full["model_id"].isin(["B1", "D0", "D1"])
        & external_full["horizon_months"].isin([12, 36])
    ].sort_values(["model_id", "horizon_months"])
    ci_summary = bootstrap_intervals[
        bootstrap_intervals["metric"].isin(
            [
                "auc_improvement",
                "auc_progression",
                "brier_improvement",
                "brier_progression",
                "multistate_brier",
                "calibration_slope_improvement",
                "calibration_slope_progression",
                "observed_expected_improvement",
                "observed_expected_progression",
            ]
        )
    ]
    d1d0_summary = model_comparisons[
        model_comparisons["comparison"].eq("D1_vs_D0_matched_rows")
    ]
    endpoint_primary = endpoint_comparison[
        endpoint_comparison["model_id"].isin(["B1", "D0", "D1"])
        & endpoint_comparison["metric"].isin(
            ["auc_improvement", "auc_progression", "multistate_brier"]
        )
    ]
    b3d3 = external_full[
        external_full["model_id"].isin(["B3", "D3"])
    ]
    highest_shift = numeric_shift.nlargest(
        15, "absolute_standardized_mean_difference"
    )

    locked_report = f"""# Phase 4 Locked OASIS-3 External Validation

Generated: {GENERATED_AT}

## Decision

**GO TO PHASE 5 WITH RESTRICTIONS**

Transportability classification: **{transportability}**

This is untouched external validation of the frozen ADNI models using the
provisional OASIS-3 `CDRTOT` endpoint. Predictions were saved, hashed, and
independently regenerated before the first outcome access.

## Untouched primary results

{md(primary_summary, ["model_id", "horizon_months", "rows", "participants", "improvement_events", "progression_events", "auc_improvement", "auc_progression", "c_index_improvement", "c_index_progression", "brier_improvement", "brier_progression", "multistate_brier", "integrated_brier_improvement", "integrated_brier_progression", "integrated_multistate_brier", "calibration_intercept_improvement", "calibration_slope_improvement", "observed_expected_improvement", "calibration_intercept_progression", "calibration_slope_progression", "observed_expected_progression"])}

The B1 12-month progression estimate is based on only **18 events** and is
explicitly low precision.

## Participant-level confidence intervals

{md(ci_summary)}

## D1 versus D0 on identical rows

{md(d1d0_summary)}

## Internal-to-external change

{md(external_internal[external_internal["model_id"].isin(["B1", "D0", "D1"])])}

## B3 and D3 sensitivities

{md(b3d3, ["model_id", "horizon_months", "rows", "participants", "improvement_events", "progression_events", "auc_improvement", "auc_progression", "brier_improvement", "brier_progression", "multistate_brier", "calibration_slope_improvement", "calibration_slope_progression", "observed_expected_improvement", "observed_expected_progression"])}

## Reconstructed-CDR sensitivity

{md(endpoint_primary)}

## Major predictor shifts

{md(highest_shift, ["model_id", "horizon_months", "feature", "adni_mean", "oasis_mean", "standardized_mean_difference", "adni_missing_percent", "oasis_missing_percent"])}

All interpretations retain the provisional-endpoint, sparse-event,
death/dropout, and case-mix restrictions. Phase 5 was not started.
"""
    (REPORT_ROOT / "phase4_locked_external_validation_report.md").write_text(
        locked_report, encoding="utf-8"
    )

    calibration_report = f"""# Phase 4 Untouched Calibration Report

Generated: {GENERATED_AT}

Untouched CDRTOT results were frozen before any updating.

{md(primary_summary, ["model_id", "horizon_months", "calibration_intercept_improvement", "calibration_slope_improvement", "observed_expected_improvement", "calibration_intercept_progression", "calibration_slope_progression", "observed_expected_progression", "calibration_intercept_remaining", "calibration_slope_remaining", "observed_expected_remaining"])}

Calibration curves are in
`results/plots/phase4/untouched_oasis_calibration_panels.png` at 1200 DPI.
"""
    (REPORT_ROOT / "phase4_untouched_calibration_report.md").write_text(
        calibration_report, encoding="utf-8"
    )

    transport_report = f"""# Phase 4 D1 versus D0 Transportability

Generated: {GENERATED_AT}

Classification: **{transportability}**

{md(d1d0_summary)}

## Paired participant-bootstrap intervals

{md(paired_d1_d0_ci)}

Minimum D1 external cause/horizon AUC: {minimum_external_auc:.4f}.
Median ADNI-minus-OASIS D1 AUC: {median_auc_degradation:.4f}.

No model was selected or modified using OASIS results.
"""
    (REPORT_ROOT / "phase4_d1_vs_d0_transportability_report.md").write_text(
        transport_report, encoding="utf-8"
    )

    endpoint_report = f"""# Phase 4 Reconstructed-CDR Endpoint Sensitivity

Generated: {GENERATED_AT}

The same frozen predictions were evaluated after reconstructing global CDR from
the six box scores. No model was refit.

## Eligibility and event counts

{md(endpoint_counts)}

## Metric changes

{md(endpoint_primary)}
"""
    (REPORT_ROOT / "phase4_endpoint_sensitivity_report.md").write_text(
        endpoint_report, encoding="utf-8"
    )

    recalibration_report = f"""# Phase 4 Recalibration Report

Generated: {GENERATED_AT}

These are model-updating results, not untouched external validation.
Calibration-in-the-large and intercept/slope updating used five-fold
participant-grouped cross-fitting. Predictor coefficients remain frozen.

{md(recalibration_comparison, ["model_id", "horizon_months", "method", "auc_improvement", "auc_progression", "brier_improvement", "brier_progression", "multistate_brier", "calibration_intercept_improvement", "calibration_slope_improvement", "observed_expected_improvement", "calibration_intercept_progression", "calibration_slope_progression", "observed_expected_progression"])}
"""
    (REPORT_ROOT / "phase4_recalibration_report.md").write_text(
        recalibration_report, encoding="utf-8"
    )

    shift_report = f"""# Phase 4 Distribution-Shift Report

Generated: {GENERATED_AT}

## Largest standardized predictor shifts

{md(highest_shift)}

## Event-rate shift

{md(event_rate_shift[event_rate_shift["model_id"].isin(["B1", "D0", "D1"])])}
"""
    (REPORT_ROOT / "phase4_distribution_shift_report.md").write_text(
        shift_report, encoding="utf-8"
    )

    model_card = f"""# Updated External-Validation Model Card

Generated: {GENERATED_AT}

Frozen package: `{rel(LOCK)}`

Models evaluated: B1, D0, D1, B3, D3.

Primary horizons: 12 and 36 months. B1/D1 24 and 60 months are sensitivities.

External endpoint: OASIS-3 `CDRTOT`, treated as a provisional de facto global
CDR. Six-box reconstructed global CDR is a secondary endpoint sensitivity.

Transportability: **{transportability}**.

The model remains limited by sparse 12-month progression events, incomplete
death/dropout characterization, clinical overlap between MMSE and CDR,
participant-correlated landmarks, and cohort case-mix shift. Recalibration
results are explicitly model updating. MRI and fluid biomarkers are not part
of this locked validation.
"""
    (REPORT_ROOT / "phase4_updated_model_card.md").write_text(
        model_card, encoding="utf-8"
    )

    blocker_update = f"""# Phase 4 Frozen-Package Blocker Audit — Resolved

Generated: {GENERATED_AT}

The former D0/B3/D3 package blocker is resolved by
`models/phase3/phase4_lock_v2`. Stage A and independent prediction QC passed.
All 14 required bundles were present; all 28 point/curve files were frozen
before outcome access.

Decision: **RESOLVED**.
"""
    (
        REPORT_ROOT / "phase4_frozen_package_blocker_audit.md"
    ).write_text(blocker_update, encoding="utf-8")

    handoff = f"""# Phase 4 to Phase 5 Handoff

Generated: {GENERATED_AT}

## Decision

**GO TO PHASE 5 WITH RESTRICTIONS**

Phase 4 transportability classification: **{transportability}**.

Phase 5 may use the completed Phase 4 results but must retain:

1. provisional OASIS `CDRTOT` endpoint language;
2. explicit low precision for 12-month B1 progression (18 events);
3. untouched results as primary and recalibration as model updating;
4. participant-grouped handling of repeated landmarks;
5. no retrospective Phase 3 model selection using OASIS results.

Phase 5 was not started automatically.
"""
    (REPORT_ROOT / "phase4_phase5_handoff.md").write_text(
        handoff, encoding="utf-8"
    )

    run_path = TABLE_ROOT / "phase4_run_metadata.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.update(
        {
            "outcomes_first_accessed_at_utc": outcomes_first_accessed_at,
            "oasis_record_level_outcomes_accessed": True,
            "stage_d_untouched_results_frozen": True,
            "untouched_results_freeze_metadata": rel(untouched_freeze_path),
            "stage_e_endpoint_sensitivity_complete": True,
            "stage_f_recalibration_complete": True,
            "participant_bootstrap_replicates": BOOTSTRAPS,
            "transportability_classification": transportability,
            "provisional_final_decision": "GO TO PHASE 5 WITH RESTRICTIONS",
            "independent_final_qc_pending": True,
            "phase4_complete": False,
            "phase5_started": False,
            "decision": "HOLD FOR VALIDATION AUDIT",
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "joblib": joblib.__version__,
            },
        }
    )
    save_json(run, run_path)
    print(
        json.dumps(
            {
                "untouched_model_horizons": len(untouched_metrics),
                "primary_model_horizons": len(primary_summary),
                "bootstrap_replicates_per_primary_model_horizon": BOOTSTRAPS,
                "baseline_event_counts": "PASS",
                "reconstructed_endpoint_complete": True,
                "cross_fitted_recalibration_complete": True,
                "transportability": transportability,
                "provisional_decision": "GO TO PHASE 5 WITH RESTRICTIONS",
                "independent_final_qc_pending": True,
                "phase5_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
