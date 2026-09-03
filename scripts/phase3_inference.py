#!/usr/bin/env python3
"""Deterministic inference for the frozen Phase 3 ADNI models.

This module only loads already-fitted objects. It contains no call to ``fit``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from phase3_survival_core import paired_cumulative_incidence


DEFAULT_RANGES = {
    "age_at_landmark": (18.0, 110.0),
    "education_years": (0.0, 40.0),
    "apoe_e4_dose": (0.0, 2.0),
    "index_mmse": (0.0, 30.0),
    "current_mmse": (0.0, 30.0),
    "mmse_change_from_index": (-30.0, 30.0),
    "elapsed_years_from_index": (0.0, 30.0),
}


def load_frozen_bundle(path: Path | str) -> dict[str, Any]:
    bundle = joblib.load(Path(path))
    required = {
        "preprocessor",
        "improvement_model",
        "progression_model",
        "continuous_features",
        "categorical_features",
        "horizon_days",
        "model_version_id",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Frozen bundle is missing keys: {sorted(missing)}")
    if bundle.get("development_cohort") != "ADNI":
        raise ValueError("Only a frozen ADNI development bundle is accepted.")
    if bundle.get("external_outcomes_accessed") is not False:
        raise ValueError("Bundle lacks the external-outcome lock certification.")
    return bundle


def _quality_flags(frame: pd.DataFrame, required: list[str]) -> list[str]:
    flags: list[str] = []
    for _, row in frame.iterrows():
        row_flags: list[str] = []
        missing = [feature for feature in required if pd.isna(row.get(feature))]
        if missing:
            row_flags.append("imputed:" + ",".join(missing))
        for feature, (lower, upper) in DEFAULT_RANGES.items():
            if feature not in required or pd.isna(row.get(feature)):
                continue
            value = float(row[feature])
            if value < lower or value > upper:
                row_flags.append(f"outside_expected_range:{feature}")
        if "sex" in required and not pd.isna(row.get("sex")):
            if str(row["sex"]) not in {"Female", "Male"}:
                row_flags.append("unseen_category:sex")
        flags.append(";".join(row_flags) if row_flags else "none")
    return flags


def predict_from_frozen(
    frame: pd.DataFrame,
    artifact_path: Path | str,
) -> pd.DataFrame:
    """Return coherent transition probabilities without refitting anything."""

    bundle = load_frozen_bundle(artifact_path)
    required = list(bundle["continuous_features"]) + list(
        bundle["categorical_features"]
    )
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")
    preprocessor = bundle["preprocessor"]
    transformed = preprocessor.transform(frame)
    expected_order = list(bundle["transformed_feature_order"])
    if transformed.columns.tolist() != expected_order:
        raise ValueError("Transformed feature order differs from the frozen schema.")
    horizon_days = float(bundle["horizon_days"])
    pi, pp, pr = paired_cumulative_incidence(
        bundle["improvement_model"],
        bundle["progression_model"],
        transformed.to_numpy(),
        np.array([horizon_days]),
    )
    result = pd.DataFrame(
        {
            "improvement_probability": pi[:, 0],
            "progression_probability": pp[:, 0],
            "remaining_cdr_0_5_probability": pr[:, 0],
            "prediction_horizon_months": int(bundle["horizon_months"]),
            "model_id": bundle["model_id"],
            "model_version_id": bundle["model_version_id"],
            "quality_flags": _quality_flags(frame, required),
        },
        index=frame.index,
    )
    result["probability_sum"] = result[
        [
            "improvement_probability",
            "progression_probability",
            "remaining_cdr_0_5_probability",
        ]
    ].sum(axis=1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    predictions = predict_from_frozen(frame, args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(
        json.dumps(
            {
                "rows": len(predictions),
                "output": str(args.output),
                "model_version_id": predictions["model_version_id"].iloc[0]
                if len(predictions)
                else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
