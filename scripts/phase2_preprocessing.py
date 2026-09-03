#!/usr/bin/env python3
"""Leakage-safe preprocessing components for Phase 3.

This module deliberately contains no outcome model.  A preprocessor can be fit
only on an explicitly labelled ADNI training partition, serialized with its
training provenance, and then applied unchanged to ADNI validation or OASIS-3.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


@dataclass(frozen=True)
class PreprocessingProvenance:
    fit_cohort: str
    fit_partition: str
    fit_participant_count: int
    fit_row_count: int
    continuous_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    fit_timestamp_utc: str
    sklearn_version: str
    external_outcomes_accessed: bool = False
    outcome_columns_used: bool = False


class ClinicalCorePreprocessor:
    """Training-fold-only clinical-core transformer with provenance checks."""

    def __init__(
        self,
        continuous_features: Iterable[str],
        categorical_features: Iterable[str],
    ) -> None:
        self.continuous_features = tuple(continuous_features)
        self.categorical_features = tuple(categorical_features)
        numeric_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=False)),
                ("scale", RobustScaler(with_centering=True, with_scaling=True)),
            ]
        )
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=np.float64,
                    ),
                ),
            ]
        )
        self.transformer = ColumnTransformer(
            [
                ("continuous", numeric_pipe, list(self.continuous_features)),
                ("categorical", categorical_pipe, list(self.categorical_features)),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.provenance: PreprocessingProvenance | None = None

    @property
    def required_features(self) -> tuple[str, ...]:
        return self.continuous_features + self.categorical_features

    def _validate_schema(self, frame: pd.DataFrame) -> None:
        missing = sorted(set(self.required_features) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing preprocessing features: {missing}")

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        cohort: str,
        partition_label: str,
        participant_column: str = "participant_id",
    ) -> "ClinicalCorePreprocessor":
        if cohort.lower() != "adni":
            raise ValueError("Primary preprocessing may be fit only on ADNI.")
        if "train" not in partition_label.lower():
            raise ValueError("partition_label must explicitly identify a training partition.")
        self._validate_schema(frame)
        if participant_column not in frame:
            raise ValueError(f"Missing participant grouping column {participant_column!r}.")
        forbidden = {
            "outcome_code",
            "horizon_status",
            "first_transition",
            "transition_time_days",
            "event_by_horizon_flag",
        }
        if forbidden.intersection(self.required_features):
            raise ValueError("Outcome-derived fields cannot be preprocessing features.")
        self.transformer.fit(frame.loc[:, self.required_features])
        self.provenance = PreprocessingProvenance(
            fit_cohort="adni",
            fit_partition=partition_label,
            fit_participant_count=int(frame[participant_column].astype(str).nunique()),
            fit_row_count=int(len(frame)),
            continuous_features=self.continuous_features,
            categorical_features=self.categorical_features,
            fit_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            sklearn_version=sklearn.__version__,
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.provenance is None:
            raise RuntimeError("Preprocessor must be fit on an ADNI training partition first.")
        self._validate_schema(frame)
        values = self.transformer.transform(frame.loc[:, self.required_features])
        return pd.DataFrame(
            values,
            columns=self.transformer.get_feature_names_out(),
            index=frame.index,
        )

    def save(self, path: Path) -> tuple[Path, Path]:
        if self.provenance is None:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        provenance_path = path.with_suffix(".provenance.json")
        provenance_path.write_text(
            json.dumps(asdict(self.provenance), indent=2), encoding="utf-8"
        )
        return path, provenance_path


def load_preprocessor(path: Path) -> ClinicalCorePreprocessor:
    loaded = joblib.load(path)
    if not isinstance(loaded, ClinicalCorePreprocessor):
        raise TypeError("Serialized object is not a ClinicalCorePreprocessor.")
    if loaded.provenance is None or loaded.provenance.fit_cohort != "adni":
        raise ValueError("Serialized preprocessor lacks valid ADNI-only fit provenance.")
    return loaded
