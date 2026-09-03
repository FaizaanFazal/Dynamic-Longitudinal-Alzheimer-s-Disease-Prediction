#!/usr/bin/env python3
"""Core survival methods for Phase 3 ADNI development.

The implementation is deliberately small and auditable:

* ridge-penalized cause-specific Cox regression with Breslow ties;
* Breslow baseline-hazard estimation;
* coherent cumulative-incidence calculation from paired hazards;
* Kaplan-Meier censoring weights and competing-risk-aware fixed-horizon metrics;
* participant-level bootstrap helpers.

No function in this module reads project data or fits on an external cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


EPS = 1e-12


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _cox_components(
    beta: np.ndarray,
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return Breslow partial log-likelihood and gradient.

    Rows are sorted by descending follow-up time, making every risk set a
    cumulative prefix. Event weights and risk-set weights support the
    participant-normalized pooled-landmark sensitivity.
    """

    order = np.argsort(-time, kind="mergesort")
    t = time[order]
    e = event[order].astype(bool)
    w = weight[order]
    z = x[order]
    eta = np.clip(z @ beta, -40.0, 40.0)
    wrisk = w * np.exp(eta)
    cum_risk = np.cumsum(wrisk)
    cum_xrisk = np.cumsum(wrisk[:, None] * z, axis=0)

    unique_t, first, counts = np.unique(t, return_index=True, return_counts=True)
    # np.unique sorts ascending; recover the end index of each time group.
    end_by_time = {
        float(tt): int(start + count - 1)
        for tt, start, count in zip(unique_t, first, counts)
    }
    event_indices = np.flatnonzero(e)
    if not len(event_indices):
        return 0.0, np.zeros(x.shape[1], dtype=float)

    event_times = t[event_indices]
    ll = float(np.sum(w[event_indices] * eta[event_indices]))
    grad = np.sum((w[event_indices, None] * z[event_indices]), axis=0)
    for tt in np.unique(event_times):
        idx = event_indices[event_times == tt]
        d = float(np.sum(w[idx]))
        end = end_by_time[float(tt)]
        risk = max(float(cum_risk[end]), EPS)
        ll -= d * np.log(risk)
        grad -= d * cum_xrisk[end] / risk
    return ll, grad


def cox_partial_loglik_per_event(
    beta: np.ndarray,
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    weight: np.ndarray | None = None,
) -> float:
    """Unpenalized validation partial log-likelihood per weighted event."""

    if weight is None:
        weight = np.ones(len(time), dtype=float)
    ll, _ = _cox_components(
        _as_float_array(beta),
        _as_float_array(x),
        _as_float_array(time),
        np.asarray(event, dtype=bool),
        _as_float_array(weight),
    )
    denominator = float(np.sum(weight[np.asarray(event, dtype=bool)]))
    return ll / denominator if denominator > 0 else np.nan


@dataclass
class RidgeCauseSpecificCox:
    """Ridge cause-specific Cox model with a frozen Breslow baseline hazard."""

    alpha: float
    coef_: np.ndarray | None = None
    baseline_times_: np.ndarray | None = None
    baseline_hazard_increments_: np.ndarray | None = None
    converged_: bool = False
    iterations_: int = 0
    objective_: float = np.nan
    effective_df_: float = np.nan
    information_: np.ndarray | None = None
    covariance_: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "RidgeCauseSpecificCox":
        x = _as_float_array(x)
        time = _as_float_array(time)
        event = np.asarray(event, dtype=bool)
        weight = (
            np.ones(len(time), dtype=float)
            if sample_weight is None
            else _as_float_array(sample_weight)
        )
        if x.ndim != 2 or len(x) != len(time):
            raise ValueError("X must be a 2D array aligned to time.")
        if not np.isfinite(x).all() or not np.isfinite(time).all():
            raise ValueError("Cox inputs must be finite after preprocessing.")
        if np.any(time < 0) or np.any(weight <= 0):
            raise ValueError("Times must be nonnegative and weights positive.")
        if int(event.sum()) == 0:
            raise ValueError("Cannot fit a cause-specific model without events.")

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            ll, grad = _cox_components(beta, x, time, event, weight)
            penalty = 0.5 * float(self.alpha) * float(beta @ beta)
            return -ll + penalty, -grad + float(self.alpha) * beta

        result = minimize(
            objective,
            np.zeros(x.shape[1], dtype=float),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50},
        )
        self.coef_ = np.asarray(result.x, dtype=float)
        self.converged_ = bool(result.success)
        self.iterations_ = int(result.nit)
        self.objective_ = float(result.fun)
        self._fit_baseline(x, time, event, weight)
        self._fit_information(x, time, event, weight)
        return self

    def _fit_baseline(
        self,
        x: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        weight: np.ndarray,
    ) -> None:
        if self.coef_ is None:
            raise RuntimeError("Model coefficients are unavailable.")
        eta = np.clip(x @ self.coef_, -40.0, 40.0)
        risk_score = weight * np.exp(eta)
        times: list[float] = []
        increments: list[float] = []
        for tt in np.sort(np.unique(time[event])):
            at_event = event & np.isclose(time, tt, rtol=0.0, atol=1e-10)
            d = float(np.sum(weight[at_event]))
            risk = float(np.sum(risk_score[time >= tt - 1e-10]))
            times.append(float(tt))
            increments.append(d / max(risk, EPS))
        self.baseline_times_ = np.asarray(times, dtype=float)
        self.baseline_hazard_increments_ = np.asarray(increments, dtype=float)

    def _fit_information(
        self,
        x: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        weight: np.ndarray,
    ) -> None:
        """Observed information and penalized model-based covariance."""

        if self.coef_ is None:
            return
        info = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        eta = np.clip(x @ self.coef_, -40.0, 40.0)
        wrisk = weight * np.exp(eta)
        for tt in np.unique(time[event]):
            at_event = event & np.isclose(time, tt, rtol=0.0, atol=1e-10)
            at_risk = time >= tt - 1e-10
            d = float(np.sum(weight[at_event]))
            rw = wrisk[at_risk]
            rz = x[at_risk]
            denom = max(float(rw.sum()), EPS)
            mean = (rw[:, None] * rz).sum(axis=0) / denom
            second = np.einsum("i,ij,ik->jk", rw, rz, rz) / denom
            info += d * (second - np.outer(mean, mean))
        penalized = info + float(self.alpha) * np.eye(x.shape[1])
        self.information_ = info
        self.covariance_ = np.linalg.pinv(penalized)
        self.effective_df_ = float(np.trace(info @ np.linalg.pinv(penalized)))

    def predict_risk_score(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fit.")
        return np.exp(np.clip(_as_float_array(x) @ self.coef_, -40.0, 40.0))

    def cumulative_baseline_hazard(self, times: np.ndarray) -> np.ndarray:
        if self.baseline_times_ is None or self.baseline_hazard_increments_ is None:
            raise RuntimeError("Baseline hazard is unavailable.")
        query = _as_float_array(times)
        cumulative = np.cumsum(self.baseline_hazard_increments_)
        idx = np.searchsorted(self.baseline_times_, query, side="right") - 1
        return np.where(idx >= 0, cumulative[np.maximum(idx, 0)], 0.0)


def paired_cumulative_incidence(
    improvement_model: RidgeCauseSpecificCox,
    progression_model: RidgeCauseSpecificCox,
    x: np.ndarray,
    evaluation_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine paired cause-specific hazards into coherent state probabilities.

    At each unique baseline-hazard jump, total conditional transition
    probability is ``1-exp(-(dH_I*r_I + dH_P*r_P))``. It is allocated to each
    cause in proportion to its cause-specific hazard contribution. Remaining
    probability is the product survival. The three outputs therefore sum to
    one up to floating-point precision.
    """

    x = _as_float_array(x)
    evaluation_times = np.sort(np.unique(_as_float_array(evaluation_times)))
    n = len(x)
    if n == 0:
        shape = (0, len(evaluation_times))
        return np.empty(shape), np.empty(shape), np.empty(shape)
    ri = improvement_model.predict_risk_score(x)
    rp = progression_model.predict_risk_score(x)
    ti = improvement_model.baseline_times_
    tp = progression_model.baseline_times_
    dhi = improvement_model.baseline_hazard_increments_
    dhp = progression_model.baseline_hazard_increments_
    if ti is None or tp is None or dhi is None or dhp is None:
        raise RuntimeError("Both cause-specific models require baseline hazards.")

    jump_times = np.unique(np.concatenate([ti, tp]))
    jump_times = jump_times[jump_times <= float(evaluation_times.max()) + 1e-10]
    map_i = {float(t): float(h) for t, h in zip(ti, dhi)}
    map_p = {float(t): float(h) for t, h in zip(tp, dhp)}
    survival = np.ones(n, dtype=float)
    cif_i = np.zeros(n, dtype=float)
    cif_p = np.zeros(n, dtype=float)
    out_i = np.zeros((n, len(evaluation_times)), dtype=float)
    out_p = np.zeros_like(out_i)
    out_s = np.zeros_like(out_i)
    jump_index = 0
    for q, eval_time in enumerate(evaluation_times):
        while jump_index < len(jump_times) and jump_times[jump_index] <= eval_time + 1e-10:
            tt = float(jump_times[jump_index])
            hi = map_i.get(tt, 0.0) * ri
            hp = map_p.get(tt, 0.0) * rp
            total = hi + hp
            transition = -np.expm1(-total)
            allocation_i = np.divide(
                hi, total, out=np.zeros_like(total), where=total > EPS
            )
            allocation_p = np.divide(
                hp, total, out=np.zeros_like(total), where=total > EPS
            )
            cif_i += survival * transition * allocation_i
            cif_p += survival * transition * allocation_p
            survival *= np.exp(-total)
            jump_index += 1
        out_i[:, q] = cif_i
        out_p[:, q] = cif_p
        out_s[:, q] = survival
    total = out_i + out_p + out_s
    out_i /= np.maximum(total, EPS)
    out_p /= np.maximum(total, EPS)
    out_s /= np.maximum(total, EPS)
    return out_i, out_p, out_s


@dataclass
class KaplanMeierCensoring:
    times_: np.ndarray
    survival_: np.ndarray

    @classmethod
    def fit(
        cls, time: np.ndarray, censoring_event: np.ndarray
    ) -> "KaplanMeierCensoring":
        time = _as_float_array(time)
        censoring_event = np.asarray(censoring_event, dtype=bool)
        unique = np.sort(np.unique(time[censoring_event]))
        survival = 1.0
        values: list[float] = []
        for tt in unique:
            risk = int(np.sum(time >= tt - 1e-10))
            d = int(np.sum(censoring_event & np.isclose(time, tt, atol=1e-10)))
            survival *= 1.0 - d / max(risk, 1)
            values.append(max(survival, 1e-6))
        return cls(unique.astype(float), np.asarray(values, dtype=float))

    def at(self, query: np.ndarray, *, left_limit: bool = False) -> np.ndarray:
        query = _as_float_array(query)
        side = "left" if left_limit else "right"
        idx = np.searchsorted(self.times_, query, side=side) - 1
        return np.where(idx >= 0, self.survival_[np.maximum(idx, 0)], 1.0)


def fixed_horizon_ipcw(
    time: np.ndarray,
    event_type: np.ndarray,
    horizon: float,
    censoring_model: KaplanMeierCensoring,
) -> np.ndarray:
    """Competing-risk Brier/calibration weights at a fixed horizon."""

    time = _as_float_array(time)
    event_type = np.asarray(event_type, dtype=str)
    any_event = (event_type != "none") & (time <= horizon + 1e-10)
    event_free = time >= horizon - 1e-10
    weights = np.zeros(len(time), dtype=float)
    if np.any(any_event):
        weights[any_event] = 1.0 / np.maximum(
            censoring_model.at(time[any_event], left_limit=True), 1e-6
        )
    if np.any(event_free):
        weights[event_free] = 1.0 / np.maximum(
            censoring_model.at(np.full(event_free.sum(), horizon)), 1e-6
        )
    return weights


def weighted_auc(
    y: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> float:
    """Weighted Mann-Whitney AUC with exact tie handling."""

    y = np.asarray(y, dtype=bool)
    prediction = _as_float_array(prediction)
    weight = _as_float_array(weight)
    use = np.isfinite(prediction) & (weight > 0)
    y, prediction, weight = y[use], prediction[use], weight[use]
    pos_total = float(weight[y].sum())
    neg_total = float(weight[~y].sum())
    if pos_total <= 0 or neg_total <= 0:
        return np.nan
    order = np.argsort(prediction, kind="mergesort")
    y, prediction, weight = y[order], prediction[order], weight[order]
    numerator = 0.0
    negative_below = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and prediction[end] == prediction[start]:
            end += 1
        group_pos = float(weight[start:end][y[start:end]].sum())
        group_neg = float(weight[start:end][~y[start:end]].sum())
        numerator += group_pos * (negative_below + 0.5 * group_neg)
        negative_below += group_neg
        start = end
    return numerator / (pos_total * neg_total)


def weighted_logistic_calibration(
    y: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Return CITL, slope, O/E, weighted observed rate, weighted expected rate."""

    y = np.asarray(y, dtype=float)
    prediction = np.clip(_as_float_array(prediction), 1e-6, 1 - 1e-6)
    weight = _as_float_array(weight)
    use = np.isfinite(y) & np.isfinite(prediction) & (weight > 0)
    y, prediction, weight = y[use], prediction[use], weight[use]
    if not len(y) or y.min() == y.max():
        return np.nan, np.nan, np.nan, np.nan, np.nan
    observed = float(np.average(y, weights=weight))
    expected = float(np.average(prediction, weights=weight))
    citl = float(
        np.log(np.clip(observed, 1e-6, 1 - 1e-6) / np.clip(1 - observed, 1e-6, 1))
        - np.log(np.clip(expected, 1e-6, 1 - 1e-6) / np.clip(1 - expected, 1e-6, 1))
    )
    oe = observed / max(expected, EPS)
    lp = np.log(prediction / (1 - prediction))

    def obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = theta[0] + theta[1] * lp
        mu = expit(np.clip(eta, -30, 30))
        loss = -np.sum(weight * (y * np.log(mu + EPS) + (1 - y) * np.log(1 - mu + EPS)))
        grad = np.array(
            [np.sum(weight * (mu - y)), np.sum(weight * (mu - y) * lp)]
        )
        return float(loss), grad

    result = minimize(
        obj,
        np.array([0.0, 1.0]),
        method="BFGS",
        jac=True,
        options={"maxiter": 200, "gtol": 1e-7},
    )
    return citl, float(result.x[1]), oe, observed, expected


def harrell_c_index(
    time: np.ndarray, target_event: np.ndarray, risk: np.ndarray
) -> float:
    """Harrell cause-specific concordance (competing events are censored)."""

    time = _as_float_array(time)
    target_event = np.asarray(target_event, dtype=bool)
    risk = _as_float_array(risk)
    concordant = 0.0
    comparable = 0.0
    for i in np.flatnonzero(target_event):
        controls = time > time[i] + 1e-10
        if not np.any(controls):
            continue
        comparable += float(controls.sum())
        concordant += float(np.sum(risk[i] > risk[controls]))
        concordant += 0.5 * float(np.sum(risk[i] == risk[controls]))
    return concordant / comparable if comparable else np.nan


def horizon_metric_record(
    *,
    probability_improvement: np.ndarray,
    probability_progression: np.ndarray,
    probability_remaining: np.ndarray,
    time: np.ndarray,
    event_type: np.ndarray,
    horizon: float,
    ipcw: np.ndarray,
) -> dict[str, float]:
    """Competing-risk-aware horizon metrics for all three states."""

    event_type = np.asarray(event_type, dtype=str)
    yi = (event_type == "improvement") & (time <= horizon + 1e-10)
    yp = (event_type == "progression") & (time <= horizon + 1e-10)
    yr = (time >= horizon - 1e-10) & (event_type == "none")
    known = ipcw > 0
    denom = max(float(ipcw.sum()), EPS)
    brier_i = float(np.sum(ipcw * (yi - probability_improvement) ** 2) / denom)
    brier_p = float(np.sum(ipcw * (yp - probability_progression) ** 2) / denom)
    brier_r = float(np.sum(ipcw * (yr - probability_remaining) ** 2) / denom)
    multi = brier_i + brier_p + brier_r
    ci_i, slope_i, oe_i, obs_i, exp_i = weighted_logistic_calibration(
        yi, probability_improvement, ipcw
    )
    ci_p, slope_p, oe_p, obs_p, exp_p = weighted_logistic_calibration(
        yp, probability_progression, ipcw
    )
    return {
        "auc_improvement": weighted_auc(yi, probability_improvement, ipcw),
        "auc_progression": weighted_auc(yp, probability_progression, ipcw),
        "c_index_improvement": harrell_c_index(
            time, event_type == "improvement", probability_improvement
        ),
        "c_index_progression": harrell_c_index(
            time, event_type == "progression", probability_progression
        ),
        "brier_improvement": brier_i,
        "brier_progression": brier_p,
        "brier_remaining": brier_r,
        "multistate_brier": multi,
        "calibration_intercept_improvement": ci_i,
        "calibration_slope_improvement": slope_i,
        "observed_expected_improvement": oe_i,
        "observed_rate_improvement": obs_i,
        "expected_rate_improvement": exp_i,
        "calibration_intercept_progression": ci_p,
        "calibration_slope_progression": slope_p,
        "observed_expected_progression": oe_p,
        "observed_rate_progression": obs_p,
        "expected_rate_progression": exp_p,
        "known_rows": int(known.sum()),
        "ipcw_effective_n": float((ipcw.sum() ** 2) / max(np.sum(ipcw**2), EPS)),
    }


def calibration_bins(
    y: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """Equal-frequency calibration bins using nonzero IPCW rows."""

    frame = pd.DataFrame(
        {"outcome": np.asarray(y, dtype=float), "prediction": prediction, "weight": weight}
    )
    frame = frame[frame["weight"].gt(0) & frame["prediction"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    rank = frame["prediction"].rank(method="first", pct=True)
    frame["bin"] = np.minimum((rank * bins).astype(int), bins - 1) + 1
    rows = []
    for b, group in frame.groupby("bin", observed=True):
        rows.append(
            {
                "bin": int(b),
                "rows": int(len(group)),
                "weight_sum": float(group["weight"].sum()),
                "mean_prediction": float(
                    np.average(group["prediction"], weights=group["weight"])
                ),
                "observed_probability": float(
                    np.average(group["outcome"], weights=group["weight"])
                ),
                "prediction_min": float(group["prediction"].min()),
                "prediction_max": float(group["prediction"].max()),
            }
        )
    return pd.DataFrame(rows)


def normal_interval(estimate: float, standard_error: float) -> tuple[float, float]:
    z = float(norm.ppf(0.975))
    return estimate - z * standard_error, estimate + z * standard_error
