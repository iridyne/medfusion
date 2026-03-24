"""Advanced post-training analysis helpers for build-results.

Provides dependency-light survival analysis and SHAP-style variable importance
artifacts for MedFusion postprocessing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _coerce_event_array(values: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.Series):
        raw = values.to_numpy()
    else:
        raw = np.asarray(values)

    result = np.zeros(len(raw), dtype=int)
    positive_tokens = {"1", "true", "yes", "event", "dead", "deceased"}
    for index, value in enumerate(raw):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            result[index] = 0
        elif isinstance(value, (bool, np.bool_)):
            result[index] = int(value)
        elif isinstance(value, (int, np.integer)):
            result[index] = int(value > 0)
        elif isinstance(value, (float, np.floating)):
            result[index] = int(value > 0)
        else:
            result[index] = int(str(value).strip().lower() in positive_tokens)
    return result


def _compute_concordance_index(
    survival_times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> float | None:
    comparable = 0
    concordant = 0.0

    for i in range(len(survival_times)):
        for j in range(i + 1, len(survival_times)):
            if events[i] == 1 and survival_times[i] < survival_times[j]:
                comparable += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1.0
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
            elif events[j] == 1 and survival_times[j] < survival_times[i]:
                comparable += 1
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1.0
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5

    if comparable == 0:
        return None
    return float(concordant / comparable)


def _kaplan_meier_curve(
    survival_times: np.ndarray,
    events: np.ndarray,
) -> list[dict[str, float | int]]:
    if len(survival_times) == 0:
        return [{"time": 0.0, "survival_probability": 1.0, "n_at_risk": 0, "events": 0, "censored": 0}]

    order = np.argsort(survival_times)
    survival_times = survival_times[order]
    events = events[order]

    unique_times = np.unique(survival_times)
    n_at_risk = len(survival_times)
    survival_prob = 1.0
    curve = [{"time": 0.0, "survival_probability": 1.0, "n_at_risk": int(n_at_risk), "events": 0, "censored": 0}]

    for time_value in unique_times:
        time_mask = survival_times == time_value
        events_at_time = int(events[time_mask].sum())
        censored_at_time = int(time_mask.sum() - events_at_time)

        if n_at_risk > 0 and events_at_time > 0:
            survival_prob *= 1.0 - (events_at_time / n_at_risk)

        curve.append(
            {
                "time": float(time_value),
                "survival_probability": float(survival_prob),
                "n_at_risk": int(n_at_risk),
                "events": events_at_time,
                "censored": censored_at_time,
            }
        )
        n_at_risk -= int(time_mask.sum())

    return curve


def _median_survival_time(curve: list[dict[str, float | int]]) -> float | None:
    for point in curve:
        if float(point["survival_probability"]) <= 0.5:
            return float(point["time"])
    return None


def _compute_logrank_p_value(
    times: np.ndarray,
    events: np.ndarray,
    groups: np.ndarray,
) -> float | None:
    unique_event_times = np.unique(times[events == 1])
    if len(unique_event_times) == 0:
        return None

    observed_minus_expected = 0.0
    variance = 0.0

    for time_value in unique_event_times:
        at_risk = times >= time_value
        event_mask = (times == time_value) & (events == 1)

        n1 = int(np.sum(at_risk & (groups == 1)))
        n0 = int(np.sum(at_risk & (groups == 0)))
        d1 = int(np.sum(event_mask & (groups == 1)))
        d0 = int(np.sum(event_mask & (groups == 0)))
        n = n1 + n0
        d = d1 + d0

        if n <= 1 or d == 0:
            continue

        expected_1 = d * (n1 / n)
        var_1 = (n1 * n0 * d * (n - d)) / ((n**2) * (n - 1)) if n > 1 else 0.0
        observed_minus_expected += d1 - expected_1
        variance += var_1

    if variance <= 0:
        return None

    z_score = observed_minus_expected / math.sqrt(variance)
    # Two-sided p-value for N(0,1) using erfc.
    return float(math.erfc(abs(z_score) / math.sqrt(2.0)))


def _plot_kaplan_meier(
    low_curve: list[dict[str, float | int]],
    high_curve: list[dict[str, float | int]],
    save_path: Path,
    title: str,
    low_label: str,
    high_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(
        [float(point["time"]) for point in low_curve],
        [float(point["survival_probability"]) for point in low_curve],
        where="post",
        linewidth=2,
        label=low_label,
        color="#1f77b4",
    )
    ax.step(
        [float(point["time"]) for point in high_curve],
        [float(point["survival_probability"]) for point in high_curve],
        where="post",
        linewidth=2,
        label=high_label,
        color="#d62728",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_risk_distribution(
    risk_scores: np.ndarray,
    groups: np.ndarray,
    cutoff: float,
    save_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(risk_scores[groups == 0], bins=20, alpha=0.6, color="#1f77b4", label="Low risk")
    ax.hist(risk_scores[groups == 1], bins=20, alpha=0.6, color="#d62728", label="High risk")
    ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.5, label=f"cutoff={cutoff:.3f}")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_survival_artifacts(
    output_dir: Path,
    metadata_frame: pd.DataFrame | None,
    risk_scores: np.ndarray,
    *,
    time_column: str | None,
    event_column: str | None,
    split: str,
    cutoff: float | None = None,
    cutoff_source: str = "current_split_median",
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    payload: dict[str, Any] = {
        "enabled": True,
        "available": False,
        "split": split,
        "time_column": time_column,
        "event_column": event_column,
    }
    if metadata_frame is None or not time_column or not event_column:
        payload["reason"] = "survival_columns_not_configured"
        return payload, {}, {}
    if time_column not in metadata_frame.columns or event_column not in metadata_frame.columns:
        payload["reason"] = "survival_columns_missing_in_metadata"
        return payload, {}, {}

    times = pd.to_numeric(metadata_frame[time_column], errors="coerce").to_numpy(dtype=float)
    events = _coerce_event_array(metadata_frame[event_column])
    risk_scores = np.asarray(risk_scores, dtype=float).reshape(-1)

    valid_mask = np.isfinite(times) & np.isfinite(risk_scores)
    if valid_mask.sum() < 4:
        payload["reason"] = "not_enough_valid_survival_samples"
        return payload, {}, {}

    times = times[valid_mask]
    events = events[valid_mask]
    risk_scores = risk_scores[valid_mask]

    if cutoff is None:
        cutoff = float(np.median(risk_scores))
        cutoff_source = "current_split_median"

    groups = (risk_scores >= cutoff).astype(int)
    low_curve = _kaplan_meier_curve(times[groups == 0], events[groups == 0])
    high_curve = _kaplan_meier_curve(times[groups == 1], events[groups == 1])
    c_index = _compute_concordance_index(times, events, risk_scores)
    logrank_p = _compute_logrank_p_value(times, events, groups)

    survival_dir = output_dir / "visualizations" / "survival"
    survival_dir.mkdir(parents=True, exist_ok=True)
    km_plot_path = survival_dir / "kaplan_meier.png"
    risk_distribution_path = survival_dir / "risk_score_distribution.png"
    _plot_kaplan_meier(
        low_curve,
        high_curve,
        save_path=km_plot_path,
        title=f"Kaplan-Meier Survival Curve ({split})",
        low_label=f"Low risk (n={(groups == 0).sum()})",
        high_label=f"High risk (n={(groups == 1).sum()})",
    )
    _plot_risk_distribution(
        risk_scores,
        groups,
        cutoff=cutoff,
        save_path=risk_distribution_path,
        title=f"Risk Score Distribution ({split})",
    )

    payload.update(
        {
            "available": True,
            "cutoff": round(float(cutoff), 6),
            "cutoff_source": cutoff_source,
            "sample_count": int(len(times)),
            "event_count": int(events.sum()),
            "event_rate": round(float(events.mean()), 4),
            "c_index": round(float(c_index), 4) if c_index is not None else None,
            "logrank_p_value": round(float(logrank_p), 6) if logrank_p is not None else None,
            "risk_score_summary": {
                "min": round(float(np.min(risk_scores)), 4),
                "max": round(float(np.max(risk_scores)), 4),
                "mean": round(float(np.mean(risk_scores)), 4),
                "median": round(float(np.median(risk_scores)), 4),
                "std": round(float(np.std(risk_scores)), 4),
            },
            "groups": {
                "low": {
                    "count": int((groups == 0).sum()),
                    "event_count": int(events[groups == 0].sum()),
                    "curve": low_curve,
                    "median_survival_time": _median_survival_time(low_curve),
                },
                "high": {
                    "count": int((groups == 1).sum()),
                    "event_count": int(events[groups == 1].sum()),
                    "curve": high_curve,
                    "median_survival_time": _median_survival_time(high_curve),
                },
            },
            "artifacts": {
                "kaplan_meier_plot_path": str(km_plot_path),
                "risk_score_distribution_plot_path": str(risk_distribution_path),
            },
        }
    )

    survival_json_path = output_dir / "survival.json"
    _write_json(survival_json_path, payload)

    artifact_paths = {
        "survival_json_path": str(survival_json_path),
        "kaplan_meier_plot_path": str(km_plot_path),
        "risk_score_distribution_plot_path": str(risk_distribution_path),
    }
    metric_updates = {
        "c_index": round(float(c_index), 4) if c_index is not None else None,
        "logrank_p_value": round(float(logrank_p), 6) if logrank_p is not None else None,
        "risk_cutoff": round(float(cutoff), 6),
    }
    return payload, artifact_paths, metric_updates


def _plot_shap_bar(
    feature_names: list[str],
    importance: np.ndarray,
    save_path: Path,
    top_k: int,
) -> None:
    order = np.argsort(importance)[-top_k:]
    ordered_names = [feature_names[index] for index in order]
    ordered_values = importance[order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(order))))
    ax.barh(range(len(order)), ordered_values, color="#4c78a8")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(ordered_names)
    ax.set_xlabel("Mean |contribution|")
    ax.set_title("Global Variable Importance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_shap_beeswarm(
    feature_names: list[str],
    contribution_matrix: np.ndarray,
    feature_values: np.ndarray,
    save_path: Path,
    top_k: int,
    max_samples: int,
) -> None:
    importance = np.mean(np.abs(contribution_matrix), axis=0)
    order = np.argsort(importance)[-top_k:]

    if contribution_matrix.shape[0] > max_samples:
        sample_indices = np.linspace(0, contribution_matrix.shape[0] - 1, max_samples, dtype=int)
        contribution_matrix = contribution_matrix[sample_indices]
        feature_values = feature_values[sample_indices]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(order))))
    for row_index, feature_index in enumerate(order):
        contributions = contribution_matrix[:, feature_index]
        values = feature_values[:, feature_index]
        if np.max(values) > np.min(values):
            normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        else:
            normalized_values = np.full_like(values, 0.5, dtype=float)
        jitter = np.random.default_rng(42).normal(0, 0.08, size=len(contributions))
        ax.scatter(
            contributions,
            np.full(len(contributions), row_index) + jitter,
            c=normalized_values,
            cmap="coolwarm",
            s=18,
            alpha=0.75,
            edgecolors="none",
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[index] for index in order])
    ax.set_xlabel("Contribution (surrogate SHAP value)")
    ax.set_title("SHAP-style Beeswarm")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_shap_artifacts(
    output_dir: Path,
    feature_names: list[str],
    tabular_data: np.ndarray,
    model_scores: np.ndarray,
    y_true: np.ndarray,
    *,
    positive_class_label: str | None,
    max_display: int = 10,
    max_samples: int = 200,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    payload: dict[str, Any] = {
        "enabled": True,
        "available": False,
    }

    tabular_data = np.asarray(tabular_data, dtype=float)
    model_scores = np.asarray(model_scores, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)

    if len(tabular_data) != len(model_scores) or len(y_true) != len(model_scores):
        payload["reason"] = "shape_mismatch"
        return payload, {}, {}
    if len(model_scores) < 8:
        payload["reason"] = "not_enough_samples"
        return payload, {}, {}

    if tabular_data.ndim == 1:
        tabular_data = tabular_data.reshape(-1, 1)

    combined_feature_names = ["model_score"] + list(feature_names or [])
    if tabular_data.shape[1] != len(feature_names):
        combined_feature_names = ["model_score"] + [f"feature_{index}" for index in range(tabular_data.shape[1])]

    X = np.concatenate([model_scores.reshape(-1, 1), tabular_data], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_binary = (y_true == 1).astype(int)

    try:
        if np.unique(y_binary).size > 1:
            surrogate = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)
            surrogate.fit(X_scaled, y_binary)
            coefficients = surrogate.coef_[0]
            contribution_matrix = X_scaled * coefficients.reshape(1, -1)
            method = "logistic_surrogate_shap"
            intercept = float(surrogate.intercept_[0])
        else:
            raise ValueError("binary target collapsed")
    except Exception:
        surrogate = Ridge(alpha=1.0, random_state=42)
        surrogate.fit(X_scaled, model_scores)
        coefficients = surrogate.coef_.reshape(-1)
        contribution_matrix = X_scaled * coefficients.reshape(1, -1)
        method = "ridge_surrogate_shap"
        intercept = float(np.ravel(surrogate.intercept_)[0]) if np.ndim(surrogate.intercept_) else float(surrogate.intercept_)

    global_importance = np.mean(np.abs(contribution_matrix), axis=0)
    order = np.argsort(global_importance)[::-1]

    shap_dir = output_dir / "visualizations" / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    shap_bar_path = shap_dir / "shap_bar.png"
    shap_beeswarm_path = shap_dir / "shap_beeswarm.png"
    _plot_shap_bar(combined_feature_names, global_importance, shap_bar_path, top_k=max_display)
    _plot_shap_beeswarm(
        combined_feature_names,
        contribution_matrix,
        X,
        shap_beeswarm_path,
        top_k=max_display,
        max_samples=max_samples,
    )

    ranked_features = [
        {
            "feature": combined_feature_names[index],
            "mean_abs_contribution": round(float(global_importance[index]), 6),
            "coefficient": round(float(coefficients[index]), 6),
        }
        for index in order
    ]

    payload.update(
        {
            "available": True,
            "method": method,
            "target": positive_class_label or "positive_class",
            "sample_count": int(X.shape[0]),
            "feature_count": int(X.shape[1]),
            "intercept": round(intercept, 6),
            "features": ranked_features,
            "artifacts": {
                "shap_bar_plot_path": str(shap_bar_path),
                "shap_beeswarm_plot_path": str(shap_beeswarm_path),
            },
        }
    )

    shap_json_path = output_dir / "shap_summary.json"
    _write_json(shap_json_path, payload)
    artifact_paths = {
        "shap_summary_json_path": str(shap_json_path),
        "shap_bar_plot_path": str(shap_bar_path),
        "shap_beeswarm_plot_path": str(shap_beeswarm_path),
    }
    metric_updates = {
        "shap_method": method,
        "top_global_feature": ranked_features[0]["feature"] if ranked_features else None,
    }
    return payload, artifact_paths, metric_updates
