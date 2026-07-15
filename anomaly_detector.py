"""
anomaly_detector.py

An unsupervised anomaly detection engine for catalogue/inventory data —
flags parts whose combination of price, stock level, and order velocity
looks statistically "off," without needing pre-labeled examples of what
"broken" looks like.

Design notes (why this isn't just IsolationForest.fit_predict()):
- Feature engineering: raw price/stock aren't comparable across
  categories, so features are computed relative to each part's category
  (e.g. price deviation from category mean) rather than as raw values.
- Baseline comparison: a simple z-score rule is implemented alongside
  IsolationForest so the ML model's value-add can actually be judged,
  not assumed.
- Evaluation: since this is unsupervised, ground truth doesn't normally
  exist. Here, synthetic anomalies are deliberately injected into demo
  data so precision/recall can be computed — a technique used in practice
  to sanity-check unsupervised models before trusting them on real data.
- Persistence: the trained pipeline is saved/loaded with joblib, since a
  real deployment would train offline and score new records at request
  time.

Author: Tal Faynberg
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "price_z_in_category",
    "stock_z_in_category",
    "order_velocity_z_in_category",
    "price_to_stock_ratio",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build category-relative features so a €5 outlier in a €6-avg category
    is weighted the same as a €500 outlier in a €600-avg category.
    """
    out = df.copy()

    def zscore(group: pd.Series) -> pd.Series:
        std = group.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std

    out["price_z_in_category"] = out.groupby("category")["unit_price"].transform(zscore)
    out["stock_z_in_category"] = out.groupby("category")["stock_qty"].transform(zscore)
    out["order_velocity_z_in_category"] = out.groupby("category")["order_velocity"].transform(zscore)
    out["price_to_stock_ratio"] = out["unit_price"] / out["stock_qty"].replace(0, np.nan)
    out["price_to_stock_ratio"] = out["price_to_stock_ratio"].fillna(out["price_to_stock_ratio"].median())

    return out


@dataclass
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    flagged_count: int
    true_anomaly_count: int


class AnomalyDetector:
    """Wraps feature scaling + IsolationForest into one fit/predict unit."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", IsolationForest(contamination=contamination, random_state=random_state)),
        ])
        self._fitted = False

    def fit(self, df_features: pd.DataFrame) -> "AnomalyDetector":
        self.pipeline.fit(df_features[FEATURE_COLUMNS])
        self._fitted = True
        return self

    def predict(self, df_features: pd.DataFrame) -> pd.Series:
        """Returns 1 for anomaly, 0 for normal (relabeled from sklearn's -1/1)."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")
        raw = self.pipeline.predict(df_features[FEATURE_COLUMNS])
        return pd.Series(np.where(raw == -1, 1, 0), index=df_features.index, name="is_anomaly")

    def anomaly_score(self, df_features: pd.DataFrame) -> pd.Series:
        """Lower score = more anomalous (raw IsolationForest convention)."""
        return pd.Series(
            self.pipeline.decision_function(df_features[FEATURE_COLUMNS]),
            index=df_features.index,
            name="anomaly_score",
        )

    def save(self, path: str) -> None:
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        instance = cls()
        instance.pipeline = joblib.load(path)
        instance._fitted = True
        return instance


def zscore_baseline(df_features: pd.DataFrame, threshold: float = 2.5) -> pd.Series:
    """Simple baseline: flag if ANY single feature z-score exceeds threshold."""
    z_cols = ["price_z_in_category", "stock_z_in_category", "order_velocity_z_in_category"]
    flagged = (df_features[z_cols].abs() > threshold).any(axis=1)
    return flagged.astype(int).rename("is_anomaly")


def evaluate(predicted: pd.Series, ground_truth: pd.Series) -> EvaluationResult:
    return EvaluationResult(
        precision=round(precision_score(ground_truth, predicted, zero_division=0), 3),
        recall=round(recall_score(ground_truth, predicted, zero_division=0), 3),
        f1=round(f1_score(ground_truth, predicted, zero_division=0), 3),
        flagged_count=int(predicted.sum()),
        true_anomaly_count=int(ground_truth.sum()),
    )


def _make_synthetic_dataset(n_normal: int = 200, n_anomalies: int = 15, seed: int = 7) -> pd.DataFrame:
    """Generate normal catalogue data plus deliberately injected anomalies with ground-truth labels."""
    rng = np.random.default_rng(seed)
    categories = ["filters", "brakes", "cooling", "engine", "electrical"]
    category_price_base = {"filters": 8, "brakes": 25, "cooling": 40, "engine": 150, "electrical": 60}

    rows = []
    for i in range(n_normal):
        cat = rng.choice(categories)
        base = category_price_base[cat]
        rows.append({
            "part_number": f"N{i:04d}",
            "category": cat,
            "unit_price": max(0.5, rng.normal(base, base * 0.15)),
            "stock_qty": max(0, int(rng.normal(80, 20))),
            "order_velocity": max(0, rng.normal(10, 3)),
            "is_true_anomaly": 0,
        })

    for i in range(n_anomalies):
        cat = rng.choice(categories)
        base = category_price_base[cat]
        anomaly_type = rng.choice(["price_spike", "stock_glut", "dead_stock"])
        if anomaly_type == "price_spike":
            price, stock, velocity = base * rng.uniform(4, 8), rng.normal(80, 20), rng.normal(10, 3)
        elif anomaly_type == "stock_glut":
            price, stock, velocity = rng.normal(base, base * 0.15), rng.uniform(500, 900), rng.normal(1, 0.5)
        else:  # dead_stock
            price, stock, velocity = rng.normal(base, base * 0.15), rng.normal(80, 20), 0

        rows.append({
            "part_number": f"A{i:04d}",
            "category": cat,
            "unit_price": max(0.5, price),
            "stock_qty": max(0, int(stock)),
            "order_velocity": max(0, velocity),
            "is_true_anomaly": 1,
        })

    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    raw = _make_synthetic_dataset()
    features = engineer_features(raw)
    ground_truth = raw["is_true_anomaly"]

    # --- Baseline ----------------------------------------------------------
    baseline_pred = zscore_baseline(features)
    baseline_eval = evaluate(baseline_pred, ground_truth)

    # --- Isolation Forest ----------------------------------------------------
    contamination_rate = ground_truth.mean()  # informed by known anomaly rate in this demo
    detector = AnomalyDetector(contamination=contamination_rate).fit(features)
    model_pred = detector.predict(features)
    model_eval = evaluate(model_pred, ground_truth)

    print("=== Baseline (single-feature z-score > 2.5) ===")
    print(f"Precision: {baseline_eval.precision}  Recall: {baseline_eval.recall}  F1: {baseline_eval.f1}")
    print(f"Flagged {baseline_eval.flagged_count} of {len(raw)} (true anomalies: {baseline_eval.true_anomaly_count})\n")

    print("=== IsolationForest (multivariate) ===")
    print(f"Precision: {model_eval.precision}  Recall: {model_eval.recall}  F1: {model_eval.f1}")
    print(f"Flagged {model_eval.flagged_count} of {len(raw)} (true anomalies: {model_eval.true_anomaly_count})\n")

    if baseline_eval.f1 >= model_eval.f1:
        print("Note: on this dataset the simple baseline matches or beats IsolationForest.")
        print("That's a legitimate outcome, not a bug — with only 4 engineered features and")
        print("well-separated anomaly types, a univariate rule can be competitive. The value of")
        print("the ML model shows up more as feature count and interaction complexity grow.\n")

    # --- Inspect top flagged anomalies with scores --------------------------
    scores = detector.anomaly_score(features)
    result = raw.assign(anomaly_score=scores, flagged=model_pred).sort_values("anomaly_score").head(8)
    print("Most anomalous records (lowest score = most anomalous):")
    print(result[["part_number", "category", "unit_price", "stock_qty", "order_velocity", "anomaly_score", "flagged"]]
          .to_string(index=False))

    # --- Persistence demo ----------------------------------------------------
    model_path = "anomaly_model.joblib"
    detector.save(model_path)
    reloaded = AnomalyDetector.load(model_path)
    print(f"\nModel saved to {model_path} and reloaded successfully: "
          f"{(reloaded.predict(features) == model_pred).all()}")
    Path(model_path).unlink()  # cleanup for repeatable demo runs
