"""
ticket_classifier.py

A lightweight text classifier that auto-categorizes short free-text
issue/ticket descriptions (e.g. "part not found", "wrong price", "stock
mismatch") into a fixed set of categories using scikit-learn.

Intended as a small, honest demonstration of the ML workflow (TF-IDF +
linear classifier, train/test split, evaluation) applied to a realistic
use case: auto-triaging data-quality issue reports instead of manually
categorizing every one, which is what the categorization logic in my
validation tooling did by hand-coded rules. This version learns it from
labeled examples instead.

Author: Tal Faynberg
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


@dataclass
class TrainedClassifier:
    pipeline: Pipeline
    report: str

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.pipeline.predict(texts))

    def predict_proba_top(self, texts: list[str]) -> list[tuple[str, float]]:
        probs = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_
        results = []
        for row in probs:
            best_idx = row.argmax()
            results.append((classes[best_idx], round(float(row[best_idx]), 3)))
        return results


def train_ticket_classifier(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> TrainedClassifier:
    """Train a TF-IDF + logistic regression classifier on labeled ticket text."""
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    report = classification_report(y_test, y_pred, zero_division=0)

    return TrainedClassifier(pipeline=pipeline, report=report)


if __name__ == "__main__":
    # --- Demo with synthetic labeled ticket data --------------------------
    data = pd.DataFrame({
        "text": [
            "part number not found in catalogue",
            "missing part in the system export",
            "cannot locate component in database",
            "price does not match between systems",
            "unit price mismatch on invoice",
            "cost discrepancy for this part",
            "stock quantity is wrong",
            "inventory count does not match warehouse",
            "on-hand quantity looks incorrect",
            "status code is invalid for this item",
            "part marked active but should be superseded",
            "status field shows an unrecognized code",
            "part number not found for this SKU",
            "component missing from the export file",
            "price is off by a large margin",
            "invoice cost is different from catalogue price",
            "quantity on hand seems too low",
            "stock count mismatch after last sync",
            "status looks outdated for this record",
            "invalid status code returned",
        ],
        "label": [
            "missing_part", "missing_part", "missing_part",
            "price_mismatch", "price_mismatch", "price_mismatch",
            "stock_mismatch", "stock_mismatch", "stock_mismatch",
            "status_issue", "status_issue", "status_issue",
            "missing_part", "missing_part",
            "price_mismatch", "price_mismatch",
            "stock_mismatch", "stock_mismatch",
            "status_issue", "status_issue",
        ],
    })

    result = train_ticket_classifier(data["text"].tolist(), data["label"].tolist())

    print("Evaluation on held-out test split:")
    print(result.report)

    new_tickets = [
        "the part cannot be found anywhere in the system",
        "the price shown does not match what was invoiced",
        "warehouse count is lower than what the system shows",
    ]
    predictions = result.predict_proba_top(new_tickets)
    print("Predictions on new tickets:")
    for ticket, (label, confidence) in zip(new_tickets, predictions):
        print(f"  [{label:16s} ({confidence:.0%})]  {ticket}")
