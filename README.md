# Data Processing & Automation Toolkit

![Run scripts](https://github.com/<your-username>/<repo-name>/actions/workflows/run-scripts.yml/badge.svg)

Six standalone Python scripts demonstrating the kind of data quality,
reconciliation, ETL, and reporting work I do day to day, plus one
ML-adjacent piece. Originally built as VBA/Excel tooling for production
catalogue-vs-ERP checks; rewritten here in Python as generalized,
reusable modules with synthetic demo data (no real company data).

## Scripts

### `data_reconciliation.py`
Compares two part/product datasets on a shared key and flags status,
stock, and price mismatches, plus records missing from either source.

### `data_quality_validator.py`
A small rule-engine for data quality checks. Define rules once
(`not_null`, `no_duplicates`, `in_range`, `allowed_values`, or custom),
run them in one pass, get a severity-tagged violation report.

### `bom_comparator.py`
Diffs two Bill-of-Materials revisions: added/removed components,
quantity changes, plus a side-by-side view for manual review.

### `etl_pipeline.py`
Extracts records from multiple source formats (CSV, fixed-width text)
into one normalized schema, logging per-record errors instead of
failing the whole batch.

### `sql_reporting.py`
Loads part/order data into a local SQLite database and runs
parameterized SQL for common reports: top movers, low stock, and
category-relative price outliers.

### `ticket_classifier.py`
A TF-IDF + logistic regression classifier that auto-categorizes short
issue/ticket text into fixed categories, trained and evaluated on
labeled examples with scikit-learn.

### `anomaly_detector.py`
An unsupervised anomaly detection engine (IsolationForest) for flagging
catalogue/inventory records that look statistically off — price spikes,
stock gluts, dead stock — without needing pre-labeled examples. Includes
category-relative feature engineering, a z-score baseline for comparison,
evaluation against synthetic injected anomalies, and model persistence
via joblib.

## Run locally

```bash
pip install -r requirements.txt
python data_reconciliation.py
python data_quality_validator.py
python bom_comparator.py
python etl_pipeline.py
python sql_reporting.py
python ticket_classifier.py
python anomaly_detector.py
```

Each script runs standalone with built-in synthetic data — no external
files needed to see it work.

## Background

These patterns originated from building and maintaining a VBA-based
validation tool that consolidated and cross-checked product data across
a catalogue system, an ERP system, and several reporting sheets, producing
categorized issue reports for a data team to action. This repo reframes
that logic in a general, portfolio-friendly form, plus a couple of
adjacent skills (SQL reporting, basic ML).
