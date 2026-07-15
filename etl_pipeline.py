"""
etl_pipeline.py

A small, extensible ETL pipeline for ingesting product/part data from
multiple source formats (CSV, fixed-width text, pipe-delimited) into a
single normalized schema, with per-record error logging instead of
hard failures.

Generalized from a binary/structured .data file parser I built for
importing supplier catalogue exports (TecDoc-format files), which used
a custom binary reader with a Shell.Application COM fallback. This
version focuses on the common text-based formats most ETL work involves.

Author: Tal Faynberg
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass
class ExtractError:
    source: str
    line_number: int
    raw_line: str
    reason: str


@dataclass
class ExtractResult:
    records: list[dict] = field(default_factory=list)
    errors: list[ExtractError] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def error_report(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self.errors])


# --- Extractors -------------------------------------------------------------

def extract_csv(text: str, source_name: str = "csv") -> ExtractResult:
    """Parse standard CSV text into normalized records."""
    result = ExtractResult()
    reader = csv.DictReader(io.StringIO(text))
    for i, row in enumerate(reader, start=2):  # header is line 1
        try:
            result.records.append(_normalize_row(row))
        except (ValueError, KeyError) as exc:
            result.errors.append(ExtractError(source_name, i, str(row), str(exc)))
    return result


def extract_fixed_width(
    text: str,
    field_specs: list[tuple[str, int, int]],
    source_name: str = "fixed_width",
) -> ExtractResult:
    """
    Parse fixed-width text. field_specs is a list of
    (field_name, start_col, end_col) using 0-based, end-exclusive slicing.
    """
    result = ExtractResult()
    for i, line in enumerate(text.strip().splitlines(), start=1):
        try:
            row = {name: line[start:end].strip() for name, start, end in field_specs}
            result.records.append(_normalize_row(row))
        except (ValueError, IndexError) as exc:
            result.errors.append(ExtractError(source_name, i, line, str(exc)))
    return result


def _normalize_row(row: dict) -> dict:
    """Coerce common fields to expected types; raises on bad data."""
    return {
        "part_number": row["part_number"].strip(),
        "description": row.get("description", "").strip(),
        "quantity": int(row["quantity"]),
        "unit_price": float(row["unit_price"]),
    }


# --- Pipeline orchestration --------------------------------------------------

@dataclass
class PipelineResult:
    data: pd.DataFrame
    total_extracted: int
    total_errors: int
    errors: pd.DataFrame


def run_pipeline(
    sources: list[tuple[str, Callable[[], ExtractResult]]],
) -> PipelineResult:
    """
    Run extraction across multiple sources and combine into one dataset.

    Parameters
    ----------
    sources : list of (label, extractor_fn) pairs, where extractor_fn takes
        no arguments and returns an ExtractResult (use functools.partial or
        a lambda to bind arguments).
    """
    all_records: list[dict] = []
    all_errors: list[pd.DataFrame] = []

    for label, extractor_fn in sources:
        result = extractor_fn()
        all_records.extend(result.records)
        if result.errors:
            err_df = result.error_report()
            err_df["source_label"] = label
            all_errors.append(err_df)

    combined = pd.DataFrame(all_records).drop_duplicates(subset="part_number")
    errors_df = pd.concat(all_errors, ignore_index=True) if all_errors else pd.DataFrame()

    return PipelineResult(
        data=combined.reset_index(drop=True),
        total_extracted=len(combined),
        total_errors=len(errors_df),
        errors=errors_df,
    )


if __name__ == "__main__":
    # --- Demo: two different source formats feeding one pipeline --------
    csv_source = """part_number,description,quantity,unit_price
P100,Water Pump,25,42.50
P101,Brake Pad Set,,18.99
P102,Oil Filter,150,6.75"""

    fw_specs = [
        ("part_number", 0, 4),
        ("description", 4, 22),
        ("quantity", 22, 26),
        ("unit_price", 26, 34),
    ]

    def _pad(value: str, width: int) -> str:
        return value.ljust(width)[:width]

    fixed_width_rows = [
        ("P200", "Air Filter", "80", "12.30"),
        ("P201", "Spark Plug", "500", "1.85"),
    ]
    fixed_width_source = "\n".join(
        _pad(pn, 4) + _pad(desc, 18) + _pad(qty, 4) + _pad(price, 8)
        for pn, desc, qty, price in fixed_width_rows
    )

    result = run_pipeline([
        ("supplier_csv_export", lambda: extract_csv(csv_source, "supplier_csv_export")),
        ("legacy_fixed_width_feed", lambda: extract_fixed_width(fixed_width_source, fw_specs, "legacy_feed")),
    ])

    print(f"Extracted {result.total_extracted} clean records, {result.total_errors} errors\n")
    print(result.data.to_string(index=False))
    if result.total_errors:
        print("\nErrors:")
        print(result.errors.to_string(index=False))
