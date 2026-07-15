"""
data_reconciliation.py

Compares product/part records across two data sources (e.g. an internal
catalogue vs an ERP export) and flags discrepancies by category.

This is a generalized version of a reconciliation pattern I built to
cross-check a product catalogue against ERP order/stock data at scale
(originally implemented in VBA against CCU + JDE exports; this is the
Python/pandas equivalent).

Author: Tal Faynberg
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ReconciliationReport:
    """Holds categorized discrepancy results from a reconciliation run."""

    status_discrepancies: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_issues: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_issues: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_in_source_b: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_in_source_a: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> pd.DataFrame:
        """One-line-per-category count, ready to print or export."""
        rows = [
            ("Status discrepancies", len(self.status_discrepancies)),
            ("Stock issues", len(self.stock_issues)),
            ("Price issues", len(self.price_issues)),
            ("Missing in Source B", len(self.missing_in_source_b)),
            ("Missing in Source A", len(self.missing_in_source_a)),
        ]
        return pd.DataFrame(rows, columns=["Category", "Count"])


def reconcile(
    source_a: pd.DataFrame,
    source_b: pd.DataFrame,
    key: str = "part_number",
    price_tolerance: float = 0.01,
) -> ReconciliationReport:
    """
    Reconcile two part/product datasets on a shared key.

    Parameters
    ----------
    source_a, source_b : pd.DataFrame
        Must each contain columns: [key, 'status', 'stock_qty', 'unit_price'].
    key : str
        Column name used to join the two sources.
    price_tolerance : float
        Absolute price difference above which a mismatch is flagged.

    Returns
    -------
    ReconciliationReport
    """
    merged = source_a.merge(
        source_b, on=key, how="outer", suffixes=("_a", "_b"), indicator=True
    )

    missing_in_b = merged[merged["_merge"] == "left_only"][[key]]
    missing_in_a = merged[merged["_merge"] == "right_only"][[key]]
    both = merged[merged["_merge"] == "both"].copy()

    status_mismatch = both[both["status_a"] != both["status_b"]][
        [key, "status_a", "status_b"]
    ]

    stock_mismatch = both[both["stock_qty_a"] != both["stock_qty_b"]][
        [key, "stock_qty_a", "stock_qty_b"]
    ].assign(
        delta=lambda d: d["stock_qty_a"] - d["stock_qty_b"]
    )

    price_mismatch = both[
        (both["unit_price_a"] - both["unit_price_b"]).abs() > price_tolerance
    ][[key, "unit_price_a", "unit_price_b"]].assign(
        delta=lambda d: (d["unit_price_a"] - d["unit_price_b"]).round(2)
    )

    return ReconciliationReport(
        status_discrepancies=status_mismatch.reset_index(drop=True),
        stock_issues=stock_mismatch.reset_index(drop=True),
        price_issues=price_mismatch.reset_index(drop=True),
        missing_in_source_b=missing_in_b.reset_index(drop=True),
        missing_in_source_a=missing_in_a.reset_index(drop=True),
    )


def export_report(report: ReconciliationReport, path: str = "reconciliation_report.xlsx") -> None:
    """Write each discrepancy category to its own sheet, plus a summary tab."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        report.summary().to_excel(writer, sheet_name="Summary", index=False)
        report.status_discrepancies.to_excel(writer, sheet_name="Status_Discrepancies", index=False)
        report.stock_issues.to_excel(writer, sheet_name="Stock_Issues", index=False)
        report.price_issues.to_excel(writer, sheet_name="Price_Issues", index=False)
        report.missing_in_source_b.to_excel(writer, sheet_name="Missing_in_B", index=False)
        report.missing_in_source_a.to_excel(writer, sheet_name="Missing_in_A", index=False)


if __name__ == "__main__":
    # --- Demo with synthetic data ---------------------------------------
    catalogue = pd.DataFrame({
        "part_number": ["P001", "P002", "P003", "P004", "P006"],
        "status": ["active", "active", "superseded", "active", "active"],
        "stock_qty": [120, 45, 0, 300, 15],
        "unit_price": [12.50, 8.75, 4.20, 33.00, 19.99],
    })

    erp_export = pd.DataFrame({
        "part_number": ["P001", "P002", "P003", "P005", "P006"],
        "status": ["active", "discontinued", "superseded", "active", "active"],
        "stock_qty": [120, 45, 0, 60, 10],
        "unit_price": [12.50, 8.75, 4.50, 22.00, 19.99],
    })

    report = reconcile(catalogue, erp_export)

    print("Reconciliation Summary")
    print(report.summary().to_string(index=False))
    print("\nStatus discrepancies:")
    print(report.status_discrepancies.to_string(index=False))
    print("\nStock issues:")
    print(report.stock_issues.to_string(index=False))
