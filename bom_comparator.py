"""
bom_comparator.py

Compares two versions of a Bill of Materials (BOM) — e.g. an old vs new
revision, or a system-of-record BOM vs a supplier-provided BOM — and
reports added, removed, and quantity-changed components.

This is a Python/pandas generalization of BOM comparison tooling I built
(originally two Excel/VBA tools: a flat-list diff and a side-by-side
visual comparison) for cross-checking product structures across systems.

Author: Tal Faynberg
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class BomDiff:
    added: pd.DataFrame = field(default_factory=pd.DataFrame)
    removed: pd.DataFrame = field(default_factory=pd.DataFrame)
    qty_changed: pd.DataFrame = field(default_factory=pd.DataFrame)
    unchanged_count: int = 0

    def has_changes(self) -> bool:
        return not (self.added.empty and self.removed.empty and self.qty_changed.empty)

    def summary(self) -> str:
        return (
            f"Added: {len(self.added)} | "
            f"Removed: {len(self.removed)} | "
            f"Qty changed: {len(self.qty_changed)} | "
            f"Unchanged: {self.unchanged_count}"
        )


def compare_bom(
    bom_old: pd.DataFrame,
    bom_new: pd.DataFrame,
    component_col: str = "component_id",
    qty_col: str = "quantity",
) -> BomDiff:
    """
    Compare two BOMs (each a flat list of component_id -> quantity).

    Parameters
    ----------
    bom_old, bom_new : pd.DataFrame
        Must contain columns [component_col, qty_col].
    """
    old_indexed = bom_old.set_index(component_col)[qty_col]
    new_indexed = bom_new.set_index(component_col)[qty_col]

    added_ids = new_indexed.index.difference(old_indexed.index)
    removed_ids = old_indexed.index.difference(new_indexed.index)
    common_ids = old_indexed.index.intersection(new_indexed.index)

    added = bom_new[bom_new[component_col].isin(added_ids)].reset_index(drop=True)
    removed = bom_old[bom_old[component_col].isin(removed_ids)].reset_index(drop=True)

    common_old = old_indexed.loc[common_ids]
    common_new = new_indexed.loc[common_ids]
    changed_mask = common_old != common_new

    qty_changed = pd.DataFrame({
        component_col: common_ids[changed_mask],
        "old_qty": common_old[changed_mask].values,
        "new_qty": common_new[changed_mask].values,
    })
    qty_changed["delta"] = qty_changed["new_qty"] - qty_changed["old_qty"]

    return BomDiff(
        added=added,
        removed=removed,
        qty_changed=qty_changed.reset_index(drop=True),
        unchanged_count=int((~changed_mask).sum()),
    )


def side_by_side(
    bom_old: pd.DataFrame,
    bom_new: pd.DataFrame,
    component_col: str = "component_id",
    qty_col: str = "quantity",
) -> pd.DataFrame:
    """Build a single side-by-side view for visual review/export to Excel."""
    merged = bom_old.merge(
        bom_new, on=component_col, how="outer", suffixes=("_old", "_new")
    )
    merged["status"] = "unchanged"
    merged.loc[merged[f"{qty_col}_old"].isna(), "status"] = "added"
    merged.loc[merged[f"{qty_col}_new"].isna(), "status"] = "removed"
    merged.loc[
        (merged[f"{qty_col}_old"].notna())
        & (merged[f"{qty_col}_new"].notna())
        & (merged[f"{qty_col}_old"] != merged[f"{qty_col}_new"]),
        "status",
    ] = "qty_changed"
    return merged.sort_values(["status", component_col]).reset_index(drop=True)


if __name__ == "__main__":
    # --- Demo with synthetic BOM data ------------------------------------
    bom_v1 = pd.DataFrame({
        "component_id": ["C-100", "C-101", "C-102", "C-103"],
        "quantity": [2, 1, 4, 1],
    })

    bom_v2 = pd.DataFrame({
        "component_id": ["C-100", "C-101", "C-103", "C-104"],
        "quantity": [2, 3, 1, 5],
    })

    diff = compare_bom(bom_v1, bom_v2)
    print(diff.summary())
    print("\nAdded components:")
    print(diff.added.to_string(index=False))
    print("\nRemoved components:")
    print(diff.removed.to_string(index=False))
    print("\nQuantity changes:")
    print(diff.qty_changed.to_string(index=False))

    print("\nSide-by-side view:")
    print(side_by_side(bom_v1, bom_v2).to_string(index=False))
