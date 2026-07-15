"""
sql_reporting.py

A small SQL-backed reporting tool: loads part/order data into a local
SQLite database, then runs parameterized SQL queries to answer common
catalogue/inventory questions (stock aging, top movers, price outliers).

Demonstrates schema design, parameterized queries (no string-formatted
SQL), and wrapping raw SQL results back into pandas for further use —
the kind of query work I'd otherwise run directly against JDE/CCU tables.

Author: Tal Faynberg
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS parts (
    part_number   TEXT PRIMARY KEY,
    description   TEXT NOT NULL,
    category      TEXT NOT NULL,
    unit_price    REAL NOT NULL,
    stock_qty     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    part_number   TEXT NOT NULL REFERENCES parts(part_number),
    order_date    TEXT NOT NULL,
    qty_ordered   INTEGER NOT NULL
);
"""


class InventoryDB:
    """Thin wrapper around a SQLite connection for this reporting demo."""

    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def load_parts(self, parts_df: pd.DataFrame) -> None:
        parts_df.to_sql("parts", self.conn, if_exists="append", index=False)

    def load_orders(self, orders_df: pd.DataFrame) -> None:
        orders_df.to_sql("orders", self.conn, if_exists="append", index=False)

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def close(self) -> None:
        self.conn.close()

    # --- Prebuilt reports -----------------------------------------------

    def top_movers(self, limit: int = 5) -> pd.DataFrame:
        return self.query(
            """
            SELECT p.part_number, p.description, SUM(o.qty_ordered) AS total_ordered
            FROM orders o
            JOIN parts p ON p.part_number = o.part_number
            GROUP BY p.part_number, p.description
            ORDER BY total_ordered DESC
            LIMIT ?
            """,
            (limit,),
        )

    def low_stock(self, threshold: int) -> pd.DataFrame:
        return self.query(
            """
            SELECT part_number, description, stock_qty
            FROM parts
            WHERE stock_qty < ?
            ORDER BY stock_qty ASC
            """,
            (threshold,),
        )

    def price_outliers_by_category(self, z_threshold: float = 1.5) -> pd.DataFrame:
        """Flag parts priced far from their category's average (simple z-score)."""
        parts = self.query("SELECT * FROM parts")
        stats = parts.groupby("category")["unit_price"].agg(["mean", "std"]).reset_index()
        merged = parts.merge(stats, on="category")
        merged["z_score"] = (merged["unit_price"] - merged["mean"]) / merged["std"]
        return merged[merged["z_score"].abs() > z_threshold][
            ["part_number", "description", "category", "unit_price", "z_score"]
        ].sort_values("z_score", key=abs, ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # --- Demo with synthetic parts + order history -----------------------
    parts = pd.DataFrame({
        "part_number": ["P001", "P002", "P003", "P004", "P005", "P006"],
        "description": ["Water Pump", "Brake Pad Set", "Oil Filter", "Air Filter", "Timing Belt", "Cabin Filter (Premium)"],
        "category": ["cooling", "brakes", "filters", "filters", "engine", "filters"],
        "unit_price": [42.50, 18.99, 6.75, 9.20, 210.00, 68.00],
        "stock_qty": [12, 3, 150, 60, 5, 40],
    })

    orders = pd.DataFrame({
        "part_number": ["P001", "P001", "P003", "P003", "P003", "P002"],
        "order_date": ["2026-05-01", "2026-06-10", "2026-05-15", "2026-06-01", "2026-06-20", "2026-06-05"],
        "qty_ordered": [10, 15, 40, 35, 50, 8],
    })

    with closing(InventoryDB()) as db:
        db.load_parts(parts)
        db.load_orders(orders)

        print("Top movers:")
        print(db.top_movers().to_string(index=False))

        print("\nLow stock (< 10 units):")
        print(db.low_stock(threshold=10).to_string(index=False))

        print("\nPrice outliers by category:")
        print(db.price_outliers_by_category(z_threshold=1.0).to_string(index=False))
