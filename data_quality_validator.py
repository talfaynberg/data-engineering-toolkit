"""
data_quality_validator.py

A small, extensible rule-based data quality validator for tabular data.
Define rules once, run them against any DataFrame, and get a structured
report of violations grouped by severity.

Grew out of catalogue data-quality checks I used to run manually/in VBA
(missing prices, invalid status codes, duplicate part numbers, out-of-range
quantities). This generalizes that into a reusable framework.

Author: Tal Faynberg
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import pandas as pd


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class Rule:
    name: str
    check: Callable[[pd.DataFrame], pd.Series]  # returns boolean mask of FAILING rows
    severity: Severity
    description: str = ""


@dataclass
class Violation:
    rule_name: str
    severity: Severity
    row_index: int
    description: str


@dataclass
class ValidationReport:
    violations: list[Violation] = field(default_factory=list)

    def add(self, rule: Rule, failing_rows: pd.Index) -> None:
        for idx in failing_rows:
            self.violations.append(
                Violation(rule.name, rule.severity, idx, rule.description)
            )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "row": v.row_index,
                    "rule": v.rule_name,
                    "severity": v.severity.value,
                    "description": v.description,
                }
                for v in self.violations
            ]
        )

    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.ERROR)

    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.WARNING)

    def passed(self) -> bool:
        """No ERROR-level violations means the dataset passes."""
        return self.error_count() == 0


class DataQualityValidator:
    """Register rules, then run them all against a DataFrame in one pass."""

    def __init__(self) -> None:
        self._rules: list[Rule] = []

    def add_rule(self, rule: Rule) -> "DataQualityValidator":
        self._rules.append(rule)
        return self  # allows chaining

    def run(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()
        for rule in self._rules:
            failing_mask = rule.check(df)
            failing_rows = df.index[failing_mask]
            report.add(rule, failing_rows)
        return report


# --- Convenience rule builders --------------------------------------------

def rule_not_null(column: str, severity: Severity = Severity.ERROR) -> Rule:
    return Rule(
        name=f"not_null_{column}",
        check=lambda df: df[column].isna(),
        severity=severity,
        description=f"'{column}' must not be empty",
    )


def rule_no_duplicates(column: str, severity: Severity = Severity.ERROR) -> Rule:
    return Rule(
        name=f"no_duplicates_{column}",
        check=lambda df: df[column].duplicated(keep=False),
        severity=severity,
        description=f"'{column}' must be unique",
    )


def rule_in_range(column: str, low: float, high: float, severity: Severity = Severity.WARNING) -> Rule:
    return Rule(
        name=f"range_{column}",
        check=lambda df: ~df[column].between(low, high),
        severity=severity,
        description=f"'{column}' should be between {low} and {high}",
    )


def rule_allowed_values(column: str, allowed: set, severity: Severity = Severity.ERROR) -> Rule:
    return Rule(
        name=f"allowed_values_{column}",
        check=lambda df: ~df[column].isin(allowed),
        severity=severity,
        description=f"'{column}' must be one of {sorted(allowed)}",
    )


if __name__ == "__main__":
    # --- Demo with synthetic catalogue data ------------------------------
    data = pd.DataFrame({
        "part_number": ["P001", "P002", "P002", "P004", None],
        "status_code": ["001", "008", "999", "001", "011"],
        "unit_price": [12.50, -3.00, 8.75, 45.00, 19.99],
    })

    validator = (
        DataQualityValidator()
        .add_rule(rule_not_null("part_number"))
        .add_rule(rule_no_duplicates("part_number"))
        .add_rule(rule_allowed_values("status_code", {"001", "008", "009", "011"}))
        .add_rule(rule_in_range("unit_price", 0, 10_000, severity=Severity.ERROR))
    )

    report = validator.run(data)

    print(f"Passed: {report.passed()}")
    print(f"Errors: {report.error_count()}  |  Warnings: {report.warning_count()}\n")
    print(report.to_dataframe().to_string(index=False))
