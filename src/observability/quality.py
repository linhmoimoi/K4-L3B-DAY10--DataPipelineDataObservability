from __future__ import annotations

from datetime import date, datetime
import math
from pathlib import Path
import re
from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

from core.config import Settings
from core.utils import write_json


_FALLBACK_FRESHNESS_DAYS = 180
_MAX_STALE_RATIO = 0.25


def _freshness_threshold(settings: Settings) -> int:
    value = getattr(settings, "freshness_threshold_days", None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return _FALLBACK_FRESHNESS_DAYS


def _json_safe(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def evaluate_freshness_sla(df: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Evaluate the allowed fraction of records older than the freshness limit."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Freshness SLA requires a pandas DataFrame.")
    if df.empty:
        raise ValueError("Freshness SLA cannot be evaluated for an empty DataFrame.")
    if "age_days" not in df.columns:
        raise KeyError("Freshness SLA requires the 'age_days' column.")

    age_days = pd.to_numeric(df["age_days"], errors="coerce")
    invalid_count = int(age_days.isna().sum())
    if invalid_count:
        raise ValueError(f"Freshness SLA cannot evaluate {invalid_count} missing or non-numeric age_days value(s).")

    threshold_days = _freshness_threshold(settings)
    stale_rows = int((age_days > threshold_days).sum())
    total_rows = int(len(df))
    stale_ratio = stale_rows / total_rows
    return {
        "threshold_days": threshold_days,
        "stale_rows": stale_rows,
        "total_rows": total_rows,
        "stale_ratio": stale_ratio,
        "stale_percent": stale_ratio * 100,
        "max_stale_ratio": _MAX_STALE_RATIO,
        "is_fresh": stale_ratio <= _MAX_STALE_RATIO,
    }


def build_freshness_report(df: pd.DataFrame, settings: Settings, report_path) -> dict[str, Any]:
    """Build and persist freshness statistics plus the oldest/newest dates."""
    report = evaluate_freshness_sla(df, settings)
    parsed_dates: list[pd.Timestamp] = []
    if "published" in df.columns:
        published = pd.to_datetime(df["published"], errors="coerce", utc=True).dropna()
        parsed_dates = list(published)
    report["latest_published"] = max(parsed_dates).date().isoformat() if parsed_dates else None
    report["oldest_published"] = min(parsed_dates).date().isoformat() if parsed_dates else None

    destination = Path(report_path) if report_path is not None else Path(settings.paths.freshness_report)
    write_json(destination, _json_safe(report))
    return report


def _quality_report_path(settings: Settings, stage: Any) -> Path:
    paths = settings.paths
    stage_name = str(stage).strip()
    normalized_stage = stage_name.casefold()
    if "corrupt" in normalized_stage and getattr(paths, "corrupted_quality_report", None):
        return Path(paths.corrupted_quality_report)
    if "baseline" in normalized_stage and getattr(paths, "baseline_quality_report", None):
        return Path(paths.baseline_quality_report)

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", stage_name).strip("_-").lower() or "quality"
    quality_dir = Path(getattr(paths, "quality_dir", Path("data") / "quality"))
    return quality_dir / f"{slug}_quality_report.json"


def _expectation_result(name: str, column: str | None, expectation: Any, batch: Any) -> dict[str, Any]:
    validation = batch.validate(expectation, result_format="SUMMARY")
    serialized = validation.to_json_dict()
    raw_result = serialized.get("result") or {}

    # Keep useful metrics while excluding examples of unexpected values, which
    # can contain full paper text or other input data.
    detail_fields = (
        "observed_value",
        "element_count",
        "missing_count",
        "missing_percent",
        "unexpected_count",
        "unexpected_percent",
        "unexpected_percent_nonmissing",
        "unexpected_percent_total",
    )
    details = {key: _json_safe(raw_result[key]) for key in detail_fields if key in raw_result}
    exception_info = serialized.get("exception_info") or {}
    if exception_info.get("raised_exception"):
        details["error"] = exception_info.get("exception_message") or "Great Expectations raised an exception."

    success = bool(serialized.get("success", False)) and not bool(exception_info.get("raised_exception"))
    entry: dict[str, Any] = {
        "name": name,
        "column": column,
        "status": "passed" if success else "failed",
        "success": success,
        "result": details,
    }
    return entry


def run_data_quality_checks(df: pd.DataFrame, settings: Settings, report_name: str) -> dict[str, Any]:
    """Run Great Expectations checks and the freshness SLA, then save reports."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Quality checks require a pandas DataFrame.")

    # Great Expectations 1.x fluent API with an in-memory context; no project
    # configuration or validation results are persisted by GX itself.
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    data_source = context.data_sources.add_pandas(name="papers_source")
    data_asset = data_source.add_dataframe_asset(name="papers_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("papers_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    checks: list[tuple[str, str | None, Any]] = [
        (
            "ExpectTableRowCountToBeBetween",
            None,
            gx.expectations.ExpectTableRowCountToBeBetween(min_value=5, max_value=5000),
        ),
        (
            "ExpectColumnValuesToNotBeNull",
            "paper_id",
            gx.expectations.ExpectColumnValuesToNotBeNull(column="paper_id"),
        ),
        (
            "ExpectColumnValuesToNotBeNull",
            "title",
            gx.expectations.ExpectColumnValuesToNotBeNull(column="title"),
        ),
        (
            "ExpectColumnValuesToNotBeNull",
            "text_for_embedding",
            gx.expectations.ExpectColumnValuesToNotBeNull(column="text_for_embedding"),
        ),
        (
            "ExpectColumnValuesToBeUnique",
            "paper_id",
            gx.expectations.ExpectColumnValuesToBeUnique(column="paper_id"),
        ),
        (
            "ExpectColumnValueLengthsToBeBetween",
            "summary",
            gx.expectations.ExpectColumnValueLengthsToBeBetween(column="summary", min_value=30),
        ),
    ]
    expectation_results = [
        _expectation_result(name, column, expectation, batch)
        for name, column, expectation in checks
    ]

    freshness = build_freshness_report(df, settings, settings.paths.freshness_report)
    all_expectations_pass = all(item["success"] for item in expectation_results)
    result = {
        "success": bool(all_expectations_pass and freshness["is_fresh"]),
        "stage": _json_safe(report_name),
        "expectations": expectation_results,
        "freshness": _json_safe(freshness),
    }
    write_json(_quality_report_path(settings, report_name), result)
    return result
