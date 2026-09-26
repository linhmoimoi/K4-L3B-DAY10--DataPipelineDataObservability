from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


_DROP_FRACTION = 0.20
_SUMMARY_FRACTION = 0.20
_NOISE_FRACTION = 0.20
_TITLE_FRACTION = 0.20
_STALE_FRACTION = 0.20
_DUPLICATE_FRACTION = 0.10
_NOISE_TEXT = "###@@@%%%^^^&&&***"


def _is_missing(value: Any) -> bool:
    try:
        missing = pd.isna(value)
        return bool(missing) if isinstance(missing, (bool,)) or type(missing).__name__ == "bool_" else False
    except (TypeError, ValueError):
        return False


def _text(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _log_value(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return _log_value(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [_log_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _log_value(item) for key, item in value.items()}
    return str(value)


def _ceil_fraction(total: int, fraction: float) -> int:
    if total <= 0:
        return 0
    return min(total, math.ceil(total * fraction))


def _stable_positions(df: pd.DataFrame) -> list[int]:
    """Return deterministic row positions, using paper_id and title as keys."""
    positions = list(range(len(df)))
    paper_ids = df["paper_id"].tolist() if "paper_id" in df.columns else [""] * len(df)
    titles = df["title"].tolist() if "title" in df.columns else [""] * len(df)
    return sorted(
        positions,
        key=lambda position: (
            _text(paper_ids[position]).casefold(),
            _text(paper_ids[position]),
            _text(titles[position]).casefold(),
            _text(titles[position]),
            position,
        ),
    )


def _rebuild_embedding_text(df: pd.DataFrame, position: int) -> None:
    if "summary_chars" in df.columns:
        df.at[position, "summary_chars"] = len(_text(df.at[position, "summary"]))

    if "text_for_embedding" not in df.columns:
        return

    fields = (
        ("Title", "title"),
        ("Authors", "authors_joined"),
        ("Published", "published"),
        ("Categories", "categories_joined"),
        ("Summary", "summary"),
    )
    sections: list[str] = []
    for label, column in fields:
        value = _text(df.at[position, column]) if column in df.columns else ""
        sections.append(f"{label}: {value}" if value else f"{label}:")
    df.at[position, "text_for_embedding"] = "\n".join(sections)


def _date_key(value: Any) -> pd.Timestamp | pd.NaT:
    if _is_missing(value):
        return pd.NaT
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    except (OverflowError, TypeError, ValueError):
        return pd.NaT
    return pd.NaT if pd.isna(parsed) else parsed


def _format_published_like(original: Any, target_date: date) -> Any:
    if isinstance(original, pd.Timestamp):
        target = pd.Timestamp(target_date)
        return target.tz_localize(original.tz) if original.tz is not None else target
    if isinstance(original, datetime):
        return datetime.combine(target_date, time.min, tzinfo=original.tzinfo)
    if isinstance(original, date):
        return target_date
    return target_date.isoformat()


def _change_entry(paper_id: Any, **details: Any) -> dict[str, Any]:
    return {"paper_id": _text(paper_id), **details}


def corrupt_clean_dataframe(clean_df: pd.DataFrame, log_path) -> pd.DataFrame:
    """Return a corrupted copy and write a JSON log of each applied mutation."""
    if not isinstance(clean_df, pd.DataFrame):
        raise TypeError("corrupt_clean_dataframe requires a pandas DataFrame.")
    required_columns = {"paper_id", "title", "summary", "published"}
    missing_columns = sorted(required_columns - set(clean_df.columns))
    if missing_columns:
        raise ValueError(
            "Cannot corrupt the clean dataframe; missing required columns: "
            + ", ".join(missing_columns)
            + "."
        )

    original_rows = len(clean_df)
    corrupted = clean_df.copy(deep=True).reset_index(drop=True)
    reference_date = datetime.now(timezone.utc).date()
    corruption_log: list[dict[str, Any]] = []

    # 1. Drop the most recently published fifth of the input, breaking date ties
    # by ascending paper_id and retaining stable input order for any remaining tie.
    drop_count = _ceil_fraction(len(corrupted), _DROP_FRACTION)
    if drop_count:
        sort_frame = pd.DataFrame(
            {
                "_position": range(len(corrupted)),
                "_published": pd.to_datetime(corrupted["published"], errors="coerce", utc=True),
                "_paper_id": corrupted["paper_id"].map(_text).str.casefold(),
            }
        )
        ordered_positions = sort_frame.sort_values(
            ["_published", "_paper_id", "_position"],
            ascending=[False, True, True],
            kind="mergesort",
            na_position="last",
        )["_position"].tolist()
        dropped_positions = ordered_positions[:drop_count]
        drop_changes = [
            _change_entry(
                corrupted.at[position, "paper_id"],
                published=_log_value(corrupted.at[position, "published"]),
                reason="Removed as one of the most recently published records.",
            )
            for position in dropped_positions
        ]
        retained_positions = [position for position in range(len(corrupted)) if position not in set(dropped_positions)]
        corrupted = corrupted.iloc[retained_positions].reset_index(drop=True)
    else:
        drop_changes = []
    corruption_log.append(
        {
            "type": "drop_latest_records",
            "parameters": {
                "fraction": _DROP_FRACTION,
                "rounding": "ceil",
                "sort": "published descending, paper_id ascending, stable input order",
                "requested_drop_count": drop_count,
            },
            "affected_rows": len(drop_changes),
            "affected_paper_ids": [change["paper_id"] for change in drop_changes],
            "changes": drop_changes,
            **({"reason": "The input dataframe is empty."} if original_rows == 0 else {}),
        }
    )

    # Use one stable ordering for deterministic subset selection. Summary and
    # noise subsets are disjoint whenever the remaining rows allow it.
    stable_positions = _stable_positions(corrupted)

    # 2. Blank summaries.
    blank_candidates = [position for position in stable_positions if _text(corrupted.at[position, "summary"])]
    blank_count = _ceil_fraction(len(corrupted), _SUMMARY_FRACTION)
    blank_positions = blank_candidates[:blank_count]
    blank_changes: list[dict[str, Any]] = []
    for position in blank_positions:
        before = _text(corrupted.at[position, "summary"])
        corrupted.at[position, "summary"] = ""
        _rebuild_embedding_text(corrupted, position)
        blank_changes.append(_change_entry(corrupted.at[position, "paper_id"], before=before, after=""))
    corruption_log.append(
        {
            "type": "blank_summary",
            "parameters": {
                "fraction": _SUMMARY_FRACTION,
                "rounding": "ceil",
                "target_count": blank_count,
            },
            "affected_rows": len(blank_changes),
            "affected_paper_ids": [change["paper_id"] for change in blank_changes],
            "changes": blank_changes,
            **({"reason": "No non-empty summaries were available."} if not blank_changes else {}),
        }
    )

    # 3. Inject explicit, non-semantic noise into a separate deterministic subset.
    noise_count = _ceil_fraction(len(corrupted), _NOISE_FRACTION)
    blank_position_set = set(blank_positions)
    noise_candidates = [
        position
        for position in stable_positions
        if position not in blank_position_set and _text(corrupted.at[position, "summary"])
    ]
    if len(noise_candidates) < noise_count:
        extra_candidates = [position for position in stable_positions if position not in noise_candidates]
        noise_candidates.extend(extra_candidates[: noise_count - len(noise_candidates)])
    noise_positions = noise_candidates[:noise_count]
    noise_changes: list[dict[str, Any]] = []
    for position in noise_positions:
        before = _text(corrupted.at[position, "summary"])
        after = f"{before} {_NOISE_TEXT}".strip()
        corrupted.at[position, "summary"] = after
        _rebuild_embedding_text(corrupted, position)
        noise_changes.append(
            _change_entry(
                corrupted.at[position, "paper_id"],
                noise=_NOISE_TEXT,
                before=before,
                after=after,
            )
        )
    corruption_log.append(
        {
            "type": "inject_noise",
            "parameters": {
                "fraction": _NOISE_FRACTION,
                "rounding": "ceil",
                "noise_string": _NOISE_TEXT,
                "target_count": noise_count,
                "avoids_blank_summary_rows_when_possible": True,
            },
            "affected_rows": len(noise_changes),
            "affected_paper_ids": [change["paper_id"] for change in noise_changes],
            "changes": noise_changes,
            **({"reason": "No rows were available for noise injection."} if not noise_changes else {}),
        }
    )

    # 4. Truncate a deterministic fifth of the eligible titles to seven chars.
    title_candidates = [position for position in stable_positions if len(_text(corrupted.at[position, "title"])) >= 8]
    title_count = _ceil_fraction(len(title_candidates), _TITLE_FRACTION)
    title_positions = title_candidates[:title_count]
    title_changes: list[dict[str, Any]] = []
    for position in title_positions:
        before = _text(corrupted.at[position, "title"])
        after = before[:7]
        corrupted.at[position, "title"] = after
        _rebuild_embedding_text(corrupted, position)
        title_changes.append(
            _change_entry(
                corrupted.at[position, "paper_id"],
                before=before,
                after=after,
                length_before=len(before),
                length_after=len(after),
            )
        )
    corruption_log.append(
        {
            "type": "truncate_title",
            "parameters": {
                "fraction_of_eligible_rows": _TITLE_FRACTION,
                "rounding": "ceil",
                "minimum_original_length": 8,
                "maximum_result_length": 7,
                "eligible_rows": len(title_candidates),
                "target_count": title_count,
            },
            "affected_rows": len(title_changes),
            "affected_paper_ids": [change["paper_id"] for change in title_changes],
            "changes": title_changes,
            **({"reason": "No titles were at least eight characters long."} if not title_changes else {}),
        }
    )

    # 5. Set a deterministic subset's publication dates to exactly one year
    # before the UTC reference date and update derived age/text columns.
    stale_count = _ceil_fraction(len(corrupted), _STALE_FRACTION)
    stale_candidates = [
        position
        for position in stable_positions
        if not pd.isna(_date_key(corrupted.at[position, "published"]))
    ]
    stale_positions = stale_candidates[:stale_count]
    stale_date = reference_date - timedelta(days=365)
    stale_changes: list[dict[str, Any]] = []
    for position in stale_positions:
        before_value = corrupted.at[position, "published"]
        before_date = _date_key(before_value)
        if pd.isna(before_date):
            continue
        after_value = _format_published_like(before_value, stale_date)
        corrupted.at[position, "published"] = after_value
        if "age_days" in corrupted.columns:
            corrupted.at[position, "age_days"] = (reference_date - stale_date).days
        _rebuild_embedding_text(corrupted, position)
        stale_changes.append(
            _change_entry(
                corrupted.at[position, "paper_id"],
                before=_log_value(before_value),
                after=_log_value(after_value),
                days_lagged=365,
            )
        )
    corruption_log.append(
        {
            "type": "stale_date",
            "parameters": {
                "fraction": _STALE_FRACTION,
                "rounding": "ceil",
                "reference_date_utc": reference_date.isoformat(),
                "days_lagged": 365,
                "target_date": stale_date.isoformat(),
                "target_count": stale_count,
                "valid_date_candidates": len(stale_candidates),
            },
            "affected_rows": len(stale_changes),
            "affected_paper_ids": [change["paper_id"] for change in stale_changes],
            "changes": stale_changes,
            **(
                {"reason": "No rows had a valid published date."}
                if not stale_changes
                else {}
            ),
        }
    )

    # 6. Append copies of a deterministic subset, preserving the original IDs.
    rows_before_duplicates = len(corrupted)
    duplicate_count = _ceil_fraction(rows_before_duplicates, _DUPLICATE_FRACTION)
    duplicate_positions = _stable_positions(corrupted)[:duplicate_count]
    duplicate_changes = [
        _change_entry(
            corrupted.at[position, "paper_id"],
            source_row_position=position,
            duplicate_copy_number=1,
        )
        for position in duplicate_positions
    ]
    if duplicate_positions:
        duplicates = corrupted.iloc[duplicate_positions].copy(deep=True)
        corrupted = pd.concat([corrupted, duplicates], ignore_index=True)
    corruption_log.append(
        {
            "type": "duplicate_rows",
            "parameters": {
                "fraction": _DUPLICATE_FRACTION,
                "rounding": "ceil",
                "target_duplicate_rows": duplicate_count,
                "rows_before_duplication": rows_before_duplicates,
                "rows_after_duplication": len(corrupted),
            },
            "affected_rows": len(duplicate_changes),
            "affected_paper_ids": [change["paper_id"] for change in duplicate_changes],
            "changes": duplicate_changes,
            **({"reason": "No rows were available to duplicate."} if not duplicate_changes else {}),
        }
    )

    log_payload = {
        "input_rows": int(original_rows),
        "output_rows": int(len(corrupted)),
        "corruptions": corruption_log,
    }
    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return corrupted
