from __future__ import annotations

from datetime import datetime, time, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Settings, load_settings
from core.utils import read_json, write_json, write_text
from evaluation.metrics import EvaluationBundle, evaluate_pipeline
from ingestion.cleaning import build_clean_dataframe
from ingestion.corruption import corrupt_clean_dataframe
from ingestion.crossref import load_raw_records
from observability.quality import evaluate_freshness_sla, run_data_quality_checks
from retrieval.index import LocalEmbeddingIndex


_METRIC_NAMES = (
    "retrieval_hit_rate",
    "mean_token_f1",
    "judge_accuracy",
    "mean_judge_score",
)
_CORRUPTION_TYPES = (
    "drop_latest_records",
    "blank_summary",
    "inject_noise",
    "truncate_title",
    "stale_date",
    "duplicate_rows",
)


class PipelineStageError(RuntimeError):
    """An error annotated with the pipeline stage that could not complete."""

    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _save_dataframe(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    serializable = json.loads(df.to_json(orient="records", force_ascii=False, date_format="iso"))
    write_json(json_path, serializable)


def _required_metrics(metrics: Any, label: str) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise ValueError(f"{label} metrics artifact must contain a JSON object.")
    missing = [name for name in _METRIC_NAMES if name not in metrics]
    if missing:
        raise ValueError(f"{label} metrics are missing: {', '.join(missing)}.")
    invalid = [
        name
        for name in _METRIC_NAMES
        if isinstance(metrics[name], bool)
        or not isinstance(metrics[name], (int, float))
        or not math.isfinite(float(metrics[name]))
    ]
    if invalid:
        raise ValueError(f"{label} metrics are not finite numeric values: {', '.join(invalid)}.")
    return metrics


def _validate_test_set(test_set: Any, clean_df: pd.DataFrame) -> list[dict[str, Any]]:
    if not isinstance(test_set, list) or len(test_set) != 10:
        raise ValueError("The existing baseline evaluation set must be a JSON array of exactly ten questions.")
    paper_ids = set(clean_df["paper_id"].astype(str)) if "paper_id" in clean_df.columns else set()
    seen_ids: set[str] = set()
    allowed_types = {"summary", "authors", "date", "categories"}
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(test_set, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation question {index} is not a JSON object.")
        for field in ("id", "question", "ground_truth"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Evaluation question {index} has an empty or invalid {field}.")
        if item["id"] in seen_ids:
            raise ValueError(f"Evaluation set contains duplicate ID {item['id']}.")
        seen_ids.add(item["id"])
        if item.get("question_type") not in allowed_types:
            raise ValueError(f"Evaluation question {item['id']} has an unsupported question_type.")
        doc_ids = item.get("ground_truth_doc_ids")
        if (
            not isinstance(doc_ids, list)
            or len(doc_ids) != 1
            or not isinstance(doc_ids[0], str)
            or doc_ids[0] not in paper_ids
        ):
            raise ValueError(f"Evaluation question {item['id']} does not reference one paper in the baseline dataset.")
        validated.append(item)
    return validated


def _judge_details(metrics: dict[str, Any], answers: list[dict[str, Any]] | None) -> dict[str, Any]:
    judges = [item.get("judge", {}) for item in (answers or []) if isinstance(item, dict)]
    fallback_reasons = sorted(
        {
            judge.get("reasoning", "")
            for judge in judges
            if isinstance(judge, dict) and "Fallback heuristic judge" in judge.get("reasoning", "")
        }
    )
    if judges:
        fallback_answer_count = sum(
            1
            for judge in judges
            if isinstance(judge, dict) and "Fallback heuristic judge" in judge.get("reasoning", "")
        )
        if fallback_answer_count == 0:
            mode = "LLM"
        elif fallback_answer_count == len(judges):
            mode = "heuristic fallback"
        else:
            mode = f"mixed ({fallback_answer_count}/{len(judges)} heuristic fallback)"
    else:
        recorded_mode = metrics.get("judge_mode")
        mode = "unknown" if not recorded_mode else str(recorded_mode).replace("_", " ")
    reason = metrics.get("judge_fallback_reason") or "; ".join(fallback_reasons)
    if not reason and mode == "unknown":
        reason = "The existing baseline metrics do not record judge mode or fallback reason."
    return {
        "mode": mode,
        "fallback_reason": str(reason) if reason else "No heuristic fallback was recorded.",
    }


def _quality_label(quality: dict[str, Any]) -> str:
    expectations = quality.get("expectations", [])
    passed = sum(1 for item in expectations if item.get("success", False))
    total = len(expectations)
    status = "PASS" if quality.get("success", False) else "FAIL"
    return f"{status} ({passed}/{total} expectations)"


def _freshness_label(freshness: dict[str, Any]) -> str:
    status = "FRESH" if freshness.get("is_fresh") else "STALE"
    stale = freshness.get("stale_rows", "?")
    total = freshness.get("total_rows", "?")
    ratio = freshness.get("stale_ratio")
    ratio_text = f"{float(ratio):.1%}" if isinstance(ratio, (int, float)) else "unknown"
    return f"{status} ({stale}/{total} stale; {ratio_text})"


def _expectation_map(quality: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(item.get("name", "expectation")), str(item.get("column") or "")): item
        for item in quality.get("expectations", [])
    }


def _write_comparison_report(
    report_path: Path,
    baseline_metrics: dict[str, Any],
    corrupted_metrics: dict[str, Any],
    repaired_metrics: dict[str, Any],
    baseline_quality: dict[str, Any],
    corrupted_quality: dict[str, Any],
    repaired_quality: dict[str, Any],
    freshness: dict[str, dict[str, Any]],
    corruption_log: dict[str, Any],
    evaluator_modes: dict[str, dict[str, Any]],
    artifact_paths: dict[str, str],
    evaluation_set_hash: str,
) -> None:
    lines = [
        "# Corruption and Repair Comparison",
        "",
        "## Evaluation setup",
        "",
        f"- Shared evaluation set: `{artifact_paths['evaluation_set']}` (SHA-256 `{evaluation_set_hash[:16]}`)",
        "- Baseline, corrupted, and repaired states used the same evaluation set.",
        f"- Rows before corruption: {corruption_log['input_rows']}",
        f"- Rows after corruption and duplicate injection: {corruption_log['output_rows']}",
        f"- Corruption log: `{artifact_paths['corruption_log']}`",
        "",
        "## Baseline / Corrupted / Repaired",
        "",
        "| Metric/signal | Baseline | Corrupted | Repaired |",
        "|---|---:|---:|---:|",
    ]
    for name in _METRIC_NAMES:
        lines.append(
            f"| `{name}` | {float(baseline_metrics[name]):.4f} | "
            f"{float(corrupted_metrics[name]):.4f} | {float(repaired_metrics[name]):.4f} |"
        )
    lines.extend(
        [
            f"| Quality checks | {_quality_label(baseline_quality)} | {_quality_label(corrupted_quality)} | {_quality_label(repaired_quality)} |",
            f"| Freshness status | {_freshness_label(freshness['baseline'])} | {_freshness_label(freshness['corrupted'])} | {_freshness_label(freshness['repaired'])} |",
            "",
            "## Corruptions applied",
            "",
            "| Corruption | Affected rows |",
            "|---|---:|",
        ]
    )
    corruption_by_type = {item.get("type"): item for item in corruption_log.get("corruptions", [])}
    for corruption_type in _CORRUPTION_TYPES:
        entry = corruption_by_type.get(corruption_type, {})
        lines.append(f"| `{corruption_type}` | {entry.get('affected_rows', 0)} |")
    lines.extend(["", "## Quality expectations affected", ""])
    baseline_expectations = _expectation_map(baseline_quality)
    corrupted_expectations = _expectation_map(corrupted_quality)
    repaired_expectations = _expectation_map(repaired_quality)
    expectation_keys = sorted(set(baseline_expectations) | set(corrupted_expectations) | set(repaired_expectations))
    affected_expectations: list[str] = []
    for key in expectation_keys:
        b = baseline_expectations.get(key, {}).get("status", "missing")
        c = corrupted_expectations.get(key, {}).get("status", "missing")
        r = repaired_expectations.get(key, {}).get("status", "missing")
        if len({b, c, r}) > 1 or c == "failed" or r == "failed":
            label = key[0] + (f" (`{key[1]}`)" if key[1] else "")
            affected_expectations.append(f"- {label}: baseline **{b}** → corrupted **{c}** → repaired **{r}**")
    lines.extend(affected_expectations or ["- No expectation status changed across the three states."])
    lines.extend(["", "## Degradation and recovery", "", "For these metrics, positive degradation means Baseline − Corrupted; positive recovery means Repaired − Corrupted.", "", "| Metric | Degradation | Recovery | Repaired vs Baseline |", "|---|---:|---:|---:|"])
    not_fully_restored: list[str] = []
    for name in _METRIC_NAMES:
        baseline_value = float(baseline_metrics[name])
        corrupted_value = float(corrupted_metrics[name])
        repaired_value = float(repaired_metrics[name])
        if repaired_value + 1e-12 < baseline_value:
            not_fully_restored.append(f"`{name}` remains {baseline_value - repaired_value:.4f} below Baseline")
        lines.append(
            f"| `{name}` | {baseline_value - corrupted_value:+.4f} | "
            f"{repaired_value - corrupted_value:+.4f} | {repaired_value - baseline_value:+.4f} |"
        )
    lines.append("")
    if not_fully_restored:
        lines.append("Metrics not fully restored: " + "; ".join(not_fully_restored) + ".")
    elif all(float(repaired_metrics[name]) >= float(baseline_metrics[name]) for name in _METRIC_NAMES):
        lines.append("All listed evaluation metrics match or exceed Baseline in this run.")
    else:
        lines.append("No overall recovery conclusion is drawn from the available metrics.")
    if not repaired_quality.get("success", False):
        lines.append("Repaired quality checks still fail; data quality is not fully restored.")
    if not freshness["repaired"].get("is_fresh", False):
        lines.append("Repaired freshness SLA still fails; freshness is not fully restored.")

    lines.extend(["", "## Evaluator", ""])
    for state in ("baseline", "corrupted", "repaired"):
        details = evaluator_modes[state]
        lines.append(f"- {state.title()}: {details['mode']}; reason: {details['fallback_reason']}")

    lines.extend(["", "## Limitations", ""])
    if any("fallback" in details["mode"].lower() for details in evaluator_modes.values()):
        lines.append(
            "- Judge accuracy and mean judge score include heuristic fallback results; "
            "they are not exclusively LLM-as-judge measurements."
        )
    lines.append("- Results are measured on the existing fixed evaluation set shown above.")
    if not_fully_restored:
        lines.append("- Metrics below Baseline after repair: " + "; ".join(not_fully_restored) + ".")
    if not repaired_quality.get("success", False) or not freshness["repaired"].get("is_fresh", False):
        lines.append("- Repaired quality or freshness still has failed checks; inspect the stage reports for details.")
    elif not not_fully_restored:
        lines.append("- On this run, all listed metrics and repaired quality/freshness checks match the baseline status.")

    lines.extend(["", "## Artifacts", ""])
    for key, path in artifact_paths.items():
        lines.append(f"- {key}: `{path}`")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(report_path, "\n".join(lines))


def repair_from_raw_snapshot(settings: Settings) -> pd.DataFrame:
    """Rebuild and persist a repaired dataframe from the immutable raw records file."""
    raw_path = Path(settings.paths.raw_records_json)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw records snapshot does not exist: {raw_path}")
    try:
        records = load_raw_records(raw_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read raw records snapshot {raw_path}: {type(exc).__name__}: {exc}") from exc
    if not records:
        raise ValueError(f"Raw records snapshot contains no usable records: {raw_path}")

    today_utc = datetime.now(timezone.utc).date()
    run_date = datetime.combine(today_utc, time.min, tzinfo=timezone.utc)
    repaired_df = build_clean_dataframe(records, run_date)
    if repaired_df.empty:
        raise ValueError("Cleaning the raw records snapshot produced no repaired rows.")
    _save_dataframe(
        repaired_df,
        Path(settings.paths.repaired_clean_csv),
        Path(settings.paths.repaired_clean_json),
    )
    return repaired_df


def run_corruption_flow_pipeline(settings: Settings) -> dict[str, Any]:
    """Compare baseline metrics with corrupted evaluation and raw-snapshot repair."""
    stage = "baseline preflight"
    try:
        paths = settings.paths
        required_paths = {
            "baseline metrics": Path(paths.baseline_metrics),
            "baseline clean JSON": Path(paths.clean_json),
            "baseline evaluation set": Path(paths.eval_testset),
            "raw records snapshot": Path(paths.raw_records_json),
        }
        missing = [f"{label}: {path.as_posix()}" for label, path in required_paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing required artifacts: " + "; ".join(missing))

        baseline_metrics = _required_metrics(read_json(Path(paths.baseline_metrics)), "Baseline")
        clean_df = pd.read_json(paths.clean_json)
        if clean_df.empty:
            raise ValueError("Baseline clean dataframe is empty.")
        if "paper_id" not in clean_df.columns:
            raise ValueError("Baseline clean dataframe is missing paper_id.")
        test_set_path = Path(paths.eval_testset)
        evaluation_set = _validate_test_set(read_json(test_set_path), clean_df)
        evaluation_set_hash = _sha256(test_set_path)
        raw_records_path = Path(paths.raw_records_json)
        raw_response_path = Path(paths.raw_api_response)
        raw_records_hash = _sha256(raw_records_path)
        raw_response_hash = _sha256(raw_response_path) if raw_response_path.is_file() else None
        raw_records = load_raw_records(raw_records_path)
        if not raw_records:
            raise ValueError("The raw records snapshot contains no usable records.")
        collection_names = {
            settings.baseline_collection_name,
            settings.corrupted_collection_name,
            settings.repaired_collection_name,
        }
        if len(collection_names) != 3:
            raise ValueError("Baseline, corrupted, and repaired Chroma collections must have distinct names.")

        baseline_answers: list[dict[str, Any]] | None = None
        baseline_answers_path = Path(paths.baseline_answers)
        if baseline_answers_path.is_file():
            loaded_answers = read_json(baseline_answers_path)
            if not isinstance(loaded_answers, list):
                raise ValueError("Baseline answers artifact must contain a JSON array.")
            baseline_answers = loaded_answers
            expected_ids = {item["id"] for item in evaluation_set}
            answer_ids = {item.get("id") for item in baseline_answers if isinstance(item, dict)}
            if answer_ids and answer_ids != expected_ids:
                raise ValueError("Baseline answers do not correspond to the existing baseline evaluation set.")
        if baseline_metrics.get("samples") not in (None, len(evaluation_set)):
            raise ValueError("Baseline metric sample count does not match the existing evaluation set.")

        baseline_quality = run_data_quality_checks(clean_df, settings, "baseline")
        baseline_freshness = evaluate_freshness_sla(clean_df, settings)
        if not baseline_quality.get("success", False):
            raise ValueError("Baseline quality gate failed; refusing to compare against an invalid baseline.")

        stage = "corruption"
        corrupted_df = corrupt_clean_dataframe(clean_df, paths.corruption_log)
        _save_dataframe(
            corrupted_df,
            Path(paths.corrupted_clean_csv),
            Path(paths.corrupted_clean_json),
        )
        corruption_log = read_json(Path(paths.corruption_log))
        if not isinstance(corruption_log, dict):
            raise ValueError("Corruption log is not a JSON object.")
        logged_types = {item.get("type") for item in corruption_log.get("corruptions", []) if isinstance(item, dict)}
        if not set(_CORRUPTION_TYPES).issubset(logged_types):
            raise ValueError("Corruption log does not contain all six required corruption types.")
        if corruption_log.get("input_rows") != len(clean_df) or corruption_log.get("output_rows") != len(corrupted_df):
            raise ValueError("Corruption log row counts do not match the dataframe artifacts.")

        stage = "corrupted quality and freshness"
        corrupted_quality = run_data_quality_checks(corrupted_df, settings, "corrupted")
        corrupted_freshness = evaluate_freshness_sla(corrupted_df, settings)

        stage = "corrupted index"
        corrupted_index = LocalEmbeddingIndex.build(
            corrupted_df,
            settings,
            embeddings_output_path=Path(paths.corrupted_embeddings_json),
        )
        if corrupted_index.collection_name != settings.corrupted_collection_name:
            raise RuntimeError(f"Unexpected corrupted collection name: {corrupted_index.collection_name}")
        if len(corrupted_index.documents) != len(corrupted_df):
            raise RuntimeError("Corrupted Chroma index does not contain every corrupted dataframe row.")

        stage = "corrupted evaluation"
        corrupted_bundle: EvaluationBundle = evaluate_pipeline(
            settings,
            corrupted_index,
            test_set_path,
            paths.corrupted_metrics,
            paths.corrupted_answers,
        )
        corrupted_metrics = _required_metrics(corrupted_bundle.summary, "Corrupted")
        if _sha256(test_set_path) != evaluation_set_hash:
            raise RuntimeError("The shared evaluation set changed during corrupted evaluation.")

        stage = "repair from raw snapshot"
        repaired_df = repair_from_raw_snapshot(settings)

        stage = "repaired quality and freshness"
        repaired_quality = run_data_quality_checks(repaired_df, settings, "repaired")
        repaired_freshness = evaluate_freshness_sla(repaired_df, settings)

        stage = "repaired index"
        repaired_index = LocalEmbeddingIndex.build(
            repaired_df,
            settings,
            embeddings_output_path=Path(paths.repaired_embeddings_json),
        )
        if repaired_index.collection_name != settings.repaired_collection_name:
            raise RuntimeError(f"Unexpected repaired collection name: {repaired_index.collection_name}")
        if len(repaired_index.documents) != len(repaired_df):
            raise RuntimeError("Repaired Chroma index does not contain every repaired dataframe row.")

        stage = "repaired evaluation"
        repaired_bundle: EvaluationBundle = evaluate_pipeline(
            settings,
            repaired_index,
            test_set_path,
            paths.repaired_metrics,
            paths.repaired_answers,
        )
        repaired_metrics = _required_metrics(repaired_bundle.summary, "Repaired")
        if _sha256(test_set_path) != evaluation_set_hash:
            raise RuntimeError("The shared evaluation set changed during repaired evaluation.")
        if _sha256(raw_records_path) != raw_records_hash:
            raise RuntimeError("Raw records snapshot changed during the corruption flow.")
        if raw_response_hash is not None and _sha256(raw_response_path) != raw_response_hash:
            raise RuntimeError("Raw API response snapshot changed during the corruption flow.")

        stage = "report generation"
        root = Path(paths.project_dir)
        artifact_paths = {
            "baseline_metrics": _relative_path(Path(paths.baseline_metrics), root),
            "baseline_answers": _relative_path(baseline_answers_path, root) if baseline_answers_path.is_file() else "not available",
            "corruption_log": _relative_path(Path(paths.corruption_log), root),
            "corrupted_clean_csv": _relative_path(Path(paths.corrupted_clean_csv), root),
            "corrupted_clean_json": _relative_path(Path(paths.corrupted_clean_json), root),
            "corrupted_embeddings_manifest": _relative_path(Path(paths.corrupted_embeddings_json), root),
            "corrupted_metrics": _relative_path(Path(paths.corrupted_metrics), root),
            "corrupted_answers": _relative_path(Path(paths.corrupted_answers), root),
            "repaired_clean_csv": _relative_path(Path(paths.repaired_clean_csv), root),
            "repaired_clean_json": _relative_path(Path(paths.repaired_clean_json), root),
            "repaired_embeddings_manifest": _relative_path(Path(paths.repaired_embeddings_json), root),
            "repaired_metrics": _relative_path(Path(paths.repaired_metrics), root),
            "repaired_answers": _relative_path(Path(paths.repaired_answers), root),
            "evaluation_set": _relative_path(test_set_path, root),
            "comparison_report": _relative_path(Path(paths.comparison_report), root),
        }
        evaluator_modes = {
            "baseline": _judge_details(baseline_metrics, baseline_answers),
            "corrupted": _judge_details(corrupted_metrics, corrupted_bundle.answers),
            "repaired": _judge_details(repaired_metrics, repaired_bundle.answers),
        }
        _write_comparison_report(
            Path(paths.comparison_report),
            baseline_metrics,
            corrupted_metrics,
            repaired_metrics,
            baseline_quality,
            corrupted_quality,
            repaired_quality,
            {
                "baseline": baseline_freshness,
                "corrupted": corrupted_freshness,
                "repaired": repaired_freshness,
            },
            corruption_log,
            evaluator_modes,
            artifact_paths,
            evaluation_set_hash,
        )

        return {
            "baseline_metrics": baseline_metrics,
            "corrupted_metrics": corrupted_metrics,
            "repaired_metrics": repaired_metrics,
            "baseline_quality": baseline_quality,
            "corrupted_quality": corrupted_quality,
            "repaired_quality": repaired_quality,
            "freshness": {
                "baseline": baseline_freshness,
                "corrupted": corrupted_freshness,
                "repaired": repaired_freshness,
            },
            "corruption_log": corruption_log,
            "rows": {
                "baseline": len(clean_df),
                "corrupted": len(corrupted_df),
                "repaired": len(repaired_df),
            },
            "evaluator_modes": evaluator_modes,
            "artifact_paths": artifact_paths,
            "evaluation_set_sha256": evaluation_set_hash,
            "raw_snapshots_unchanged": True,
        }
    except PipelineStageError:
        raise
    except Exception as exc:
        raise PipelineStageError(stage, f"{type(exc).__name__}: {exc}") from exc


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return "n/a"


def _print_comparison(result: dict[str, Any]) -> None:
    print(f"{'Metric':<22} | {'Baseline':>12} | {'Corrupted':>12} | {'Repaired':>12}")
    print("-" * 67)
    for name in _METRIC_NAMES:
        print(
            f"{name:<22} | {_format_metric(result['baseline_metrics'][name]):>12} | "
            f"{_format_metric(result['corrupted_metrics'][name]):>12} | "
            f"{_format_metric(result['repaired_metrics'][name]):>12}"
        )
    for label, field in (("Quality checks", "quality"), ("Freshness status", "freshness")):
        if field == "quality":
            values = [_quality_label(result[f"{state}_quality"]) for state in ("baseline", "corrupted", "repaired")]
        else:
            values = [_freshness_label(result["freshness"][state]) for state in ("baseline", "corrupted", "repaired")]
        print(f"{label:<22} | {values[0]:>12} | {values[1]:>12} | {values[2]:>12}")


def main() -> None:
    settings = load_settings()
    try:
        result = run_corruption_flow_pipeline(settings)
    except PipelineStageError as exc:
        print(f"Pipeline failed at stage '{exc.stage}': {exc.reason}")
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"Pipeline failed at stage 'initialization': {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    _print_comparison(result)
    print(f"Comparison report: {result['artifact_paths']['comparison_report']}")
