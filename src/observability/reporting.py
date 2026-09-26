from __future__ import annotations

from pathlib import Path
from typing import Any

from core.utils import write_text


def generate_phase1_report(
    report_path,
    source_summary: dict[str, Any],
    metrics: dict[str, Any] | None,
    quality: dict[str, Any],
    freshness: dict[str, Any],
    run_status: dict[str, Any] | None = None,
) -> None:
    """Write a baseline report using results produced by this pipeline run."""
    index_config = source_summary.get("index_configuration", {})
    artifact_paths = source_summary.get("artifact_paths", {})
    metrics = metrics or {}
    run_status = run_status or {}
    freshness_ratio = freshness.get("stale_ratio")
    if isinstance(freshness_ratio, (int, float)):
        freshness_display = f"{freshness_ratio:.1%}"
    else:
        freshness_display = "unavailable"

    lines = [
        "# Phase 1 Baseline Report",
        "",
        "## Source and data",
        "",
        f"- Source: {source_summary.get('source', 'unknown')}",
        f"- Raw records: {source_summary.get('raw_record_count', 'unknown')}",
        f"- Clean rows: {source_summary.get('clean_row_count', 'unknown')}",
        f"- Ingest status: {run_status.get('ingest', 'unknown')}",
        f"- Clean status: {run_status.get('clean', 'unknown')}",
        "",
        "## Data quality",
        "",
        f"- Gate status: {'passed' if quality.get('success') else 'failed'}",
    ]
    for expectation in quality.get("expectations", []):
        name = expectation.get("name", "expectation")
        column = expectation.get("column")
        label = f"{name} (`{column}`)" if column else name
        lines.append(f"- {label}: {expectation.get('status', 'unknown')}")
    lines.extend(
        [
            f"- Freshness: {'fresh' if freshness.get('is_fresh') else 'stale'}",
            f"- Stale rows: {freshness.get('stale_rows', 'unknown')} / {freshness.get('total_rows', 'unknown')}",
            f"- Stale ratio: {freshness_display}",
            f"- Freshness threshold: {freshness.get('threshold_days', 'unknown')} days",
            f"- Quality/freshness status: {run_status.get('quality_freshness', quality.get('success', 'unknown'))}",
            f"- Category coverage: {run_status.get('category_coverage', 'unknown')}",
            f"- Preflight status: {run_status.get('preflight', 'unknown')}",
            f"- Index status: {run_status.get('index', 'unknown')}",
            f"- Evaluation status: {run_status.get('evaluation', 'unknown')}",
            f"- Report status: {run_status.get('report', 'in_progress')}",
            "",
            "## Baseline index",
            "",
            f"- Backend: {index_config.get('backend', 'unknown')}",
            f"- Collection: {index_config.get('collection_name', 'unknown')}",
            f"- Embedding model: {index_config.get('embedding_model', 'unknown')}",
            f"- Indexed documents: {index_config.get('indexed_documents', 'unknown')}",
            "",
            "## Evaluation metrics",
            "",
        ]
    )
    if run_status.get("evaluation") != "passed":
        lines.append("- evaluation_not_run")
        if run_status.get("evaluation_reason"):
            lines.append(f"- Reason: {run_status['evaluation_reason']}")
        lines.append("- No new baseline metrics were produced in this run.")
    for metric_name in ("retrieval_hit_rate", "mean_token_f1", "judge_accuracy", "mean_judge_score"):
        if metric_name in metrics:
            lines.append(f"- {metric_name}: {metrics[metric_name]}")
    if metrics.get("judge_mode"):
        lines.append(f"- Judge mode: {metrics['judge_mode']}")
        if metrics.get("judge_fallback_reason"):
            lines.append(f"- Judge fallback reason: {metrics['judge_fallback_reason']}")
        if metrics.get("judge_fallback_affected_metrics"):
            lines.append("- Metrics affected by heuristic fallback: " + ", ".join(metrics["judge_fallback_affected_metrics"]))
    lines.extend(["", "## Artifacts", ""])
    for name, path in artifact_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    write_text(Path(report_path), "\n".join(lines))


def generate_corruption_report(
    report_path,
    baseline_metrics: dict[str, Any],
    corrupted_metrics: dict[str, Any],
    repaired_metrics: dict[str, Any],
    corrupted_quality: dict[str, Any],
    repaired_quality: dict[str, Any],
    corrupted_freshness: dict[str, Any],
    repaired_freshness: dict[str, Any],
) -> None:
    """TODO(student): viet markdown report so sanh baseline/corrupted/repaired."""
    raise NotImplementedError("Student task: implement corruption comparison report.")
