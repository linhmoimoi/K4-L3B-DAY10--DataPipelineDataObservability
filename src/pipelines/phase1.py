from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from core.config import Settings, load_settings
from core.utils import read_json, write_json
from evaluation.metrics import evaluate_pipeline
from evaluation.testset import assess_test_set_coverage, build_test_set
from ingestion.cleaning import build_clean_dataframe
from ingestion.crossref import enrich_missing_categories, fetch_source_records, load_raw_records
from observability.quality import run_data_quality_checks
from observability.reporting import generate_phase1_report
from retrieval.index import LocalEmbeddingIndex


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_or_fetch_records(settings: Settings) -> tuple[list[Any], str]:
    raw_records_path = Path(settings.paths.raw_records_json)
    raw_response_path = Path(settings.paths.raw_api_response)
    refresh_source = bool(getattr(settings, "refresh_source", False))

    if not refresh_source and raw_records_path.is_file() and raw_response_path.is_file():
        try:
            records = load_raw_records(raw_records_path)
        except (OSError, ValueError, TypeError):
            records = []
        if records:
            return records, "Crossref saved records snapshot"

    # fetch_source_records uses the configured Crossref query and transparently
    # falls back to raw_api_response when the network or API is unavailable.
    records = fetch_source_records(settings)
    missing_artifacts = [
        path.as_posix()
        for path in (raw_response_path, raw_records_path)
        if not path.is_file()
    ]
    if missing_artifacts:
        raise RuntimeError(
            "Ingestion did not produce the required raw artifacts: " + ", ".join(missing_artifacts)
        )
    if not records:
        raise RuntimeError("Ingestion returned no valid Crossref records.")
    return records, "Crossref fetch_source_records (REST API with saved-response fallback)"


def _can_reuse_test_set(test_set: Any, paper_ids: set[str]) -> bool:
    if not isinstance(test_set, list) or len(test_set) != 10:
        return False
    seen_ids: set[str] = set()
    for item in test_set:
        if not isinstance(item, dict):
            return False
        if not all(
            isinstance(item.get(key), str) and item[key].strip()
            for key in ("id", "question", "ground_truth")
        ):
            return False
        if item["id"] in seen_ids or item.get("question_type") not in {
            "summary",
            "authors",
            "date",
            "categories",
        }:
            return False
        seen_ids.add(item["id"])
        doc_ids = item.get("ground_truth_doc_ids")
        if (
            not isinstance(doc_ids, list)
            or len(doc_ids) != 1
            or not isinstance(doc_ids[0], str)
            or doc_ids[0] not in paper_ids
        ):
            return False
    return True


def _quality_failure_message(quality: dict[str, Any]) -> str:
    failed = [
        f"{item.get('name', 'expectation')}[{item.get('column')}]"
        for item in quality.get("expectations", [])
        if not item.get("success", False)
    ]
    freshness = quality.get("freshness", {})
    if not freshness.get("is_fresh", False):
        failed.append(
            "freshness SLA "
            f"(stale={freshness.get('stale_rows')}/{freshness.get('total_rows')}, "
            f"ratio={freshness.get('stale_ratio')})"
        )
    details = ", ".join(failed) if failed else "quality validation reported success=false"
    return f"Baseline quality gate failed: {details}. See {quality.get('stage', 'baseline')} quality artifact."


def run_phase1_pipeline(settings: Settings) -> dict[str, Any]:
    """Run the baseline data pipeline through evaluation and report generation."""
    paths = settings.paths
    report_path = Path(paths.baseline_report)
    status: dict[str, Any] = {
        "ingest": "not_run", "clean": "not_run", "quality_freshness": "not_run",
        "category_coverage": "not_run", "preflight": "not_run", "index": "not_run",
        "evaluation": "not_run", "report": "not_written",
    }
    source = "unknown"
    records: list[Any] = []
    clean_df = None
    quality: dict[str, Any] = {}
    freshness: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    artifact_paths: dict[str, str] = {}
    try:
        records, source = _load_or_fetch_records(settings)
        status["ingest"] = "passed"
        records, category_lineage = enrich_missing_categories(records, Path(paths.raw_records_json))
        current_category_keys = {record.paper_id.casefold() for record in records}
        category_success = sum(
            1 for key, item in category_lineage["records"].items()
            if key in current_category_keys and item.get("status") == "success"
        )

        clean_df = build_clean_dataframe(records, datetime.now(timezone.utc))
        for clean_path in (Path(paths.clean_csv), Path(paths.clean_json)):
            clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(paths.clean_csv, index=False, encoding="utf-8")
        clean_payload = json.loads(clean_df.to_json(orient="records", force_ascii=False, date_format="iso"))
        write_json(Path(paths.clean_json), clean_payload)
        status["clean"] = "passed"

        quality = run_data_quality_checks(clean_df, settings, "baseline")
        freshness = quality.get("freshness") if isinstance(quality.get("freshness"), dict) else {}
        status["quality_freshness"] = "passed" if quality.get("success") else "failed"
        if not quality.get("success", False):
            raise RuntimeError(_quality_failure_message(quality))

        coverage = assess_test_set_coverage(clean_df)
        status["category_coverage"] = "; ".join(
            f"{kind}={count}" for kind, count in coverage["valid_ground_truth_counts"].items()
        )
        status["preflight"] = coverage["status"]
        if coverage["status"] != "passed":
            missing = ", ".join(
                f"{kind}: {count} missing (have {coverage['valid_ground_truth_counts'][kind]}, need 2)"
                for kind, count in coverage["missing_per_type"].items() if count
            )
            raise RuntimeError(f"Evaluation preflight failed: insufficient ground truth coverage: {missing}.")

        test_set_path = Path(paths.eval_testset)
        test_set = build_test_set(clean_df, test_set_path)
        index = LocalEmbeddingIndex.build(clean_df, settings, embeddings_output_path=Path(paths.embeddings_json))
        if len(index.documents) != len(clean_df):
            raise RuntimeError(f"Baseline index contains {len(index.documents)} documents for {len(clean_df)} clean rows.")
        status["index"] = "passed"

        evaluation = evaluate_pipeline(settings, index, test_set_path, paths.baseline_metrics, paths.baseline_answers)
        metrics = evaluation.summary
        required_metrics = ("retrieval_hit_rate", "mean_token_f1", "judge_accuracy", "mean_judge_score")
        missing_metrics = [name for name in required_metrics if name not in metrics]
        if missing_metrics:
            raise RuntimeError("Baseline evaluation omitted required metrics: " + ", ".join(missing_metrics))
        status["evaluation"] = "passed"

        project_root = Path(paths.project_dir)
        artifact_paths = {
            "raw_api_response": _relative_path(Path(paths.raw_api_response), project_root),
            "raw_records": _relative_path(Path(paths.raw_records_json), project_root),
            "category_enrichment": _relative_path(Path(paths.raw_records_json).parent / "category_enrichment.json", project_root),
            "clean_csv": _relative_path(Path(paths.clean_csv), project_root),
            "clean_json": _relative_path(Path(paths.clean_json), project_root),
            "embeddings_manifest": _relative_path(Path(paths.embeddings_json), project_root),
            "test_set": _relative_path(test_set_path, project_root),
            "baseline_metrics": _relative_path(Path(paths.baseline_metrics), project_root),
            "baseline_answers": _relative_path(Path(paths.baseline_answers), project_root),
            "quality_report": _relative_path(Path(paths.baseline_quality_report), project_root),
            "freshness_report": _relative_path(Path(paths.freshness_report), project_root),
            "phase1_report": _relative_path(report_path, project_root),
        }
        source_summary = {
            "source": source, "raw_record_count": len(records), "clean_row_count": len(clean_df),
            "category_enrichment_successes": category_success,
            "index_configuration": {"backend": "chroma", "collection_name": index.collection_name,
                                    "embedding_model": settings.embedding_model, "indexed_documents": len(index.documents)},
            "artifact_paths": artifact_paths,
        }
        status["report"] = "passed"
        generate_phase1_report(report_path, source_summary, metrics, quality, freshness, status)
        return {"source_summary": source_summary, "clean_rows": len(clean_df), "quality": quality,
                "freshness": freshness, "baseline_metrics": metrics, "artifact_paths": artifact_paths,
                "test_set_questions": len(test_set)}
    except Exception as exc:
        status["evaluation"] = "evaluation_not_run" if status["evaluation"] == "not_run" else status["evaluation"]
        status["evaluation_reason"] = str(exc)
        if not artifact_paths:
            project_root = Path(paths.project_dir)
            artifact_paths = {"raw_api_response": _relative_path(Path(paths.raw_api_response), project_root),
                              "raw_records": _relative_path(Path(paths.raw_records_json), project_root),
                              "category_enrichment": _relative_path(Path(paths.raw_records_json).parent / "category_enrichment.json", project_root),
                              "clean_csv": _relative_path(Path(paths.clean_csv), project_root),
                              "clean_json": _relative_path(Path(paths.clean_json), project_root),
                              "test_set": _relative_path(Path(paths.eval_testset), project_root),
                              "baseline_metrics": _relative_path(Path(paths.baseline_metrics), project_root),
                              "baseline_answers": _relative_path(Path(paths.baseline_answers), project_root),
                              "phase1_report": _relative_path(report_path, project_root)}
        summary = {"source": source, "raw_record_count": len(records),
                   "clean_row_count": len(clean_df) if clean_df is not None else 0,
                   "index_configuration": {}, "artifact_paths": artifact_paths,
                   "category_coverage_counts": coverage.get("valid_ground_truth_counts", {})}
        try:
            status["report"] = "written_failure_report"
            generate_phase1_report(report_path, summary, None, quality, freshness, status)
        except Exception:
            status["report"] = "failed_to_write"
        raise


def main() -> None:
    settings = load_settings()
    try:
        result = run_phase1_pipeline(settings)
    except Exception as exc:
        print(f"Phase 1 baseline pipeline failed: {exc}")
        raise

    print("Phase 1 baseline pipeline completed.")
    print(f"Source: {result['source_summary']['source']}")
    print(f"Clean rows: {result['clean_rows']}")
    print(f"Freshness: {result['freshness']['is_fresh']}")
    for metric in ("retrieval_hit_rate", "mean_token_f1", "judge_accuracy", "mean_judge_score"):
        print(f"{metric}: {result['baseline_metrics'][metric]}")
    print(f"Report: {result['artifact_paths']['phase1_report']}")
