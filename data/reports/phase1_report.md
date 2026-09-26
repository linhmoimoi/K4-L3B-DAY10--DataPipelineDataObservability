# Phase 1 Baseline Report

## Source and data

- Source: Crossref saved records snapshot
- Raw records: 24
- Clean rows: 24
- Ingest status: passed
- Clean status: passed

## Data quality

- Gate status: passed
- ExpectTableRowCountToBeBetween: passed
- ExpectColumnValuesToNotBeNull (`paper_id`): passed
- ExpectColumnValuesToNotBeNull (`title`): passed
- ExpectColumnValuesToNotBeNull (`text_for_embedding`): passed
- ExpectColumnValuesToBeUnique (`paper_id`): passed
- ExpectColumnValueLengthsToBeBetween (`summary`): passed
- Freshness: fresh
- Stale rows: 0 / 24
- Stale ratio: 0.0%
- Freshness threshold: 180 days
- Quality/freshness status: passed
- Category coverage: summary=24; authors=24; date=24; categories=21
- Preflight status: passed
- Index status: passed
- Evaluation status: passed
- Report status: passed

## Baseline index

- Backend: chroma
- Collection: papers-baseline
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Indexed documents: 24

## Evaluation metrics

- retrieval_hit_rate: 1.0
- mean_token_f1: 0.39045783373362836
- judge_accuracy: 0.9
- mean_judge_score: 4.4
- Judge mode: llm

## Artifacts

- raw_api_response: `data/raw/crossref_response.json`
- raw_records: `data/raw/crossref_records.json`
- category_enrichment: `data/raw/category_enrichment.json`
- clean_csv: `data/clean/papers_clean.csv`
- clean_json: `data/clean/papers_clean.json`
- embeddings_manifest: `data/embeddings/papers_embeddings.json`
- test_set: `data/eval/test_set.json`
- baseline_metrics: `data/results/baseline_metrics.json`
- baseline_answers: `data/results/baseline_answers.json`
- quality_report: `data/quality/baseline_quality_report.json`
- freshness_report: `data/quality/freshness_report.json`
- phase1_report: `data/reports/phase1_report.md`
