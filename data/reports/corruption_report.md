# Corruption and Repair Comparison

## Evaluation setup

- Shared evaluation set: `data/eval/test_set.json` (SHA-256 `421411a437ddb57a`)
- Baseline, corrupted, and repaired states used the same evaluation set.
- Rows before corruption: 24
- Rows after corruption and duplicate injection: 21
- Corruption log: `data/results/corruption_log.json`

## Baseline / Corrupted / Repaired

| Metric/signal | Baseline | Corrupted | Repaired |
|---|---:|---:|---:|
| `retrieval_hit_rate` | 1.0000 | 0.6000 | 1.0000 |
| `mean_token_f1` | 1.0000 | 0.5095 | 1.0000 |
| `judge_accuracy` | 1.0000 | 0.5000 | 1.0000 |
| `mean_judge_score` | 5.0000 | 3.0000 | 5.0000 |
| Quality checks | PASS (6/6 expectations) | FAIL (4/6 expectations) | PASS (6/6 expectations) |
| Freshness status | FRESH (0/24 stale; 0.0%) | STALE (6/21 stale; 28.6%) | FRESH (0/24 stale; 0.0%) |

## Corruptions applied

| Corruption | Affected rows |
|---|---:|
| `drop_latest_records` | 5 |
| `blank_summary` | 4 |
| `inject_noise` | 4 |
| `truncate_title` | 4 |
| `stale_date` | 4 |
| `duplicate_rows` | 2 |

## Quality expectations affected

- ExpectColumnValueLengthsToBeBetween (`summary`): baseline **passed** → corrupted **failed** → repaired **passed**
- ExpectColumnValuesToBeUnique (`paper_id`): baseline **passed** → corrupted **failed** → repaired **passed**

## Degradation and recovery

For these metrics, positive degradation means Baseline − Corrupted; positive recovery means Repaired − Corrupted.

| Metric | Degradation | Recovery | Repaired vs Baseline |
|---|---:|---:|---:|
| `retrieval_hit_rate` | +0.4000 | +0.4000 | +0.0000 |
| `mean_token_f1` | +0.4905 | +0.4905 | +0.0000 |
| `judge_accuracy` | +0.5000 | +0.5000 | +0.0000 |
| `mean_judge_score` | +2.0000 | +2.0000 | +0.0000 |

All listed evaluation metrics match or exceed Baseline in this run.

## Evaluator

- Baseline: heuristic fallback; reason: Fallback heuristic judge used because the LLM evaluator was unavailable (RuntimeError).
- Corrupted: heuristic fallback; reason: Fallback heuristic judge used because the LLM evaluator was unavailable (RuntimeError).
- Repaired: heuristic fallback; reason: Fallback heuristic judge used because the LLM evaluator was unavailable (RuntimeError).

## Limitations

- Judge accuracy and mean judge score include heuristic fallback results; they are not exclusively LLM-as-judge measurements.
- Results are measured on the existing fixed evaluation set shown above.
- On this run, all listed metrics and repaired quality/freshness checks match the baseline status.

## Artifacts

- baseline_metrics: `data/results/baseline_metrics.json`
- baseline_answers: `data/results/baseline_answers.json`
- corruption_log: `data/results/corruption_log.json`
- corrupted_clean_csv: `data/clean/papers_clean_corrupted.csv`
- corrupted_clean_json: `data/clean/papers_clean_corrupted.json`
- corrupted_embeddings_manifest: `data/embeddings/papers_embeddings_corrupted.json`
- corrupted_metrics: `data/results/corrupted_metrics.json`
- corrupted_answers: `data/results/corrupted_answers.json`
- repaired_clean_csv: `data/clean/papers_clean_repaired.csv`
- repaired_clean_json: `data/clean/papers_clean_repaired.json`
- repaired_embeddings_manifest: `data/embeddings/papers_embeddings_repaired.json`
- repaired_metrics: `data/results/repaired_metrics.json`
- repaired_answers: `data/results/repaired_answers.json`
- evaluation_set: `data/eval/test_set.json`
- comparison_report: `data/reports/corruption_report.md`
