# Báo cáo cá nhân — Day 10: Data Pipeline & Data Observability

## 1. Thông tin cá nhân

| Thông tin | Nội dung |
| --- | --- |
| Họ và tên | Nguyễn Ngọc Linh |
| MSSV | 2A202602480 |
| Khóa/lớp | K4-L3B-DAY10 |
| Tên nhóm | MEME |
| Vai trò chính | Phụ trách toàn bộ pipeline và deliverable của bài lab |
| Repository | `K4-L3B-DAY10-MEME-DataPipelineDataObservability` |
| Ngày xác nhận | 2026-09-26 |

## 2. Vai trò và phạm vi công việc

Phạm vi đảm nhiệm gồm ingestion/raw data, cleaning/data modeling, data quality/freshness, evaluation test set, embedding/ChromaDB, baseline pipeline, corruption/repair, báo cáo và tích hợp. Bảng dưới phân biệt phần đã hoàn tất trong lần sanity check thành công với phần chưa chạy trong yêu cầu hiện tại.

| Module/deliverable | File/hàm đã thực hiện hoặc kiểm tra | Trạng thái và bằng chứng |
| --- | --- | --- |
| Ingestion và raw data | `src/ingestion/crossref.py`: `load_raw_records`, `fetch_source_records`, `enrich_missing_categories`; `data/raw/` | Đã đọc snapshot hiện có trong lần chạy; báo cáo phase 1 ghi nguồn `Crossref saved records snapshot` và 24 raw records. |
| Cleaning/data modeling | `src/ingestion/cleaning.py`: `build_clean_dataframe`; `data/clean/papers_clean.csv`, `papers_clean.json` | Đã chạy; hai file được cập nhật 2026-09-26 12:13 và có 24 dòng clean theo báo cáo phase 1. |
| Quality/freshness | `src/observability/quality.py`: `run_data_quality_checks`, `evaluate_freshness_sla`; `data/quality/baseline_quality_report.json`, `freshness_report.json` | Đã chạy; quality gate passed, freshness fresh, 0/24 stale, ngưỡng 180 ngày trong artifact. |
| Evaluation test set | `src/evaluation/testset.py`: `assess_test_set_coverage`, `build_test_set`; `data/eval/test_set.json` | Đã tạo/cập nhật 2026-09-26 12:13; phase report ghi preflight và evaluation passed, metrics có 10 samples. |
| Embedding và ChromaDB | `src/retrieval/index.py`: `LocalEmbeddingIndex.build`; `data/embeddings/papers_embeddings.json`, `data/chroma/` | Đã chạy; phase report ghi collection `papers-baseline`, model `sentence-transformers/all-MiniLM-L6-v2`, 24 indexed documents. |
| Baseline pipeline | `script/run_phase1.py`, `src/pipelines/phase1.py`: `run_phase1_pipeline`, `main`; `src/evaluation/metrics.py`: `evaluate_pipeline` | Hoàn thành với exit code 0 khi đặt `LLM_MODEL=gemini-3.5-flash-lite` tạm thời; baseline metrics và answers cập nhật 2026-09-26 12:15. Ragas chưa chạy vì `RUN_RAGAS` không được bật. |
| Corruption và repair | `src/pipelines/corruption_flow.py`: `run_corruption_flow_pipeline`, `repair_from_raw_snapshot` | Không chạy trong yêu cầu sanity check này. Artifact corruption/repaired đã có trong repository nhưng không được tái xác minh trong lần này. |
| Báo cáo và tích hợp | `src/observability/reporting.py`: `generate_phase1_report`; `docs/TEAM.md`; báo cáo này | Phase report ghi ingest, clean, quality/freshness, preflight, index, evaluation và report đều passed; hồ sơ nhóm và báo cáo cá nhân được cập nhật theo artifact mới. |

### Phần hỗ trợ ngoài phạm vi module đơn lẻ

Tích hợp và đối chiếu hợp đồng dữ liệu giữa raw records, clean dataset, quality gate, test set, vector index và evaluator; kiểm tra trạng thái artifact đầu ra của baseline.

## 3. Kết quả theo vai trò

| Nhiệm vụ | File/hàm/artifact | Kết quả và cách xác minh |
| --- | --- | --- |
| Chạy baseline từ entrypoint được yêu cầu | `python script/run_phase1.py` | Entry point đã được sửa để tìm package trong `src/`. Lần chạy cuối với model override kết thúc exit code 0 và ghi `Phase 1 baseline pipeline completed.` |
| Nạp raw, làm sạch và quality/freshness | `src/pipelines/phase1.py`, clean artifacts và quality reports | Phase report ghi 24 raw records, 24 clean rows, quality gate passed, freshness fresh, 0/24 stale. |
| Đánh giá baseline | `src/evaluation/metrics.py`, `data/results/baseline_metrics.json`, `baseline_answers.json` | Metrics và answers được cập nhật 2026-09-26 12:15; 10 samples, `judge_mode=llm`. |

### Artifact và metric hiện có

Năm artifact baseline bắt buộc đều tồn tại và thuộc lần chạy cuối: `data/clean/papers_clean.csv`, `data/clean/papers_clean.json`, `data/eval/test_set.json`, `data/results/baseline_metrics.json`, `data/reports/phase1_report.md`. Metrics mới có `samples=10`, `retrieval_hit_rate=1.0`, `mean_token_f1=0.39045783373362836`, `judge_accuracy=0.9`, `mean_judge_score=4.4`, `judge_mode=llm`. `data/results/baseline_answers.json` được cập nhật cùng thời điểm. Ragas được artifact đánh dấu `skipped` vì chưa bật `RUN_RAGAS`.

## 4. Giải thích phần kỹ thuật đã thực hiện

### Vấn đề cần giải quyết

Entry point chạy từ project root ban đầu không tự tìm package bên trong `src/`. Sau khi sửa đường dẫn import và bổ sung credential trong môi trường, model mặc định `gemini-2.5-flash` trả 404 cho tài khoản này. Lần chạy thành công dùng model khả dụng `gemini-3.5-flash-lite` qua biến môi trường tạm.

### Cách triển khai

`script/run_phase1.py` thêm `src/` (tính từ vị trí file script) vào `sys.path` trước khi import `pipelines.phase1`. Pipeline đọc raw snapshot, enrich category metadata, tạo clean dataframe và test set, chạy quality/freshness, build index, gọi evaluator qua provider Google, rồi ghi metrics và phase report. `LLM_MODEL` được đặt trong phiên PowerShell cho lần chạy thành công; `.env` không được chỉnh sửa trong bước kiểm tra này.

### Input, output và contract

| Thành phần | Mô tả |
| --- | --- |
| Input | Raw snapshot `data/raw/crossref_records.json` và response; cấu hình pipeline từ settings/môi trường. |
| Output đã xác minh | Clean CSV/JSON, test set, embeddings manifest/index, quality/freshness report, baseline metrics/answers và `data/reports/phase1_report.md`. |
| Module phụ thuộc | `src/core/config.py`, `src/ingestion/`, `src/observability/`, `src/evaluation/`, `src/retrieval/`. |
| Module sử dụng output | Embedding/index và evaluation trong `src/pipelines/phase1.py`. |
| Điều kiện lỗi thực tế | Model `gemini-2.5-flash` trả 404 cho tài khoản này; `gemini-3.8-flash` trả 503 do tải cao; `gemini-3.7-flash` chạm quota miễn phí 20 lượt gọi/ngày. |

### Cách xác minh

```powershell
$env:PATH = (Resolve-Path .venv/Scripts).Path + ';' + $env:PATH
$env:LLM_MODEL = 'gemini-3.5-flash-lite'
python script/run_phase1.py
```

- **Kết quả thực tế:** exit code 0, không có exception. Pipeline dùng Google LLM cho QA/judge và ghi `judge_mode=llm`.
- **Artifact/log:** `data/reports/phase1_report.md` ghi mọi stage passed; clean data, test set, baseline metrics/answers và quality/freshness report được ghi trong lần chạy.

## 5. Một quyết định kỹ thuật quan trọng

- **Bối cảnh:** Sau khi credential được bổ sung, model đang được cấu hình trả 404 và các model mới hơn lần lượt gặp lỗi dịch vụ/quota.
- **Các phương án đã cân nhắc:** Tiếp tục dùng `gemini-2.5-flash`; thử `gemini-3.8-flash`, `gemini-3.7-flash`; hoặc dùng model ổn định `gemini-3.5-flash-lite` cho bộ evaluation nhiều lượt gọi.
- **Phương án đã chọn:** Đặt `LLM_MODEL=gemini-3.5-flash-lite` trong phiên chạy, giữ provider Google và không sửa `.env`.
- **Lý do:** Model này hỗ trợ GenerateContent và đủ khả năng cho luồng QA/judge; cấu hình tạm cho phép kiểm chứng pipeline thật với credential đã cung cấp.
- **Bằng chứng:** Lệnh exit code 0; `data/results/baseline_metrics.json` có 10 samples và `judge_mode=llm`; `data/reports/phase1_report.md` ghi evaluation passed. Model này được [Google liệt kê là ổn định](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite).

## 6. Một lỗi hoặc blocker

- **Triệu chứng:** Sau khi key được bổ sung, lần chạy không override model trả `GoogleModelNotFoundError` với HTTP 404: `This model models/gemini-2.5-flash is no longer available to new users.`
- **Các lỗi thử model khác:** `gemini-3.8-flash` trả `503 UNAVAILABLE` với thông báo `This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.`; `gemini-3.7-flash` trả `429 RESOURCE_EXHAUSTED`, quota `generate_content_free_tier_requests`, giới hạn 20 lượt/ngày cho model đó.
- **Nguyên nhân:** Model mặc định không khả dụng cho tài khoản; các lỗi 503 và 429 đến từ dịch vụ/quota của provider.
- **Cách xử lý và xác minh:** Override `LLM_MODEL=gemini-3.5-flash-lite` trong phiên PowerShell rồi chạy đúng `python script/run_phase1.py`; exit code 0, metrics/answers mới và phase report thành công.
- **Vấn đề còn lại:** Chạy lại với cấu hình model cũ và không override vẫn sẽ gặp lỗi 404. Cần cập nhật `LLM_MODEL` trong môi trường chạy trước lần tái hiện tiếp theo. Ragas chưa chạy; corruption/repair không thuộc lần kiểm tra này.

## 7. Hiểu biết về luồng end-to-end

Raw Crossref response/records được parse và chuẩn hóa thành clean records; model dữ liệu tạo `age_days` và `text_for_embedding`. Quality checks xác minh completeness/uniqueness/độ dài và freshness SLA trước khi tạo test set và index embedding. Evaluation dùng các ground-truth document IDs để đo retrieval và chất lượng câu trả lời. Giữ nguyên một test set giữa baseline, corrupted và repaired giúp đối chiếu cùng câu hỏi và tài liệu đích. Repair cần được đối chiếu với raw source, quality/freshness reports và metric sau repair. Corruption/repair không được chạy trong sanity check lần này.

## 8. Phân tích kết quả

| Metric/signal | Baseline lần chạy cuối | Nhận xét |
| --- | ---: | --- |
| `retrieval_hit_rate` | 1.0 | 10 samples trong `data/results/baseline_metrics.json`. |
| `mean_token_f1` | 0.39045783373362836 | Giá trị từ evaluation với Google LLM. |
| `judge_accuracy` | 0.9 | `judge_mode=llm`, không phải heuristic fallback. |
| `mean_judge_score` | 4.4 | `judge_mode=llm`, không phải heuristic fallback. |
| Quality/freshness | Passed; fresh; 0/24 stale | Ghi trong quality report và phase report mới. |

Metrics mới thấp/cao khác với artifact mock cũ là do chế độ trả lời và judge khác nhau; không gán nguyên nhân chi tiết nếu chưa so sánh cùng điều kiện. Các artifact corruption/repaired sẵn có không được chạy lại hoặc đối chiếu trong phạm vi yêu cầu.

## 9. Điều học được và hướng cải thiện

1. Entrypoint cần hỗ trợ cấu trúc `src/` để lệnh khởi chạy trực tiếp hoạt động ổn định.
2. Chọn model được tài khoản hỗ trợ và có quota phù hợp là một phần của khả năng tái hiện evaluation; lỗi 404, 503 và 429 có nguyên nhân khác nhau.
3. Artifact metric cần được đối chiếu với timestamp, `judge_mode` và run report trước khi dùng làm bằng chứng.

Nếu có thêm thời gian, cập nhật cấu hình model dùng khi nộp bài rồi chạy lại không cần override; sau đó đo thêm Ragas nếu quota cho phép.

## 10. Cam kết của thành viên

- [x] Nội dung phản ánh phạm vi và trạng thái đã đối chiếu với artifact.
- [x] Kết luận về lần sanity check có bằng chứng và ghi rõ exit code.
- [x] Chỉ tuyên bố pipeline thành công cho lần chạy exit code 0 với model override đã ghi rõ.
- [x] Báo cáo không chứa nội dung `.env`, API key, token hoặc secret.
- [x] Báo cáo có phần phản ánh cá nhân, không sao chép báo cáo nhóm.

**Họ và tên:** Nguyễn Ngọc Linh  
**Ngày xác nhận:** 2026-09-26
