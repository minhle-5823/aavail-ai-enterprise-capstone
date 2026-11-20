# AAVAIL: Reproducible Customer Churn Workflow

**Project type:** Machine-learning capstone, FastAPI service, reproducible EDA, monitoring, automated tests, Docker.

> **Data provenance and honesty:** The included bootstrap uses **synthetic demo data**, not AAVAIL customer history. Demo charts and metrics have **no business validity**. To submit real AAVAIL findings, download or supply the official `aavail-target.csv`, retrain, and regenerate the figures. Never label synthetic results as real performance.

## 1. Business problem

AAVAIL wants to explain and monitor a churn gap between customers in Singapore and the United States. This repository implements ingestion, data-quality auditing and market-specific imputation; a binary churn predictor; comparison against a simple baseline; REST predictions for each market or all markets; feedback-based evaluation; and tests. **Churn = `1` when `is_subscriber` equals `0/False`.** Cross-sectional associations in EDA do not prove causality.

## 2. Contents

```text
.
├── aavail/
│   ├── ingestion.py         # data validation, invalid ages, missingness, imputation
│   ├── modeling.py          # 3 trained/evaluated models, serialize winner
│   ├── visualize.py         # missingness, churn rates, baseline comparison
│   ├── api.py               # /health /predict /markets/summary /feedback /monitor
│   ├── auditlog.py          # append-only JSONL, test isolation
│   └── monitoring.py        # labeled feedback-based performance
├── scripts/
│   ├── download_data.py     # explicit source URL, validation
│   ├── generate_demo.py     # synthetic fixture generation ONLY
│   ├── bootstrap_demo.py    # Docker/bootstrap training
│   └── train.py             # reproducible command-line training
├── tests/                  # ingestion, model, API, logs and monitoring tests
├── run_tests.py            # one command for all tests
├── Dockerfile              # runnable image, tests as part of build
└── .github/workflows/ci.yml # CI tests and Docker build/smoke test
```

The model uses only `country`, `age`, `subscriber_type`, and `num_streams`. `customer_id` and `customer_name` are excluded from modeling; no raw personal identifiers are written to prediction logs. CSV label/country errors and duplicate IDs are audited; invalid ages become missing; missing numeric values use market medians, missing plans use market modes. The model pipeline has extra fallback imputers for API inputs. The ingestion report lists dropped rows and excluded columns.

## 3. Install and prepare

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Option A: Official AAVAIL public capstone sample (recommended for grading).** The course-public starter dataset is available at https://github.com/aavail/ai-workflow-capstone-soln/tree/master/data . Download it as follows, or provide your own authorized class dataset:

```bash
python -m scripts.download_data --output data/aavail-target.csv
python -m scripts.train --data data/aavail-target.csv --output models
```

**Option B: Offline demonstration only**, with a completely synthetic fixture:

```bash
python -m scripts.generate_demo
python -m scripts.train --data data/demo_synthetic.csv --output models --demo
```

Model/EDA outputs saved under `models/`:
- `ingestion_audit.json`: counts of missing data before and after, invalid ages, duplicate/invalid records, excluded columns.
- `metrics.json`: split metadata, candidate comparisons, holdout ROC-AUC/F1, US and SG churn summaries.
- `missing_values.png`, `market_churn.png`, `eda_churn_factors.png`, `model_vs_baseline.png`: figures with titles, axis labels and the data-mode disclaimer.
- `model.joblib`: persisted selected classifier; **load only artifacts created by your own trusted environment**.

## 4. Run the API

```bash
uvicorn aavail.api:app --host 127.0.0.1 --port 8080
```

Browse `http://127.0.0.1:8080/docs` for live OpenAPI documentation.

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/markets/summary
curl 'http://127.0.0.1:8080/markets/summary?country=singapore'
curl -X POST http://127.0.0.1:8080/predict?country=all \
  -H 'Content-Type: application/json' \
  -d '{"customers":[{"country":"singapore","age":30,"subscriber_type":"aavail_premium","num_streams":10},{"country":"united_states","age":35,"subscriber_type":"aavail_basic","num_streams":19}]}'
curl -X POST 'http://127.0.0.1:8080/predict?country=singapore' \
  -H 'Content-Type: application/json' \
  -d '{"customers":[{"country":"singapore","age":30,"subscriber_type":"aavail_premium","num_streams":10},{"country":"united_states","age":35,"subscriber_type":"aavail_basic","num_streams":19}]}'
curl http://127.0.0.1:8080/monitor
```

Each prediction returns a `prediction_id`. When an eventual true outcome is available, use `POST /feedback` with `{"prediction_id":"...","actual_churn":0}` to support real performance monitoring. Until at least five labeled outcomes arrive, the monitor explicitly reports insufficient data. Logs live under `logs/` and do not contain personal names.

## 5. Run ALL unit tests with one command

```bash
python run_tests.py
```

The test suite creates a completely separate model and log directory in pytest `tmp_path`. Tests never overwrite `models/` or `logs/` and never depend on a running server; they use FastAPI's in-memory test client. CI also runs these tests and attempts a Docker smoke test.

## 6. Docker

```bash
docker build -t aavail-capstone .
docker run --rm -p 8080:8080 aavail-capstone
```

The image is **self-contained and runs a synthetic demo if the official dataset is missing**. To run with real AAVAIL data, first place authorized `aavail-target.csv` in `data/` before building. Because this sample project ignores `data/aavail-target.csv` in Git, cloning from GitHub alone always starts in demo mode. Verify via `/markets/summary` (`data_mode`). Local Docker execution must be verified on your own computer; a successful `docker build` is NOT asserted in this package.

## 7. Evidence for peer-grading checklist

| Rubric | Evidence |
|---|---|
| API unit tests | `tests/test_api.py` |
| Model unit tests | `tests/test_model.py` |
| Logging unit tests | `tests/test_logging.py` |
| Single test command | `python run_tests.py` |
| Performance monitoring | `aavail/monitoring.py`, `GET /monitor`, `POST /feedback` |
| Read/write isolation | `tests/conftest.py` and `tmp_path` test fixtures |
| Country-specific/all API | `POST /predict?country=singapore` or `country=all` |
| Automated ingestion | `aavail/ingestion.py`, `scripts/download_data.py`, `scripts/train.py` |
| Multiple models | `DummyClassifier`, `LogisticRegression`, `RandomForestClassifier` |
| EDA charts | `docs/EDA.md`, `docs/figures/eda_churn_factors.png`, `models/missing_values.png` |
| Docker | `Dockerfile` and `.github/workflows/ci.yml` |
| Baseline comparison figure | `models/model_vs_baseline.png` |

**Reviewer's reproducibility check:** `python -m scripts.generate_demo && python -m scripts.train --data data/demo_synthetic.csv --demo && python run_tests.py` runs offline. To obtain authentic results, use Section 3 Option A.

## 8. Known limits and suggested next steps

- Demo-only metrics cannot be used to recommend investment in a specific market.
- Recorded statuses and plan/streams features support associations, not causal explanations for churn.
- Monitor endpoint is a local proof of concept; a real deployment needs authorization, TLS, rate limits, secrets management, retention policies, scheduled retraining and stronger drift monitoring.
- Current model selection uses a single stratified holdout; cross-validation and temporal validation should be added before production deployment.
- Docker image compilation, network fetching and CI execution are to be verified in the GitHub environment.

## 9. Submission

Review `docs/GITHUB_UPLOAD_INSTRUCTIONS_VI.md` for Vietnamese instructions. A synthetic demo example is shown in `docs/EDA.md`. After retraining on authentic data, replace the images in `docs/figures/` and the example observations.

Create a **PUBLIC GitHub repository**, upload the project contents, confirm that `README.md`, `Dockerfile`, `tests/` and CI are visible, then paste the actual link of YOUR repo into the assignment `URL` field. Do not paste the upstream source's GitHub URL. Suggested project title: **AAVAIL Customer Churn Prediction: End-to-End AI Workflow**.

**Upstream data/source reference:** [AAVAIL AI Enterprise Workflow capstone starter/solution files](https://github.com/aavail/ai-workflow-capstone-soln). This project is an independently written sample implementation, not a copy of the upstream application.
