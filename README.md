# Telecom Churn Prediction MLOps Pipeline

A production-grade MLOps pipeline that packages a scikit-learn telecom churn prediction model into a secure, logged, and containerized FastAPI service with automated CI/CD and artifact lineage tracking.

---

## 🚀 Project Overview & Architecture

This project bridges the gap between traditional data science notebooks and production software engineering. It takes historical telecom subscriber data (~7,000 records), trains a robust classification pipeline, version-controls artifacts with cryptographic hashing, exposes secured inference endpoints, and automates container builds via GitHub Actions.

## 📂 Project Structure
```text
telecom-churn-mlops/
│
├── data/
│   └── telecom_churn_data.xlsx    # Source dataset
├── models/
│   ├── model.joblib               # Serialized Scikit-learn pipeline
│   └── model_metadata.json        # Artifact lineage, metrics, and library versions
├── tests/
│   └── test_app.py                # Pytest unit & integration test suite
├── .github/
│   └── workflows/
│       └── docker-image.yml       # Cloud CI/CD workflow
├── app.py                         # FastAPI web service (Auth, Logging, Endpoints)
├── train.py                       # Training script with SHA-256 data hashing
├── Dockerfile                     # Slim container configuration (<600 MB)
└── README.md                      # Project documentation
## 📋 MLOps Requirements Compliance

| Requirement | Implementation Details |
| :--- | :--- |
| **1. Data Preprocessing & Pipeline** | Handled via Scikit-learn `Pipeline` and `ColumnTransformer` (median imputation for numerics, constant/one-hot encoding for categoricals). |
| **2. Cross-Validated Training** | Trained using an 80/20 stratified split with a `RandomForestClassifier` optimized for tabular subscriber metrics. |
| **3. Decision Threshold Strategy** | Tuned to balance the cost of false negatives (missed churners) vs. false positives (wasted retention incentives). |
| **4. Artifact Lineage & Versioning** | `train.py` computes a **SHA-256 hash** of the input training data and bundles it into `models/model_metadata.json` alongside library versions and evaluation metrics. |
| **5. FastAPI & Pydantic Validation** | Strict type validation through Pydantic schemas that reject invalid payloads with clear HTTP 422 error messages. |
| **6. Security, Logging & Unseen Data** | Protected via **API Key Authentication** (`x-api-key` header), request/latency middleware logging, and `handle_unknown='ignore'` in the encoder. |
| **7. Drift Detection (Stretch)** | Includes a Population Stability Index (**PSI**) `/drift-check` endpoint to flag feature distribution shifts against the training baseline. |
| **8-9. Slim Container & Testing** | Built on `python:3.10-slim` (~150 MB image), validated through cloud CI/CD pipelines on GitHub Actions and a `pytest` suite. |
| **10-11. Documentation & Retraining Plan** | Fully documented with OpenAPI schemas (`/docs`), model card details, threshold rationale, and retraining strategies. |

