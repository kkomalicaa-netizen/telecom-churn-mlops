# Telecom Churn Prediction & MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) pipeline and FastAPI service for predicting customer churn in the telecommunications sector.

## 🚀 Features & Architecture
- **Scikit-learn Pipeline**: Automated preprocessing and robust feature handling for numeric and categorical customer attributes.
- **Artifact Versioning & Hashing**: Automatically generates SHA-256 data hashes and saves trackable version metadata (`model_metadata.json`) during training[cite: 1].
- **Drift Detection Endpoint**: Built-in `/drift-check` endpoint utilizing Population Stability Index (PSI) logic to flag data distribution shifts in production[cite: 1].
- **FastAPI Service**: High-performance asynchronous API validation powered by Pydantic models with an exposed `/model-info` metadata route[cite: 1].
- **Containerization Ready**: Includes a multi-stage Dockerfile optimized for deployment.

## 📂 Project Structure
```text
Ml_Internship_Project/
│
├── models/
│   ├── model.joblib
│   └── model_metadata.json
│
├── train.py          # Training script with hashing and artifact generation
├── app.py            # FastAPI service (Prediction, Metadata, & Drift check)
├── Dockerfile        # Container build instructions
└── requirements.txt  # Python dependencies