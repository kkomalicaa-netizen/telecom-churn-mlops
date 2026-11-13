import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os
import hashlib
import json
import sklearn

def get_file_hash(filepath):
    """Computes SHA-256 hash of the training data file for artifact versioning."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def train_model():
    print("1. Loading dataset...")
    data_path = r'data/telecom_churn_data.xlsx'
    try:
        df = pd.read_excel(data_path)
    except FileNotFoundError:
        # Fallback relative path check
        data_path = 'telecom_churn_data.xlsx'
        try:
            df = pd.read_excel(data_path)
        except FileNotFoundError:
            print(f"Dataset not found! Please make sure 'telecom_churn_data.xlsx' is inside your 'data/' folder.")
            return

    # Compute training data hash for lineage tracking (Requirement 4)
    print("Computing training data hash...")
    data_hash = get_file_hash(data_path)

    # Drop columns that are IDs, administrative, or leaking the answer
    cols_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude', 'Churn Reason', 'Churn Score', 'CLTV', 'Churn Label']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    print("2. Cleaning data...")
    # Fix 'Total Charges': coerce errors to NaN
    if 'Total Charges' in df.columns:
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    # Target variable is 'Churn Value' (1 for churn, 0 for stay)
    if 'Churn Value' in df.columns:
        y = df['Churn Value']
        X = df.drop('Churn Value', axis=1)
    else:
        print("Error: 'Churn Value' column not found in dataset!")
        return

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric features found: {numeric_features}")
    print(f"Categorical features found: {categorical_features}")

    print("3. Building the preprocessing and training pipeline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # handle_unknown='ignore' fulfills requirement for unseen category management (Requirement 6 / 4 integration)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split data into Training set (80%) and Testing set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("4. Training the model...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Model trained successfully! Test Accuracy: {accuracy * 100:.2f}% | ROC-AUC: {roc_auc:.4f}")

    print("5. Saving the model and metadata artifacts to 'models/'...")
    os.makedirs('models', exist_ok=True)
    
    # Save model pipeline
    model_path = 'models/model.joblib'
    joblib.dump(model_pipeline, model_path)

    # Capture library versions for full reproducibility tracking
    library_versions = {
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__
    }

    # Bundle metadata (Requirement 4)
    metadata = {
        "training_data_sha256": data_hash,
        "library_versions": library_versions,
        "metrics": {
            "test_accuracy": float(accuracy),
            "roc_auc": float(roc_auc)
        }
    }

    # Save metadata JSON
    metadata_path = 'models/model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Done! Model and version metadata saved successfully.")

if __name__ == '__main__':
    train_model()
