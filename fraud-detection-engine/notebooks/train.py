# Trains a fraud detection pipeline and logs the run to MLflow.
# Output: models/fraud_model.pkl (scaler + classifier bundled together)

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

mlflow.set_experiment("fraud-detection")

df = pd.read_csv("notebooks/creditcard.csv")

print(f"Dataset: {df.shape[0]:,} rows | Fraud rate: {df['Class'].mean()*100:.2f}%")

# Drop Time — it's a sequential counter with no predictive signal
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

# stratify=y preserves the 0.17% fraud ratio in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():
    # Bundle scaler + model so production inference never skips the transformation step
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",  # critical: compensates for 0.17% fraud rate
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # fraud probability score

    auc    = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_params({"n_estimators": 100, "class_weight": "balanced"})
    mlflow.log_metrics({
        "auc_roc":         auc,
        "precision_fraud": report["1"]["precision"],
        "recall_fraud":    report["1"]["recall"],
        "f1_fraud":        report["1"]["f1-score"],
    })

    joblib.dump(pipeline, "models/fraud_model.pkl")
    mlflow.log_artifact("models/fraud_model.pkl")

    print(f"\nAUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))