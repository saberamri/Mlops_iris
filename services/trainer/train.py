import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MODEL_DIR = "/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- MLflow config (via variables docker-compose)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-demo")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train + log
n_estimators = 100

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    # Log params/metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)

    # Log model inside MLflow artifacts
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Save also to shared volume for the API (like before)
    joblib.dump(model, f"{MODEL_DIR}/model.pkl")

    print(f"✅ Run ID: {run.info.run_id}")
    print(f"✅ accuracy={acc:.3f}")
    print("✅ saved: /models/model.pkl")
