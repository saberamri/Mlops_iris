from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Dossier où le modèle sera sauvegardé
MODEL_DIR = "/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Charger les données
X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entraînement
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model accuracy: {accuracy:.3f}")

# Sauvegarde du modèle
joblib.dump(model, f"{MODEL_DIR}/model.pkl")
print("Model saved in /models/model.pkl")