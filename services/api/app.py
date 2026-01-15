import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "/models/model.pkl"

app = FastAPI(title="MLOps API", version="1.0")

model = None  # on va charger le modèle ici


class PredictRequest(BaseModel):
    # Iris dataset = 4 features
    features: list[float]


@app.on_event("startup")
def load_model():
    """
    Cette fonction s'exécute au démarrage du conteneur API.
    Elle charge le modèle depuis le volume Docker monté dans /models.
    """
    global model
    if not os.path.exists(MODEL_PATH):
        # L'API peut tourner même si le modèle n'existe pas encore,
        # mais /predict ne marchera pas tant que trainer n'a pas produit le fichier.
        print(f"⚠️ Model not found at {MODEL_PATH}. Run trainer first.")
        model = None
        return

    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")


@app.get("/health")
def health():
    """
    Endpoint simple pour vérifier que l'API répond.
    On indique aussi si le modèle est chargé ou non.
    """
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Reçoit des features et renvoie la prédiction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run trainer first.")

    if len(req.features) != 4:
        raise HTTPException(status_code=400, detail="Iris model expects 4 features.")

    pred = model.predict([req.features])[0]
    return {"prediction": int(pred)}
