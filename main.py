# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# -----------------------------
# Load trained pipeline
# -----------------------------
model = joblib.load("lung_cancer_pipeline.pkl")

app = FastAPI(title="Lung Cancer Prediction API")


# -----------------------------
# Input schema
# -----------------------------
class LungCancerInput(BaseModel):
    age: int
    smoking: str
    yellow_fingers: str
    anxiety: str
    chronic_disease: str
    fatigue: str


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_lung_cancer(data: LungCancerInput):

    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).max()

    return {
        "prediction": prediction,
        "confidence": round(float(probability), 3)
    }