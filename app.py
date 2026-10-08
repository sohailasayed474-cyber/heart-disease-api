from typing import Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

model = joblib.load("heart_pipeline.joblib")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease from 13 clinical features (UCI Cleveland). Educational project, not a medical device.",
    version="1.0.0",
)


class Patient(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 62, "sex": 1, "cp": 4, "trestbps": 140, "chol": 268, "fbs": 0,
        "restecg": 2, "thalach": 160, "exang": 0, "oldpeak": 3.6,
        "slope": 3, "ca": 2, "thal": 3}})

    age: float = Field(..., gt=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., gt=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., gt=0, le=700, description="Cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., gt=0, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)")
    ca: Optional[float] = Field(None, ge=0, le=3, description="Major vessels colored (0-3), optional")
    thal: Optional[float] = Field(None, description="3 = normal, 6 = fixed defect, 7 = reversible defect, optional")

    @field_validator("thal")
    @classmethod
    def check_thal(cls, v):
        if v is not None and v not in (3, 6, 7):
            raise ValueError("thal must be 3, 6 or 7")
        return v


@app.get("/")
def home():
    return {"message": "Heart Disease API is running. Go to /docs to test it."}


@app.post("/predict")
def predict(p: Patient):
    row = p.model_dump()
    row["age_group"] = str(pd.cut([p.age], bins=[0, 40, 55, 65, 120],
                                  labels=["<40", "40-55", "55-65", "65+"])[0])
    row["chol_age_ratio"] = p.chol / p.age
    row["hr_reserve"] = (220 - p.age) - p.thalach
    row["high_bp"] = int(p.trestbps >= 140)
    row = {k: (np.nan if v is None else v) for k, v in row.items()}

    df = pd.DataFrame([row])
    pred = int(model.predict(df)[0])
    return {
        "prediction": pred,
        "diagnosis": "Heart disease likely" if pred == 1 else "No heart disease likely",
        "probability_of_disease": round(float(model.predict_proba(df)[0][1]), 3),
    }