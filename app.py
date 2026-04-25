from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Permitir que la web se comunique con el servidor
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo y el escalador
model = joblib.load('./modelo_xgb_erc.pkl')
scaler = joblib.load('./scaler_erc.pkl')

class ClinicaData(BaseModel):
    hemo: float
    rbcc: float
    sc: float
    bu: float
    al: float
    sg: float
    htn: int
    dm: int
    bgr: float
    age: float


@app.post("/predict")
async def predict(data: ClinicaData):
    input_df = pd.DataFrame([data.dict()])
    scaled_data = scaler.transform(input_df)
    
    # Obtener las probabilidades [prob_sano, prob_enfermo]
    probabilities = model.predict_proba(scaled_data)[0]
    
    # La predicción final (clase con mayor probabilidad)
    prediction = int(model.predict(scaled_data)[0])
    
    # Confianza: si es 1, enviamos prob_enfermo; si es 0, enviamos prob_sano
    confidence = probabilities[prediction] * 100
    
    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }