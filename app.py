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
    # Convertir datos a DataFrame (mismo orden que el entrenamiento)
    input_df = pd.DataFrame([data.dict()])
    
    # Escalar y predecir
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    
    return {"prediction": int(prediction[0])}