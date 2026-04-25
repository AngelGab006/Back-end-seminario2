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
    try:
        # pre procesar datos
        dict_data = data.dict()
        columnas_frontend = ['hemo', 'rbcc', 'sc', 'bu', 'al', 'sg', 'htn', 'dm', 'bgr', 'age']
        input_df = pd.DataFrame([dict_data])[columnas_frontend]
        input_df = input_df.rename(columns={
            'htn': 'htn_yes',
            'dm': 'dm_yes'
        })
        
        # Escalar los datos
        scaled_data = scaler.transform(input_df)
        
        # Realizar la predicción
        prediction = model.predict(scaled_data)
        
        return {"prediction": int(prediction[0])}