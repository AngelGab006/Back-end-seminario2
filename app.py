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
        dict_data = data.dict()
        input_df = pd.DataFrame([dict_data])
        
        # Renombrar columnas para que coincidan con get_dummies del entrenamiento
        input_df = input_df.rename(columns={
            'htn': 'htn_yes',
            'dm': 'dm_yes'
        })
        
        # Reordenar las columnas exactamente como las vio el modelo
        columnas_entrenamiento = [
            'age', 'sg', 'al', 'bgr', 'bu', 'sc', 'hemo', 'rbcc', 'htn_yes', 'dm_yes'
        ]
        input_df = input_df[columnas_entrenamiento]
        
        # Escalar los datos
        scaled_data = scaler.transform(input_df)

        # Obtener predicción de clase y probabilidades
        prediction = int(model.predict(scaled_data)[0])
        probabilities = model.predict_proba(scaled_data)[0] # [prob_sano, prob_enfermo]
        
        # Calcular confianza basado en la clase predicha
        confidence = round(probabilities[prediction] * 100, 2)
        
        return {
            "prediction": prediction,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"error": str(e)}, 500
