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
        
        input_df = input_df.rename(columns={
            'htn': 'htn_yes',
            'dm': 'dm_yes'
        })
        
        # re ordenar
        columnas_entrenamiento = [
            'hemo', 'rbcc', 'sc', 'bu', 'al', 'sg', 'htn_yes', 'dm_yes', 'bgr', 'age'
        ]
        input_df = input_df[columnas_entrenamiento]
        
        # Escalar
        scaled_data = scaler.transform(input_df)

        # Predecir
        prediction = model.predict(scaled_data)
        
        return {"prediction": int(prediction[0])}
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"error": str(e)}, 500