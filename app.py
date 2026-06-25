from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load Model
model = joblib.load("mobile_account_model.pkl")

# Input Schema
class InputData(BaseModel):
    REF_AREA: str
    SEX: str
    AGE: str
    URBANISATION: str
    COMP_BREAKDOWN_1: str
    COMP_BREAKDOWN_2: str
    COMP_BREAKDOWN_3: str
    TIME_PERIOD: int

@app.get("/")
def home():
    return {"message": "Mobile Account Prediction API"}

@app.post("/predict")
def predict(data: InputData):

    input_df = pd.DataFrame([{
        "REF_AREA": data.REF_AREA,
        "SEX": data.SEX,
        "AGE": data.AGE,
        "URBANISATION": data.URBANISATION,
        "COMP_BREAKDOWN_1": data.COMP_BREAKDOWN_1,
        "COMP_BREAKDOWN_2": data.COMP_BREAKDOWN_2,
        "COMP_BREAKDOWN_3": data.COMP_BREAKDOWN_3,
        "TIME_PERIOD": data.TIME_PERIOD
    }])

    prediction = model.predict(input_df)

    return {
        "Predicted_OBS_VALUE": float(prediction[0])
    }