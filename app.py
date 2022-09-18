#gender, age, hypertension heart_disease avg_glucose_lvl bmi smoking status
import uvicorn
from fastapi import FastAPI
from Inputs import IFeatures
import pickle

app = FastAPI()
model = pickle.load(open(r"G:\Projects_to_compete\HearDiseasePredictor\randfor96.pkl","rb"))
scaler = pickle.load(open(r"G:\Projects_to_compete\HearDiseasePredictor\datascaler.pkl","rb"))

@app.get('/')
def index():
    return {'message':'Hello Its working'}

if __name__ == '__main__':
    uvicorn.run(app,)
