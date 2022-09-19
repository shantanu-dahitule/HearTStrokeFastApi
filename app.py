#gender, age, hypertension heart_disease avg_glucose_lvl bmi smoking status
import uvicorn
from fastapi import FastAPI
from Inputs import IFeatures
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import heartstrokenewtry
# scaler = StandardScaler()
app = FastAPI()
model = pickle.load(open(r"G:\Projects_to_compete\HearDiseasePredictor\randfor96.pkl","rb"))
#model = pickle.load(open(r"G:\Projects_to_compete\HearDiseasePredictor\linearReg.pkl","rb"))
scaler = pickle.load(open(r"G:\Projects_to_compete\HearDiseasePredictor\datascaler.pkl","rb"))

@app.get('/')
def index():
    return {'message':'Hello Its working'}

@app.post('/predict')
def predictStroke(data:IFeatures):
    data=data.dict()
    gender=data['gender']
    age=data['age']
    hypertension = data['hypertension']
    heart_disease=data['heart_disease']
    avg_glucose_lvl= data['avg_glucose_lvl']
    bmi=data['bmi']
    smoking_status = data['smoking_status']
    c = scaler.fit_transform([[gender,age,hypertension,heart_disease,avg_glucose_lvl,bmi,smoking_status]])
    #c=[[gender,age,hypertension,heart_disease,avg_glucose_lvl,bmi,smoking_status]]
    # d=[[1,68,1,1,45.5,112.2,2]]
    # d=scaler.fit_transform(d)
    classify = model.predict(c)
    #classify = classify.tolist()
    if(classify>0.5):
        classify="Stroke may occure"
    else:
        classify="Stroke may not occure"
    return {
        'prediction': classify
    }
if __name__ == '__main__':
    uvicorn.run(app,debug=True)
