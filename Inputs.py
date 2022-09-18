from pydantic import BaseModel

class IFeatures(BaseModel):
    gender: int
    age: int
    hypertension: int 
    heart_disease: int 
    avg_glucose_lvl: float 
    bmi: float 
    smoking_status: int