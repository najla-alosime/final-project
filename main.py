from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

app = FastAPI()

# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to SRS Talent Accelerators api"}

# get request
@app.get("/items/")
def create_item(item: dict):
    return {"item": item}

model = joblib.load('rf_model.joblib')
# scaler = joblib.load('Models/scaler.joblib')

# Define a Pydantic model for input data validation

class InputFeatures(BaseModel):
    
    Education: str
    Gender: str
    EverBenched: str
    ExperienceInCurrentDomain: int
    



def preprocessing(input_features: InputFeatures):
    dict_f = {
        'Education': input_features.Education,
        'Gender': input_features.Gender,
        'EverBenched': input_features.EverBenched,
        'ExperienceInCurrentDomain': input_features.ExperienceInCurrentDomain,
        
    }

    # # Convert dictionary values to a list in the correct order
    # features_list = [dict_f[key] for key in sorted(dict_f)]
    # # Scale the input features
    # scaled_features = scaler.transform([features_list])
    # return scaled_features

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}


