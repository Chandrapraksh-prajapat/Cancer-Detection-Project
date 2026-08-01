from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

app = FastAPI()

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# model = mlflow.pyfunc.load_model(
# model_uri="models:/Cancer_Detection_Model/1")

# Run id  of best model :ffad0d30a3484f379f6709acf033f18d

model = mlflow.pyfunc.load_model(
    "mlruns/1/models/m-792749f364f54db490a5ac17a9324b71/artifacts"
)

class CancerData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


@app.post("/predict")
def predict(data: CancerData):

    df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(df)
    print(df)

    print("prediction = ", prediction)
    print("type(model) = ", type(model))
    return {
        "prediction": prediction.tolist()
    }
   
{
"radius_mean": 17.99,
"texture_mean": 10.38,
"perimeter_mean": 122.8,
"area_mean": 1001.0,
"smoothness_mean": 0.1184,
"compactness_mean": 0.2776,
"concavity_mean": 0.3001,
"concave_points_mean": 0.1471,
"symmetry_mean": 0.2419,
"fractal_dimension_mean": 0.07871,
"radius_se": 1.095,
"texture_se": 0.9053,
"perimeter_se": 8.589,
"area_se": 153.4,
"smoothness_se": 0.006399,
"compactness_se": 0.04904,
"concavity_se": 0.05373,
"concave_points_se": 0.01587,
"symmetry_se": 0.03003,
"fractal_dimension_se": 0.006193,
"radius_worst": 25.38,
"texture_worst": 17.33,
"perimeter_worst": 184.6,
"area_worst": 2019.0,
"smoothness_worst": 0.1622,
"compactness_worst": 0.6656,
"concavity_worst": 0.7119,
"concave_points_worst": 0.2654,
"symmetry_worst": 0.4601,
"fractal_dimension_worst": 0.1189
}