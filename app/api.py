from fastapi import FastAPI
from pydantic import BaseModel
from skops.io import load
import pandas as pd


app = FastAPI(
    title="Démonstration du modèle de prédiction de survie sur le Titanic",
    description="Application de prédiction de survie sur le Titanic via FastAPI",
)

from skops.io import get_untrusted_types

unknown_types = get_untrusted_types(file="model.skops")
model = load("model.skops", trusted=unknown_types)


class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    Embarked: str


@app.get("/")
def home() -> dict:
    return {"message": "Bienvenue sur l'API Titanic"}


@app.post("/predict")
def predict(passenger: Passenger) -> dict:
    df = pd.DataFrame(
        [
            {
                "Pclass": passenger.Pclass,
                "Sex": passenger.Sex,
                "Age": passenger.Age,
                "Fare": passenger.Fare,
                "Embarked": passenger.Embarked,
            }
        ]
    )

    prediction = int(model.predict(df)[0])

    return {
        "survived_prediction": prediction,
        "input": passenger.model_dump(),
    }