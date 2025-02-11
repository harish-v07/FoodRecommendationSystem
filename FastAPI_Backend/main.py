from fastapi import FastAPI
from pydantic import BaseModel, Field, conlist
from typing import List, Optional
import pandas as pd
from model import recommend, output_recommended_recipes

# Load the dataset
dataset = pd.read_csv('../Data/dataset.csv', compression='gzip')

app = FastAPI()

class Params(BaseModel):
    n_neighbors: int = Field(default=5, ge=1)
    return_distance: bool = Field(default=False)

class PredictionIn(BaseModel):
    nutrition_input: conlist(float, min_length=9, max_length=9) = Field(..., description="Nutrition input must have exactly 9 items")
    ingredients: List[str] = Field(default_factory=list, description="List of ingredients")
    params: Optional[Params] = Field(default=None, description="Optional parameters")

class Recipe(BaseModel):
    Name: str
    CookTime: str
    PrepTime: str
    TotalTime: str
    RecipeIngredientParts: List[str]
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    RecipeInstructions: List[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = Field(default=None, description="Recommended recipes")

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict/", response_model=PredictionOut)
def update_item(prediction_input: PredictionIn):
    recommendation_dataframe = recommend(
        dataset,
        prediction_input.nutrition_input,
        prediction_input.ingredients,
        prediction_input.params.dict() if prediction_input.params else {}
    )
    output = output_recommended_recipes(recommendation_dataframe)
    
    # Convert numerical time fields to strings
    for recipe in output:
        recipe["CookTime"] = str(recipe["CookTime"])
        recipe["PrepTime"] = str(recipe["PrepTime"])
        recipe["TotalTime"] = str(recipe["TotalTime"])
    
    return {"output": output}
