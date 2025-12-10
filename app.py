from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model.predict import ToxicityClassifier
from fastapi.responses import FileResponse
import os
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Загрузка модели")
classifier = ToxicityClassifier()
print("модель загружена")


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict(input: TextInput):
    result = classifier.predict(input.text)
    return {"prediction": result}


@app.get("/")
def home():
    return FileResponse(os.path.join("frontend", "index.html"))


app.mount("/static", StaticFiles(directory="frontend"), name="static")
