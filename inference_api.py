# inference_api.py
from fastapi import FastAPI, Query, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from utils.processing import clean_text
import joblib
import torch
import numpy as np
from Models.trainers import DNNModel

app = FastAPI()

# Load vectorizer
vectorizer = joblib.load("artifacts/vectorizer.pkl")

# Load models
lr_model = joblib.load("artifacts/lr_model.pkl")
xgb_model = joblib.load("artifacts/xgb_model.pkl")

# Load DNN
dnn_model = DNNModel(input_dim=5000)
dnn_model.load_state_dict(torch.load("artifacts/dnn_model.pt", map_location=torch.device("cpu")))
dnn_model.eval()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

class InferenceRequest(BaseModel):
    text: str
    model: str = Query("lr", enum=["lr", "xgb", "dnn"])

@app.post("/predict")
def predict(req: InferenceRequest):
    cleaned = clean_text(req.text)
    vector = vectorizer.transform([cleaned]).toarray()

    if req.model == "lr":
        prob = lr_model.predict_proba(vector)[0][1]
    elif req.model == "xgb":
        prob = xgb_model.predict_proba(vector)[0][1]
    elif req.model == "dnn":
        with torch.no_grad():
            tensor = torch.tensor(vector, dtype=torch.float32)
            prob = float(dnn_model(tensor).numpy()[0][0])
    else:
        return {"error": "Invalid model selected"}

    return {
        "model": req.model,
        "phishing_probability": prob,
        "is_phishing": bool(prob > 0.5)
    }

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request, text: str = Form(...), model: str = Form(...)):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned]).toarray()

    if model == "lr":
        prob = lr_model.predict_proba(vector)[0][1]
    elif model == "xgb":
        prob = xgb_model.predict_proba(vector)[0][1]
    elif model == "dnn":
        with torch.no_grad():
            tensor = torch.tensor(vector, dtype=torch.float32)
            prob = float(dnn_model(tensor).numpy()[0][0])
    else:
        prob = 0.0

    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": f"Phishing Probability: {prob:.2f} — {'Phishing' if prob > 0.5 else 'Legit'}"
    })
