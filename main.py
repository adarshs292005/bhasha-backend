import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Bhasha ISL Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and classes
model = None
classes = []

@app.on_event("startup")
def load_ai_model():
    global model, classes
    print("Starting up: Downloading model from Supabase...")
    
    local_model_path = "isl_rf_model.joblib"
    
    # 1. Download model artifact from Supabase storage bucket
    try:
        res = supabase.storage.from_("bhasha-models").download("v1/isl_rf_model.joblib")
        with open(local_model_path, 'wb+') as f:
            f.write(res)
        model = joblib.load(local_model_path)
        print("✅ Model loaded into memory.")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")

    # 2. Fetch active vocabulary classes from the database registry
    try:
        response = supabase.table("model_versions").select("classes").eq("version_id", "v1").execute()
        classes = response.data[0]["classes"]
        print(f"✅ Vocabulary loaded: {classes}")
    except Exception as e:
        print(f"❌ Failed to fetch classes: {e}")

class LandmarksPayload(BaseModel):
    landmarks: list[float]

@app.post("/predict")
def predict_sign(payload: LandmarksPayload):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    if len(payload.landmarks) != 63:
        raise HTTPException(status_code=400, detail=f"Expected 63 coordinates, got {len(payload.landmarks)}")
    
    X = np.array(payload.landmarks).reshape(1, -1)
    probabilities = model.predict_proba(X)[0]
    
    best_index = np.argmax(probabilities)
    confidence = probabilities[best_index]
    predicted_word = classes[best_index]
    
    return {
        "prediction": predicted_word,
        "confidence": float(confidence)
    }