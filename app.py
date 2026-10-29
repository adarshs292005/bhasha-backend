
import json
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATHS AND MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "v2"

MODEL_PATH = MODEL_DIR / "bhasha_character_rf_v2.joblib"
CLASSES_PATH = MODEL_DIR / "bhasha_character_classes_v2.json"
METADATA_PATH = MODEL_DIR / "bhasha_character_model_metadata_v2.json"


# Check required files
for file_path in [MODEL_PATH, CLASSES_PATH, METADATA_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# Load trained model
model = joblib.load(MODEL_PATH)


# Load class labels
with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    raw_classes = json.load(file)


# Load model metadata
with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)


# Support dictionary or list class formats
if isinstance(raw_classes, dict):
    classes = [
        label
        for label, index in sorted(
            raw_classes.items(),
            key=lambda item: item[1]
        )
    ]
else:
    classes = raw_classes


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Bhasha ML Backend",
    description=(
        "Bhasha Indian Sign Language character "
        "recognition API"
    ),
    version="2.0.0",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

# Add deployed frontend URL through the FRONTEND_URL
# environment variable when it is available.
frontend_url = os.environ.get("FRONTEND_URL")

if frontend_url:
    allowed_origins.append(frontend_url.rstrip("/"))


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST VALIDATION
# ============================================================

class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=126,
        max_length=126
    )

    hand_count: int = Field(
        ...,
        ge=1,
        le=2
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Bhasha ML Backend",
        "status": "online",
        "backend_version": "2.0",
        "model_version": metadata.get(
            "model_version",
            "unknown"
        ),
        "features": 126,
        "classes": len(classes),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/v2/health")
def health():
    return {
        "status": "healthy",
        "backend_version": "2.0",
        "model_loaded": model is not None,
        "model_version": metadata.get(
            "model_version",
            "unknown"
        ),
        "features": 126,
        "classes": len(classes),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/api/v2/model-info")
def model_info():
    return {
        "backend_version": "2.0",
        "model_version": metadata.get(
            "model_version",
            "unknown"
        ),
        "model_type": metadata.get(
            "model_type",
            "RandomForestClassifier"
        ),
        "features": 126,
        "classes": len(classes),
        "class_names": classes,
    }


# ============================================================
# CHARACTER PREDICTION
# ============================================================

@app.post("/api/v2/predict")
def predict(request: PredictionRequest):

    # Validate feature count
    if len(request.features) != 126:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected 126 features, "
                f"received {len(request.features)}."
            ),
        )

    # Validate hand count
    if request.hand_count not in (1, 2):
        raise HTTPException(
            status_code=400,
            detail="hand_count must be 1 or 2.",
        )

    # Convert features to NumPy array
    features = np.asarray(
        request.features,
        dtype=np.float32
    ).reshape(1, -1)

    # Reject invalid numerical values
    if not np.isfinite(features).all():
        raise HTTPException(
            status_code=400,
            detail="Features contain NaN or Infinity.",
        )

    try:
        # Predict class probabilities
        probabilities = model.predict_proba(features)[0]

        # Find highest-probability prediction
        predicted_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_index]
        )

        # Validate predicted class index
        if predicted_index >= len(classes):
            raise HTTPException(
                status_code=500,
                detail="Model class index is invalid.",
            )

        # Get predicted character label
        label = str(classes[predicted_index])

        # Accept predictions at or above 50% confidence
        accepted = confidence >= 0.50

        return {
            "label": label,
            "confidence": confidence,
            "accepted": accepted,
            "model_version": metadata.get(
                "model_version",
                "unknown"
            ),
            "backend_version": "2.0",
        }

    except HTTPException:
        raise

    except Exception as error:
        print("Prediction error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        )


# ============================================================
# SERVER STARTUP
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )

