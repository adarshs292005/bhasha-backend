import json
import os
import warnings
from pathlib import Path
from typing import List

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Character model paths
MODEL_DIR = BASE_DIR / "models" / "v2"

MODEL_PATH = MODEL_DIR / "bhasha_character_rf_v2.joblib"
CLASSES_PATH = MODEL_DIR / "bhasha_character_classes_v2.json"
METADATA_PATH = MODEL_DIR / "bhasha_character_model_metadata_v2.json"

# Word model paths
WORD_MODEL_DIR = BASE_DIR / "models" / "word"

WORD_MODEL_PATH = WORD_MODEL_DIR / "bhasha_word_rf.joblib"
WORD_CLASSES_PATH = WORD_MODEL_DIR / "word_classes.json"


# ============================================================
# VALIDATE REQUIRED FILES
# ============================================================

character_files = [
    MODEL_PATH,
    CLASSES_PATH,
    METADATA_PATH,
]

for file_path in character_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required character model file not found: {file_path}"
        )

word_files = [
    WORD_MODEL_PATH,
    WORD_CLASSES_PATH,
]

for file_path in word_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required word model file not found: {file_path}"
        )


# ============================================================
# LOAD CHARACTER MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    raw_classes = json.load(file)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

if isinstance(raw_classes, dict):
    classes = [
        label
        for label, index in sorted(
            raw_classes.items(),
            key=lambda item: item[1],
        )
    ]
else:
    classes = raw_classes


# ============================================================
# LOAD WORD MODEL
# ============================================================

# The model was trained with scikit-learn 1.6.1.
# A warning may appear if your local version differs.
word_model = joblib.load(WORD_MODEL_PATH)

with open(WORD_CLASSES_PATH, "r", encoding="utf-8") as file:
    word_classes = json.load(file)

if not isinstance(word_classes, list):
    raise ValueError(
        "word_classes.json must contain a JSON list of labels."
    )

if len(word_classes) != len(word_model.classes_):
    raise ValueError(
        "Word model class count does not match word_classes.json."
    )

# Verify the model's class ordering matches the label file.
model_word_classes = [
    str(label) for label in word_model.classes_
]

if model_word_classes != word_classes:
    raise ValueError(
        "Word model class ordering does not match word_classes.json. "
        "Check the saved class labels before using the word endpoint."
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Bhasha ML Backend",
    description=(
        "Bhasha sign-language character and word recognition API"
    ),
    version="2.1.0",
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
        max_length=126,
    )

    hand_count: int = Field(
        ...,
        ge=1,
        le=2,
    )


# ============================================================
# SHARED PREDICTION HELPER
# ============================================================

def predict_with_model(
    selected_model,
    feature_values: List[float],
    selected_classes: List[str],
    confidence_threshold: float,
):
    """Run prediction with a selected model."""

    features = np.asarray(
        feature_values,
        dtype=np.float32,
    ).reshape(1, -1)

    if not np.isfinite(features).all():
        raise HTTPException(
            status_code=400,
            detail="Features contain NaN or Infinity.",
        )

    try:
        probabilities = selected_model.predict_proba(features)[0]

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        if predicted_index >= len(selected_classes):
            raise HTTPException(
                status_code=500,
                detail="Model class index is invalid.",
            )

        return {
            "label": str(selected_classes[predicted_index]),
            "confidence": confidence,
            "accepted": confidence >= confidence_threshold,
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
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Bhasha ML Backend",
        "status": "online",
        "backend_version": "2.1",
        "character_model_loaded": model is not None,
        "word_model_loaded": word_model is not None,
        "character_classes": len(classes),
        "word_classes": len(word_classes),
        "features": 126,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/v2/health")
def health():
    return {
        "status": "healthy",
        "backend_version": "2.1",
        "model_loaded": model is not None,
        "word_model_loaded": word_model is not None,
        "model_version": metadata.get(
            "model_version",
            "unknown",
        ),
        "character_classes": len(classes),
        "word_classes": len(word_classes),
        "features": 126,
    }


# ============================================================
# CHARACTER MODEL INFORMATION
# ============================================================

@app.get("/api/v2/model-info")
def model_info():
    return {
        "backend_version": "2.1",
        "model_version": metadata.get(
            "model_version",
            "unknown",
        ),
        "model_type": metadata.get(
            "model_type",
            "RandomForestClassifier",
        ),
        "features": 126,
        "classes": len(classes),
        "class_names": classes,
    }


# ============================================================
# WORD MODEL INFORMATION
# ============================================================

@app.get("/api/v2/word-model-info")
def word_model_info():
    return {
        "model_type": type(word_model).__name__,
        "features": 126,
        "classes": len(word_classes),
        "class_names": word_classes,
        "feature_normalization": (
            "Wrist origin; scale by maximum XY-plane distance "
            "from wrist; left hand followed by right hand."
        ),
    }


# ============================================================
# CHARACTER PREDICTION
# EXISTING ENDPOINT — KEEP BEHAVIOR UNCHANGED
# ============================================================

@app.post("/api/v2/predict")
def predict(request: PredictionRequest):
    if len(request.features) != 126:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected 126 features, "
                f"received {len(request.features)}."
            ),
        )

    if request.hand_count not in (1, 2):
        raise HTTPException(
            status_code=400,
            detail="hand_count must be 1 or 2.",
        )

    result = predict_with_model(
        selected_model=model,
        feature_values=request.features,
        selected_classes=classes,
        confidence_threshold=0.50,
    )

    return {
        **result,
        "model_version": metadata.get(
            "model_version",
            "unknown",
        ),
        "backend_version": "2.1",
    }


# ============================================================
# WORD PREDICTION
# NEW ENDPOINT
# ============================================================

@app.post("/api/v2/predict-word")
def predict_word(request: PredictionRequest):
    if len(request.features) != 126:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected 126 features, "
                f"received {len(request.features)}."
            ),
        )

    if request.hand_count not in (1, 2):
        raise HTTPException(
            status_code=400,
            detail="hand_count must be 1 or 2.",
        )

    # IMPORTANT:
    # This endpoint expects features prepared using the word model's
    # training normalization, not the current character normalizer.
    result = predict_with_model(
        selected_model=word_model,
        feature_values=request.features,
        selected_classes=word_classes,
        confidence_threshold=0.50,
    )

    return {
        **result,
        "backend_version": "2.1",
        "model_type": "word",
    }


# ============================================================
# SERVER STARTUP
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
    )

