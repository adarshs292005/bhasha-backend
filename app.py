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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models" / "v2"

MODEL_PATH = MODEL_DIR / "bhasha_character_rf_v2.joblib"
CLASSES_PATH = MODEL_DIR / "bhasha_character_classes_v2.json"
METADATA_PATH = MODEL_DIR / "bhasha_character_model_metadata_v2.json"


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not CLASSES_PATH.exists():
    raise FileNotFoundError(
        f"Classes file not found: {CLASSES_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_PATH}"
    )


model = joblib.load(MODEL_PATH)


with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    raw_classes = json.load(file)


with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)


# ============================================================
# CLASS MAPPING
# ============================================================

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
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Bhasha ML Backend",
    description="Bhasha Indian Sign Language character recognition API",
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

frontend_url = os.environ.get("FRONTEND_URL")

if frontend_url:
    allowed_origins.append(
        frontend_url.rstrip("/")
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
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
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Bhasha ML Backend",
        "status": "online",
        "backend_version": "2.0",
        "model_version": metadata.get(
            "model_version",
            "unknown",
        ),
        "features": 126,
        "classes": len(classes),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/v2/health")
def health():
    return {
        "status": "healthy",
        "backend_version": "2.0",
        "model_loaded": model is not None,
        "model_version": metadata.get(
            "model_version",
            "unknown",
        ),
        "features": 126,
        "classes": len(classes),
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/api/v2/model-info")
def model_info():
    return {
        "backend_version": "2.0",
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
# PREDICTION
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

    features = np.asarray(
        request.features,
        dtype=np.float32,
    ).reshape(1, -1)

    if not np.isfinite(features).all():
        raise HTTPException(
            status_code=400,
            detail="Features contain NaN or Infinity.",
        )

    try:
        probabilities = model.predict_proba(
            features
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_index]
        )

        if predicted_index >= len(classes):
            raise HTTPException(
                status_code=500,
                detail="Model class index is invalid.",
            )

        label = str(
            classes[predicted_index]
        )

        accepted = confidence >= 0.50

        return {
            "label": label,
            "confidence": confidence,
            "accepted": accepted,
            "model_version": metadata.get(
                "model_version",
                "unknown",
            ),
            "backend_version": "2.0",
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            "Prediction error:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            "8000",
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )