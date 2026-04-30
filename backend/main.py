# We create the FastAPI application that serves house-price predictions
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    HouseFeatures,
    TextPredictionRequest,
    PredictionResponse,
    HealthResponse,
)
from predictor import HousePredictor
from nlp_extractor import NLPExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We instantiate our core objects at module level so they are shared across requests
predictor = HousePredictor()
extractor = NLPExtractor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # We warm up the predictor at startup to avoid a cold-load delay on the first request
    logger.info("Warming up model…")
    try:
        predictor._load()
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
    yield


app = FastAPI(
    title="HOU53-bot API",
    description=(
        "House price prediction API for the Ames, Iowa dataset. "
        "Accepts either structured feature inputs or a natural-language description."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# We enable CORS so the React frontend can reach this API from any origin in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────
# We define the main prediction endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    # We expose model metrics alongside the status so callers can verify model quality
    try:
        return predictor.health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_from_features(features: HouseFeatures):
    # We convert the Pydantic model to a plain dict with original column names
    try:
        feature_dict = features.to_feature_dict()
        result = predictor.predict(feature_dict)
        return PredictionResponse(**result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please run train.py before starting the API.",
        )
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/text", response_model=PredictionResponse, tags=["Predictions"])
async def predict_from_text(request: TextPredictionRequest):
    # We parse the free-text description into structured features, then predict
    try:
        extracted = extractor.extract(request.description)
        result = predictor.predict(extracted)
        # We attach the extracted features so the frontend can display them
        result['extracted_features'] = {
            k: v for k, v in extracted.items()
            if v is not None and k in (
                'BedroomAbvGr', 'FullBath', 'HalfBath', 'GrLivArea',
                'YearBuilt', 'YearRemodAdd', 'GarageCars', 'GarageArea',
                'TotalBsmtSF', 'Fireplaces', 'PoolArea', 'OverallQual',
                'CentralAir', 'Neighborhood', 'BldgType', 'HouseStyle',
                'Foundation', 'LotArea', 'WoodDeckSF', 'OpenPorchSF',
                '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1',
            )
        }
        return PredictionResponse(**result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please run train.py before starting the API.",
        )
    except Exception as exc:
        logger.exception("Text prediction error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/features", tags=["Meta"])
async def list_features():
    # We return the feature schema so the frontend can build dynamic forms
    try:
        predictor._load()
        return {
            "feature_columns": predictor._meta.get("feature_cols", []),
            "numerical":       predictor._meta.get("num_cols", []),
            "categorical":     predictor._meta.get("cat_cols", []),
            "top_features":    predictor._meta.get("top_features", {}),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
