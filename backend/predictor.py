# We load the trained pipeline once and reuse it across all prediction requests
import os
import json
from typing import Any

import numpy as np
import pandas as pd
import joblib

MODEL_PATH = os.getenv('MODEL_PATH', '/app/backend/model.joblib')
META_PATH  = os.getenv('META_PATH',  '/app/backend/feature_names.json')

# We keep concise human-readable descriptions for the most influential features
FEATURE_DESCRIPTIONS: dict[str, str] = {
    'OverallQual':   'Overall quality of materials and finishes (1–10)',
    'GrLivArea':     'Above-ground living area in square feet',
    'GarageArea':    'Garage size in square feet',
    'GarageCars':    'Garage capacity in number of cars',
    'TotalBsmtSF':   'Total basement area in square feet',
    'YearBuilt':     'Year the house was originally built',
    'YearRemodAdd':  'Year of the most recent remodel',
    '1stFlrSF':      'First-floor area in square feet',
    'FullBath':      'Number of full bathrooms above grade',
    'TotRmsAbvGrd':  'Total rooms above grade (excluding bathrooms)',
    'Fireplaces':    'Number of fireplaces',
    'LotArea':       'Lot size in square feet',
    'Neighborhood':  'Physical location within Ames city limits',
    'ExterQual':     'Quality of the exterior material',
    'KitchenQual':   'Kitchen quality rating',
    'BsmtFinSF1':    'Finished basement area (type 1) in square feet',
    'MasVnrArea':    'Masonry veneer area in square feet',
    'OpenPorchSF':   'Open porch area in square feet',
    'WoodDeckSF':    'Wood deck area in square feet',
    'BedroomAbvGr':  'Number of bedrooms above basement level',
    'HalfBath':      'Number of half bathrooms above grade',
    'LotFrontage':   'Linear feet of street connected to the property',
    'OverallCond':   'Overall condition rating (1–10)',
    'MSSubClass':    'Building class',
    'MSZoning':      'General zoning classification',
    'BldgType':      'Type of dwelling',
    'HouseStyle':    'Style of dwelling',
    'Foundation':    'Type of foundation',
    'CentralAir':    'Central air conditioning (Y/N)',
    'GarageType':    'Garage location type',
    'GarageFinish':  'Interior finish of the garage',
    'SaleType':      'Type of sale',
    'SaleCondition': 'Condition of sale',
}

# We map quality codes to a friendly label for display purposes
QUALITY_LABELS: dict[str, str] = {
    'Ex': 'Excellent', 'Gd': 'Good', 'TA': 'Average',
    'Fa': 'Fair',       'Po': 'Poor', 'Missing': 'Not present',
}


class HousePredictor:
    def __init__(self) -> None:
        self._model_data: dict | None = None
        self._meta: dict | None = None

    # We use lazy loading so the model is only read from disk on the first request
    def _load(self) -> None:
        if self._model_data is None:
            self._model_data = joblib.load(MODEL_PATH)
            with open(META_PATH) as f:
                self._meta = json.load(f)

    def predict(self, features: dict[str, Any]) -> dict:
        self._load()

        pipeline     = self._model_data['pipeline']
        feature_cols = self._model_data['feature_cols']

        # We build a one-row DataFrame aligned with the training column order
        row = {col: features.get(col, np.nan) for col in feature_cols}
        df  = pd.DataFrame([row])

        log_price = float(pipeline.predict(df)[0])
        price     = float(np.expm1(log_price))

        explanation = self._build_explanation(features)

        return {
            'predicted_price': round(price, 2),
            'price_range': {
                'low':  round(price * 0.88, 2),
                'high': round(price * 1.12, 2),
            },
            'explanation':    explanation,
            'model_metrics':  self._model_data.get('metrics', {}),
        }

    def _build_explanation(self, features: dict[str, Any]) -> list[dict]:
        # We return a ranked list of the features that matter most for this input
        top_features = self._meta.get('top_features', {})
        result = []

        for feat, importance in list(top_features.items())[:20]:
            raw_val = features.get(feat)
            if raw_val is None:
                continue

            display_val = (
                QUALITY_LABELS.get(str(raw_val), str(raw_val))
                if feat not in ('GrLivArea', 'LotArea', 'GarageArea',
                                'TotalBsmtSF', '1stFlrSF', 'BsmtFinSF1')
                else f"{int(raw_val):,} sq ft"
                if isinstance(raw_val, (int, float)) and not np.isnan(raw_val)
                else str(raw_val)
            )

            result.append({
                'feature':     feat,
                'value':       raw_val if not (isinstance(raw_val, float) and np.isnan(raw_val)) else None,
                'display':     display_val,
                'importance':  round(importance, 4),
                'description': FEATURE_DESCRIPTIONS.get(feat, feat),
            })

        return result[:10]

    def health(self) -> dict:
        self._load()
        return {
            'status':  'ok',
            'metrics': self._model_data.get('metrics', {}),
        }
