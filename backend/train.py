# We import everything we need to build and persist our training pipeline
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings('ignore')

# We resolve paths from environment variables so Docker can override them easily
DATA_PATH  = os.getenv('DATA_PATH',  '/app/data/raw/house_prices.csv')
MODEL_PATH = os.getenv('MODEL_PATH', '/app/backend/model.joblib')
META_PATH  = os.getenv('META_PATH',  '/app/backend/feature_names.json')


def load_data(path: str) -> pd.DataFrame:
    # We skip the leading blank row and treat '?' as NaN consistently
    df = pd.read_csv(path, na_values='?', skiprows=1)
    print(f"  Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def get_column_groups(df: pd.DataFrame):
    drop_cols  = {'Id', 'SalePrice'}
    feat_cols  = [c for c in df.columns if c not in drop_cols]
    num_cols   = df[feat_cols].select_dtypes(include='number').columns.tolist()
    cat_cols   = df[feat_cols].select_dtypes(include='object').columns.tolist()
    return feat_cols, num_cols, cat_cols


def build_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    # We use median imputation for numerics and constant fill for categoricals
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
    ])

    # We tune XGBoost conservatively to avoid overfitting on a ~1 000-row training set
    regressor = XGBRegressor(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    return Pipeline([('preprocessor', preprocessor), ('regressor', regressor)])


def aggregate_importances(model: Pipeline, num_cols: list, cat_cols: list) -> dict:
    # We roll up one-hot-encoded importances back to the original feature names
    preprocessor = model.named_steps['preprocessor']
    regressor     = model.named_steps['regressor']
    cat_feature_names = list(
        preprocessor.named_transformers_['cat']['encoder']
        .get_feature_names_out(cat_cols)
    )
    all_names = num_cols + cat_feature_names
    raw_imp   = regressor.feature_importances_

    aggregated = {}
    for col in num_cols:
        idx = all_names.index(col)
        aggregated[col] = float(raw_imp[idx])
    for col in cat_cols:
        aggregated[col] = float(
            sum(imp for name, imp in zip(all_names, raw_imp) if name.startswith(col + '_'))
        )

    return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))


def train():
    print("═══════════════════════════════════════")
    print(" HOU53-bot — Model Training")
    print("═══════════════════════════════════════")

    df = load_data(DATA_PATH)

    feat_cols, num_cols, cat_cols = get_column_groups(df)

    X = df[feat_cols]
    # We apply log1p to the target to make it more normally distributed
    y = np.log1p(df['SalePrice'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    pipeline = build_pipeline(num_cols, cat_cols)

    # We run a 5-fold cross-validation before the final fit
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1,
    )
    print(f"  CV RMSE (log): {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)

    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    print(f"  Test RMSE : ${rmse:>10,.0f}")
    print(f"  Test MAE  : ${mae:>10,.0f}")
    print(f"  Test R²   :  {r2:.4f}")

    importances = aggregate_importances(pipeline, num_cols, cat_cols)

    print("\n  Top 10 features:")
    for feat, imp in list(importances.items())[:10]:
        print(f"    {feat:<22} {imp:.4f}")

    # We persist the full pipeline so the backend can load it without retraining
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    joblib.dump({
        'pipeline':     pipeline,
        'num_cols':     num_cols,
        'cat_cols':     cat_cols,
        'feature_cols': feat_cols,
        'metrics': {
            'rmse':         rmse,
            'mae':          mae,
            'r2':           r2,
            'cv_rmse_mean': float(-cv_scores.mean()),
            'cv_rmse_std':  float(cv_scores.std()),
        },
    }, MODEL_PATH)

    with open(META_PATH, 'w') as f:
        json.dump({
            'feature_cols': feat_cols,
            'num_cols':     num_cols,
            'cat_cols':     cat_cols,
            'top_features': importances,
        }, f, indent=2)

    print(f"\n  Model  → {MODEL_PATH}")
    print(f"  Meta   → {META_PATH}")
    print("  Training complete!")


if __name__ == '__main__':
    train()
