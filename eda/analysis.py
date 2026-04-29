# We import all the libraries we need for our analysis
import os
import sys
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings('ignore')

# We define paths relative to the project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'house_prices.csv')
PLOTS_DIR = os.path.join(ROOT_DIR, 'eda', 'plots')
MODEL_DIR = os.path.join(ROOT_DIR, 'backend')

os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# We load and inspect the dataset
# ─────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    # We read the CSV skipping the leading blank line and treating '?' as NaN
    df = pd.read_csv(path, na_values='?', skiprows=1)
    print(f"  Dataset shape: {df.shape}")
    return df


def eda_overview(df: pd.DataFrame) -> None:
    print("\n── 1. General overview ──────────────────────────────")
    print(f"  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    print(f"  Target (SalePrice) — min: ${df['SalePrice'].min():,.0f}  "
          f"mean: ${df['SalePrice'].mean():,.0f}  max: ${df['SalePrice'].max():,.0f}")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\n  Columns with missing values ({len(missing)} total):")
    for col, cnt in missing.items():
        pct = cnt / len(df) * 100
        print(f"    {col:<20} {cnt:>5}  ({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────
# We generate and save the main EDA plots
# ─────────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('SalePrice Distribution', fontsize=14, fontweight='bold')

    axes[0].hist(df['SalePrice'], bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
    axes[0].set_title('Original Scale')
    axes[0].set_xlabel('SalePrice ($)')
    axes[0].set_ylabel('Count')

    axes[1].hist(np.log1p(df['SalePrice']), bins=60, color='darkorange', edgecolor='white', linewidth=0.4)
    axes[1].set_title('Log-Transformed')
    axes[1].set_xlabel('log(SalePrice + 1)')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'target_distribution.png'), dpi=120)
    plt.close()
    print("  Saved: target_distribution.png")


def plot_numeric_correlations(df: pd.DataFrame) -> None:
    # We pick the 15 numerical features most correlated with SalePrice
    num_df = df.select_dtypes(include='number').drop(columns=['Id'])
    corr = num_df.corr()['SalePrice'].drop('SalePrice').abs().sort_values(ascending=False)
    top_feats = corr.head(15).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_feats)))
    ax.barh(top_feats[::-1], corr[top_feats[::-1]], color=colors)
    ax.set_title('Top 15 Numerical Features — Pearson |r| with SalePrice', fontweight='bold')
    ax.set_xlabel('|Pearson correlation|')
    ax.axvline(0, color='grey', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'numeric_correlations.png'), dpi=120)
    plt.close()
    print("  Saved: numeric_correlations.png")


def plot_key_scatter(df: pd.DataFrame) -> None:
    # We visualise the four features with the strongest linear relationship to price
    features = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, feat in zip(axes, features):
        ax.scatter(df[feat], df['SalePrice'], alpha=0.35, s=14, color='steelblue')
        ax.set_xlabel(feat)
        ax.set_ylabel('SalePrice ($)')
        ax.set_title(f'{feat} vs SalePrice')

    fig.suptitle('Key Feature Relationships with SalePrice', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'key_scatter.png'), dpi=120)
    plt.close()
    print("  Saved: key_scatter.png")


def plot_neighborhood_price(df: pd.DataFrame) -> None:
    # We show median sale price per neighbourhood to spot location effects
    nbhd = (df.groupby('Neighborhood')['SalePrice']
              .median()
              .sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(nbhd)))
    ax.bar(nbhd.index, nbhd.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title('Median SalePrice by Neighbourhood', fontsize=13, fontweight='bold')
    ax.set_xlabel('Neighbourhood')
    ax.set_ylabel('Median SalePrice ($)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'neighborhood_price.png'), dpi=120)
    plt.close()
    print("  Saved: neighborhood_price.png")


def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(missing.index[::-1], missing.values[::-1], color='salmon', edgecolor='white')
    ax.set_title('Missing Values per Column', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of missing values')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'missing_values.png'), dpi=120)
    plt.close()
    print("  Saved: missing_values.png")


def plot_feature_importance(importances: dict, save_path: str) -> None:
    # We display the top 20 features ranked by XGBoost gain importance
    top = dict(list(importances.items())[:20])
    feats = list(top.keys())[::-1]
    vals = list(top.values())[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(feats)))
    ax.barh(feats, vals, color=colors)
    ax.set_title('Top 20 Features — XGBoost Importance (gain)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ─────────────────────────────────────────────────────────────────
# We build the preprocessing pipeline and train the XGBoost model
# ─────────────────────────────────────────────────────────────────

def get_column_groups(df: pd.DataFrame):
    drop_cols = {'Id', 'SalePrice'}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    num_cols = df[feature_cols].select_dtypes(include='number').columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include='object').columns.tolist()
    return feature_cols, num_cols, cat_cols


def build_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    # We chain imputation → scaling for numeric and imputation → encoding for categorical
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
    ])

    # We use XGBoost with carefully tuned hyper-parameters for this dataset size
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


def evaluate(model, X_train, X_test, y_train, y_test, feature_cols, num_cols, cat_cols):
    print("\n── 3. Model evaluation ──────────────────────────────")

    # We run 5-fold CV on the training set before touching the test set
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf,
                                scoring='neg_root_mean_squared_error', n_jobs=-1)
    print(f"  5-fold CV RMSE (log space): {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"  Test RMSE : ${rmse:>10,.0f}")
    print(f"  Test MAE  : ${mae:>10,.0f}")
    print(f"  Test R²   :  {r2:.4f}")

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'cv_rmse_mean': float(-cv_scores.mean()),
        'cv_rmse_std': float(cv_scores.std()),
    }


def aggregate_feature_importances(model, num_cols, cat_cols):
    # We group one-hot columns back to their original categorical name
    preprocessor = model.named_steps['preprocessor']
    regressor     = model.named_steps['regressor']

    cat_names = list(
        preprocessor.named_transformers_['cat']['encoder']
        .get_feature_names_out(cat_cols)
    )
    all_names = num_cols + cat_names
    raw_imp   = regressor.feature_importances_

    aggregated = {}
    for col in num_cols:
        idx = all_names.index(col)
        aggregated[col] = float(raw_imp[idx])

    for col in cat_cols:
        prefix = col + '_'
        total  = sum(imp for name, imp in zip(all_names, raw_imp) if name.startswith(prefix))
        aggregated[col] = float(total)

    return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))


def train_and_save(df: pd.DataFrame):
    print("\n── 2. Training ─────────────────────────────────────")

    feature_cols, num_cols, cat_cols = get_column_groups(df)

    X = df[feature_cols]
    # We log-transform the target to reduce skewness and improve model performance
    y = np.log1p(df['SalePrice'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    pipeline = build_pipeline(num_cols, cat_cols)

    metrics = evaluate(pipeline, X_train, X_test, y_train, y_test,
                       feature_cols, num_cols, cat_cols)

    importances = aggregate_feature_importances(pipeline, num_cols, cat_cols)

    print("\n  Top 10 features by importance:")
    for feat, imp in list(importances.items())[:10]:
        print(f"    {feat:<22} {imp:.4f}")

    # We save the trained pipeline and associated metadata for the backend
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'model.joblib')
    meta_path  = os.path.join(MODEL_DIR, 'feature_names.json')

    joblib.dump({
        'pipeline':     pipeline,
        'num_cols':     num_cols,
        'cat_cols':     cat_cols,
        'feature_cols': feature_cols,
        'metrics':      metrics,
    }, model_path)
    print(f"\n  Model saved → {model_path}")

    with open(meta_path, 'w') as f:
        json.dump({
            'feature_cols': feature_cols,
            'num_cols':     num_cols,
            'cat_cols':     cat_cols,
            'top_features': importances,
        }, f, indent=2)
    print(f"  Metadata saved → {meta_path}")

    return pipeline, importances


# ─────────────────────────────────────────────────────────────────
# We run everything end-to-end
# ─────────────────────────────────────────────────────────────────

def main():
    print("═══════════════════════════════════════════════════")
    print("  HOU53-bot — EDA & Model Training")
    print("═══════════════════════════════════════════════════")

    print("\n── 0. Loading data ─────────────────────────────────")
    df = load_data(DATA_PATH)

    eda_overview(df)

    print("\n── Generating EDA plots ────────────────────────────")
    plot_target_distribution(df)
    plot_numeric_correlations(df)
    plot_key_scatter(df)
    plot_neighborhood_price(df)
    plot_missing_values(df)

    pipeline, importances = train_and_save(df)

    imp_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
    plot_feature_importance(importances, imp_path)

    print("\n═══════════════════════════════════════════════════")
    print("  Done! All outputs saved.")
    print("═══════════════════════════════════════════════════")


if __name__ == '__main__':
    main()
