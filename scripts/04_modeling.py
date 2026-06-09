"""
04. 모델 학습 및 성능 비교
- 단일 모델: Linear Regression, Ridge, Decision Tree, KNN
- 앙상블 모델: Bagging, Random Forest, Gradient Boosting, XGBoost
- Global + Cluster별 학습
- RMSE/MAE/R² 비교
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, BaggingRegressor,
                               GradientBoostingRegressor)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

df = pd.read_pickle('data/final.pkl')
df = df[(df['days_until_next'] <= 365) & (df['visit_count'] > 1)].copy()

feature_cols = [
    'prev_gap', 'gap_avg', 'gap_std', 'gap_ma',
    'km_diff_avg', 'km_diff_std', 'km_diff_ma',
    'total_spend', 'avg_spend', 'max_spend', 'spend_ma',
    'visit_count', 'visit_freq_per_year', 'service_diversity',
    'n_services', 'vehicle_age', 'DayAvgDrivingKm',
    'visits_90d', 'spend_90d', 'visits_180d', 'spend_180d',
    'visits_365d', 'spend_365d',
    'last_service_month', 'in_month', 'in_dayofweek',
    'has_oil', 'has_brake', 'has_tire', 'has_battery', 'has_coolant',
    'oil_ratio', 'brake_ratio', 'tire_ratio', 'source_enc'
]

def prepare_data(data, features):
    X = data[features].copy()
    for c in X.columns:
        if X[c].dtype in ['float64', 'int64']:
            X[c] = X[c].fillna(X[c].median())
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        if X[c].dtype in ['float64', 'int64']:
            X[c] = X[c].fillna(X[c].median())
    return X, data['days_until_next'].values

def evaluate(name, X_tr, X_te, y_tr, y_te, model, use_scaled):
    if use_scaled:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
    else:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    return {'Model': name, 'RMSE': round(rmse, 2), 'MAE': round(mae, 2), 'R2': round(r2, 4)}

models = [
    ('Linear Regression', LinearRegression(), True),
    ('Ridge Regression', Ridge(alpha=1.0), True),
    ('Decision Tree', DecisionTreeRegressor(max_depth=10, random_state=42), False),
    ('KNN (k=15)', KNeighborsRegressor(n_neighbors=15), True),
    ('Bagging', BaggingRegressor(n_estimators=100, random_state=42, n_jobs=-1), False),
    ('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1), False),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42), False),
    ('XGBoost', xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1), False),
]

all_results = []

for label, data in [('Global', df)] + [(f'Cluster {c}', df[df['cluster'] == c]) for c in sorted(df['cluster'].unique())]:
    print(f"\n--- {label} (n={len(data)}) ---")
    X, y = prepare_data(data, feature_cols)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    for name, model, use_s in models:
        res = evaluate(name, X_tr_s if use_s else X_tr, X_te_s if use_s else X_te, y_tr, y_te, model, use_s)
        res['Group'] = label
        all_results.append(res)
        print(f"  {name:25s} | RMSE: {res['RMSE']:7.2f} | R²: {res['R2']:7.4f}")

results_df = pd.DataFrame(all_results)
pivot_rmse = results_df.pivot_table(index='Model', columns='Group', values='RMSE')
pivot_r2 = results_df.pivot_table(index='Model', columns='Group', values='R2')

print("\n\n=== RMSE 비교 ===")
print(pivot_rmse.round(2).to_string())
print("\n=== R² 비교 ===")
print(pivot_r2.round(4).to_string())

results_df.to_csv('results/model_comparison.csv', index=False)
print("\n저장: results/model_comparison.csv")
