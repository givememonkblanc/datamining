"""
07. Level 2 (차량 단위) 모든 모델 성능 비교 재검증
- Linear Regression, Ridge, Decision Tree, KNN
- Bagging, Random Forest, Gradient Boosting, XGBoost
- Feature Set [C] 기준: gap_ma 포함
- RMSE/MAE/R² 비교 → 어떤 모델이 MAE 최적인지 확인
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

df = pd.read_pickle('/tmp/df_final.pkl')

# 차량 단위 집계 (Level 2, Feature Set [C])
car_features = df.groupby('carNo').agg(
    drivingKm=('drivingKm','mean'),
    log_drivingKm=('log_drivingKm','mean'),
    DayAvgDrivingKm=('DayAvgDrivingKm','mean'),
    vehicle_age=('vehicle_age','first'),
    cluster=('cluster','first'),
    avg_spend=('avg_spend','mean'),
    max_spend=('max_spend','max'),
    total_spend=('total_spend','sum'),
    n_services=('n_services','mean'),
    service_diversity=('service_diversity','mean'),
    oil_ratio=('oil_ratio','mean'),
    brake_ratio=('brake_ratio','mean'),
    tire_ratio=('tire_ratio','mean'),
    visits_90d=('visits_90d','sum'),
    visits_180d=('visits_180d','sum'),
    visits_365d=('visits_365d','sum'),
    gap_avg=('gap_avg','mean'),
    gap_std=('gap_std','mean'),
    gap_ma=('gap_ma','mean'),
    km_diff_avg=('km_diff_avg','mean'),
    spend_ma=('spend_ma','mean'),
    visit_count=('visit_count','max'),
    visit_freq_per_year=('visit_freq_per_year','mean'),
    target=('days_until_next','mean')
).reset_index()

# 3회↑ 방문 차량만
car_visit_counts = df.groupby('carNo').size().reset_index(name='n_visits')
eligible = car_visit_counts[car_visit_counts['n_visits'] >= 3]['carNo']
car_features = car_features[car_features['carNo'].isin(eligible)].copy()

print(f"Level 2 대상 차량: {len(car_features):,}대 (3회↑ 방문)")

feature_cols = [
    'drivingKm','log_drivingKm','DayAvgDrivingKm','vehicle_age','cluster',
    'avg_spend','max_spend','total_spend','n_services','service_diversity',
    'oil_ratio','brake_ratio','tire_ratio',
    'visits_90d','visits_180d','visits_365d',
    'gap_avg','gap_std','gap_ma','km_diff_avg','spend_ma',
    'visit_count','visit_freq_per_year'
]

X = car_features[feature_cols].copy()
for c in X.columns:
    if X[c].dtype in ['float64','int64']:
        X[c] = X[c].fillna(X[c].median())
X = X.replace([np.inf, -np.inf], np.nan)
for c in X.columns:
    if X[c].dtype in ['float64','int64']:
        X[c] = X[c].fillna(X[c].median())
y = car_features['target'].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

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

print(f"\n{'모델':25s} | {'RMSE':>8s} | {'MAE':>8s} | {'R²':>8s}")
print('-' * 55)

results = []
for name, model, use_s in models:
    X_tr_f = X_tr_s if use_s else X_tr
    X_te_f = X_te_s if use_s else X_te
    model.fit(X_tr_f, y_tr)
    y_pred = model.predict(X_te_f)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    print(f"{name:25s} | {rmse:8.2f} | {mae:8.2f} | {r2:8.4f}")

# Sort by MAE
results_df = pd.DataFrame(results).sort_values('MAE')
print(f"\n--- MAE 기준 정렬 ---")
print(f"{'순위':5s} {'모델':25s} | {'RMSE':>8s} | {'MAE':>8s} | {'R²':>8s}")
print('-' * 55)
for i, (_, row) in enumerate(results_df.iterrows(), 1):
    print(f"{i:5d} {row['Model']:25s} | {row['RMSE']:8.2f} | {row['MAE']:8.2f} | {row['R2']:8.4f}")

results_df.to_csv('results/level2_comparison.csv', index=False)
print(f"\n저장: results/level2_comparison.csv")
