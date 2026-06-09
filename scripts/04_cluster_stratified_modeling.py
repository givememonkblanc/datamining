"""
04-2. 군집별 맞춤 모델 검증
- Global XGBoost vs Cluster별 XGBoost 비교
- 동일한 cluster test split에서 성능 비교
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

FEATURE_COLS = [
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


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = data[FEATURE_COLS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
    return X, data['days_until_next'].values


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        'R2': round(r2_score(y_true, y_pred), 4),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        'MAE': round(mean_absolute_error(y_true, y_pred), 2),
    }


df = pd.read_pickle('data/final.pkl').copy()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

global_X_train, global_y_train = prepare_data(train_df)
global_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
global_model.fit(global_X_train, global_y_train)

summary_rows: list[dict[str, float | str | int]] = []

for cluster_id in sorted(df['cluster'].unique()):
    cluster_name = f'행태군 {chr(65 + cluster_id)}'
    cluster_train = train_df[train_df['cluster'] == cluster_id].copy()
    cluster_test = test_df[test_df['cluster'] == cluster_id].copy()

    X_train_local, y_train_local = prepare_data(cluster_train)
    X_test, y_test = prepare_data(cluster_test)

    local_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    local_model.fit(X_train_local, y_train_local)

    global_pred = global_model.predict(X_test)
    local_pred = local_model.predict(X_test)

    global_metrics = evaluate(y_test, global_pred)
    local_metrics = evaluate(y_test, local_pred)

    summary_rows.append({
        'Cluster': cluster_name,
        'TrainN': len(cluster_train),
        'TestN': len(cluster_test),
        'Global_R2': global_metrics['R2'],
        'Local_R2': local_metrics['R2'],
        'Global_RMSE': global_metrics['RMSE'],
        'Local_RMSE': local_metrics['RMSE'],
        'Global_MAE': global_metrics['MAE'],
        'Local_MAE': local_metrics['MAE'],
        'R2_Gain': round(local_metrics['R2'] - global_metrics['R2'], 4),
        'RMSE_Change': round(local_metrics['RMSE'] - global_metrics['RMSE'], 2),
    })

results_df = pd.DataFrame(summary_rows)
results_df.to_csv('results/cluster_xgboost_global_vs_local.csv', index=False)

print('=== Global vs Cluster-specific XGBoost ===')
print(results_df.to_string(index=False))
print('\n저장: results/cluster_xgboost_global_vs_local.csv')

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

x = np.arange(len(results_df))
width = 0.35

axes[0].bar(x - width / 2, results_df['Global_R2'], width, label='Global XGBoost', color='#2980b9')
axes[0].bar(x + width / 2, results_df['Local_R2'], width, label='Cluster XGBoost', color='#e67e22')
axes[0].axhline(0, color='#7f8c8d', linewidth=1, alpha=0.6)
axes[0].set_xticks(x)
axes[0].set_xticklabels(results_df['Cluster'])
axes[0].set_ylabel('R²')
axes[0].set_title('군집별 XGBoost: Global vs Local')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.25, axis='y')

axes[1].bar(x - width / 2, results_df['Global_RMSE'], width, label='Global XGBoost', color='#2980b9')
axes[1].bar(x + width / 2, results_df['Local_RMSE'], width, label='Cluster XGBoost', color='#e67e22')
axes[1].set_xticks(x)
axes[1].set_xticklabels(results_df['Cluster'])
axes[1].set_ylabel('RMSE (일)')
axes[1].set_title('군집별 XGBoost RMSE 비교')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.25, axis='y')

plt.tight_layout()
plt.savefig('/home/ryzen395/datamining/cluster_xgboost_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ cluster_xgboost_comparison.png')
