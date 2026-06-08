"""
03. 주행거리 + 방문빈도 기반 K-Means 클러스터링
- 클러스터링용 전처리 규칙 강화
- log_drivingKm, visit_freq_per_year 기반 군집화
- 그룹별 특성 분석
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

df = pd.read_pickle('data/features.pkl').copy()

# 클러스터링용 전처리 규칙 강화
required_cols = ['drivingKm', 'visit_freq_per_year', 'vehicle_age', 'gap_ma', 'service_diversity', 'avg_spend', 'days_until_next']
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required_cols)
df = df[df['drivingKm'] > 100]
df = df[df['visit_freq_per_year'] > 0]
df = df[df['vehicle_age'].between(0, 30)]

km_upper = df['drivingKm'].quantile(0.995)
freq_upper = df['visit_freq_per_year'].quantile(0.995)
gap_upper = df['gap_ma'].quantile(0.995)

df['drivingKm'] = df['drivingKm'].clip(lower=100, upper=km_upper)
df['visit_freq_per_year'] = df['visit_freq_per_year'].clip(lower=0, upper=freq_upper)
df['gap_ma'] = df['gap_ma'].clip(lower=0, upper=gap_upper)

df['log_drivingKm'] = np.log1p(df['drivingKm'])
df['AnnualAvgDrivingKm'] = df['drivingKm'] / df['vehicle_age'].clip(lower=1)
df['DayAvgDrivingKm'] = df['AnnualAvgDrivingKm'] / 365.0
day_upper = df['DayAvgDrivingKm'].quantile(0.995)
df['DayAvgDrivingKm'] = df['DayAvgDrivingKm'].clip(lower=0, upper=day_upper)
df['log_DayAvgDrivingKm'] = np.log1p(df['DayAvgDrivingKm'])
df['log_AnnualAvgDrivingKm'] = np.log1p(df['AnnualAvgDrivingKm'])

# 클러스터링 입력 변수 구성 (연간 방문빈도 + 로그 주행거리)
df['cluster_visit_freq'] = df['visit_freq_per_year']

cluster_features = [
    'log_drivingKm',
    'cluster_visit_freq',
]

cluster_df = df[cluster_features].replace([np.inf, -np.inf], np.nan).copy()
for col in cluster_features:
    cluster_df[col] = cluster_df[col].fillna(cluster_df[col].median())

# K-Means 클러스터링
X = cluster_df.values
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_s)

label_map = {idx: f'행태군 {chr(65 + idx)}' for idx in range(k)}
df['cluster_label'] = df['cluster'].map(label_map)

# 클러스터별 특성 요약
summary = df.groupby('cluster').agg(
    count=('days_until_next','count'),
    label=('cluster_label', 'first'),
    km=('drivingKm','mean'),
    day_km=('DayAvgDrivingKm','mean'),
    visit_freq=('visit_freq_per_year','mean'),
    gap_ma=('gap_ma','mean'),
    service_diversity=('service_diversity','mean'),
    avg_spend=('avg_spend','mean'),
    target=('days_until_next','mean')
).round(1)

print("=== 클러스터별 특성 ===")
print(summary.to_string())
print(f"\n총: {len(df)}행, {df['carNo'].nunique()}대 차량")
print(f"전처리 상한값 → drivingKm: {km_upper:.1f}, visit_freq_per_year: {freq_upper:.2f}, gap_ma: {gap_upper:.1f}, DayAvgDrivingKm: {day_upper:.1f}")

df.to_pickle('data/final.pkl')
print("저장: data/final.pkl")
