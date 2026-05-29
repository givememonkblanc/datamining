"""
03. 이상치 처리 및 K-Means 클러스터링
- 이상치 제거 (DayAvgDrivingKm, drivingKm)
- log_drivingKm, log_DayAvgDrivingKm 기반 군집화
- 그룹별 특성 분석
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

df = pd.read_pickle('data/features.pkl')

# 이상치 처리
df = df[df['DayAvgDrivingKm'] > 0]
df = df[df['drivingKm'] > 100]

km_999 = df['drivingKm'].quantile(0.999)
df['drivingKm'] = df['drivingKm'].clip(upper=km_999)
df['log_drivingKm'] = np.log1p(df['drivingKm'])

df['AnnualAvgDrivingKm'] = df['drivingKm'] / df['vehicle_age'].clip(lower=1)
df['DayAvgDrivingKm'] = df['AnnualAvgDrivingKm'] / 365.0

day_99 = df['DayAvgDrivingKm'].quantile(0.99)
df['DayAvgDrivingKm'] = df['DayAvgDrivingKm'].clip(upper=day_99)
df['log_DayAvgDrivingKm'] = np.log1p(df['DayAvgDrivingKm'])
df['log_AnnualAvgDrivingKm'] = np.log1p(df['AnnualAvgDrivingKm'])
df = df[df['vehicle_age'] <= 30]

# K-Means 클러스터링
X = df[['log_drivingKm', 'log_DayAvgDrivingKm']].values
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_s)

# 클러스터별 특성 요약
summary = df.groupby('cluster').agg(
    count=('days_until_next','count'),
    km=('drivingKm','mean'),
    day_km=('DayAvgDrivingKm','mean'),
    age=('vehicle_age','mean'),
    freq=('visit_freq_per_year','mean'),
    target=('days_until_next','mean'),
    n_services=('n_services','mean')
).round(1)

print("=== 클러스터별 특성 ===")
print(summary.to_string())
print(f"\n총: {len(df)}행, {df['carNo'].nunique()}대 차량")

df.to_pickle('data/final.pkl')
print("저장: data/final.pkl")
