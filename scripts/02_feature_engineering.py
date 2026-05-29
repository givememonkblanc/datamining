"""
02. 파생변수 생성
- 차대번호 10번째 자리 → 생산년도 매핑
- 연간/일평균 주행거리
- 정비이력 이동통계 (이동평균, 이동표준편차)
- 주행거리/금액 이동통계
"""
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

df = pd.read_pickle('data/visits.pkl')

# =============================================
# 1) 차대번호 파생변수
# =============================================
for i in range(11):
    df[f'chassis_{i+1}'] = df['chassisNo'].astype(str).str[i]

# 생산년도 매핑 (VIN 10번째 자리)
def resolve_prod_year(row):
    char = str(row['chassisNo'])[9] if len(str(row['chassisNo'])) > 9 else '0'
    char = char.upper()
    service_year = row['inDay'].year
    
    codes80 = {'A':1980,'B':1981,'C':1982,'D':1983,'E':1984,'F':1985,
               'G':1986,'H':1987,'J':1988,'K':1989,'L':1990,'M':1991,
               'N':1992,'P':1993,'R':1994,'S':1995,'T':1996,'V':1997,
               'W':1998,'X':1999,'Y':2000}
    codes00 = {'1':2001,'2':2002,'3':2003,'4':2004,'5':2005,
               '6':2006,'7':2007,'8':2008,'9':2009}
    codes10 = {'A':2010,'B':2011,'C':2012,'D':2013,'E':2014,'F':2015,
               'G':2016,'H':2017,'J':2018,'K':2019,'L':2020,'M':2021,
               'N':2022,'P':2023,'R':2024,'S':2025,'T':2026,'V':2027,
               'W':2028,'X':2029,'Y':2030}
    
    if char in codes00:
        return codes00[char]
    y80 = codes80.get(char, 0)
    y10 = codes10.get(char, 0)
    candidates = [y for y in [y80, y10] if 0 < y <= service_year and service_year - y <= 35]
    return max(candidates) if candidates else max(1, service_year - 5)

df['prod_year'] = df.apply(resolve_prod_year, axis=1)
df['vehicle_age'] = (df['inDay'].dt.year - df['prod_year']).clip(lower=0)

# =============================================
# 2) 주행거리 파생변수
# =============================================
df['AnnualAvgDrivingKm'] = df['drivingKm'] / df['vehicle_age'].clip(lower=1)
df['DayAvgDrivingKm'] = df['AnnualAvgDrivingKm'] / 365.0

# 로그 변환 (클러스터링 용)
for col in ['DayAvgDrivingKm', 'drivingKm', 'AnnualAvgDrivingKm']:
    df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    df[f'log_{col}'] = df[f'log_{col}'].replace([np.inf, -np.inf], np.nan).fillna(0)

# =============================================
# 3) 정비이력/금액 이동통계 (차량별)
# =============================================
all_feat = []

for car_no, grp in df.groupby('carNo', sort=False):
    grp = grp.sort_values('inDay').reset_index(drop=True)
    
    gaps, km_diffs, amts, all_services = [], [], [], []
    
    for idx, row in grp.iterrows():
        feat = {
            'carNo': car_no,
            'inDay': row['inDay'],
            'days_until_next': row['days_until_next'],
            'drivingKm': row['drivingKm'],
            'repaiAmt': row['repaiAmt'],
            'n_services': row['n_services'],
            'prod_year': row['prod_year'],
            'vehicle_age': row['vehicle_age'],
            'DayAvgDrivingKm': row['DayAvgDrivingKm'],
            'log_DayAvgDrivingKm': row['log_DayAvgDrivingKm'],
            'log_drivingKm': row['log_drivingKm'],
            'AnnualAvgDrivingKm': row['AnnualAvgDrivingKm'],
            'log_AnnualAvgDrivingKm': row['log_AnnualAvgDrivingKm'],
            'source_enc': 1 if row['source'] == 'KBEAD' else 0,
            'in_month': row['inDay'].month,
            'in_quarter': row['inDay'].quarter,
            'in_dayofweek': row['inDay'].dayofweek,
            'last_service_month': row['inDay'].month,
        }
        
        services = row['service_items']
        feat['has_oil'] = int(any('오일' in s for s in services))
        feat['has_brake'] = int(any('브레이크' in s or s == '제동' for s in services))
        feat['has_tire'] = int(any('타이어' in s for s in services))
        feat['has_battery'] = int(any('배터리' in s for s in services))
        feat['has_coolant'] = int(any('부동액' in s for s in services))
        feat['visit_count'] = idx + 1
        
        if idx > 0:
            last = grp.iloc[idx-1]
            gap = (row['inDay'] - last['inDay']).days
            gaps.append(gap)
            
            feat['prev_gap'] = gap
            feat['gap_avg'] = np.mean(gaps)
            feat['gap_std'] = np.std(gaps) if len(gaps) > 1 else 0
            feat['gap_ma'] = np.mean(gaps[-3:])
            
            km_diff = max(0, row['drivingKm'] - last['drivingKm']) if row['drivingKm'] > 0 and last['drivingKm'] > 0 else 0
            km_diffs.append(km_diff)
            feat['km_diff_avg'] = np.mean(km_diffs)
            feat['km_diff_std'] = np.std(km_diffs) if len(km_diffs) > 1 else 0
            feat['km_diff_ma'] = np.mean(km_diffs[-3:])
            
            amts.append(last['repaiAmt'])
            feat['total_spend'] = np.sum(amts)
            feat['avg_spend'] = np.mean(amts)
            feat['max_spend'] = np.max(amts)
            feat['spend_ma'] = np.mean(amts[-3:])
            
            all_svc = [s for prev_idx in range(idx) for s in grp.iloc[prev_idx]['service_items']]
            feat['service_diversity'] = len(set(all_svc))
            feat['oil_ratio'] = sum(1 for s in all_svc if '오일' in s) / max(1, len(all_svc))
            feat['brake_ratio'] = sum(1 for s in all_svc if s in ['브레이크 패드/라이닝슈', '제동']) / max(1, len(all_svc))
            feat['tire_ratio'] = sum(1 for s in all_svc if '타이어' in s) / max(1, len(all_svc))
            
            tenure = (row['inDay'] - grp.iloc[0]['inDay']).days
            feat['visit_freq_per_year'] = idx / max(1, tenure / 365.25)
            
            for window, label in [(90, '90d'), (180, '180d'), (365, '365d')]:
                mask = prev = grp[(grp['inDay'] >= row['inDay'] - pd.Timedelta(days=window)) & (grp['inDay'] < row['inDay'])]
                feat[f'visits_{label}'] = len(mask)
                feat[f'spend_{label}'] = mask['repaiAmt'].sum()
            
            feat['last_service_month'] = last['inDay'].month
        else:
            feat.update({'prev_gap': 30, 'gap_avg': 30, 'gap_std': 0, 'gap_ma': 30,
                         'km_diff_avg': 0, 'km_diff_std': 0, 'km_diff_ma': 0,
                         'total_spend': 0, 'avg_spend': 0, 'max_spend': 0, 'spend_ma': 0,
                         'service_diversity': 0, 'oil_ratio': 0, 'brake_ratio': 0, 'tire_ratio': 0,
                         'visit_freq_per_year': 0, 'visits_90d': 0, 'spend_90d': 0,
                         'visits_180d': 0, 'spend_180d': 0, 'visits_365d': 0, 'spend_365d': 0})
        
        all_feat.append(feat)

feat_df = pd.DataFrame(all_feat)
for c in feat_df.columns:
    if feat_df[c].dtype in ['float64', 'int64']:
        feat_df[c] = feat_df[c].fillna(feat_df[c].median())
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
for c in feat_df.columns:
    if feat_df[c].dtype in ['float64', 'int64']:
        feat_df[c] = feat_df[c].fillna(feat_df[c].median())

feat_df.to_pickle('data/features.pkl')
print(f"파생변수 완료: {feat_df.shape}, 저장: data/features.pkl")
