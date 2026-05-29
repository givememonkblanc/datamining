"""
차량 정비 데이터 분석 - 01. 데이터 전처리
- 데이터 로드 (CP949 인코딩)
- 방문 단위 집계 (서비스 항목 → 방문)
- Target 정의 (다음 방문까지의 일수)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
kbead = pd.read_csv('data/DB적재_KBEAD.csv', encoding='cp949')
nice = pd.read_csv('data/DB적재_NICE.csv', encoding='cp949')
kbead['source'] = 'KBEAD'
nice['source'] = 'NICE'

df = pd.concat([kbead, nice], ignore_index=True)
df = df.dropna(subset=['carNo'])
df['inDay'] = pd.to_datetime(df['inDay'], errors='coerce')
df['outDay'] = pd.to_datetime(df['outDay'], errors='coerce')
df = df.dropna(subset=['inDay'])

# 방문 단위 집계 (여러 서비스 항목 → 하나의 방문으로)
visit = df.groupby(['carNo', 'inDay']).agg(
    outDay=('outDay', 'first'),
    drivingKm=('drivingKm', 'first'),
    repaiAmt=('repaiAmt', 'sum'),
    carName=('carName', 'first'),
    chassisNo=('chassisNo', 'first'),
    source=('source', 'first'),
    service_items=('CATEGORY_NM_DTL_03', lambda x: list(x.str.strip())),
    n_services=('CATEGORY_NM_DTL_03', 'count')
).reset_index()

# Target 생성: 다음 방문까지의 일수
visit = visit.sort_values(['carNo', 'inDay']).reset_index(drop=True)
visit['next_visit_date'] = visit.groupby('carNo')['inDay'].shift(-1)
visit['days_until_next'] = (visit['next_visit_date'] - visit['inDay']).dt.days
visit = visit.dropna(subset=['days_until_next'])
visit = visit[(visit['days_until_next'] >= 1) & (visit['days_until_next'] <= 1095)]

print(f"방문 단위 집계: {len(visit)}건, 차량: {visit['carNo'].nunique()}대")
visit.to_pickle('data/visits.pkl')
print("저장: data/visits.pkl")
