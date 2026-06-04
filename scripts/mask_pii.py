"""
PII 마스킹 스크립트
- carNo, chassisNo → '****' 로 대체
- source 문자열 KBEAD/NICE → A/B 로 대체
- CSV 파일명 변경 (DB적재_KBEAD.csv → DB적재_A.csv)
"""
import pandas as pd
import os

base = 'data'

# 1) KBEAD CSV 마스킹
kbead = pd.read_csv(f'{base}/DB적재_KBEAD.csv', encoding='cp949')
print(f"[KBEAD 원본] {len(kbead)}행, columns: {list(kbead.columns)}")
kbead['carNo'] = '****'
kbead['chassisNo'] = '****'
kbead.to_csv(f'{base}/DB적재_A.csv', encoding='cp949', index=False)
print(f"[KBEAD 처리] → DB적재_A.csv 저장 완료")

# 2) NICE CSV 마스킹
nice = pd.read_csv(f'{base}/DB적재_NICE.csv', encoding='cp949')
print(f"[NICE 원본] {len(nice)}행")
nice['carNo'] = '****'
nice['chassisNo'] = '****'
nice.to_csv(f'{base}/DB적재_B.csv', encoding='cp949', index=False)
print(f"[NICE 처리] → DB적재_B.csv 저장 완료")

# 3) 기존 원본 파일 삭제
os.remove(f'{base}/DB적재_KBEAD.csv')
os.remove(f'{base}/DB적재_NICE.csv')
print("[삭제] 원본 CSV 파일 삭제 완료")
