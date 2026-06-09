"""
06. Gradient Boosting 오차 분석 — 맞춘 것과 못 맞춘 것의 패턴
- Residual 분포 및 Feature별 오차 패턴 시각화
- 잘 맞힌 구간 vs 못 맞힌 구간 특성 비교
- Cluster별/Feature별 편향(Bias) 분석
"""
import pandas as pd, numpy as np, matplotlib, warnings, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

OUT = '/home/ryzen395/datamining'
df = pd.read_pickle('/tmp/df_final.pkl')
level1_df = df[(df['days_until_next'] <= 365) & (df['visit_count'] > 1)].copy()

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

X = level1_df[feature_cols].copy()
for c in X.columns:
    if X[c].dtype in ['float64', 'int64']:
        X[c] = X[c].fillna(X[c].median())
X = X.replace([np.inf, -np.inf], np.nan)
for c in X.columns:
    if X[c].dtype in ['float64', 'int64']:
        X[c] = X[c].fillna(X[c].median())
y = level1_df['days_until_next'].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
te_idx = X_te.index  # 테스트셋 원본 인덱스
te_df = level1_df.loc[te_idx].copy()

model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

# Residual = 실제 - 예측 (양수 = 과소예측 = 실제보다 짧게 예측)
residuals = y_te - y_pred
abs_error = np.abs(residuals)
te_df['residual'] = residuals
te_df['abs_error'] = abs_error
te_df['predicted'] = y_pred
te_df['actual'] = y_te

r2 = r2_score(y_te, y_pred)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))
mae = mean_absolute_error(y_te, y_pred)

print(f"Test set: n={len(y_te)}")
print(f"Gradient Boosting: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
print(f"Residual: mean={residuals.mean():.2f}, std={residuals.std():.2f}")
print(f"  median={np.median(residuals):.2f}, skew={pd.Series(residuals).skew():.2f}")

# ============================================================
# Chart 1: Residual Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(residuals, bins=80, color='#3498db', alpha=0.7, edgecolor='white', linewidth=0.3)
ax.axvline(0, color='#e74c3c', linestyle='--', linewidth=1.5, label='0 (완벽예측)')
ax.axvline(residuals.mean(), color='#2c3e50', linestyle=':', linewidth=1.5,
           label=f'평균 오차={residuals.mean():.1f}일')
ax.set_xlabel('잔차 (Residual = 실제 - 예측, 일)', fontsize=11)
ax.set_ylabel('빈도', fontsize=11)
ax.set_title('Gradient Boosting 잔차 분포', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Add annotation for interpretation
ann = (f'R²={r2:.3f}, RMSE={rmse:.1f}일\n'
       f'MAE={mae:.1f}일\n'
       f'Residual Std={residuals.std():.1f}일\n'
       f'Skew={pd.Series(residuals).skew():.2f}')
ax.text(0.97, 0.95, ann, transform=ax.transAxes, ha='right', va='top',
        fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', alpha=0.9, edgecolor='#d0d0d0'))

# Cumulative error curve
ax = axes[1]
sorted_ae = np.sort(abs_error)
cumulative = np.arange(1, len(sorted_ae)+1) / len(sorted_ae)
ax.plot(sorted_ae, cumulative, color='#2ecc71', linewidth=2)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5)
ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
p50 = np.percentile(abs_error, 50)
p80 = np.percentile(abs_error, 80)
p90 = np.percentile(abs_error, 90)
ax.axvline(p50, color='orange', linestyle='--', alpha=0.6, label=f'P50 = {p50:.0f}일')
ax.axvline(p80, color='red', linestyle='--', alpha=0.6, label=f'P80 = {p80:.0f}일')
ax.axvline(p90, color='purple', linestyle='--', alpha=0.6, label=f'P90 = {p90:.0f}일')
ax.set_xlabel('절대오차 (일)', fontsize=11)
ax.set_ylabel('누적 비율', fontsize=11)
ax.set_title('절대오차 누적 분포 — 오차 규모 이해', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/error_residual_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n→ error_residual_dist.png')

# ============================================================
# Chart 2: Residuals vs Predicted (heteroscedasticity check)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
from scipy.stats import gaussian_kde
xy_ev = np.vstack([y_pred, residuals])
z_ev = gaussian_kde(xy_ev)(xy_ev)
idx_ev = z_ev.argsort()
sc = ax.scatter(y_pred[idx_ev], residuals[idx_ev], c=z_ev[idx_ev], s=8, cmap='viridis', alpha=0.5, edgecolors='none')
ax.axhline(0, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)

# Smooth trend line (loess-like using binned average)
bins = np.linspace(0, 365, 20)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_mean = []
for i in range(len(bins)-1):
    mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
    if mask.sum() > 0:
        bin_mean.append(residuals[mask].mean())
    else:
        bin_mean.append(np.nan)
ax.plot(bin_centers, bin_mean, color='#e74c3c', linewidth=2.5, linestyle='-', marker='o', markersize=6, label='구간 평균 잔차')

ax.set_xlabel('예측값 (일)', fontsize=12)
ax.set_ylabel('잔차 (일)', fontsize=12)
ax.set_title('잔차 vs 예측값 — 예측 구간별 오차 패턴', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.colorbar(sc, ax=ax, label='밀도')

plt.tight_layout()
plt.savefig(f'{OUT}/error_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ error_vs_predicted.png')

# ============================================================
# Chart 3: Error by Cluster
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

colors_cls = {0:'#3498db', 1:'#2ecc71', 2:'#e74c3c', 3:'#95a5a6'}
cluster_names = {0:'행태군 A', 1:'행태군 B', 2:'행태군 C', 3:'행태군 D'}

# 3a: Residual boxplot by cluster
ax = axes[0]
box_data = [te_df[te_df['cluster']==c]['residual'].values for c in sorted(cluster_names)]
bp = ax.boxplot(box_data, labels=[cluster_names[c] for c in sorted(cluster_names)],
                patch_artist=True, widths=0.5)
for patch, c in zip(bp['boxes'], sorted(cluster_names)):
    patch.set_facecolor(colors_cls[c])
    patch.set_alpha(0.6)
ax.axhline(0, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
ax.set_ylabel('잔차 (일)', fontsize=11)
ax.set_title('Cluster별 잔차 분포', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

# 3b: MAE by cluster
ax = axes[1]
cls_stats = te_df.groupby('cluster').agg(
    n=('residual','count'),
    bias=('residual','mean'),
    mae=('abs_error','mean'),
    rmse=('residual', lambda x: np.sqrt(np.mean(x**2))),
    r2=('actual', lambda x: 1 - np.sum((x - te_df.loc[x.index,'predicted'])**2) / np.sum((x - x.mean())**2))
).round(2)

ax2 = ax.twinx()
bars = ax.bar(range(len(cls_stats)), cls_stats['mae'], color=[colors_cls[i] for i in cls_stats.index], alpha=0.7, width=0.5, label='MAE')
ax2.plot(range(len(cls_stats)), cls_stats['bias'], 'ro-', markersize=8, linewidth=2, label='Bias (평균잔차)')
ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)

ax.set_xticks(range(len(cls_stats)))
ax.set_xticklabels([cluster_names[c] for c in cls_stats.index], fontsize=10)
ax.set_ylabel('MAE (일)', fontsize=11, color='#2c3e50')
ax2.set_ylabel('Bias (일)', fontsize=11, color='#e74c3c')
ax.set_title('Cluster별 MAE & Bias', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

# combine legends
l1, _ = ax.get_legend_handles_labels()
l2, _ = ax2.get_legend_handles_labels()
ax.legend(l1+l2, ['MAE', 'Bias (평균잔차)'], fontsize=9, loc='upper left')

# 3c: Summary table
ax = axes[2]
ax.axis('off')

cls_stats_display = cls_stats.copy()
cls_stats_display['n'] = cls_stats_display['n'].astype(int)
cls_stats_display.columns = ['표본수', 'Bias(일)', 'MAE(일)', 'RMSE(일)', 'R²']
cls_stats_display.index = [cluster_names[c] for c in cls_stats_display.index]

table_lines = ['Cluster별 오차 요약\n', '━━━━━━━━━━━━━━━━━━━━━━━━\n']
for label, row in cls_stats_display.iterrows():
    table_lines.append(f'● {label}\n')
    table_lines.append(f'  표본수: {int(row["표본수"]):>5d}건\n')
    table_lines.append(f'  Bias:   {row["Bias(일)"]:>+6.1f}일\n')
    table_lines.append(f'  MAE:    {row["MAE(일)"]:>6.1f}일\n')
    table_lines.append(f'  RMSE:   {row["RMSE(일)"]:>6.1f}일\n')
    table_lines.append(f'  R²:     {row["R²"]:>+7.3f}\n')

ax.text(0.05, 0.95, ''.join(table_lines), transform=ax.transAxes,
        fontsize=9.5, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', alpha=0.9))
ax.set_title('Cluster별 오차 수치', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}/error_by_cluster.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ error_by_cluster.png')

# ============================================================
# Chart 4: Error by Key Features (scatter + binned trend)
# ============================================================
key_features = [
    ('gap_ma', '최근 평균 방문간격 (일)', '일'),
    ('visit_freq_per_year', '연간 방문빈도 (회/년)', '회/년'),
    ('vehicle_age', '차량 연식 (년)', '년'),
    ('DayAvgDrivingKm', '일평균 주행거리 (km)', 'km'),
    ('oil_ratio', '오일 정비 비율', ''),
    ('visit_count', '총 방문 횟수', '회'),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, (fname, flabel, funit) in enumerate(key_features):
    ax = axes[i]
    vals = te_df[fname].values
    # Scatter with density
    xy_f = np.vstack([vals, residuals])
    z_f = gaussian_kde(xy_f)(xy_f)
    idx_f = z_f.argsort()
    sc = ax.scatter(vals[idx_f], residuals[idx_f], c=z_f[idx_f], s=6, cmap='viridis', alpha=0.4, edgecolors='none')
    ax.axhline(0, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.6)

    # Binned average
    nbins = 15
    bins = np.linspace(vals.min(), vals.max(), nbins+1)
    bin_c = (bins[:-1] + bins[1:]) / 2
    bin_m = []
    for j in range(nbins):
        m = (vals >= bins[j]) & (vals < bins[j+1])
        if m.sum() > 0:
            bin_m.append(residuals[m].mean())
        else:
            bin_m.append(np.nan)
    ax.plot(bin_c, bin_m, color='#e74c3c', linewidth=2.5, marker='o', markersize=5, label='구간 평균')

    ax.set_xlabel(f'{flabel}', fontsize=10)
    ax.set_ylabel('잔차 (일)', fontsize=10)
    ax.set_title(f'잔차 vs {fname}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)
    # Show correlation
    corr = np.corrcoef(vals, residuals)[0,1]
    ax.text(0.97, 0.06, f'상관 r={corr:.3f}', transform=ax.transAxes,
            ha='right', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

plt.tight_layout()
plt.savefig(f'{OUT}/error_by_features.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ error_by_features.png')

# ============================================================
# Chart 5: Error buckets — well-predicted vs poorly-predicted
# ============================================================
# Define buckets based on absolute error
te_df['error_bucket'] = pd.cut(te_df['abs_error'],
    bins=[-1, 15, 30, 60, 120, 999],
    labels=['±15일 이내 (정밀)', '±30일 이내 (양호)', '±60일 이내 (보통)', '±120일 이내 (큰 오차)', '±120일 초과 (매우 큰 오차)'])

bucket_summary = te_df.groupby('error_bucket', observed=True).agg(
    건수=('residual','count'),
    비율=('residual', lambda x: f'{len(x)/len(te_df)*100:.1f}%'),
    평균실제값=('actual','mean'),
    평균예측값=('predicted','mean'),
    평균잔차=('residual','mean'),
    평균_gap_ma=('gap_ma','mean'),
    평균_visit_freq=('visit_freq_per_year','mean'),
    평균_vehicle_age=('vehicle_age','mean'),
    평균_DayAvgDrivingKm=('DayAvgDrivingKm','mean'),
    평균_visit_count=('visit_count','mean'),
    오일비율=('oil_ratio','mean'),
    cluster_0_ratio=('cluster', lambda x: (x==0).mean()),
    cluster_1_ratio=('cluster', lambda x: (x==1).mean()),
    cluster_2_ratio=('cluster', lambda x: (x==2).mean()),
    cluster_3_ratio=('cluster', lambda x: (x==3).mean()),
).round(2)

print('\n=== Error Bucket Analysis ===')
print(bucket_summary.to_string())

# Visualize bucket characteristics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a: Bucket size
ax = axes[0,0]
bucket_counts = te_df['error_bucket'].value_counts(sort=False)
colors_bucket = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
bars = ax.barh(range(len(bucket_counts)), bucket_counts.values, color=colors_bucket, alpha=0.8)
ax.set_yticks(range(len(bucket_counts)))
ax.set_yticklabels(bucket_counts.index, fontsize=9)
ax.set_xlabel('건수', fontsize=11)
ax.set_title('오차 구간별 분포', fontsize=13, fontweight='bold')
for bar, val in zip(bars, bucket_counts.values):
    pct = val / len(te_df) * 100
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
            f'{val:,}건 ({pct:.1f}%)', fontsize=9, va='center')
ax.grid(True, alpha=0.2, axis='x')

# 5b: Key features across buckets
ax = axes[0,1]
bucket_order = bucket_counts.index
feat_compare = ['평균_gap_ma', '평균_visit_freq', '평균_DayAvgDrivingKm', '평균_visit_count']
feat_labels = ['gap_ma', '방문빈도', '일평균주행', '방문횟수']

# Normalize for visualization
bucket_norm = bucket_summary[feat_compare].copy()
for col in bucket_norm.columns:
    bucket_norm[col] = (bucket_norm[col] - bucket_norm[col].min()) / (bucket_norm[col].max() - bucket_norm[col].min() + 1e-6)

x_pos = np.arange(len(bucket_norm))
width = 0.2
for i, col in enumerate(feat_compare):
    ax.bar(x_pos + i*width - 1.5*width, bucket_norm[col], width, alpha=0.75, label=feat_labels[i])
ax.set_xticks(x_pos)
ax.set_xticklabels(bucket_order, fontsize=8, rotation=20)
ax.set_ylabel('정규화된 값 (0~1)', fontsize=10)
ax.set_title('오차 구간별 Feature 특성 (상대 비교)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.2, axis='y')

# 5c: Bucket-wise actual vs predicted
ax = axes[1,0]
bucket_means = bucket_summary[['평균실제값','평균예측값']]
x_pos = np.arange(len(bucket_means))
width = 0.3
ax.bar(x_pos - width/2, bucket_means['평균실제값'], width, color='#3498db', alpha=0.8, label='실제 평균')
ax.bar(x_pos + width/2, bucket_means['평균예측값'], width, color='#e74c3c', alpha=0.8, label='예측 평균')
ax.set_xticks(x_pos)
ax.set_xticklabels(bucket_order, fontsize=8, rotation=20)
ax.set_ylabel('방문 간격 (일)', fontsize=11)
ax.set_title('오차 구간별 실제 vs 예측 평균', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

# 5d: Cluster composition per error bucket
ax = axes[1,1]
cluster_cols = ['cluster_0_ratio', 'cluster_1_ratio', 'cluster_2_ratio', 'cluster_3_ratio']
bottom = np.zeros(len(bucket_summary))
for i, cc in enumerate(cluster_cols):
    vals = bucket_summary[cc].values
    ax.bar(range(len(bucket_summary)), vals, bottom=bottom, width=0.5,
           color=colors_cls[i], alpha=0.8, label=cluster_names[i])
    bottom += vals
ax.set_xticks(range(len(bucket_summary)))
ax.set_xticklabels(bucket_order, fontsize=8, rotation=20)
ax.set_ylabel('Cluster 비율', fontsize=11)
ax.set_title('오차 구간별 Cluster 구성비', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT}/error_buckets.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ error_buckets.png')

# ============================================================
# Chart 6: Over-prediction vs Under-prediction analysis
# ============================================================
te_df['error_type'] = '정밀 (±15일)'
te_df.loc[residuals < -30, 'error_type'] = '과대예측 (Over)'
te_df.loc[residuals > 30, 'error_type'] = '과소예측 (Under)'

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 6a: Mean feature profile by error type
ax = axes[0]
error_groups = te_df[te_df['error_type'].isin(['과대예측 (Over)', '과소예측 (Under)', '정밀 (±15일)'])]
profile_feats = ['gap_ma', 'visit_freq_per_year', 'DayAvgDrivingKm', 'vehicle_age',
                 'oil_ratio', 'visit_count', 'gap_std']
profile_data = error_groups.groupby('error_type')[profile_feats].mean()

# Normalize
profile_norm = (profile_data - profile_data.min()) / (profile_data.max() - profile_data.min() + 1e-6)

labels = ['gap_ma', '방문빈도', '일평균주행', '연식', '오일비율', '방문횟수', '간격변동성']
x_pos = np.arange(len(labels))
width = 0.25
etypes = profile_norm.index
ecolors = ['#3498db', '#e74c3c', '#2ecc71']
for i, etype in enumerate(etypes):
    ax.bar(x_pos + i*width - width, profile_norm.loc[etype].values, width,
           alpha=0.8, color=ecolors[i], label=etype)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('정규화된 값', fontsize=10)
ax.set_title('과대/과소/정밀 예측 그룹의 Feature 프로필', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

# 6b: Actual vs Predicted by error type (scatter)
ax = axes[1]
for etype, ecol in zip(etypes, ecolors):
    sub = error_groups[error_groups['error_type'] == etype]
    ax.scatter(sub['actual'], sub['predicted'], s=4, alpha=0.3, color=ecol, label=f'{etype} (n={len(sub):,})')
ax.plot([0, 365], [0, 365], '--', color='#2c3e50', linewidth=1.5, alpha=0.6, label='완벽 예측')
ax.fill_between([0, 365], [30, 395], [-30, 335], alpha=0.05, color='#2ecc71', label='±30일')
ax.set_xlabel('실제값 (일)', fontsize=11)
ax.set_ylabel('예측값 (일)', fontsize=11)
ax.set_title('오차 유형별 예측 vs 실제 산점도', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/error_over_under.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ error_over_under.png')

# ============================================================
# Chart 7: Residual distribution by target quantiles
# ============================================================
te_df['actual_bucket'] = pd.qcut(te_df['actual'], q=4, labels=['Q1 (짧은간격)', 'Q2', 'Q3', 'Q4 (긴간격)'])

fig, ax = plt.subplots(figsize=(9, 5.5))
box_data_q = [te_df[te_df['actual_bucket']==b]['residual'].values for b in ['Q1 (짧은간격)', 'Q2', 'Q3', 'Q4 (긴간격)']]
bp = ax.boxplot(box_data_q, labels=['Q1 (1~44일)', 'Q2 (44~91일)', 'Q3 (91~151일)', 'Q4 (151~365일)'],
                patch_artist=True, widths=0.5)
qcolors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
for patch, c in zip(bp['boxes'], qcolors):
    patch.set_facecolor(c)
    patch.set_alpha(0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.6)
ax.set_xlabel('실제 방문간격 구간 (분위)', fontsize=11)
ax.set_ylabel('잔차 (일)', fontsize=11)
ax.set_title('실제 방문간격 구간별 잔차 분포', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

# Add interpretation
ann_q = ('양수 = 과소예측 (실제보다 짧게 예측)\n'
         '음수 = 과대예측 (실제보다 길게 예측)\n'
         'Q1 구간: 주로 과대예측 (실제 방문이 예측보다 빠름)\n'
         'Q4 구간: 주로 과소예측 (실제 방문이 예측보다 느림)')
ax.text(0.97, 0.05, ann_q, transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', alpha=0.9, edgecolor='#d0d0d0'))

plt.tight_layout()
plt.savefig(f'{OUT}/error_by_target_quantile.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ error_by_target_quantile.png')

# ============================================================
# Print summary findings for report
# ============================================================
print('\n' + '='*70)
print('ERROR ANALYSIS SUMMARY')
print('='*70)

# Over/under prediction rates
over_mask = residuals < -30
under_mask = residuals > 30
precise_mask = abs_error <= 15
print(f'\nError Buckets (절대오차 기준):')
print(f'  정밀 (±15일): {precise_mask.sum():>7,}건 ({precise_mask.mean()*100:.1f}%)')
print(f'  양호 (±30일): {((abs_error>15)&(abs_error<=30)).sum():>7,}건 ({(abs_error>15).mean()*100 - (abs_error>30).mean()*100:.1f}%)')
print(f'  과대예측 (<-30일): {over_mask.sum():>7,}건 ({over_mask.mean()*100:.1f}%)')
print(f'  과소예측 (>+30일): {under_mask.sum():>7,}건 ({under_mask.mean()*100:.1f}%)')

# Feature contrasts
print(f'\n과대예측 vs 과소예측 Feature 비교:')
over_df = te_df[over_mask]
under_df = te_df[under_mask]
for col in ['gap_ma', 'visit_freq_per_year', 'DayAvgDrivingKm', 'vehicle_age', 'oil_ratio', 'visit_count', 'gap_std']:
    print(f'  {col:25s}: 과대={over_df[col].mean():>8.2f}  과소={under_df[col].mean():>8.2f}  차이={over_df[col].mean()-under_df[col].mean():>+8.2f}')

# Cluster bias
print(f'\nCluster별 편향(Bias):')
for c in sorted(cluster_names):
    sub = te_df[te_df['cluster']==c]
    print(f'  {cluster_names[c]}: Bias={sub["residual"].mean():+.1f}일, MAE={sub["abs_error"].mean():.1f}일, n={len(sub):,}')

# Systematic patterns
print(f'\n주요 발견:')
print(f'  1. 잔차 std = {residuals.std():.1f}일 (Target std={y_te.std():.1f}일 대비 {residuals.std()/y_te.std()*100:.0f}%)')
print(f'  2. 전체 예측의 {precise_mask.mean()*100:.1f}%가 ±15일 이내 오차')
print(f'  3. Target Q1(짧은간격): 과대예측 경향 (Bias={te_df[te_df["actual_bucket"]=="Q1 (짧은간격)"]["residual"].mean():+.1f}일)')
print(f'  4. Target Q4(긴간격): 과소예측 경향 (Bias={te_df[te_df["actual_bucket"]=="Q4 (긴간격)"]["residual"].mean():+.1f}일)')

# Save summary CSV
te_df[['actual','predicted','residual','abs_error','cluster','error_type','actual_bucket'] + feature_cols[:10]].to_csv(f'{OUT}/results/error_analysis.csv', index=False)
print(f'\n→ results/error_analysis.csv 저장 완료')

print('\nDone.')
