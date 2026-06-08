import pandas as pd, numpy as np, matplotlib, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# Ensure NanumGothic is available (add if missing)
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

df = pd.read_pickle('/tmp/df_final.pkl')
t = df['days_until_next']

# ============================================================
# Chart 1: Target 분포
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [2, 1]})

ax1.hist(t, bins=80, color='#3498db', edgecolor='white', linewidth=0.5, alpha=0.8)
ax1.axvline(t.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'평균 {t.mean():.0f}일')
ax1.axvline(t.median(), color='#2ecc71', linestyle=':', linewidth=2, label=f'중앙값 {t.median():.0f}일')
ax1.set_xlabel('방문 간격 (일)')
ax1.set_ylabel('방문 건수')
ax1.set_title('Target: 다음 방문까지의 일수 분포')
ax1.legend(fontsize=9)
ax1.set_xlim(0, 400)
ax1.grid(True, alpha=0.3, axis='y')

stats_text = (
    f'통계량\n'
    f'━━━━━━━━━━\n'
    f'평 균:  {t.mean():.1f}일\n'
    f'중앙값: {t.median():.0f}일\n'
    f'표준편차:{t.std():.1f}일\n'
    f'Q1:     {t.quantile(0.25):.0f}일\n'
    f'Q3:     {t.quantile(0.75):.0f}일\n'
    f'━━━━━━━━━━\n'
    f'CV = {t.std()/t.mean():.2f}\n'
    f'(예측 난이도 ↑)'
)
ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=11,
         fontfamily='monospace', verticalalignment='center',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#fef9e7', alpha=0.9))
ax2.axis('off')

plt.tight_layout()
plt.savefig('/home/ryzen395/datamining/eda_target_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ eda_target_dist.png')

# ============================================================
# Chart 2: 정비 카테고리 분포
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

cats = ['has_oil','has_brake','has_tire','has_battery','has_coolant']
labels = ['오일 교환','브레이크 패드','타이어','배터리','부동액']
colors_cat = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']
vals = [df[c].mean()*100 for c in cats]

bars = ax.barh(labels[::-1], vals[::-1], color=colors_cat[::-1], edgecolor='white', height=0.6)
for bar, v in zip(bars, vals[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{v:.1f}%',
            va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('해당 정비 포함 방문 비율 (%)')
ax.set_title('정비 항목별 방문 비율')
ax.set_xlim(0, 85)
ax.grid(True, alpha=0.3, axis='x')

# inset: 복수 서비스 비율
ax_inset = fig.add_axes([0.65, 0.15, 0.25, 0.25])
labels_pie = ['단일 서비스', '복수 서비스']
sizes_pie = [46.0, 54.0]
ax_inset.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', colors=['#95a5a6','#2ecc71'],
             startangle=90, textprops={'fontsize':8})
ax_inset.set_title('방문당 서비스 수', fontsize=9)

plt.tight_layout()
plt.savefig('/home/ryzen395/datamining/eda_category.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ eda_category.png')

# ============================================================
# Chart 3: 상관관계 히트맵
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))

corr_vars = ['days_until_next','drivingKm','vehicle_age','DayAvgDrivingKm','log_drivingKm',
             'repaiAmt','n_services','gap_avg','gap_ma','visit_freq_per_year','visits_90d','oil_ratio']
var_labels = ['Target\n(방문간격)','주행거리','차량연식','일평균\n주행거리','로그\n주행거리',
              '정비금액','서비스\n항목수','평균\n방문간격','이동평균\n방문간격','연간\n방문빈도','최근90일\n방문수','오일비율']

corr = df[corr_vars].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = plt.cm.RdBu_r

im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        if i >= j:
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

ax.set_xticks(range(len(corr_vars)))
ax.set_yticks(range(len(corr_vars)))
ax.set_xticklabels(var_labels, fontsize=8, rotation=30, ha='right')
ax.set_yticklabels(var_labels, fontsize=8)
ax.set_title('주요 변수 간 상관관계', fontsize=13, fontweight='bold', pad=10)

plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig('/home/ryzen395/datamining/eda_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ eda_correlation.png')

# ============================================================
# Chart 4: Cluster별 방문 패턴
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

cluster_names = {c: f'행태군 {chr(65 + c)}' for c in sorted(df['cluster'].unique())}
cluster_colors = {0:'#3498db', 1:'#2ecc71', 2:'#e74c3c', 3:'#95a5a6'}

# 4-1: Cluster별 Target 분포
ax = axes[0]
for c in sorted(df['cluster'].unique()):
    data = df[df['cluster']==c]['days_until_next']
    ax.hist(data, bins=60, alpha=0.5, color=cluster_colors[c], label=f'{c}: {cluster_names[c]}', density=True)
ax.set_xlim(0, 400)
ax.set_xlabel('방문 간격 (일)')
ax.set_ylabel('밀도')
ax.set_title('Cluster별 방문간격 분포')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2, axis='y')

# 4-2: 주행거리 vs 방문빈도
ax = axes[1]
for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster']==c].sample(min(3000, len(df[df['cluster']==c])), random_state=42)
    ax.scatter(sub['log_drivingKm'], sub['visit_freq_per_year'], s=1, alpha=0.3, color=cluster_colors[c], label=f'{c}')
ax.set_xlabel('로그 주행거리')
ax.set_ylabel('연간 방문빈도')
ax.set_title('로그 주행거리 vs 연간 방문빈도')
ax.legend(fontsize=8, title='Cluster')
ax.grid(True, alpha=0.2)

# 4-3: Cluster별 요약 테이블
ax = axes[2]
ax.axis('off')
summary = df.groupby('cluster').agg(
    차량수=('carNo','nunique'),
    일평균주행=('DayAvgDrivingKm','mean'),
    평균방문간격=('days_until_next','mean'),
    방문빈도=('visit_freq_per_year','mean'),
    서비스다양성=('service_diversity','mean')
).round(1)

table_text = 'Cluster별 요약\n'
table_text += '━━━━━━━━━━━━━━━━━━━━━━━━\n'
for c in sorted(summary.index):
    row = summary.loc[c]
    table_text += f'{c} ({cluster_names[c]:6s})\n'
    table_text += f'  차량수: {int(row["차량수"]):>5d}대\n'
    table_text += f'  일평균주행:{row["일평균주행"]:>8.1f}km\n'
    table_text += f'  방문간격: {row["평균방문간격"]:>7.1f}일\n'
    table_text += f'  방문빈도: {row["방문빈도"]:>7.1f}회/년\n'
    table_text += f'  다양성:   {row["서비스다양성"]:>7.1f}\n'
    table_text += f'  ──────────────────────\n'

ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontsize=8.5,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', alpha=0.9))
ax.set_title('Cluster별 특성 요약', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ryzen395/datamining/eda_cluster_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ eda_cluster_pattern.png')

# ============================================================
# Chart 5: 방문빈도 vs 방문간격 (EDA 핵심 인사이트)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

visit_bins = [0, 1, 2, 3, 5, 10, 20, 50, 100]
labels_bin = ['1회', '2회', '3회', '4~5회', '6~10회', '11~20회', '21~50회', '51회↑']
df['visit_bin'] = pd.cut(df['visit_count'], bins=visit_bins, labels=labels_bin)

stats = df.groupby('visit_bin', observed=True)['days_until_next'].agg(['mean','median','std'])
stats['count'] = df.groupby('visit_bin', observed=True).size()
stats = stats.reindex(labels_bin)

# 막대 + 에러바
x = np.arange(len(labels_bin))
ax.bar(x, stats['mean'], yerr=stats['std'], capsize=4, color='#3498db', edgecolor='white', alpha=0.8,
       error_kw={'linewidth':1.5, 'ecolor':'#2c3e50'})
ax2_line = ax.twinx()
ax2_line.plot(x, stats['count'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
ax2_line.set_ylabel('차량 수 (log)', fontsize=11)
ax2_line.set_yscale('log')

for i in range(len(labels_bin)):
    ax.text(i, stats['mean'].iloc[i] + 5, f'{stats["mean"].iloc[i]:.0f}일', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels_bin, fontsize=9)
ax.set_xlabel('총 방문 횟수', fontsize=11)
ax.set_ylabel('평균 방문 간격 (일)', fontsize=11)
ax.set_title('방문 횟수별 평균 방문 간격 및 차량 분포', fontsize=13, fontweight='bold')
ax.legend(['평균 방문간격'], fontsize=9, loc='upper left')
ax2_line.legend(['차량 수'], fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/ryzen395/datamining/eda_visit_freq.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ eda_visit_freq.png')

print('\n모든 EDA 차트 생성 완료')
