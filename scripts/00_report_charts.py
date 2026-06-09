import pandas as pd, numpy as np, matplotlib, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

df = pd.read_pickle('/tmp/df_final.pkl')

# ============================================================
# Chart 1: K-Means 클러스터링 전/후 비교
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

colors = {0:'#3498db', 1:'#2ecc71', 2:'#e74c3c', 3:'#95a5a6'}
cluster_names = {c: f'행태군 {chr(65 + c)}' for c in sorted(df['cluster'].unique())}

# Before/after clustering: use identical sample and axis ranges
plot_df = df.sample(min(12000, len(df)), random_state=42).copy()
x_min, x_max = plot_df['log_drivingKm'].min(), plot_df['log_drivingKm'].max()
y_min, y_max = plot_df['visit_freq_per_year'].min(), plot_df['visit_freq_per_year'].max()

axes[0].scatter(plot_df['log_drivingKm'], plot_df['visit_freq_per_year'],
                s=3, alpha=0.28, color='#7f8c8d')
axes[0].set_xlabel('로그 주행거리')
axes[0].set_ylabel('연간 방문빈도')
axes[0].set_title('클러스터링 전 분포', fontsize=13, fontweight='bold')
axes[0].set_xlim(x_min, x_max)
axes[0].set_ylim(y_min, y_max)
axes[0].grid(True, alpha=0.2)

# After clustering: colored by cluster in original feature space
for c in sorted(df['cluster'].unique()):
    sub = plot_df[plot_df['cluster'] == c]
    axes[1].scatter(sub['log_drivingKm'], sub['visit_freq_per_year'],
                    s=3, alpha=0.4, color=colors[c], label=cluster_names[c])

axes[1].set_xlabel('로그 주행거리')
axes[1].set_ylabel('연간 방문빈도')
axes[1].set_title('K-Means 클러스터링 결과 (k=4)', fontsize=13, fontweight='bold')
axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)
axes[1].legend(title='Cluster', fontsize=9)
axes[1].grid(True, alpha=0.2)

# Cluster summary table
axes[2].axis('off')
summary = df.groupby('cluster').agg(
    차량수=('carNo','nunique'),
    평균주행거리=('drivingKm','mean'),
    일평균주행=('DayAvgDrivingKm','mean'),
    방문빈도=('visit_freq_per_year','mean'),
    최근평균간격=('gap_ma','mean'),
    서비스다양성=('service_diversity','mean')
).round(1)

table_lines = ['Cluster별 특성 요약\n', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n']
for c in sorted(summary.index):
    row = summary.loc[c]
    table_lines.append(f'Cluster {c} ({cluster_names[c]})\n')
    table_lines.append(f'  차량수:     {int(row["차량수"]):>5d}대\n')
    table_lines.append(f'  평균 주행:   {row["평균주행거리"]:>7.0f}km\n')
    table_lines.append(f'  일평균 주행: {row["일평균주행"]:>6.1f}km\n')
    table_lines.append(f'  방문빈도:    {row["방문빈도"]:>6.1f}회/년\n')
    table_lines.append(f'  최근 평균간격:{row["최근평균간격"]:>6.1f}일\n')
    table_lines.append(f'  서비스 다양성:{row["서비스다양성"]:>6.1f}\n')
    table_lines.append('  ─────────────────────────\n')

axes[2].text(0.05, 0.95, ''.join(table_lines), transform=axes[2].transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', alpha=0.9))
axes[2].set_title('Cluster별 특성 요약', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ryzen395/datamining/clustering_final.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ clustering_final.png')

# ============================================================
# Chart 2: Feature Set별 성능 비교
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

feature_sets = ['[A] 기본 정보', '[B] +정비 패턴', '[C] +방문 이력']
r2_scores = [0.471, 0.582, 0.753]
colors = ['#3498db', '#2ecc71', '#e74c3c']

x = np.arange(len(feature_sets))
bars = ax.bar(x, r2_scores, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)

for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.015, f'{score:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    if i > 0:
        diff = score - r2_scores[i-1]
        ax.text(bar.get_x() + bar.get_width()/2, score/2, f'+{diff:.3f}',
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')

ax.plot(x, r2_scores, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(feature_sets, fontsize=10)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('Feature Set별 XGBoost 성능 비교', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.85)
ax.grid(True, alpha=0.3, axis='y')

note_text = 'A → B: 정비 패턴 추가\nB → C: 방문 이력 추가'
ax.text(0.98, 0.04, note_text, transform=ax.transAxes,
        ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', alpha=0.9, edgecolor='#d0d0d0'))

plt.tight_layout()
plt.savefig('/home/ryzen395/datamining/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ model_comparison.png')

# ============================================================
# Chart 3: Predicted vs Actual
# ============================================================
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

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

model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
r2_val = r2_score(y_te, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_te, y_pred))
mae_val = mean_absolute_error(y_te, y_pred)

fig, ax = plt.subplots(figsize=(8, 7))

# Scatter with density coloring
from scipy.stats import gaussian_kde
xy = np.vstack([y_te, y_pred])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
sc = ax.scatter(y_te[idx], y_pred[idx], c=z[idx], s=8, cmap='viridis', alpha=0.6, edgecolors='none')

# Perfect prediction line
lims = [0, 400]
ax.plot(lims, lims, '--', color='#e74c3c', linewidth=2, alpha=0.6, label='완벽 예측')

# +/-30일 band
ax.fill_between(lims, [l-30 for l in lims], [l+30 for l in lims], alpha=0.08, color='#2ecc71', label='±30일 오차 범위')

ax.set_xlabel('실제 방문 간격 (일)', fontsize=12)
ax.set_ylabel('예측 방문 간격 (일)', fontsize=12)
ax.set_title('Level 1 Gradient Boosting 예측값 vs 실제값', fontsize=13, fontweight='bold')
ax.set_xlim(0, 365)
ax.set_ylim(0, 365)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.colorbar(sc, ax=ax, label='밀도')
plt.tight_layout()
plt.savefig('/home/ryzen395/datamining/pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'→ pred_vs_actual.png [Gradient Boosting] (n={len(level1_df)}, R²={r2_val:.3f}, RMSE={rmse_val:.1f})')

# ============================================================
# Chart 4: Feature Importance
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))

importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

top_n = importances.tail(15)

colors_imp = plt.cm.YlOrRd(top_n['importance'].values / top_n['importance'].max())

bars = ax.barh(top_n['feature'], top_n['importance'], color=colors_imp, edgecolor='white', height=0.7)
for bar, val in zip(bars, top_n['importance']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f} ({val/top_n["importance"].max()*100:.0f}%)',
            fontsize=9, va='center')

ax.set_xlabel('중요도', fontsize=12)
ax.set_title('XGBoost 변수 중요도 (Top 15)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/ryzen395/datamining/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('→ feature_importance.png')

print('\n모든 보고서 차트 생성 완료')
