from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictors.psf4_formula import build_psf4_forecast

OUT = Path(__file__).resolve().parent
elapsed_months = 6
forecast = build_psf4_forecast(elapsed_months)
values = forecast['series']['hep_multipliers']
months = [i / 10 for i in range(len(values))]
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def draw_chart(ax):
    ax.plot(months, values, color='#2563eb', linewidth=3)
    ax.axhline(4.83, color='#64748b', linewidth=1, linestyle='--')
    ax.text(59, 4.93, '수렴값 4.83', ha='right', va='bottom', fontsize=11,
            color='#475569')
    ax.scatter([0], [values[0]], color='#2563eb', s=38, zorder=3)
    ax.annotate(f'현재 {values[0]:.3f}', (0, values[0]), xytext=(12, 14),
                textcoords='offset points', fontsize=11, color='#1e3a5f')
    ax.set(xlim=(-1, 61), ylim=(0, 5.35),
           xlabel='현재 이후 경과 개월 수 (개월)', ylabel='PSF4 가중치 (HEP 배수)')
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.grid(axis='y', alpha=0.18)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
draw_chart(ax)
ax.set_title('교육·훈련 경과에 따른 PSF4 가중치 변화\n현재 교육·훈련 후 6개월 경과 예시',
             loc='left', fontsize=15, pad=18)
fig.tight_layout()
fig.savefig(OUT / 'psf4-monthly-example.png', bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(14, 6), dpi=180)
fig.text(0.065, 0.92, '교육·훈련 경과에 따른 PSF4 가중치 변화',
         fontsize=23, color='#203b64')
fig.text(0.065, 0.86, '예시 조건: 현재 교육·훈련 후 6개월 경과',
         fontsize=14, color='#475569')
ax = fig.add_axes([0.075, 0.17, 0.53, 0.60])
draw_chart(ax)

sections = [
    (0.71, '가중치 산정',
     '현재 경과 개월 수에 미래 경과 개월 수를\n더하여 스케일링한 공식에 대입한다.'),
    (0.49, '그래프 생성',
     '향후 60개월의 가중치를 0.1개월 간격으로\n계산하고, 총 601개 값을 연결한다.'),
    (0.27, '변화 양상',
     f'현재 가중치 {values[0]:.3f}에서 비선형적으로 증가하며,\n장기간 경과할수록 4.83에 수렴한다.'),
]
for ypos, label, body in sections:
    fig.text(0.665, ypos, label, fontsize=17, color='#203b64')
    fig.text(0.665, ypos - 0.055, body, fontsize=14, va='top',
             linespacing=1.65, color='#202020')
fig.text(0.065, 0.045, '※ X축의 0개월은 현재 시점이며, Y축은 HEP 계산에 사용하는 PSF4 가중치이다.',
         fontsize=11, color='#475569')
fig.savefig(OUT / 'psf4-monthly-explained.png', bbox_inches='tight', pad_inches=0.2)
plt.close(fig)
print(f'Points: {len(values)}; current weight: {values[0]}; weight in 60 months: {values[-1]}')
