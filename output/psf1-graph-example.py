from pathlib import Path
import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictors.psf1_forecast import build_psf1_forecast, KST

data = build_psf1_forecast('2026-09-07T08:00:00+09:00', 0,
                           datetime(2026, 9, 7, 8, tzinfo=KST))
segments = data['segments']
x = [s['start_elapsed_hours'] for s in segments] + [15]
y = [s['predicted_label'] for s in segments] + [2]
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
ax.step(x, y, where='post', color='#2563eb', linewidth=3)
ax.set(xlim=(-0.2, 15.2), ylim=(-0.25, 2.45),
       xlabel='조회 시점 이후 경과 시간 (시간)', ylabel='PSF1 수준')
ax.set_yticks([0, 1, 2], ['0 (F.1.1)', '1 (F.1.2)', '2 (F.1.3)'])
ax.set_xticks(x, [f'{v:g}' for v in x])
ax.grid(axis='y', alpha=0.2)
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
for s in segments[1:]:
    start = s['start_elapsed_hours']
    ax.axvline(start, color='#94a3b8', linewidth=0.8, linestyle=':', zorder=0)
    ax.text(start, 2.24, s['start_at'][11:16], ha='center', fontsize=10)
ax.set_title('PSF1 시간 변화 예시\n기본 수준 0 · 근무 시작 및 조회 08:00 (KST)',
             loc='left', fontsize=15, pad=18)
fig.text(0.99, 0.02, '마지막 구간은 계속 유지되며, 예시에서는 15시간까지 표시',
         ha='right', fontsize=9, color='#475569')
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(Path(__file__).with_suffix('.png'), bbox_inches='tight')
