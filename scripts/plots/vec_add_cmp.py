from matplotlib.patches import Patch
from plot_utils import check_double_input
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from scipy.stats import hmean

data_gpu1, data_gpu2, out_file = check_double_input()

to_replace = {
    'VectorAddition_int16_sg8':      'int16 sg8',
    'VectorAddition_int32_sg8':      'int32 sg8',
    'VectorAddition_fp16_sg8':       'fp16  sg8',
    'VectorAddition_fp32_sg8':       'fp32  sg8',
    'VectorAddition_int16_sg16':      'int16 sg16',
    'VectorAddition_int32_sg16':      'int32 sg16',
    'VectorAddition_fp16_sg16':       'fp16  sg16',
    'VectorAddition_fp32_sg16':       'fp32  sg16',
    'VectorAddition_int16_sg32':      'int16 sg32',
    'VectorAddition_int32_sg32':      'int32 sg32',
    'VectorAddition_fp16_sg32':       'fp16  sg32',
    'VectorAddition_fp32_sg32':       'fp32  sg32',
}

data_gpu1 = data_gpu1.replace(to_replace).groupby('bench-name')['kernel-time-mean'].apply(lambda x: hmean(x)).reset_index()
data_gpu2 = data_gpu2.replace(to_replace).groupby('bench-name')['kernel-time-mean'].apply(lambda x: hmean(x)).reset_index()

data_gpu1['kernel-time-mean'] *= 1000
data_gpu2['kernel-time-mean'] *= 1000

df = pd.DataFrame()
df['type'] = data_gpu1['bench-name'].apply(lambda x: x.split()[0])
df['sg'] = data_gpu1['bench-name'].apply(lambda x: x.split()[1])
df['A770'] = data_gpu1['kernel-time-mean']

v100_values = data_gpu2['kernel-time-mean'].values
v100_mapping = dict(zip(data_gpu2['bench-name'].apply(lambda x: x.split()[0]), v100_values))
df['V100'] = df['type'].map(v100_mapping)

types = df['type'].unique()
bar_width = 1

fig, ax = plt.subplots(figsize=(10, 6))

group_width = bar_width * len(types)
num_bars = len(types)

colors = ['lightsteelblue', 'cornflowerblue', 'royalblue', 'forestgreen']

rects = []
for i, t in enumerate(types):
    a770_vals = df.loc[df['type'] == t, 'A770'].values
    a770_sg_vals = df.loc[df['type'] == t, 'sg'].values
    a770_vals[0], a770_vals[1], a770_vals[2] = a770_vals[2], a770_vals[0], a770_vals[1]

    v100_val = df.loc[df['type'] == t, 'V100'].values[0] # all the same]
    vals = np.append(a770_vals, v100_val)
    
    offset = i * (len(vals) + 1)
    rect = ax.bar(np.arange(len(vals)) + offset, vals, bar_width, color=colors)
    rects.append(rect)


tick_positions = np.arange(num_bars) * (len(vals) + 1) + (num_bars - 1) * bar_width / 2
ax.set_xticks(tick_positions)
ax.set_xticklabels(types)

legend_patches = [Patch(color=color) for color in colors]
ax.legend(legend_patches, ['A770 - sg8', 'A770 - sg16', 'A770 - sg32', 'V100'])
ax.set_xlabel('Data Types')
ax.set_ylabel('Milliseconds (ms)')
ax.set_title('Vector Addition: A770 vs. V100')

# Mostra il grafico
plt.tight_layout()
plt.savefig(out_file)