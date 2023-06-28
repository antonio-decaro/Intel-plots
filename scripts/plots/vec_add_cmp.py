from plot_utils import check_double_input
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler
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

scaler = MinMaxScaler()
data_gpu1['kernel-time-mean'] *= 1000
data_gpu2['kernel-time-mean'] *= 1000

index1 = data_gpu1['bench-name']
index2 = data_gpu1['bench-name']
time1 = data_gpu1['kernel-time-mean']
time2 = data_gpu2['kernel-time-mean']

ind = list(range(len(index1)))

# Larghezza delle barre
width = 0.35

# Crea il grafico a barre
fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.barh(ind, time1, height=width, color='b', label='A770')
bar2 = ax.barh([i + width for i in ind], time2, height=width, color='g', label='V100')

# Aggiungi etichette, titolo e legenda
ax.set_xlabel('us')
ax.set_title('Execution Time')
ax.set_yticks(ind)
ax.set_yticklabels(data_gpu1['bench-name'])
ax.legend()

min_value = min(min(time1), min(time2))
max_value = max(max(time1), max(time2))
ax.set_xlim(min_value - 0.1, max_value + 0.1)

# Mostra il grafico
plt.tight_layout()
plt.savefig(out_file)