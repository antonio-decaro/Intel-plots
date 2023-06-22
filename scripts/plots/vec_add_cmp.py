from plot_utils import check_single_input
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import hmean

df, out_plot = check_single_input()

substitutions = {
    'OptimizedVectorAddition_int16_sg8':  'SIMD int16 sg8',
    'OptimizedVectorAddition_int16_sg16': 'SIMD int16 sg16',
    'OptimizedVectorAddition_int16_sg32': 'SIMD int16 sg32',
    'OptimizedVectorAddition_int32_sg8':  'SIMD int32 sg8',
    'OptimizedVectorAddition_int32_sg16': 'SIMD int32 sg16',
    'OptimizedVectorAddition_int32_sg32': 'SIMD int32 sg32',
    'OptimizedVectorAddition_fp16_sg8':   'SIMD fp16 sg8',
    'OptimizedVectorAddition_fp16_sg16':  'SIMD fp16 sg16',
    'OptimizedVectorAddition_fp16_sg32':  'SIMD fp16 sg32',
    'OptimizedVectorAddition_fp32_sg8':   'SIMD fp32 sg8',
    'OptimizedVectorAddition_fp32_sg16':  'SIMD fp32 sg16',
    'OptimizedVectorAddition_fp32_sg32':  'SIMD fp32 sg32',
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
df['bench-name'] = df['bench-name'].replace(substitutions)

scaler = MinMaxScaler()
kernel_time = df['kernel-time-mean'].values.reshape(-1, 1)
normalized_kernel_time = scaler.fit_transform(kernel_time)
df['kernel-time-mean'] = normalized_kernel_time
# df['kernel-time-mean'] *= 1000000

# df_grouped = df.groupby('bench-name')['kernel-time-mean'].apply(lambda x: hmean(x))
df_grouped = df.groupby('bench-name')['kernel-time-mean'].mean()

plt.figure(figsize=(10, 6))  # Imposta la dimensione del grafico
bar_width = 0.1  # Larghezza delle barre

colors = ['cyan' if 'SIMD' in index else 'green' for index in df_grouped.index]
bars = plt.barh(df_grouped.index, df_grouped.values, color=colors)
# bars = plt.barh(range(len(df_grouped.index)), df_grouped.values, color=colors)

plt.ylabel('Type')
plt.xlabel('Execution Time (us)')

# plt.yticks([])
# for i, bar in enumerate(bars):
#     plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, df_grouped.index[i], ha='right', va='center')

plt.title('Vector Addition: Arc A770')

if not out_plot is None:
    plt.savefig(out_plot)
else:
    plt.show()
