from plot_utils import check_single_input
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler

df, out_plot = check_single_input()

substitutions = {
    'OptimizedVectorAddition_fp32_sg8':   'fp32 sg8',
    'OptimizedVectorAddition_fp32_sg16':  'fp32 sg16',
    'OptimizedVectorAddition_fp32_sg32':  'fp32 sg32',
    'OptimizedVectorAddition_int32_sg8':  'int32 sg8',
    'OptimizedVectorAddition_int32_sg16': 'int32 sg16',
    'OptimizedVectorAddition_int32_sg32': 'int32 sg32',
    'OptimizedVectorAddition_int64_sg8':  'int64 sg8',
    'OptimizedVectorAddition_int64_sg16': 'int64 sg16',
    'OptimizedVectorAddition_int64_sg32': 'int64 sg32',
}
df['bench-name'] = df['bench-name'].replace(substitutions)

scaler = MinMaxScaler()
# kernel_time = df['kernel-time-mean'].values.reshape(-1, 1)
# normalized_kernel_time = scaler.fit_transform(kernel_time)
# df['kernel-time-mean'] = normalized_kernel_time
df['kernel-time-mean'] *= 1000000

df_grouped = df.groupby(['bench-name', 'local-size'])['kernel-time-mean'].mean().reset_index()

plt.figure(figsize=(10, 6))  # Imposta la dimensione del grafico
bar_width = 0.1  # Larghezza delle barre

# Genera un indice per le posizioni delle barre sull'asse X
x_pos = np.arange(len(df_grouped['local-size'].unique()))


# Itera sui tipi e crea una barra per ogni tipo
for i, type in enumerate(df_grouped['bench-name'].unique()):
    data = df_grouped[df_grouped['bench-name'] == type]
    plt.bar(x_pos + i * bar_width, data['kernel-time-mean'], width=bar_width, label=type)

plt.xlabel('Local Size')
plt.ylabel('Execution Time (us)')
plt.title('Vector Addition: Arc A770')
plt.xticks(x_pos + bar_width * (len(df_grouped['bench-name'].unique()) - 1) / 2, df_grouped['local-size'].unique())
plt.legend()
plt.show()

if not out_plot is None:
    plt.savefig(out_plot)
else:
    plt.show()
