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
    'MicroBench_HostDeviceBandwidth_1D_H2D_Contiguous': '1D_H2D_C',
    'MicroBench_HostDeviceBandwidth_2D_H2D_Contiguous': '2D_H2D_C',
    'MicroBench_HostDeviceBandwidth_3D_H2D_Contiguous': '3D_H2D_C',
    'MicroBench_HostDeviceBandwidth_1D_D2H_Contiguous': '1D_D2H_C',
    'MicroBench_HostDeviceBandwidth_2D_D2H_Contiguous': '2D_D2H_C',
    'MicroBench_HostDeviceBandwidth_3D_D2H_Contiguous': '3D_D2H_C',

    'MicroBench_HostDeviceBandwidth_1D_H2D_Strided': '1D_H2D_S',
    'MicroBench_HostDeviceBandwidth_2D_H2D_Strided': '2D_H2D_S',
    'MicroBench_HostDeviceBandwidth_3D_H2D_Strided': '3D_H2D_S',
    'MicroBench_HostDeviceBandwidth_1D_D2H_Strided': '1D_D2H_S',
    'MicroBench_HostDeviceBandwidth_2D_D2H_Strided': '2D_D2H_S',
    'MicroBench_HostDeviceBandwidth_3D_D2H_Strided': '3D_D2H_S',
}

data_gpu1 = data_gpu1.replace(to_replace)
data_gpu2 = data_gpu2.replace(to_replace)

index1 = data_gpu1['bench-name']
index2 = data_gpu1['bench-name']
bandwidth1 = data_gpu1['run-time-throughput']
bandwidth2 = data_gpu2['run-time-throughput']

ind = list(range(len(index1)))

# Larghezza delle barre
width = 0.35

# Crea il grafico a barre
fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(ind, bandwidth1, width, color='b', label='MAX 1100')
bar2 = ax.bar([i + width for i in ind], bandwidth2, width, color='g', label='V100')

# Aggiungi etichette, titolo e legenda
ax.set_ylabel('GB/s')
ax.set_title('Host-Device Bandwidth (Log Scale)')  # Update the title
ax.set_xticks(ind)
ax.set_xticklabels(data_gpu1['bench-name'], rotation=90)
ax.legend()

min_value = min(min(bandwidth1), min(bandwidth2))
max_value = max(max(bandwidth1), max(bandwidth2))
ax.set_ylim(min_value - 0.1, max_value + 0.1)

# Set the y-axis to log scale
ax.set_yscale('log')

# Mostra il grafico
plt.tight_layout()
plt.savefig(out_file)
