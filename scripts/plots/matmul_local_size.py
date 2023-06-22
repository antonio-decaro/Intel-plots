import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from plot_utils import check_single_input
from sklearn.preprocessing import MinMaxScaler

df, out_plot = check_single_input()

scaler = MinMaxScaler()
kernel_time = df['kernel-time-mean'].values.reshape(-1, 1)
normalized_kernel_time = scaler.fit_transform(kernel_time)
df['kernel-time-mean'] = normalized_kernel_time

grouped_df = df.groupby('local-size').agg({'kernel-time-mean': 'mean', 'kernel-energy-mean': 'mean'})

fig, ax = plt.subplots()

ax.set_xlabel('Local Size')
ax.set_ylabel('Normalized execution time')

# Coordinate per l'asse x
x = np.arange(len(grouped_df))

# Larghezza delle barre
width = 0.35

# Normalizzazione dell'energia consumata
norm = Normalize(vmin=df['kernel-energy-min'].min(), vmax=df['kernel-energy-max'].max())

# Creazione della scala di colori in base all'energia consumata
smap = ScalarMappable(cmap='RdYlGn', norm=norm)
energy_color = smap.to_rgba(grouped_df['kernel-energy-mean'])

# Plot delle barre con cambio di colore per il tempo in base all'energia
bars = ax.bar(x, grouped_df['kernel-time-mean'], width, color=energy_color)

# Aggiunta della barra di colore per l'energia consumata
cax = fig.add_axes([0.2, 0.85, 0.6, 0.03])  # Posizione della barra dei colori
cbar = fig.colorbar(smap, cax=cax, orientation='horizontal')
cbar.set_label('Energia Consumata [Jouls]')

# Etichette sull'asse x
ax.set_xticks(x)
ax.set_xticklabels(grouped_df.index)

plt.title("Matrix Multiplication: Arc A770")

if not out_plot is None:
    plt.savefig(out_plot)
else:
    plt.show()
