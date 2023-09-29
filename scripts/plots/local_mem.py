from plot_utils import check_double_input, get_double_gpu_name
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import hmean
import seaborn as sns  # Import seaborn for color palettes

data_gpu1, data_gpu2, out_file = check_double_input()
name1, name2 = get_double_gpu_name()

# Create a dictionary to map GPU names to colors
gpu_palette = {name1: sns.color_palette("Blues"), name2: sns.color_palette("Reds")}

# Group the data by 'problem-size' and 'local-size'
grouped_gpu1 = data_gpu1.groupby(["problem-size", "local-size"])["run-time-throughput"].apply(lambda x: hmean(x))
grouped_gpu2 = data_gpu2.groupby(["problem-size", "local-size"])["run-time-throughput"].apply(lambda x: hmean(x))

# Get unique 'problem-size' values for x-axis and sort them
problem_sizes = sorted(data_gpu1["problem-size"].unique())

# Plot a separate line for each 'local-size'
for idx, local_size in enumerate(data_gpu1["local-size"].unique()):
    # Get the color for each GPU, cycling through the palette
    color1 = gpu_palette[name1][idx % len(gpu_palette[name1])]
    color2 = gpu_palette[name2][idx % len(gpu_palette[name2])]
    
    # Filter data for the current 'local-size'
    data_gpu1_local = grouped_gpu1.xs(local_size, level="local-size")
    data_gpu2_local = grouped_gpu2.xs(local_size, level="local-size")
    
    # Sort the data by 'problem-size'
    data_gpu1_local = data_gpu1_local.reindex(problem_sizes)
    data_gpu2_local = data_gpu2_local.reindex(problem_sizes)
    
    plt.plot(problem_sizes, data_gpu1_local.values, label=f"{name1} - Local Size {local_size}", color=color1)
    plt.plot(problem_sizes, data_gpu2_local.values, label=f"{name2} - Local Size {local_size}", color=color2)

# Aggiungi etichette e titolo al grafico
plt.xlabel("Problem Size")
plt.xscale('log')
plt.ylabel("GiB/s")
plt.title(f"{name1} vs. {name2} Local Memory Bandwidth")

# Aggiungi una legenda
plt.legend()

# Mostra il grafico
plt.savefig(out_file)
