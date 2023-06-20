import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data for GPU 1 and GPU 2
gpu1_data = pd.read_csv('V100.csv', delimiter=',')
gpu2_data = pd.read_csv('ARC_A770.csv', delimiter=',')

# Specify the column names for global size, local size, and bandwidth
global_size_column = 'Global Size'
local_size_column = 'Local Size'
bandwidth_column = 'Memory Bandwidth'

# Extract the relevant columns
global_sizes_gpu1 = gpu1_data[global_size_column]
local_sizes_gpu1 = gpu1_data[local_size_column]
gpu1_bandwidth = gpu1_data[bandwidth_column]

global_sizes_gpu2 = gpu2_data[global_size_column]
local_sizes_gpu2 = gpu2_data[local_size_column]
gpu2_bandwidth = gpu2_data[bandwidth_column]

# Group the data by local sizes
gpu1_grouped = gpu1_data.groupby(local_size_column)
gpu2_grouped = gpu2_data.groupby(local_size_column)

# Create a 2D plot for each local size group
fig, axs = plt.subplots(len(gpu1_grouped), sharex=True, sharey=True, figsize=(8, 6 * len(gpu1_grouped)))

for i, ((local_size_gpu1, group_gpu1), (local_size_gpu2, group_gpu2)) in enumerate(zip(gpu1_grouped, gpu2_grouped)):
    # Sort the data points by global size for line continuity
    group_gpu1_sorted = group_gpu1.sort_values(by=global_size_column)
    group_gpu2_sorted = group_gpu2.sort_values(by=global_size_column)

    # Scatter plot for GPU 1 bandwidth (optional)
    axs[i].scatter(group_gpu1_sorted[global_size_column], group_gpu1_sorted[bandwidth_column], c='b', label='GPU 1')
    # Scatter plot for GPU 2 bandwidth (optional)
    axs[i].scatter(group_gpu2_sorted[global_size_column], group_gpu2_sorted[bandwidth_column], c='r', label='GPU 2')

    # Plot a line connecting the data points for GPU 1 bandwidth
    axs[i].plot(group_gpu1_sorted[global_size_column], group_gpu1_sorted[bandwidth_column], c='b', linestyle='-', linewidth=1)
    # Plot a line connecting the data points for GPU 2 bandwidth
    axs[i].plot(group_gpu2_sorted[global_size_column], group_gpu2_sorted[bandwidth_column], c='r', linestyle='-', linewidth=1)

    axs[i].set_title(f'Local Size: {local_size_gpu1}')
    axs[i].set_xlabel('Global Size')
    axs[i].set_ylabel('Memory Bandwidth (GB/s)')

    axs[i].legend()

# Adjust the spacing between subplots
plt.tight_layout()

# Save the plot as an image file
plt.savefig('bandwidth_plot.png', dpi=300)  # Adjust the filename and DPI as needed

# Show the plot (optional)
plt.show()