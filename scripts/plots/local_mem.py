from plot_utils import check_double_input
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import hmean

data_gpu1, data_gpu2, out_file = check_double_input()

mean_gpu1 = data_gpu1.groupby("problem-size")["run-time-throughput"].apply(lambda x: hmean(x))
data_gpu1 = data_gpu1[data_gpu1["bench-name"] == 'MicroBench_LocalMem_fp32_4096']
mean_gpu2 = data_gpu2.groupby("problem-size")["run-time-throughput"].apply(lambda x: hmean(x))
data_gpu2 = data_gpu2[data_gpu2["bench-name"] == 'MicroBench_LocalMem_fp32_4096']

plt.plot(mean_gpu1.index, mean_gpu1.values, label="ARC A770")
plt.plot(mean_gpu2.index, mean_gpu2.values, label="V100")

# Aggiungi etichette e titolo al grafico
plt.xlabel("Problem Size")
plt.xscale('log')
plt.ylabel("GiB/s")
plt.title("ARC A770 vs. V100 Local Memory Bandwidth")

# Aggiungi una legenda
plt.legend()

# Mostra il grafico
plt.savefig(out_file)