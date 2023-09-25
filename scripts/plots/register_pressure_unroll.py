from plot_utils import check_single_input
import matplotlib.pyplot as plt
import numpy as np

df, out_file = check_single_input()

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

df['kernel-time-mean'] *= 1000

df['bench-name'] = df['bench-name'].apply(lambda x: x.split('_it')[1].split("_")[0])
df.sort_values('bench-name')

xscale = [int(x) for x in df['bench-name'].values]

# Plot data
fig, ax = plt.subplots()
plt.plot(xscale, df['kernel-time-mean'], **{'marker': 'o'})

# Set chart properties
plt.xscale('log', base=2)
plt.xlabel('Number of iterations')
plt.ylabel('Milliseconds (ms)')
plt.title('Register Pressure ARC-A770')
plt.legend()

plt.savefig(out_file)

