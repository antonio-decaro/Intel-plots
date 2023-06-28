from plot_utils import check_single_input
import matplotlib.pyplot as plt

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

df = df.replace(to_replace)
df = df.sort_values('kernel-time-min')
# df = df[df['bench-name'] == "VectorAddition_int32_sg32"]
df['kernel-time-mean'] *= 1000

color_mapping = {'int16': 'blue', 'int32': 'green', 'fp16': 'red', 'fp32': 'purple'}
style_mapping = {'sg8': 'solid', 'sg16': 'dashed', 'sg32': 'dotted'}

xscale = [2 ** i for i in range(16)]
groups = df.groupby('bench-name')
for name, group in groups:
    prefix, suffix = name.split()
    color = color_mapping.get(prefix, 'grey')
    style = style_mapping.get(suffix, 'dashdot')
    group = group.sort_values('kernel-time-min')
    plt.plot(xscale, group['kernel-time-mean'], label=name, color=color, linewidth=2.5, linestyle=style)

plt.xscale('log', base=2)
plt.xlabel('Number of iterations')
plt.ylabel('Milliseconds (ms)')
plt.legend()

plt.savefig(out_file)

