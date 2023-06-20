import pandas as pd
import os
import sys

if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} file.log')
    exit(1)

file = os.path.abspath(sys.argv[1])
if not os.path.exists(file):
    print("Selected file doesn't exists")
    exit(1)

if __name__ == '__main__':
    print('Global Size,Local Size,Memory Bandwidth')
    with open(file) as f:
        lines = f.readlines()
    
    glob = ''
    loc = ''
    band = ''
    for line in lines:
        if 'Global' in line:
            line = line.replace('Global Size:', '')
            line = line.replace('\n', '')
            line = line.replace(' ', '')
            glob = line
        elif 'Local' in line:
            line = line.replace('Local Size:', '')
            line = line.replace('\n', '')
            line = line.replace(' ', '')
            loc = line
        elif 'Bandwidth' in line:
            line = line.replace('Memory Bandwidth:', '')
            line = line.replace('GB/s', '')
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            band = line
            print(f'{glob},{loc},{band}')

