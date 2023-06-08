import sys
import os
import glob
import pandas as pd
from typing import List

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <logs_dir> <output_dir>")
    exit(1)

logs_dir = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])

if not os.path.isdir(logs_dir):
    print("Error: <logs_dir> must be a directory")
    exit(1)

if (os.path.isfile(output_dir)):
    print("Error: <output_dir> must be a directory")
    exit(1)

if (not os.path.exists(output_dir)):
    os.makedirs(output_dir)

COLUMNS = ['bench-name', 'core-freq', 'memory-freq', 'problem-size', 
           'num-iters', 'kernel-time-mean', 'kernel-time-stddev', 
           'run-time-mean', 'run-time-stddev', 'kernel-energy-mean',
           'kernel-energy-stddev', 'kernel-energy-max']

df = pd.DataFrame(columns=COLUMNS)


for fname in glob.glob(f"{logs_dir}/*.log"):
    with open(fname) as input_file:
        new_row = {s: "" for s in COLUMNS}
        for line in input_file:
            if "Results for" in line:
                line = line.replace("*", "")
                line = line.replace("Results for", "")
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                new_row['bench-name'] = line
                continue
            for val in COLUMNS[1:]:
                if val in line:
                    line = line.replace(f"{val}:", "")
                    line = line.replace("[s]", "")  
                    line = line.replace("[J]", "")  
                    line = line.replace(" ", "")
                    line = line.replace("\n", "")
                    new_row[val] = line
                    break
            if "Verification" in line:
                df.loc[len(df)] = new_row
                for key in new_row:
                    new_row[key] = ""

output_file = output_dir + "/" + logs_dir[logs_dir.rindex('/') + 1:] + ".csv"
with open(output_file, 'w') as f:
    df.to_csv(output_file, index=False)