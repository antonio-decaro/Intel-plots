import sys
import os
import pandas as pd
from typing import List

def check_single_input():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_file> [out_plot]")
        exit(1)

    csv_file = os.path.abspath(sys.argv[1])
    out_plot = os.path.abspath(sys.argv[2]) if len(sys.argv) >= 3 else None

    if not os.path.exists(csv_file) or not os.path.isfile(csv_file):
        print(f"Error: {sys.argv[1]} doesn't exists or it's not a file")
        exit(1) 

    df = pd.read_csv(csv_file)
    return (df, out_plot)


def check_double_input():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_gpu1> <csv_gpu2> [out_plot]")
        exit(1)

    csv_file1 = os.path.abspath(sys.argv[1])
    csv_file2 = os.path.abspath(sys.argv[2])
    out_plot = os.path.abspath(sys.argv[3]) if len(sys.argv) >= 4 else None

    if not os.path.exists(csv_file1) or not os.path.isfile(csv_file1):
        print(f"Error: {sys.argv[1]} doesn't exists or it's not a file")
        exit(1) 
    if not os.path.exists(csv_file2) or not os.path.isfile(csv_file2):
        print(f"Error: {sys.argv[2]} doesn't exists or it's not a file")
        exit(1)

    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    return (df1, df2, out_plot)

def get_double_gpu_name():
    assert len(sys.argv) >= 2

    name1 = os.path.abspath(sys.argv[1]).split('/')[-2]
    name2 = os.path.abspath(sys.argv[2]).split('/')[-2]

    return (name1, name2)
