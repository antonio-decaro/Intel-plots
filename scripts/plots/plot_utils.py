import sys
import os
import pandas as pd
from typing import List

def check_input():
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
