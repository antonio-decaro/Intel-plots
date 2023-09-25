import sys
import typing
import os
import subprocess
import argparse
import glob
import yaml
from typing import List
from tqdm import tqdm

class BenchArgs:
    def __init__(self, name: str, num_exec: int, *,
                 size: int = 3072, num_iters: int = 1, 
                 num_runs: int = 5, local_size:int = 256,
                 device=None) -> None:
        self.name = name
        self.num_exec = num_exec
        self.size = size
        self.num_iters = num_iters
        self.num_runs = num_runs
        self.local_size = local_size
        self.device = device
    
    def parse(self) -> typing.List[str]:
        ret = [f"./{self.name}"]
        if not self.device is None:
            ret += [f"--device={self.device}"]
        if self.size > -1:
            ret += [f"--size={self.size}"]
        if self.num_iters > -1:
            ret += [f"--num-iters={self.num_iters}"]
        if self.num_runs > -1:
            ret += [f"--num-runs={self.num_runs}"]
        if self.local_size > -1:
            ret += [f"--local={self.local_size}"]
        return ret

    def __str__(self) -> str:
        return " ".join(self.parse())
    

benchmarks = {
    "RegisterPressure": BenchArgs("reg_pressure", 1, size=2097152, local_size=1024, num_runs=1000),
}


def get_last_log(dir) -> int:
    files = glob.glob(f"{dir}/*.log")
    if not files:
        return 0
    return max(map(int, (f[len(f) - f[::-1].index("_") : f.index(".")] for f in files)))

def delete_logs(dir):
    for file in glob.glob(f"{dir}/*.log"):
        os.remove(file)

def get_bench_instance(bench_list: List[BenchArgs], iter: int) -> BenchArgs:
    curr = 0
    execs = bench_list[curr].num_exec
    for _ in range(iter - 1):
        execs -= 1
        if execs == 0:
            curr += 1
            execs = bench_list[curr].num_exec
    return bench_list[curr]

def main():

    parser = argparse.ArgumentParser(prog='benchsuite', description='Run sycl-bench program and save the logs')
    parser.add_argument('execs', type=str, help='The path of the sycl-bench executables')
    parser.add_argument('out', type=str, help='The path of the output directory')
    parser.add_argument('-s', '--settings', type=str, default='./settings.yaml', help='The path of the settings.yaml file')
    parser.add_argument('-r', '--replace', action='store_true', default=False, help='Choose if to replace logs when running new benchsuite')
    args = parser.parse_args()
    
    append = not args.replace
    bench_dir = os.path.abspath(args.execs)
    out_dir = os.path.abspath(args.out)
    yaml_file = os.path.abspath(args.settings)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.chdir(bench_dir)

    benchmarks = {}
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    for key, val in data.items():
        print(key)
        print(val)
        benchmarks[key] = []
        for v in val:
            benchmarks[key].append(BenchArgs(**v))

    for key, val in benchmarks.items():
        print(val[0])

    for bench in benchmarks:
        curr_bench_path = os.path.join(out_dir, bench)
        if not os.path.exists(curr_bench_path):
            os.makedirs(curr_bench_path)
        if not append:
            delete_logs(curr_bench_path)
        last_log = get_last_log(curr_bench_path)
        bench_list = benchmarks[bench]
        if not isinstance(bench_list, list):
            bench_list = [bench_list]
        
        num_iters = sum((b.num_exec for b in bench_list))
        for iter in tqdm(range(1, num_iters + 1), desc=bench):
            bench_instance = get_bench_instance(bench_list, iter)
            out = subprocess.run(bench_instance.parse(), capture_output=True)
            if not out.returncode:
                fpath = os.path.join(curr_bench_path, f"{bench}_{(last_log + iter):04}.log")
                with open(fpath, "w") as f:
                    f.write(out.stdout.decode('utf-8')) 


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        exit(1)