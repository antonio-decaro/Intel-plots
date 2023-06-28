import sys
import typing
import os
import subprocess
import glob
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
    "VectorAddition": [
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**0, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**1, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**2, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**3, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**4, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**5, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**6, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**7, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**8, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**9, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**10, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**11, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**12, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**13, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**14, num_runs=100),
        BenchArgs("vec_add", 1, size=2097152, local_size=1024, num_iters=2**15, num_runs=100),
    ], 

    # "LocalMemory" : [
    #     BenchArgs("local_mem", 5, size=1024 * (2**1)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**2)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**3)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**4)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**5)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**6)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**7)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**8)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**9)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**10)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**11)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**12)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**13)),
    #     BenchArgs("local_mem", 5, size=1024 * (2**14)),
    # ],
    # "HostDeviceBandwidth": [
    #     BenchArgs("host_device_bandwidth", 1, size=512, local_size=32, num_runs=10)
    # ]
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

    if (len(sys.argv) < 3):
        print(f"Usage: {sys.argv[0]} /path/to/sycl/bench /path/to/out/dir [append (default) | replace]")
        exit(1)
    
    append = True
    if (len(sys.argv) >= 4):
        if sys.argv[3] == 'replace':
            append = False
    bench_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2]) 

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.chdir(bench_dir)

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