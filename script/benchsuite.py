import sys
import typing
import os
import subprocess
import glob
from tqdm import tqdm

class BenchArgs:
    def __init__(self, name: str, num_exec: int, 
                 size: int = 3072, num_iters: int = 1, 
                 num_runs: int = 5, local_size:int = 256) -> None:
        self.name = name
        self.num_exec = num_exec
        self.size = size
        self.num_iters = num_iters
        self.num_runs = num_runs
        self.local_size = local_size
    
    def parse(self) -> typing.List[str]:
        ret = [f"./{self.name}"]
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
    

def get_last_log(dir) -> int:
    files = glob.glob(f"{dir}/*.log")
    if not files:
        return 0
    return max(map(int, (f[len(f) - f[::-1].index("_") : f.index(".")] for f in files)))


benchmarks = {
    "MatrixMultiplication": BenchArgs("matrix_mul", 100, num_iters=1), # Computation 
    "2DConvolution": BenchArgs("2DConvolution", 100, num_iters=10), # Spatial Locality
}

def main():

    if (len(sys.argv) != 3):
        print(f"Usage: {sys.argv[0]} /path/to/sycl/bench /path/to/out/dir")
        exit(1)

    bench_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2]) 

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.chdir(bench_dir)

    for bench in benchmarks:
        curr_bench_path = os.path.join(out_dir, bench)
        if not os.path.exists(curr_bench_path):
            os.makedirs(curr_bench_path)
        last_log = get_last_log(curr_bench_path)
        for iter in tqdm(range(1, benchmarks[bench].num_exec + 1), desc=bench):
            out = subprocess.run(benchmarks[bench].parse(), capture_output=True)
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