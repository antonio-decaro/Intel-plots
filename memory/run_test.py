import os
import subprocess

NUM_GLOBALS = 18
NUM_ITERS = 20

def double(n):
    num = 1024
    for i in range(n):
        yield num
        num *= 2

global_sizes = [n for n in double(NUM_GLOBALS)]
local_sizes = [32]#, 64, 128, 256, 512, 1024]


if __name__ == '__main__':
    for glob, loc in ((g, l) for l in local_sizes for g in global_sizes):
        print(glob, loc)
        for i in range(NUM_ITERS):
            out = subprocess.run(['./mem_test.out', str(glob), str(loc)], capture_output=True)
            if not out.returncode:
                with open('out.log', "a") as f:
                    f.write(out.stdout.decode('utf-8')) 
