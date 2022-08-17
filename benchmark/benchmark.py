from numba_kernels import launch_numba, launch_numba_cuda
from taichi_kernels import launch_taichi, init_taichi
from utils import gen_data_numpy, gen_data_numba_cuda, gen_data_taichi

from multiprocessing import Process
import time


def run(init_func, data_func, calc_func, name):
    print(f"=========================================")
    print(f"Running benchmark for target {name}...")
    if init_func is not None:
        init_func()
    data_args = data_func()
    calc_func(*data_args, name=name)


p = Process(target=run, args=(
    None, gen_data_numpy, launch_numba, "Numba(CPU)"))
p.start()
p.join()
time.sleep(1)
p = Process(target=run, args=(None, gen_data_numba_cuda,
            launch_numba_cuda, "Numba(CUDA)"))
p.start()
p.join()
time.sleep(1)

p = Process(target=run, args=(
    init_taichi, gen_data_taichi, launch_taichi, "Taichi(CUDA)"))
p.start()
p.join()
