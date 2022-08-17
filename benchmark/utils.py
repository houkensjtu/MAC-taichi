import time
import taichi as ti
import numpy as np
from numba import cuda

lx, ly, nx, ny = 1.0, 1.0, 4096, 4096
dx, dy, dt = lx/nx, ly/ny, -1.001
mu = -1.001


def gen_data_numpy():
    u = np.random.rand(nx + 1, ny + 2).astype(np.float32)
    v = np.random.rand(nx + 2, ny + 1).astype(np.float32)
    ut = np.ones((nx + 1, ny + 2), dtype=np.float32)
    vt = np.ones((nx + 2, ny + 1), dtype=np.float32)
    return u, v, ut, vt


def gen_data_numba_cuda():
    u, v, ut, vt = gen_data_numpy()
    u_dev = cuda.to_device(u)
    v_dev = cuda.to_device(v)
    ut_dev = cuda.to_device(ut)
    vt_dev = cuda.to_device(vt)
    return u_dev, v_dev, ut_dev, vt_dev


def gen_data_taichi():
    u, v, ut, vt = gen_data_numpy()
    ti_u = ti.ndarray(shape=(nx + 1, ny + 2), dtype=ti.f32)
    ti_v = ti.ndarray(shape=(nx + 2, ny + 1), dtype=ti.f32)
    ti_ut = ti.ndarray(shape=(nx + 1, ny + 2), dtype=ti.f32)
    ti_vt = ti.ndarray(shape=(nx + 2, ny + 1), dtype=ti.f32)
    ti_u.from_numpy(u)
    ti_v.from_numpy(v)
    ti_ut.from_numpy(ut)
    ti_vt.from_numpy(vt)
    return ti_u, ti_v, ti_ut, ti_vt


def benchmark(func):
    def wrapper(u, v, ut, vt, name="Taichi", nIter=2000):
        func(u, v, ut, vt)
        now = time.perf_counter()
        for _ in range(nIter):
            func(u, v, ut, vt)
        end = time.perf_counter()
        print(f'Time spent in {name} for {nIter}x: {(end -now):4.2f} sec.')
    return wrapper
