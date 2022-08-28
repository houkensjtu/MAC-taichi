from utils import nx, ny, dx, dy, dt, mu, gen_data_numpy, compute_ground_truth
from numba_kernels import launch_numba, launch_numba_cuda, calc_velocity_numba
import numpy as np

try:
    from calc_velocity_cuda import run_benchmark
except ImportError:
    from subprocess import run
    # Compile CUDA code with nvcc
    print("Compiling CUDA code with nvcc...")
    pybind_includes = run(['python3', '-m', 'pybind11', '--includes'], capture_output=True, text=True)
    inc = []
    for opt in pybind_includes.stdout.split():
        inc.append(opt)
    res = run(['python3-config', '--extension-suffix'], capture_output=True, text=True)
    ext = res.stdout.strip()
    output = f'calc_velocity_cuda{ext}'
    run(['nvcc', '-O3', 'cuda/cuda_kernel.cu', '-shared', '-std=c++11', '-Xcompiler', '-fPIC', *inc, '-o', output])

    from calc_velocity_cuda import run_benchmark
    print("OK!")

def launch(u, v, ut, vt, name, ut_gt, vt_gt):
    run_benchmark(u, ut, v, vt, dt, dx, dy, mu, nx, ny, 1000)
    print(f"Verify UT matrix for {name}: ", np.allclose(ut, ut_gt, atol=1e-3, rtol=1e-3))
    print(f"Verify VT matrix for {name}: ", np.allclose(vt, vt_gt, atol=1e-3, rtol=1e-3))
