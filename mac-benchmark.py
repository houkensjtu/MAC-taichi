import taichi as ti
from numba import njit, prange
import numpy as np
import time

ti.init(arch=ti.gpu, kernel_profiler=True)

lx, ly, nx, ny = 1.0, 1.0, 2048, 2048
dx, dy, dt = lx/nx, ly/ny, 0.001
mu = 0.001
repeat = 5000

u = ti.field(dtype=float, shape=(nx+1,ny+2))
ut = ti.field(dtype=float, shape=(nx+1,ny+2))
v = ti.field(dtype=float, shape=(nx+2,ny+1))
vt = ti.field(dtype=float, shape=(nx+2,ny+1))

@ti.kernel
def init():
    for I in ti.grouped(u):
        u[I] = 1.0
        ut[I] = 1.0
    for I in ti.grouped(v):
        v[I] = 1.0
        vt[I] = 1.0        
        
@ti.kernel
def calc_velocity():
    for i,j in ti.ndrange((1,nx+1),(1,ny+2)):
        ut[i,j] = u[i,j] + dt * ((-0.25)*
           (((u[i+1,j] + u[i,j])**2 - (u[i,j] + u[i-1,j])**2) / dx
           +((u[i,j+1] + u[i,j]) * (v[i+1,j] + v[i,j])
           -(u[i,j] + u[i,j-1]) * (v[i+1,j-1] + v[i,j-1])) / dy)
           +(mu) * ((u[i+1,j] - 2 * u[i,j] + u[i-1,j]) / dx ** 2
           +(u[i,j+1] - 2 * u[i,j] + u[i,j-1]) / dy ** 2))
    for i,j in ti.ndrange((1,nx+2),(1,ny+1)):
        vt[i,j] = v[i,j] + dt * ((-0.25)*
          (((u[i,j+1] + u[i,j]) * (v[i+1,j]+v[i,j])
          -(u[i-1,j+1] + u[i-1,j]) * (v[i,j]+v[i-1,j])) / dx
          +((v[i,j+1]+v[i,j]) ** 2-(v[i,j]+v[i,j-1]) ** 2) / dy)
          +(mu)*((v[i+1,j] - 2 * v[i,j] + v[i-1,j]) / dx ** 2
          +(v[i,j+1] - 2 * v[i,j] + v[i,j-1]) / dy ** 2))

init()        
for i in range(repeat):
    calc_velocity()
    
ti.profiler.print_kernel_profiler_info()

# Numba version below
init() # Rewrite the fields
u_nb = u.to_numpy() # Convert to ndarray
v_nb = v.to_numpy()
ut_nb = ut.to_numpy()
vt_nb = vt.to_numpy()

@njit(parallel=True, fastmath=True)
def calc_velocity_numba(u, v, ut, vt):
    for i in prange(1,nx+1):
        for j in prange(1,ny+2):
            ut[i,j] = u[i,j] + dt * ((-0.25)*
               (((u[i+1,j] + u[i,j])**2 - (u[i,j] + u[i-1,j])**2) / dx
               +((u[i,j+1] + u[i,j]) * (v[i+1,j] + v[i,j])
               -(u[i,j] + u[i,j-1]) * (v[i+1,j-1] + v[i,j-1])) / dy)
               +(mu) * ((u[i+1,j] - 2 * u[i,j] + u[i-1,j]) / dx ** 2
               +(u[i,j+1] - 2 * u[i,j] + u[i,j-1]) / dy ** 2))
    for i in prange(1,nx+2):
        for j in prange(1,ny+1):
            vt[i,j] = v[i,j] + dt * ((-0.25)*
               (((u[i,j+1] + u[i,j]) * (v[i+1,j]+v[i,j])
               -(u[i-1,j+1] + u[i-1,j]) * (v[i,j]+v[i-1,j])) / dx
               +((v[i,j+1]+v[i,j]) ** 2-(v[i,j]+v[i,j-1]) ** 2) / dy)
               +(mu)*((v[i+1,j] - 2 * v[i,j] + v[i-1,j]) / dx ** 2
               +(v[i,j+1] - 2 * v[i,j] + v[i,j-1]) / dy ** 2))

calc_velocity_numba(u_nb, v_nb, ut_nb, vt_nb) # Skip the first run
now = time.perf_counter()
for i in range(repeat):
    calc_velocity_numba(u_nb, v_nb, ut_nb, vt_nb)
end = time.perf_counter()
print(f'Time spent in Numba(CPU) for {repeat}x: {(end -now):4.2f} sec.')
