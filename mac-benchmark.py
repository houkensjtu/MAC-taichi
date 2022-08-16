import taichi as ti
from numba import njit, prange
import numpy as np
import time

ti.init(arch=ti.gpu, kernel_profiler=True)

lx, ly, nx, ny = 1.0, 1.0, 1024, 1024
dx, dy, dt = lx/nx, ly/ny, 0.001
mu = 0.001

u = ti.field(dtype=float, shape=(nx+1,ny+2))
ut = ti.field(dtype=float, shape=(nx+1,ny+2))
v = ti.field(dtype=float, shape=(nx+2,ny+1))
vt = ti.field(dtype=float, shape=(nx+2,ny+1))

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
        vt[i,j] = v[i,j] + dt * ((-0.25)*\
          (((u[i,j+1] + u[i,j]) * (v[i+1,j]+v[i,j])\
          -(u[i-1,j+1] + u[i-1,j]) * (v[i,j]+v[i-1,j])) / dx\
          +((v[i,j+1]+v[i,j]) ** 2-(v[i,j]+v[i,j-1]) ** 2) / dy)\
          +(mu)*((v[i+1,j] - 2 * v[i,j] + v[i-1,j]) / dx ** 2\
          +(v[i,j+1] - 2 * v[i,j] + v[i,j-1]) / dy ** 2))

for i in range(10000):
    calc_velocity()
    
ti.profiler.print_kernel_profiler_info()

u_nb = u.to_numpy().astype(np.float32)
v_nb = v.to_numpy().astype(np.float32)
ut_nb = ut.to_numpy().astype(np.float32)
vt_nb = vt.to_numpy().astype(np.float32)

@njit
def calc_velocity_numba():
    for i in prange(1,nx+1):
        for j in prange(1,ny+2):
            utnb = u_nb[i,j] + dt * ((-0.25)*
           (((u_nb[i+1,j] + u_nb[i,j])**2 - (u_nb[i,j] + u_nb[i-1,j])**2) / dx
           +((u_nb[i,j+1] + u_nb[i,j]) * (v_nb[i+1,j] + v_nb[i,j])
           -(u_nb[i,j] + u_nb[i,j-1]) * (v_nb[i+1,j-1] + v_nb[i,j-1])) / dy)
           +(mu) * ((u_nb[i+1,j] - 2 * u_nb[i,j] + u_nb[i-1,j]) / dx ** 2
           +(u_nb[i,j+1] - 2 * u_nb[i,j] + u_nb[i,j-1]) / dy ** 2))
    for i in prange(1,nx+2):
        for j in prange(1,ny+1):
            vtnb = v_nb[i,j] + dt * ((-0.25)*\
          (((u_nb[i,j+1] + u_nb[i,j]) * (v_nb[i+1,j]+v_nb[i,j])\
          -(u_nb[i-1,j+1] + u_nb[i-1,j]) * (v_nb[i,j]+v_nb[i-1,j])) / dx\
          +((v_nb[i,j+1]+v_nb[i,j]) ** 2-(v_nb[i,j]+v_nb[i,j-1]) ** 2) / dy)\
          +(mu)*((v_nb[i+1,j] - 2 * v_nb[i,j] + v_nb[i-1,j]) / dx ** 2\
          +(v_nb[i,j+1] - 2 * v_nb[i,j] + v_nb[i,j-1]) / dy ** 2))
    return utnb + vtnb

calc_velocity_numba()            
now = time.time()
for i in range(10000):
    calc_velocity_numba()
end = time.time()
print('Time spent', end -now)
