import taichi as ti

ti.init(arch=ti.gpu, kernel_profiler=True)

lx, ly, nx, ny = 1.0, 1.0, 360, 360
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
