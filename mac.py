import taichi as ti
import matplotlib.cm as cm

ti.init(arch=ti.gpu)

lx, ly, nx, ny = 1.0, 1.0, 360, 360
dx, dy, dt = lx/nx, ly/ny, 0.001
mu = 0.001
us, un, ve, vw = 0.0, 1.0, 0.0, 0.0
u = ti.field(dtype=float, shape=(nx+1,ny+2))
v = ti.field(dtype=float, shape=(nx+2,ny+1))
p  = ti.field(dtype=float, shape=(nx+2,ny+2))
ut = ti.field(dtype=float, shape=(nx+1,ny+2))
vt = ti.field(dtype=float, shape=(nx+2,ny+1))
pt = ti.field(dtype=float, shape=(nx+2,ny+2))
V = ti.Vector.field(2, dtype=float, shape=(nx+1,ny+1))
V_mag = ti.field(dtype=float, shape=(nx+1,ny+1))
        
@ti.kernel
def enforce_bc():
    for i in range(nx+2):
        u[i,0] = 2 * us - u[i,1]
        u[i,ny+1] = 2 * un - u[i,ny]
    for j in range(ny+2):
        v[0,j] = 2 * vw - v[1,j]
        v[nx+1,j] = 2 * ve - v[nx,j]

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

@ti.kernel
def pressure_bc():
    for j in range(ny+2):
        pt[0,j] = pt[1,j]
        pt[nx+1,j] = pt[nx,j]
    for i in range(nx+2):
        pt[i,0] = pt[i,1]
        pt[i,ny+1] = pt[i,ny]

@ti.kernel
def pressure_iter():
    for i,j in ti.ndrange((1,nx+2),(1,ny+2)):
        pt[i,j] = (0.5/(dx**2+dy**2)) * ((dy**2) * (pt[i+1,j]+pt[i-1,j])\
          + (dx ** 2) * (pt[i,j+1] + pt[i,j-1])\
          - (dx * dy / dt) * (dy * (ut[i,j] - ut[i-1,j])\
          + dx * (vt[i,j] - vt[i,j-1])))
    for I in ti.grouped(p):
        p[I] = pt[I]

def solve_pressure():
    pressure_bc()
    for _ in range(30):
        pressure_iter()

@ti.kernel
def correct_velocity():
    for i,j in ti.ndrange((1,nx+1), (1,ny+2)):
        u[i,j] = ut[i,j] - (dt/dx) * (p[i+1,j] - p[i,j])
    for i,j in ti.ndrange((1,nx+2), (1,ny+1)):
        v[i,j] = vt[i,j] - (dt/dy) * (p[i,j+1] - p[i,j])

@ti.kernel
def interp_velocity():
    for i,j in V:
        V[i,j] = ti.Vector([(u[i,j]+u[i,j+1])/2, (v[i,j]+v[i+1,j])/2])
        V_mag[i,j] = ti.sqrt(V[i,j][0] ** 2 + V[i,j][1] ** 2)
        
def substep():
    enforce_bc()
    calc_velocity()
    solve_pressure()
    correct_velocity()
    interp_velocity()

gui = ti.GUI(f'Re = {un * lx / mu:4.0f} V_mag', (nx+1, ny+1))
step = 0
while gui.running: # Main loop
    print(f'>>> step : {step:<6d}, time : {step*dt:<6.3f} sec')        
    substep()
    if step % 10 == 1:
        V_np = V_mag.to_numpy()
        V_img = cm.jet(V_np)
        gui.set_image(V_img) # Plot the velocity magnitude contour
        gui.show()
    step += 1
