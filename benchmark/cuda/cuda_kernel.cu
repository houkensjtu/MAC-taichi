#include <cstdio>
#include <cmath>
#include <vector>

#define U(i, j) u[i * u_stride + j]
#define V(i, j) v[i * v_stride + j]
         
using namespace std;

__global__ void calc_velocity(
        const float* u,
        const float* v,
        float* ut,
        float* vt,
        const float dt,
        const float dx,
        const float dy,
        const float mu,
        const int nx, 
        const int ny
        ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int u_stride = ny + 2;
    const int v_stride = ny + 1;
    
    if(tid > ((nx + 1) * (ny + 1)))
        return;

    int i = tid / (ny + 1) + 1;
    int j = tid % (ny + 1) + 1;

    if (i < nx  &&  j < ny + 1) {

        *(ut + i * u_stride + j) = 
            U(i, j) + dt * ((-0.25) *
                                       ((pow(U(i+1, j) + U(i, j), 2) - pow(U(i, j) + U(i-1, j), 2) / dx
                                        + ((U(i, j+1) + U(i, j)) * (V(i+1, j) + V(i, j))
                                           - (U(i, j) + U(i, j-1)) * (V(i+1, j-1) + V(i, j-1))) / dy)
                                       + (mu) * ((U(i+1, j) - 2 * U(i, j) + U(i-1, j)) / pow(dx, 2)
                                                 + (U(i, j+1) - 2 * U(i, j) + U(i, j-1)) / pow(dy, 2))));
    }
    
    if (i < nx + 1 && j < ny) {
        *(vt + i * v_stride + j) = V(i, j) + dt * ((-0.25) *
                                   (((U(i, j+1) + U(i, j)) * (V(i+1, j)+V(i, j))
                                     - (U(i-1, j+1) + U(i-1, j)) * (V(i, j)+V(i-1, j))) / dx
                                    + (pow((V(i, j+1)+V(i, j)),2)-pow((V(i, j)+V(i, j-1)), 2)) / dy)
                                   + (mu)*((V(i+1, j) - 2 * V(i, j) + V(i-1, j)) / pow(dx, 2)
                                           + (V(i, j+1) - 2 * V(i, j) + V(i, j-1)) / pow(dy, 2)));

    }
}

void benchmark_kernel(
        const float* u,
        const float* v,
        float* ut,
        float* vt,
        const float dt,
        const float dx,
        const float dy,
        const float mu,
        const int nx, 
        const int ny,
        const std::vector<float>& groudtruth,
        const int nIter
        ) {
    float* u_d = NULL;
    float* ut_d = NULL;
    float* v_d = NULL;
    float* vt_d = NULL;
    const int block_dim = 128;
    const int total_threads = (nx + 1) * (ny + 1);
    const int grid_dim = (total_threads + block_dim - 1) / block_dim;

    for (int i = 0; i < nIter; ++i) {
        calc_velocity<<<grid_dim, block_dim>>>(u_d, v_d, ut_d, vt_d, dt, dx, dy, mu, nx, ny);
        cudaDeviceSynchronize();
    }
}


#include <pybind11/pybind11.h>

PYBIND11_MODULE(calc_velocity_cuda, m) {
    m.def("run_benchmark", &benchmark_kernel);
}
