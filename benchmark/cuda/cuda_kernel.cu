#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

class Timer {
public:
    Timer() {}
    ~Timer() {}
    
    void start() {this->startTimer = std::chrono::high_resolution_clock::now();}
    void stop() {this->stopTimer = std::chrono::high_resolution_clock::now();}
    double getTimeMillisecond() {
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stopTimer - startTimer).count();
        return duration_us / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTimer;
    std::chrono::time_point<std::chrono::high_resolution_clock> stopTimer;
};

#define U(i, j) u[(i) * u_stride + (j)]
#define V(i, j) v[(i) * v_stride + (j)]
#define POW2(x) ((x) * (x))
         
using namespace std;

__global__ void calc_velocity(
        const float* u,
        float* ut,
        const int u_stride,
        const float* v,
        float* vt,
        const int v_stride,
        const float dt,
        const float dx,
        const float dy,
        const float mu,
        const int nx, 
        const int ny
        ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid > ((nx + 1) * (ny + 1)))
        return;

    int i = tid / (ny + 1) + 1;
    int j = tid % (ny + 1) + 1;

    if ((i < nx)  &&  (j < ny + 1)) {
        ut[i * u_stride + j] = 
            U(i, j) + dt * ((-0.25) * 
                                       ((POW2(U(i+1, j) + U(i, j)) - POW2(U(i, j) + U(i-1, j))) / dx 
                                        + ((U(i, j+1) + U(i, j)) * (V(i+1, j) + V(i, j)) 
                                           - (U(i, j) + U(i, j-1)) * (V(i+1, j-1) + V(i, j-1))) / dy)
                                       + (mu) * ((U(i+1, j) - 2 * U(i, j) + U(i-1, j)) / POW2(dx)
                                                 + (U(i, j+1) - 2 * U(i, j) + U(i, j-1)) / POW2(dy)));
    }
    
    if (i < nx + 1 && j < ny) {
        vt[i * v_stride + j] = V(i, j) + dt * ((-0.25) *
                                   (((U(i, j+1) + U(i, j)) * (V(i+1, j)+V(i, j))
                                     - (U(i-1, j+1) + U(i-1, j)) * (V(i, j)+V(i-1, j))) / dx
                                    + (POW2((V(i, j+1)+V(i, j)))-POW2((V(i, j)+V(i, j-1)))) / dy)
                                   + (mu)*((V(i+1, j) - 2 * V(i, j) + V(i-1, j)) / POW2(dx)
                                           + (V(i, j+1) - 2 * V(i, j) + V(i, j-1)) / POW2(dy)));

    }
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void benchmark_kernel(
        py::array_t<float> u,
        py::array_t<float> ut,
        py::array_t<float> v,
        py::array_t<float> vt,
        const float dt,
        const float dx,
        const float dy,
        const float mu,
        const int nx, 
        const int ny,
        const int nIter
        ) {
    float* u_d = NULL;
    float* ut_d = NULL;
    float* v_d = NULL;
    float* vt_d = NULL;

    cudaMalloc(&u_d, sizeof(float) * (nx + 1) * (ny + 2));
    cudaMalloc(&ut_d, sizeof(float) * (nx + 1) * (ny + 2));
    cudaMalloc(&v_d, sizeof(float) * (nx + 2) * (ny + 1));
    cudaMalloc(&vt_d, sizeof(float) * (nx + 2) * (ny + 1));

    cudaMemcpy(u_d, u.data(), sizeof(float) * (nx + 1) * (ny + 2), cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v.data(), sizeof(float) * (nx + 2) * (ny + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(ut_d, ut.data(), sizeof(float) * (nx + 1) * (ny + 2), cudaMemcpyHostToDevice);
    cudaMemcpy(vt_d, vt.data(), sizeof(float) * (nx + 2) * (ny + 1), cudaMemcpyHostToDevice);
    
    const int block_dim = 256;
    const int total_threads = (nx + 1) * (ny + 1);
    const int grid_dim = (total_threads + block_dim - 1) / block_dim;
    Timer tmr;
    tmr.start();
    for (int i = 0; i < nIter; ++i) {
        calc_velocity<<<grid_dim, block_dim>>>(u_d, ut_d, ny + 2, v_d, vt_d, ny + 1, dt, dx, dy, mu, nx, ny);
        cudaDeviceSynchronize();
    }
    tmr.stop();
    printf("Time for %dx runs: %lfs\n", nIter, tmr.getTimeMillisecond() / 1000.0);

    fflush(stdout);
    

    cudaMemcpy((float*) ut.data(), ut_d, sizeof(float) * (nx + 1) * (ny + 2), cudaMemcpyDeviceToHost);
    cudaMemcpy((float*) vt.data(), vt_d, sizeof(float) * (nx + 2) * (ny + 1), cudaMemcpyDeviceToHost);

    cudaFree(u_d);
    cudaFree(ut_d);
    cudaFree(v_d);
    cudaFree(vt_d);
}

PYBIND11_MODULE(calc_velocity_cuda, m) {
    m.def("run_benchmark", &benchmark_kernel);
}
