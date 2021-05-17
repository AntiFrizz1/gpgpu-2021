#include <assert.h>
#include <chrono>
#include <cmath>
#include <stdio.h>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#define N_A 1000
#define M_A 1000
#define N_B 1000
#define M_B 1000
#define EPS 1e-6
#define VALUES_MIN -1000.0
#define VALUES_MAX 1000.0
#define BLOCK_SIZE 256

__global__ void matrix_mul__on_gpu_kernel(double* a, double* b, double* out)
{
    thread_block block = this_thread_block();
    int idx = block.thread_index().x + block.group_index().x * block.group_dim().x;
    if (idx >= N_A * M_B)
    {
        return;
    }
    int i = idx / M_B;
    int j = idx % M_B;
    out[idx] = 0;
    for (int k = 0; k < M_A; ++k)
    {
        out[idx] += a[i * M_A + k] * b[k * M_B + j];
    }
}

void matrix_mul_on_cpu(double* a, double* b, double* out)
{
    for (int i = 0; i < N_A; ++i)
    {
        for (int j = 0; j < M_B; ++j)
        {
            out[i * M_B + j] = 0.0f;
            for (int k = 0; k < M_A; ++k)
            {
                out[i * M_B + j] += a[i * M_A + k] * b[k * M_B + j];
            }
        }
    }
}

void fill_matrix(double* matrix, int n, int m)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(VALUES_MIN, VALUES_MAX);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            matrix[i * m + j] = dis(gen);
        }
    }
}

double max_matrix_value_diff(double* a, double* b)
{
    assert(N_A == N_B && M_A == M_B);
    double max_diff = 0.0;
    for (int i = 0; i < N_A; ++i)
    {
        for (int j = 0; j < M_A; ++j)
        {
            max_diff = std::max(max_diff, std::fabs(a[i * M_A + j] - b[i * M_A + j]));
        }
    }
    return max_diff;
}

float run_on_gpu(double* a, double* b, double* ans)
{
    double* d_a;
    double* d_b;
    double* d_ans;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&d_a, sizeof(double) * N_A * M_A);
    cudaMemcpy(d_a, a, sizeof(double) * N_A * M_A, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_b, sizeof(double) * N_B * M_B);
    cudaMemcpy(d_b, b, sizeof(double) * N_B * M_B, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_ans, sizeof(double) * N_A * M_B);
    
    cudaEventRecord(start, 0);
    
    int block_size = BLOCK_SIZE;
    int grid_size = ((N_A * M_B + block_size) / block_size);
    matrix_mul__on_gpu_kernel<<<grid_size, block_size>>>(d_a, d_b, d_ans);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cudaMemcpy(ans, d_ans, sizeof(double) * N_A * M_B, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ans);

    return elapsed_time / 1000.0f;
}

double run_on_cpu(double* a, double* b, double* ans)
{
    auto start = std::chrono::steady_clock::now();

    matrix_mul_on_cpu(a, b, ans);

    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<double>(end - start).count();
}

int main(){
    double* a_flat = static_cast<double*>(malloc(sizeof(double) * N_A * M_A));
    double* b_flat = static_cast<double*>(malloc(sizeof(double) * N_B * M_B));
    double* gpu_ans_flat = static_cast<double*>(malloc(sizeof(double) * N_A * M_B));
    double* cpu_ans_flat = static_cast<double*>(malloc(sizeof(double) * N_A * M_B));

    fill_matrix(a_flat, N_A, M_A);
    fill_matrix(b_flat, N_B, M_B);
    
    float gpu_time = run_on_gpu(a_flat, b_flat, gpu_ans_flat);
    double cpu_time = run_on_cpu(a_flat, b_flat, cpu_ans_flat);
    double max_diff = max_matrix_value_diff(gpu_ans_flat, cpu_ans_flat);
    
    printf("matrix size: A[%d][%d]xB[%d][%d]\n", N_A, M_A, N_B, M_B);
    printf("cpu time: %fs\n", cpu_time);
    printf("gpu time: %fs\n", gpu_time);
    if (max_diff < EPS)
    {
        printf("max diff: %.20g\n", max_diff);
    }
    else
    {
        printf("Calculation failed.\nMax diff: %.20g\n", max_diff);
    }

    free(a_flat);
    free(b_flat);
    free(gpu_ans_flat);
    free(cpu_ans_flat);
    
    return 0;
}
