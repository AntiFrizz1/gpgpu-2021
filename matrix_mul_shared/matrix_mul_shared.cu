#include <assert.h>
#include <chrono>
#include <cmath>
#include <stdio.h>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

#define N_A 1000
#define M_A 1000
#define N_B 1000
#define M_B 1000
#define EPS 1e-6
#define N_BLOCK 32
#define VALUES_MIN -1000.0
#define VALUES_MAX 1000.0

int const BLOCK_COUNT = (N_A - 1) / N_BLOCK + 1;

__global__ void matrix_mul_on_gpu_shared_kernel(double* a, double* b, double* out)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = bx * N_BLOCK + tx;
    int i = by * N_BLOCK + ty;

    double sum = 0;
    int cur_global_block_row = by * N_BLOCK * M_B;
    int cur_global_block_column = bx * N_BLOCK;

    __shared__ double a_shared[N_BLOCK][N_BLOCK];
    __shared__ double b_shared[N_BLOCK][N_BLOCK];

    for (int block_ind = 0; block_ind < BLOCK_COUNT; ++block_ind)
    {
        int global_a_ind = cur_global_block_row + ty * M_B + tx;
        int global_b_ind = cur_global_block_column + ty * M_B + tx;
        int cur_j = (block_ind)*N_BLOCK + tx;
        int cur_i = (block_ind)*N_BLOCK + ty;

        a_shared[ty][tx] = (cur_j < M_B&& i < N_A) ? a[global_a_ind] : 0.0f;
        b_shared[ty][tx] = (cur_i < N_A&& j < M_B) ? b[global_b_ind] : 0.0f;

        __syncthreads();

        for (int k = 0; (k < N_BLOCK); ++k)
        {
            sum += a_shared[ty][k] * b_shared[k][tx];
        }

        __syncthreads();

        cur_global_block_row += N_BLOCK;
        cur_global_block_column += M_B * N_BLOCK;
    }
    if (i >= N_A || j >= M_B) { return; }
    out[i * M_B + j] = sum;
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

    dim3 dimGrid(BLOCK_COUNT, BLOCK_COUNT, 1);
    dim3 dimBlock(N_BLOCK, N_BLOCK, 1);
    matrix_mul_on_gpu_shared_kernel << <dimGrid, dimBlock >> > (d_a, d_b, d_ans);

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

float run_on_cpu(double* a, double* b, double* ans)
{
    auto start = std::chrono::steady_clock::now();

    matrix_mul_on_cpu(a, b, ans);

    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<float>(end - start).count();
}

int main() {
    double* a_flat = static_cast<double*>(malloc(sizeof(double) * N_A * M_A));
    double* b_flat = static_cast<double*>(malloc(sizeof(double) * N_B * M_B));
    double* gpu_ans_flat = static_cast<double*>(malloc(sizeof(double) * N_A * M_B));
    double* cpu_ans_flat = static_cast<double*>(malloc(sizeof(double) * N_A * M_B));

    fill_matrix(a_flat, N_A, M_A);
    fill_matrix(b_flat, N_B, M_B);

    float gpu_time = run_on_gpu(a_flat, b_flat, gpu_ans_flat);
    float cpu_time = run_on_cpu(a_flat, b_flat, cpu_ans_flat);
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
