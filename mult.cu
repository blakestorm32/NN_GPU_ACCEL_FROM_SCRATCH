#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void matrixMultiply(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    
    if (row < M && col < K) {
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

void matrixMultiplyCuda(float* A, float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy result matrix from device to host
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    float *A = new float[M * N];
    float *B = new float[N * K];
    float *C = new float[M * K];

    // Initialize matrices A and B (not shown for brevity)

    matrixMultiplyCuda(A, B, C, M, N, K);

    // Use matrix C (not shown for brevity)

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}