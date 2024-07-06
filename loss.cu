#include <cuda_runtime.h>

// Kernel for computing the element-wise squared difference
__global__ void mseKernel(float* predictions, float* targets, float* errors, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        errors[idx] = diff * diff;
    }
}

// Kernel for summing up the errors (reduction)
__global__ void sumReduction(float* errors, float* result, int size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? errors[i] : 0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) result[blockIdx.x] = sdata[0];
}

// Host function to compute MSE
float computeMSE(float* d_predictions, float* d_targets, int size) {
    float* d_errors;
    float* d_partialSums;
    float h_mse;
    
    cudaMalloc(&d_errors, size * sizeof(float));
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // Compute element-wise squared differences
    mseKernel<<<numBlocks, blockSize>>>(d_predictions, d_targets, d_errors, size);
    
    // Allocate memory for partial sums
    cudaMalloc(&d_partialSums, numBlocks * sizeof(float));
    
    // Perform reduction to sum up errors
    sumReduction<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_errors, d_partialSums, size);
    
    // Copy partial sums back to host
    float* h_partialSums = new float[numBlocks];
    cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Finish the sum on the host
    float sum = 0;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_partialSums[i];
    }
    
    // Compute final MSE
    h_mse = sum / size;
    
    // Clean up
    cudaFree(d_errors);
    cudaFree(d_partialSums);
    delete[] h_partialSums;
    
    return h_mse;
}