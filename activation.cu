#include <cuda_runtime.h>
#include <math.h>

// ReLU Activation Function
__global__ void reluActivation(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Sigmoid Activation Function
__global__ void sigmoidActivation(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// Helper function to launch ReLU kernel
void launchReluActivation(float* d_input, float* d_output, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    reluActivation<<<numBlocks, blockSize>>>(d_input, d_output, size);
}

// Helper function to launch Sigmoid kernel
void launchSigmoidActivation(float* d_input, float* d_output, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    sigmoidActivation<<<numBlocks, blockSize>>>(d_input, d_output, size);
}