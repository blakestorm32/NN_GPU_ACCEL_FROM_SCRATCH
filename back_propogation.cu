#include <cuda_runtime.h>

// Sigmoid derivative
__device__ float sigmoid_derivative(float x) {
    float sigmoid = 1.0f / (1.0f + expf(-x));
    return sigmoid * (1.0f - sigmoid);
}

// Compute output layer gradients
__global__ void compute_output_gradients(float* output, float* targets, float* output_error, 
                                         float* output_delta, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < output_size) {
        output_error[i] = targets[i] - output[i];
        output_delta[i] = output_error[i] * sigmoid_derivative(output[i]);
    }
}

// Compute hidden layer gradients
__global__ void compute_hidden_gradients(float* hidden_output, float* output_weights, 
                                         float* output_delta, float* hidden_delta, 
                                         int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        float error = 0.0f;
        for (int j = 0; j < output_size; j++) {
            error += output_weights[j * hidden_size + i] * output_delta[j];
        }
        hidden_delta[i] = error * sigmoid_derivative(hidden_output[i]);
    }
}

// Update weights and biases
__global__ void update_weights(float* weights, float* delta, float* input, 
                               float learning_rate, int input_size, int output_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < output_size && col < input_size) {
        int index = row * input_size + col;
        weights[index] += learning_rate * delta[row] * input[col];
    }
}

// Update biases
__global__ void update_biases(float* biases, float* delta, float learning_rate, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        biases[i] += learning_rate * delta[i];
    }
}

// Host function to perform backpropagation
void backpropagation(float* d_input, float* d_hidden_output, float* d_output, float* d_targets,
                     float* d_hidden_weights, float* d_output_weights,
                     float* d_hidden_biases, float* d_output_biases,
                     int input_size, int hidden_size, int output_size, float learning_rate) {
    
    float *d_output_error, *d_output_delta, *d_hidden_delta;
    cudaMalloc(&d_output_error, output_size * sizeof(float));
    cudaMalloc(&d_output_delta, output_size * sizeof(float));
    cudaMalloc(&d_hidden_delta, hidden_size * sizeof(float));

    // Compute output gradients
    int block_size = 256;
    int num_blocks = (output_size + block_size - 1) / block_size;
    compute_output_gradients<<<num_blocks, block_size>>>(d_output, d_targets, d_output_error, 
                                                         d_output_delta, output_size);

    // Compute hidden gradients
    num_blocks = (hidden_size + block_size - 1) / block_size;
    compute_hidden_gradients<<<num_blocks, block_size>>>(d_hidden_output, d_output_weights, 
                                                         d_output_delta, d_hidden_delta, 
                                                         hidden_size, output_size);

    // Update output weights
    dim3 block_dim(16, 16);
    dim3 grid_dim((hidden_size + block_dim.x - 1) / block_dim.x, 
                  (output_size + block_dim.y - 1) / block_dim.y);
    update_weights<<<grid_dim, block_dim>>>(d_output_weights, d_output_delta, d_hidden_output, 
                                            learning_rate, hidden_size, output_size);

    // Update hidden weights
    grid_dim.x = (input_size + block_dim.x - 1) / block_dim.x;
    grid_dim.y = (hidden_size + block_dim.y - 1) / block_dim.y;
    update_weights<<<grid_dim, block_dim>>>(d_hidden_weights, d_hidden_delta, d_input, 
                                            learning_rate, input_size, hidden_size);

    // Update output biases
    num_blocks = (output_size + block_size - 1) / block_size;
    update_biases<<<num_blocks, block_size>>>(d_output_biases, d_output_delta, learning_rate, output_size);

    // Update hidden biases
    num_blocks = (hidden_size + block_size - 1) / block_size;
    update_biases<<<num_blocks, block_size>>>(d_hidden_biases, d_hidden_delta, learning_rate, hidden_size);

    // Clean up
    cudaFree(d_output_error);
    cudaFree(d_output_delta);
    cudaFree(d_hidden_delta);
}