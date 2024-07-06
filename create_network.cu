#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>



// Add bias and apply activation function
__global__ void addBiasAndActivate(float* input, float* bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = sigmoid(input[idx] + bias[idx]);
    }
}

// Initialize weights with Xavier/Glorot initialization
__global__ void initializeWeights(float* weights, int fan_in, int fan_out, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = fan_in * fan_out;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float range = sqrtf(6.0f / (fan_in + fan_out));
        weights[idx] = curand_uniform(&state) * 2 * range - range;
    }
}

// Initialize biases to zero
__global__ void initializeBiases(float* biases, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        biases[idx] = 0.0f;
    }
}

// Network structure
struct NeuralNetwork {
    int input_size;
    int hidden_size;
    int output_size;
    float *d_hidden_weights, *d_output_weights;
    float *d_hidden_biases, *d_output_biases;
    float *d_hidden_output, *d_output;
};

// Initialize the neural network
void initializeNetwork(NeuralNetwork& nn, int input_size, int hidden_size, int output_size) {
    nn.input_size = input_size;
    nn.hidden_size = hidden_size;
    nn.output_size = output_size;

    // Allocate memory for weights and biases
    cudaMalloc(&nn.d_hidden_weights, input_size * hidden_size * sizeof(float));
    cudaMalloc(&nn.d_output_weights, hidden_size * output_size * sizeof(float));
    cudaMalloc(&nn.d_hidden_biases, hidden_size * sizeof(float));
    cudaMalloc(&nn.d_output_biases, output_size * sizeof(float));

    // Allocate memory for layer outputs
    cudaMalloc(&nn.d_hidden_output, hidden_size * sizeof(float));
    cudaMalloc(&nn.d_output, output_size * sizeof(float));

    // Initialize weights
    int block_size = 256;
    int num_blocks;

    num_blocks = (input_size * hidden_size + block_size - 1) / block_size;
    initializeWeights<<<num_blocks, block_size>>>(nn.d_hidden_weights, input_size, hidden_size, time(NULL));

    num_blocks = (hidden_size * output_size + block_size - 1) / block_size;
    initializeWeights<<<num_blocks, block_size>>>(nn.d_output_weights, hidden_size, output_size, time(NULL));

    // Initialize biases
    num_blocks = (hidden_size + block_size - 1) / block_size;
    initializeBiases<<<num_blocks, block_size>>>(nn.d_hidden_biases, hidden_size);

    num_blocks = (output_size + block_size - 1) / block_size;
    initializeBiases<<<num_blocks, block_size>>>(nn.d_output_biases, output_size);
}

// Forward pass
void forwardPass(NeuralNetwork& nn, float* d_input) {
    dim3 block_size(16, 16);
    dim3 grid_size;

    // Hidden layer
    grid_size.x = (nn.hidden_size + block_size.x - 1) / block_size.x;
    grid_size.y = (1 + block_size.y - 1) / block_size.y;
    matrixMultiply<<<grid_size, block_size>>>(d_input, nn.d_hidden_weights, nn.d_hidden_output, 
                                              1, nn.input_size, nn.hidden_size);

    int num_blocks = (nn.hidden_size + 256 - 1) / 256;
    addBiasAndActivate<<<num_blocks, 256>>>(nn.d_hidden_output, nn.d_hidden_biases, nn.hidden_size);

    // Output layer
    grid_size.x = (nn.output_size + block_size.x - 1) / block_size.x;
    grid_size.y = (1 + block_size.y - 1) / block_size.y;
    matrixMultiply<<<grid_size, block_size>>>(nn.d_hidden_output, nn.d_output_weights, nn.d_output, 
                                              1, nn.hidden_size, nn.output_size);

    num_blocks = (nn.output_size + 256 - 1) / 256;
    addBiasAndActivate<<<num_blocks, 256>>>(nn.d_output, nn.d_output_biases, nn.output_size);
}

// Backward pass
void backwardPass(NeuralNetwork& nn, float* d_input, float* d_targets, float learning_rate) {
    backpropagation(d_input, (&nn)->d_hidden_output, (&nn)->d_output, d_targets,
                     (&nn)->d_hidden_weights, (&nn)->d_output_weights,
                     (&nn)->d_hidden_biases, (&nn)->d_output_biases,
                     (&nn)->input_size, (&nn)->hidden_size, (&nn)->output_size, learning_rate)
}

// Function to train the network
void trainNetwork(NeuralNetwork& nn, float* d_input, float* d_targets, int num_samples, float learning_rate, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < num_samples; ++i) {
            forwardPass(nn, d_input + i * nn.input_size);
            backwardPass(nn, d_input + i * nn.input_size, d_targets + i * nn.output_size, learning_rate);
        }
    }
}